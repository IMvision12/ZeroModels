import keras
from keras import initializers, layers, ops

from zeromodels.base.base_attention import fused_attention


@keras.saving.register_keras_serializable(package="zeromodels")
class SigLIPAttention(keras.Layer):
    """Multi-head attention layer for SigLip model.

    This layer implements scaled dot-product multi-head attention with optional
    combined query-key-value projection for efficiency. It supports both self-attention
    and cross-attention patterns.

    Args:
        num_heads (int): Number of attention heads.
        hidden_dim (int): Dimension of each attention head.
        attention_dropout (float, optional): Dropout rate for attention weights.
            Defaults to 0.0.
        use_bias (bool, optional): Whether to use bias in linear projections.
            Defaults to True.
        combined_qkv (bool, optional): Whether to use a single combined projection
            for query, key, and value instead of separate projections. Can improve
            efficiency. Defaults to False.
        block_prefix (str, optional): Prefix for layer names. Defaults to
            "multi_head_attention".
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Attributes:
        num_heads (int): Number of attention heads.
        hidden_dim (int): Dimension of each attention head.
        dim (int): Total dimension (num_heads * hidden_dim).
        scale (float): Scaling factor for attention scores (1/sqrt(hidden_dim)).

    Returns:
        Tensor: Attention output of shape (batch_size, seq_len, dim).

    Example:
        >>> attention = SigLipAttention(
        ...     num_heads=8,
        ...     hidden_dim=64,
        ...     attention_dropout=0.1
        ... )
        >>> output = attention(inputs)  # Self-attention
        >>> output = attention(query, key=key, value=value)  # Cross-attention
    """

    def __init__(
        self,
        num_heads,
        hidden_dim,
        attention_dropout=0.0,
        use_bias=True,
        combined_qkv=False,
        block_prefix="multi_head_attention",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.attention_dropout = attention_dropout
        self.use_bias = use_bias
        self.combined_qkv = combined_qkv
        self.block_prefix = block_prefix

        self.dim = num_heads * hidden_dim
        self.scale = hidden_dim**-0.5

        self.block_prefix = block_prefix if block_prefix is not None else "blocks"
        prefix = f"{self.block_prefix}_"

        if combined_qkv:
            self.in_proj = layers.Dense(
                3 * self.dim,
                use_bias=use_bias,
                dtype=self.dtype_policy,
                name=prefix + "in_proj",
            )
        else:
            self.q_proj = layers.Dense(
                self.dim,
                use_bias=use_bias,
                dtype=self.dtype_policy,
                name=prefix + "q_proj",
            )
            self.k_proj = layers.Dense(
                self.dim,
                use_bias=use_bias,
                dtype=self.dtype_policy,
                name=prefix + "k_proj",
            )
            self.v_proj = layers.Dense(
                self.dim,
                use_bias=use_bias,
                dtype=self.dtype_policy,
                name=prefix + "v_proj",
            )

        self.out_proj = layers.Dense(
            self.dim,
            use_bias=use_bias,
            dtype=self.dtype_policy,
            name=prefix + "out_proj",
        )

        if attention_dropout > 0.0:
            self.dropout = layers.Dropout(
                attention_dropout,
                dtype=self.dtype_policy,
            )
        else:
            self.dropout = None

    def call(self, inputs, key=None, value=None, training=None):
        if key is None:
            key = inputs
        if value is None:
            value = inputs

        batch_size = ops.shape(inputs)[0]

        if self.combined_qkv:
            q_proj = self.in_proj(inputs)
            k_proj = self.in_proj(key)
            v_proj = self.in_proj(value)

            q = q_proj[..., : self.dim]
            k = k_proj[..., self.dim : 2 * self.dim]
            v = v_proj[..., 2 * self.dim :]
        else:
            q = self.q_proj(inputs)
            k = self.k_proj(key)
            v = self.v_proj(value)

        q = ops.reshape(q, (batch_size, -1, self.num_heads, self.hidden_dim))
        k = ops.reshape(k, (batch_size, -1, self.num_heads, self.hidden_dim))
        v = ops.reshape(v, (batch_size, -1, self.num_heads, self.hidden_dim))

        q = ops.transpose(q, axes=[0, 2, 1, 3])
        k = ops.transpose(k, axes=[0, 2, 1, 3])
        v = ops.transpose(v, axes=[0, 2, 1, 3])

        attn_output = fused_attention(
            q, k, v, self.scale, dropout=self.dropout, training=training
        )
        attn_output = ops.transpose(attn_output, axes=[0, 2, 1, 3])
        attn_output = ops.reshape(attn_output, (batch_size, -1, self.dim))
        output = self.out_proj(attn_output)

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "attention_dropout": self.attention_dropout,
                "use_bias": self.use_bias,
                "combined_qkv": self.combined_qkv,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SigLIPProbe(layers.Layer):
    """Learnable probe parameter layer.

    This layer creates a learnable parameter that can be used as a probe token
    or query vector in attention mechanisms. The probe is initialized using
    Glorot uniform initialization and is repeated across the batch dimension
    during the forward pass.

    Args:
        hidden_dim (int): Dimension of the probe vector.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Attributes:
        hidden_dim (int): Dimension of the probe vector.
        probe (Variable): Learnable probe parameter of shape (1, 1, hidden_dim).

    Returns:
        Tensor: SigLIPProbe tensor repeated for each batch element,
               shape (batch_size, 1, hidden_dim).

    Example:
        >>> probe_layer = SigLIPProbe(hidden_dim=512)
        >>> probe_tokens = probe_layer(inputs)  # Shape: (batch_size, 1, 512)
    """

    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.probe = self.add_weight(
            shape=(1, 1, self.hidden_dim),
            initializer=initializers.GlorotUniform(),
            dtype=self.dtype_policy.variable_dtype,
            name="probe",
        )

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        return ops.repeat(self.probe, repeats=batch_size, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


@keras.saving.register_keras_serializable(package="siglip")
class SigLIPPositionIDs(layers.Layer):
    """A Keras layer that generates position IDs for vision transformers.

    This layer creates position identifiers for a 2D grid of patches, commonly used
    in vision transformers for spatial positional encoding. It supports both 1D
    sequential position IDs and 2D coordinate-based position IDs.

    Args:
        grid_h (int): Height of the grid (number of patches vertically).
        grid_w (int): Width of the grid (number of patches horizontally).
        use_2d_positions (bool, optional): Whether to generate 2D coordinates
            (height, width) for each position. If False, generates sequential
            1D position IDs. Defaults to False.
        name (str, optional): Name of the layer.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Attributes:
        grid_h (int): Height of the grid.
        grid_w (int): Width of the grid.
        use_2d_positions (bool): Whether using 2D or 1D position encoding.
        max_length (int): Total number of positions (grid_h * grid_w).
        position_ids (tf.Variable): The position ID tensor.

    Output shape:
        - If use_2d_positions=False: (1, max_length)
        - If use_2d_positions=True: (1, max_length, 2)

    Example:
        ```python
        # Create 1D position IDs for a 14x14 grid
        pos_layer = SigLIPPositionIDs(grid_h=14, grid_w=14, use_2d_positions=False)

        # Create 2D coordinate position IDs for a 16x16 grid
        pos_layer_2d = SigLIPPositionIDs(grid_h=16, grid_w=16, use_2d_positions=True)
        ```
    """

    def __init__(self, grid_h, grid_w, use_2d_positions=False, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.use_2d_positions = use_2d_positions
        self.max_length = self.grid_h * self.grid_w

    def build(self, input_shape):
        if self.use_2d_positions:
            output_shape = (1, self.max_length, 2)
        else:
            output_shape = (1, self.max_length)

        self.position_ids = self.add_weight(
            shape=output_shape,
            initializer="zeros",
            dtype="int32",
            trainable=False,
            name="position_ids",
        )

        self._initialize_position_ids()
        super().build(input_shape)

    def _initialize_position_ids(self):
        if self.use_2d_positions:
            h_coords = ops.repeat(
                ops.arange(self.grid_h, dtype="int32"), repeats=self.grid_w
            )
            w_coords = ops.tile(
                ops.arange(self.grid_w, dtype="int32"), multiples=[self.grid_h]
            )
            coords = ops.stack([h_coords, w_coords], axis=1)
            coords = ops.expand_dims(coords, axis=0)
            self.position_ids.assign(coords)
        else:
            position_indices = ops.expand_dims(
                ops.arange(0, self.max_length, dtype="int32"), axis=0
            )
            self.position_ids.assign(position_indices)

    def call(self, inputs):
        return self.position_ids

    def compute_output_shape(self, input_shape):
        if self.use_2d_positions:
            return (1, self.max_length, 2)
        else:
            return (1, self.max_length)

    def save_own_variables(self, store):
        super().save_own_variables(store)
        store["grid_h"] = self.grid_h
        store["grid_w"] = self.grid_w
        store["use_2d_positions"] = self.use_2d_positions
        store["max_length"] = self.max_length

    def load_own_variables(self, store):
        self._initialize_position_ids()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "grid_h": self.grid_h,
                "grid_w": self.grid_w,
                "use_2d_positions": self.use_2d_positions,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SigLIPLogitScaleBias(layers.Layer):
    """Learnable logit scaling and bias layer for contrastive learning.

    This layer applies learnable scaling and bias to similarity matrices, commonly
    used in contrastive learning frameworks like CLIP and SigLip. The scaling
    parameter is initialized as log(1.0) and the bias as zero.

    Attributes:
        logit_scale (Variable): Learnable scaling parameter (stored as log value).
        logit_bias (Variable): Learnable bias parameter.

    Returns:
        Tensor: Scaled and biased similarity matrix of the same shape as input.

    Example:
        >>> scale_bias = SigLIPLogitScaleBias()
        >>> scaled_logits = scale_bias(similarity_matrix)
    """

    def build(self, input_shape):
        self.logit_scale = self.add_weight(
            shape=(),
            initializer=initializers.Constant(ops.log(1.0)),
            trainable=True,
            dtype=self.variable_dtype,
            name="logit_scale",
        )
        self.logit_bias = self.add_weight(
            shape=(),
            initializer=initializers.Zeros(),
            trainable=True,
            dtype=self.variable_dtype,
            name="logit_bias",
        )

    def call(self, similarity_matrix):
        scaled_logits = ops.multiply(similarity_matrix, ops.exp(self.logit_scale))
        return ops.add(scaled_logits, self.logit_bias)


@keras.saving.register_keras_serializable(package="siglip")
class SigLIPPositionEmbedding(layers.Layer):
    """
    Position embedding layer that can handle different grid sizes through interpolation.

    This layer creates learnable position embeddings that can be interpolated to handle
    different input sizes. It supports both 1D and 2D interpolation modes:
    - 1D interpolation: Linear interpolation for sequence-based positions
    - 2D interpolation: Bilinear interpolation for grid-based positions (e.g., image patches)

    The layer automatically detects whether to use 1D or 2D interpolation based on
    whether the position count forms a perfect square (indicating a 2D grid).

    Args:
        max_positions (int): Maximum number of positions to embed. For 2D grids,
            this should be grid_height * grid_width.
        embed_dim (int): Dimensionality of the position embeddings.
        embeddings_initializer (str or keras.initializers.Initializer): Initializer
            for the embedding weights. Defaults to "random_normal".
        name (str, optional): Name of the layer.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Input Shape:
        Position indices tensor of any shape containing integer values in range
        [0, max_positions). Typically (batch_size, sequence_length) or
        (batch_size, height, width).

    Output Shape:
        Same shape as input with an additional dimension of size embed_dim.
        If input shape is (..., ), output shape is (..., embed_dim).

    Attributes:
        max_positions (int): Maximum number of positions.
        embed_dim (int): Dimension of embeddings.
        embeddings (tf.Variable): Learnable embedding weights of shape
            (max_positions, embed_dim).

    Examples:
        ```python
        # Basic usage for sequence positions
        pos_embed = SigLIPPositionEmbedding(max_positions=100, embed_dim=256)
        position_ids = tf.range(10)  # [0, 1, 2, ..., 9]
        embeddings = pos_embed(position_ids)  # Shape: (10, 256)

        # For 2D grid positions (e.g., 4x4 image patches)
        pos_embed_2d = SigLIPPositionEmbedding(max_positions=16, embed_dim=128)
        # Position IDs for a 4x4 grid: [0, 1, 2, ..., 15]
        grid_positions = tf.range(16)
        grid_embeddings = pos_embed_2d(grid_positions)  # Shape: (16, 128)

        # The layer automatically handles size changes during loading
        # If trained on 4x4 grid (16 positions) and loaded for 8x8 grid (64 positions),
        # it will interpolate the embeddings using 2D bilinear interpolation

        # Batch processing
        batch_positions = tf.constant([[0, 1, 2], [3, 4, 5]])  # Shape: (2, 3)
        batch_embeddings = pos_embed(batch_positions)  # Shape: (2, 3, 256)
        ```

    Note:
        - When loading pretrained weights with different max_positions, the layer
          automatically interpolates embeddings to match the new size
        - 2D interpolation is used when both source and target positions form
          perfect squares (indicating grid layouts)
        - 1D linear interpolation is used for all other cases
        - Embedding dimension must match between source and target - no interpolation
          is performed across the embedding dimension
    """

    def __init__(
        self,
        max_positions,
        embed_dim,
        embeddings_initializer="random_normal",
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.max_positions = max_positions
        self.embed_dim = embed_dim
        self.embeddings_initializer = embeddings_initializer

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.max_positions, self.embed_dim),
            initializer=self.embeddings_initializer,
            trainable=True,
            name="embeddings",
        )
        super().build(input_shape)

    def call(self, inputs):
        indices = ops.cast(inputs, "int32")
        return ops.take(self.embeddings, indices, axis=0)

    def compute_output_shape(self, input_shape):
        return input_shape + (self.embed_dim,)

    def compute_output_spec(self, input_spec):
        output_shape = input_spec.shape + (self.embed_dim,)
        return keras.KerasTensor(output_shape, dtype=self.compute_dtype)

    def save_own_variables(self, store):
        super().save_own_variables(store)
        store["max_positions"] = self.max_positions
        store["embed_dim"] = self.embed_dim

    def load_own_variables(self, store):
        try:
            source_max_positions = int(store["max_positions"][...])
            source_embedding_dim = int(store["embed_dim"][...])
        except (KeyError, TypeError):
            stored_weights = store["0"]
            source_max_positions, source_embedding_dim = stored_weights.shape

        stored_embeddings = store["0"]

        if (
            source_max_positions == self.max_positions
            and source_embedding_dim == self.embed_dim
        ):
            self.embeddings.assign(stored_embeddings)
            return

        if source_embedding_dim != self.embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embed_dim}, got {source_embedding_dim}"
            )

        if source_max_positions != self.max_positions:
            import math

            source_grid_size = int(math.sqrt(source_max_positions))
            target_grid_size = int(math.sqrt(self.max_positions))

            if (
                source_grid_size * source_grid_size == source_max_positions
                and target_grid_size * target_grid_size == self.max_positions
            ):
                interpolated = self._interpolate_2d_embeddings(
                    stored_embeddings,
                    (source_grid_size, source_grid_size),
                    (target_grid_size, target_grid_size),
                )
            else:
                interpolated = self._interpolate_1d_embeddings(
                    stored_embeddings, source_max_positions, self.max_positions
                )

            self.embeddings.assign(interpolated)
        else:
            self.embeddings.assign(stored_embeddings)

    def _interpolate_2d_embeddings(self, embeddings, source_shape, target_shape):
        source_h, source_w = source_shape
        target_h, target_w = target_shape
        embed_dim = embeddings.shape[-1]

        embeddings_2d = ops.reshape(embeddings, (source_h, source_w, embed_dim))
        embeddings_2d = ops.expand_dims(embeddings_2d, axis=0)
        interpolated = ops.image.resize(
            embeddings_2d, size=(target_h, target_w), interpolation="bilinear"
        )
        interpolated = ops.squeeze(interpolated, axis=0)
        interpolated = ops.reshape(interpolated, (target_h * target_w, embed_dim))

        return interpolated

    def _interpolate_1d_embeddings(self, embeddings, source_length, target_length):
        if source_length == target_length:
            return embeddings

        source_indices = ops.linspace(0.0, float(source_length - 1), target_length)
        source_indices_int = ops.cast(ops.floor(source_indices), "int32")
        source_indices_frac = source_indices - ops.cast(source_indices_int, "float32")
        source_indices_int = ops.clip(source_indices_int, 0, source_length - 2)
        source_indices_int_next = ops.clip(source_indices_int + 1, 0, source_length - 1)
        embeddings_curr = ops.take(embeddings, source_indices_int, axis=0)
        embeddings_next = ops.take(embeddings, source_indices_int_next, axis=0)
        source_indices_frac = ops.expand_dims(source_indices_frac, axis=1)
        interpolated = embeddings_curr + source_indices_frac * (
            embeddings_next - embeddings_curr
        )

        return interpolated

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_positions": self.max_positions,
                "embed_dim": self.embed_dim,
                "embeddings_initializer": keras.utils.serialize_keras_object(
                    self.embeddings_initializer
                ),
            }
        )
        return config
