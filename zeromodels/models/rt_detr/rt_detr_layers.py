import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="zeromodels")
class RTDETRSinePositionEmbedding(layers.Layer):
    """2D sinusoidal position embedding for the RT-DETR hybrid encoder.

    Generates non-learnable sine/cosine positional encodings from a 2D
    spatial grid. The embedding dimension is split into four equal parts
    encoding height-sin, height-cos, width-sin, and width-cos.

    Reference:
        - [RT-DETR](https://arxiv.org/abs/2304.08069)

    Args:
        embed_dim: Integer, total embedding dimension. Must be
            divisible by 4. Defaults to `256`.
        temperature: Integer, temperature scaling factor.
            Defaults to `10000`.
        **kwargs: Additional keyword arguments passed to the `Layer`
            class.

    Input Shape:
        Tuple of two integers `(height, width)`.

    Output Shape:
        3D tensor: `(1, height * width, embed_dim)`.
    """

    def __init__(self, embed_dim=256, temperature=10000, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.temperature = temperature

    def call(self, height, width):
        pos_dim = self.embed_dim // 4
        dim_t = ops.cast(ops.arange(pos_dim), "float32") / pos_dim
        dim_t = 1.0 / (self.temperature**dim_t)

        grid_w = ops.cast(ops.arange(width), "float32")
        grid_h = ops.cast(ops.arange(height), "float32")
        grid_w, grid_h = ops.meshgrid(grid_w, grid_h)

        out_w = ops.reshape(grid_w, [-1, 1]) * ops.reshape(dim_t, [1, -1])
        out_h = ops.reshape(grid_h, [-1, 1]) * ops.reshape(dim_t, [1, -1])

        pos = ops.concatenate(
            [ops.sin(out_h), ops.cos(out_h), ops.sin(out_w), ops.cos(out_w)],
            axis=-1,
        )
        return ops.expand_dims(pos, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "temperature": self.temperature,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class RTDETRMultiHeadAttention(layers.Layer):
    """Multi-head attention layer for the RT-DETR transformer.

    Implements scaled dot-product multi-head attention with separate
    query, key, and value projections. Position embeddings are
    optionally added to the query and key inputs before projection.

    Reference:
        - [RT-DETR](https://arxiv.org/abs/2304.08069)

    Args:
        hidden_dim: Integer, total model dimension.
        num_heads: Integer, number of parallel attention heads.
        dropout_rate: Float, dropout on attention weights. Defaults to `0.0`.
        block_prefix: String, name prefix for sub-layers.
        **kwargs: Additional keyword arguments.

    Input Shape:
        - query: `(batch_size, seq_len_q, hidden_dim)`
        - key:   `(batch_size, seq_len_k, hidden_dim)`
        - value: `(batch_size, seq_len_k, hidden_dim)`

    Output Shape:
        3D tensor: `(batch_size, seq_len_q, hidden_dim)`.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        dropout_rate=0.0,
        block_prefix="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout_rate = dropout_rate
        self.block_prefix = block_prefix

        self.q_proj = layers.Dense(hidden_dim, name=f"{block_prefix}_q_proj")
        self.k_proj = layers.Dense(hidden_dim, name=f"{block_prefix}_k_proj")
        self.v_proj = layers.Dense(hidden_dim, name=f"{block_prefix}_v_proj")
        self.out_proj = layers.Dense(hidden_dim, name=f"{block_prefix}_out_proj")
        self.attn_dropout = layers.Dropout(dropout_rate)

    def call(self, query, key, value, training=None):
        batch_size = ops.shape(query)[0]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = ops.reshape(q, [batch_size, -1, self.num_heads, self.head_dim])
        k = ops.reshape(k, [batch_size, -1, self.num_heads, self.head_dim])
        v = ops.reshape(v, [batch_size, -1, self.num_heads, self.head_dim])

        q = ops.transpose(q, [0, 2, 1, 3])
        k = ops.transpose(k, [0, 2, 1, 3])
        v = ops.transpose(v, [0, 2, 1, 3])

        attn_weights = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2])) * self.scale
        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        attn_output = ops.matmul(attn_weights, v)
        attn_output = ops.transpose(attn_output, [0, 2, 1, 3])
        attn_output = ops.reshape(attn_output, [batch_size, -1, self.hidden_dim])
        return self.out_proj(attn_output)

    def compute_output_spec(self, query, key, value, **kwargs):
        if not self.q_proj.built:
            self.q_proj.build(query.shape)
        if not self.k_proj.built:
            self.k_proj.build(key.shape)
        if not self.v_proj.built:
            self.v_proj.build(value.shape)
        if not self.out_proj.built:
            self.out_proj.build(query.shape)
        return keras.KerasTensor(query.shape, dtype=query.dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "block_prefix": self.block_prefix,
            }
        )
        return config


def ms_deform_attn_core(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """Pure implementation of multi-scale deformable attention core.

    Args:
        value: (B, n_heads, head_dim, N_total)
        value_spatial_shapes: list of (H, W) tuples
        sampling_locations: (B, Len_q, n_heads, L, P, 2) in [0, 1]
        attention_weights: (B, Len_q, n_heads, L*P) after softmax
    """
    B = ops.shape(value)[0]
    n_heads = ops.shape(value)[1]
    head_dim = ops.shape(value)[2]
    Len_q = ops.shape(sampling_locations)[1]
    L = len(value_spatial_shapes)
    P = ops.shape(sampling_locations)[4]

    sampling_grids = 2 * sampling_locations - 1

    sizes = [h * w for h, w in value_spatial_shapes]
    split_indices = []
    cum = 0
    for s in sizes[:-1]:
        cum += s
        split_indices.append(cum)
    value_list = ops.split(value, split_indices, axis=3)

    sampling_value_list = []
    for lid, (H, W) in enumerate(value_spatial_shapes):
        value_l = ops.reshape(value_list[lid], [B * n_heads, head_dim, H, W])
        value_l = ops.transpose(value_l, [0, 2, 3, 1])
        val_flat = ops.reshape(value_l, [B * n_heads, H * W, head_dim])

        grid_l = sampling_grids[:, :, :, lid, :, :]
        grid_l = ops.transpose(grid_l, [0, 2, 1, 3, 4])
        grid_l = ops.reshape(grid_l, [B * n_heads, Len_q, P, 2])

        grid_x = grid_l[..., 0]
        grid_y = grid_l[..., 1]

        W_f = ops.cast(W, grid_x.dtype)
        H_f = ops.cast(H, grid_y.dtype)
        ix = ((grid_x + 1) * W_f - 1) / 2.0
        iy = ((grid_y + 1) * H_f - 1) / 2.0

        ix0 = ops.cast(ops.floor(ix), "int32")
        iy0 = ops.cast(ops.floor(iy), "int32")
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        fx = ix - ops.cast(ix0, ix.dtype)
        fy = iy - ops.cast(iy0, iy.dtype)

        valid_00 = ops.cast((ix0 >= 0) & (ix0 < W) & (iy0 >= 0) & (iy0 < H), ix.dtype)
        valid_01 = ops.cast((ix1 >= 0) & (ix1 < W) & (iy0 >= 0) & (iy0 < H), ix.dtype)
        valid_10 = ops.cast((ix0 >= 0) & (ix0 < W) & (iy1 >= 0) & (iy1 < H), ix.dtype)
        valid_11 = ops.cast((ix1 >= 0) & (ix1 < W) & (iy1 >= 0) & (iy1 < H), ix.dtype)

        ix0_c = ops.clip(ix0, 0, W - 1)
        ix1_c = ops.clip(ix1, 0, W - 1)
        iy0_c = ops.clip(iy0, 0, H - 1)
        iy1_c = ops.clip(iy1, 0, H - 1)

        def _gather(val_flat, iy, ix, BN, Len_q, P, H, W, head_dim):
            idx = iy * W + ix
            idx_flat = ops.reshape(idx, [BN, Len_q * P])
            idx_flat = ops.expand_dims(idx_flat, axis=-1)
            idx_flat = ops.repeat(idx_flat, head_dim, axis=-1)
            gathered = ops.take_along_axis(val_flat, idx_flat, axis=1)
            return ops.reshape(gathered, [BN, Len_q, P, head_dim])

        BN = B * n_heads
        v00 = _gather(val_flat, iy0_c, ix0_c, BN, Len_q, P, H, W, head_dim)
        v01 = _gather(val_flat, iy0_c, ix1_c, BN, Len_q, P, H, W, head_dim)
        v10 = _gather(val_flat, iy1_c, ix0_c, BN, Len_q, P, H, W, head_dim)
        v11 = _gather(val_flat, iy1_c, ix1_c, BN, Len_q, P, H, W, head_dim)

        v00 = v00 * ops.expand_dims(valid_00, axis=-1)
        v01 = v01 * ops.expand_dims(valid_01, axis=-1)
        v10 = v10 * ops.expand_dims(valid_10, axis=-1)
        v11 = v11 * ops.expand_dims(valid_11, axis=-1)

        fx = ops.expand_dims(fx, axis=-1)
        fy = ops.expand_dims(fy, axis=-1)

        w00 = (1 - fx) * (1 - fy)
        w01 = fx * (1 - fy)
        w10 = (1 - fx) * fy
        w11 = fx * fy

        sampled = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11
        sampled = ops.transpose(sampled, [0, 3, 1, 2])
        sampling_value_list.append(sampled)

    sampling_values = ops.stack(sampling_value_list, axis=-2)
    sampling_values = ops.reshape(
        sampling_values, [B * n_heads, head_dim, Len_q, L * P]
    )

    attn = ops.transpose(attention_weights, [0, 2, 1, 3])
    attn = ops.reshape(attn, [B * n_heads, 1, Len_q, L * P])

    output = ops.sum(sampling_values * attn, axis=-1)
    output = ops.reshape(output, [B, n_heads * head_dim, Len_q])
    output = ops.transpose(output, [0, 2, 1])
    return output


@keras.saving.register_keras_serializable(package="zeromodels")
class RTDETRMultiScaleDeformableAttention(layers.Layer):
    """Multi-scale deformable attention for RT-DETR decoder cross-attention.

    Each query attends to a small set of learned sampling locations
    around reference points across multiple feature levels. Uses
    bilinear interpolation to sample values at continuous locations.

    Reference:
        - [Deformable DETR](https://arxiv.org/abs/2010.04159)
        - [RT-DETR](https://arxiv.org/abs/2304.08069)

    Args:
        hidden_dim: Integer, model dimension. Defaults to `256`.
        n_levels: Integer, number of feature levels. Defaults to `3`.
        n_heads: Integer, number of attention heads. Defaults to `8`.
        n_points: Integer, sampling points per head per level. Defaults
            to `4`.
        spatial_shapes: List of (height, width) tuples.
        level_start_index: List of start indices per level.
        **kwargs: Additional keyword arguments.

    Input Shape:
        - query: `(batch_size, num_queries, hidden_dim)`
        - reference_points: `(batch_size, num_queries, n_levels, 2 or 4)`
        - input_flatten: `(batch_size, total_tokens, hidden_dim)`

    Output Shape:
        3D tensor: `(batch_size, num_queries, hidden_dim)`.
    """

    def __init__(
        self,
        hidden_dim=256,
        n_levels=3,
        n_heads=8,
        n_points=4,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.spatial_shapes = spatial_shapes or []
        self.level_start_index = level_start_index or [0]

        self.sampling_offsets = layers.Dense(
            self.n_heads * self.n_levels * self.n_points * 2,
            name="sampling_offsets",
        )
        self.attention_weights_proj = layers.Dense(
            self.n_heads * self.n_levels * self.n_points,
            name="attention_weights",
        )
        self.value_proj = layers.Dense(self.hidden_dim, name="value_proj")
        self.output_proj = layers.Dense(self.hidden_dim, name="output_proj")

    def call(
        self,
        query,
        reference_points,
        input_flatten,
        position_embeddings=None,
    ):
        if position_embeddings is not None:
            query = query + position_embeddings

        input_spatial_shapes = self.spatial_shapes
        N = ops.shape(query)[0]
        Len_q = ops.shape(query)[1]
        head_dim = self.hidden_dim // self.n_heads

        value = self.value_proj(input_flatten)
        value = ops.reshape(value, [N, -1, self.n_heads, head_dim])
        value = ops.transpose(value, [0, 2, 3, 1])

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = ops.reshape(
            sampling_offsets,
            [N, Len_q, self.n_heads, self.n_levels, self.n_points, 2],
        )

        attention_weights = self.attention_weights_proj(query)
        attention_weights = ops.reshape(
            attention_weights,
            [N, Len_q, self.n_heads, self.n_levels * self.n_points],
        )
        attention_weights = ops.softmax(attention_weights, axis=-1)

        num_coords = reference_points.shape[-1]
        if num_coords == 2:
            spatial_shapes_wh = [[w, h] for h, w in input_spatial_shapes]
            offset_normalizer = ops.cast(
                ops.convert_to_tensor(spatial_shapes_wh, dtype="float32"),
                sampling_offsets.dtype,
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif num_coords == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"reference_points last dim must be 2 or 4, got {num_coords}"
            )

        output = ms_deform_attn_core(
            value, input_spatial_shapes, sampling_locations, attention_weights
        )
        return self.output_proj(output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "n_levels": self.n_levels,
                "n_heads": self.n_heads,
                "n_points": self.n_points,
                "spatial_shapes": self.spatial_shapes,
                "level_start_index": self.level_start_index,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class RTDETRDecoderLayer(layers.Layer):
    """Single RT-DETR decoder layer with self-attention, deformable
    cross-attention, and feedforward network.

    Uses post-norm architecture: residual -> dropout -> add -> layernorm.

    Reference:
        - [RT-DETR](https://arxiv.org/abs/2304.08069)

    Args:
        hidden_dim: Integer, model dimension. Defaults to `256`.
        num_heads: Integer, attention heads. Defaults to `8`.
        dim_feedforward: Integer, FFN intermediate dim. Defaults to `1024`.
        dropout_rate: Float, dropout rate. Defaults to `0.0`.
        activation: String, FFN activation. Defaults to `"relu"`.
        n_levels: Integer, feature levels for deformable attn. Defaults
            to `3`.
        n_points: Integer, sampling points. Defaults to `4`.
        spatial_shapes: List of (H, W) tuples.
        level_start_index: List of start indices.
        block_prefix: String, name prefix.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        hidden_dim=256,
        num_heads=8,
        dim_feedforward=1024,
        dropout_rate=0.0,
        activation="relu",
        n_levels=3,
        n_points=4,
        spatial_shapes=None,
        level_start_index=None,
        block_prefix="decoder_layers_0",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.n_levels = n_levels
        self.n_points = n_points
        self.spatial_shapes = spatial_shapes or []
        self.level_start_index = level_start_index or [0]
        self.block_prefix = block_prefix

        bp = block_prefix
        self.self_attn = RTDETRMultiHeadAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout_rate=0.0,
            block_prefix=f"{bp}_self_attn",
            name=f"{bp}_self_attn",
        )
        self.self_attn_layer_norm = layers.LayerNormalization(
            epsilon=1e-5,
            name=f"{bp}_self_attn_layer_norm",
        )

        self.encoder_attn = RTDETRMultiScaleDeformableAttention(
            hidden_dim=self.hidden_dim,
            n_levels=self.n_levels,
            n_heads=self.num_heads,
            n_points=self.n_points,
            spatial_shapes=self.spatial_shapes,
            level_start_index=self.level_start_index,
            name=f"{bp}_encoder_attn",
        )
        self.encoder_attn_layer_norm = layers.LayerNormalization(
            epsilon=1e-5,
            name=f"{bp}_encoder_attn_layer_norm",
        )

        self.fc1 = layers.Dense(
            self.dim_feedforward,
            activation=self.activation,
            name=f"{bp}_fc1",
        )
        self.fc2 = layers.Dense(self.hidden_dim, name=f"{bp}_fc2")
        self.final_layer_norm = layers.LayerNormalization(
            epsilon=1e-5,
            name=f"{bp}_final_layer_norm",
        )

    def call(
        self,
        hidden_states,
        encoder_hidden_states,
        query_pos,
        reference_points,
        training=None,
    ):
        q = k = hidden_states + query_pos
        residual = hidden_states
        attn_out = self.self_attn(q, k, hidden_states, training=training)
        hidden_states = self.self_attn_layer_norm(residual + attn_out)

        residual = hidden_states
        cross_out = self.encoder_attn(
            hidden_states,
            reference_points,
            encoder_hidden_states,
            position_embeddings=query_pos,
        )
        hidden_states = self.encoder_attn_layer_norm(residual + cross_out)

        residual = hidden_states
        ff_out = self.fc2(self.fc1(hidden_states))
        hidden_states = self.final_layer_norm(residual + ff_out)

        return hidden_states

    def compute_output_spec(self, *args, **kwargs):
        return keras.KerasTensor(args[0].shape, dtype=args[0].dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "dim_feedforward": self.dim_feedforward,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
                "n_levels": self.n_levels,
                "n_points": self.n_points,
                "spatial_shapes": self.spatial_shapes,
                "level_start_index": self.level_start_index,
                "block_prefix": self.block_prefix,
            }
        )
        return config
