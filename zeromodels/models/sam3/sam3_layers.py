import math

import keras
from keras import layers, ops

from .sam3_utils import apply_rotary_pos_emb_2d


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM3LearnableEmbedding(layers.Layer):
    """Zero-initialized learnable embedding broadcast across the batch.

    Used for query embeddings, reference points, and presence tokens in
    the DETR decoder. Optionally applies sigmoid to the embeddings.

    Args:
        num_embeddings (int): Number of embedding vectors.
        embed_dim (int): Dimension of each embedding vector.
        apply_sigmoid (bool): Whether to apply sigmoid activation.
            Defaults to ``False``.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """

    def __init__(self, num_embeddings, embed_dim, apply_sigmoid=False, **kwargs):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embed_dim = embed_dim
        self.apply_sigmoid = apply_sigmoid

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.num_embeddings, self.embed_dim),
            initializer="zeros",
        )
        self.built = True

    def call(self, batch_ref):
        batch_size = ops.shape(batch_ref)[0]
        emb = self.embeddings
        if self.apply_sigmoid:
            emb = ops.sigmoid(emb)
        emb = ops.expand_dims(emb, 0)
        return ops.broadcast_to(emb, (batch_size,) + ops.shape(emb)[1:])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_embeddings": self.num_embeddings,
                "embed_dim": self.embed_dim,
                "apply_sigmoid": self.apply_sigmoid,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM3AddPositionEmbedding(layers.Layer):
    """Learnable absolute position embedding for the ViT backbone.

    Stores a flat ``(num_patches, hidden_dim)`` embedding table
    initialized at the pretrained grid resolution. When the inference
    grid differs from the pretrained grid, embeddings are tiled and
    cropped to match.

    Args:
        num_patches (int): Number of pretrained position embeddings.
        hidden_dim (int): Embedding dimension.
        pretrain_grid (int): Spatial grid size used during pretraining.
        grid_size (int): Spatial grid size at inference time.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """

    def __init__(self, num_patches, hidden_dim, pretrain_grid, grid_size, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.pretrain_grid = pretrain_grid
        self.grid_size = grid_size

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.num_patches, self.hidden_dim),
            initializer="zeros",
        )
        self.built = True

    def call(self, x):
        pos = ops.reshape(
            self.embeddings,
            (1, self.pretrain_grid, self.pretrain_grid, self.hidden_dim),
        )
        if self.grid_size != self.pretrain_grid:
            pos = ops.transpose(pos, (0, 3, 1, 2))
            repeat_h = self.grid_size // self.pretrain_grid + 1
            repeat_w = self.grid_size // self.pretrain_grid + 1
            pos = ops.tile(pos, (1, 1, repeat_h, repeat_w))
            pos = pos[:, :, : self.grid_size, : self.grid_size]
            pos = ops.transpose(pos, (0, 2, 3, 1))
        pos = ops.reshape(pos, (1, self.grid_size * self.grid_size, self.hidden_dim))
        return x + pos

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_patches": self.num_patches,
                "hidden_dim": self.hidden_dim,
                "pretrain_grid": self.pretrain_grid,
                "grid_size": self.grid_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM3ViTRoPEAttention(layers.Layer):
    """Multi-head self-attention with 2-D rotary position embeddings.

    Applies RoPE to queries and keys before computing scaled dot-product
    attention. Used in the ViT backbone of SAM3.

    Args:
        hidden_dim (int): Input and output feature dimension.
        num_heads (int): Number of attention heads.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """

    def __init__(self, hidden_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

    def build(self, input_shape):
        self.q_proj = layers.Dense(self.hidden_dim, name="q_proj")
        self.q_proj.build(input_shape)
        self.k_proj = layers.Dense(self.hidden_dim, name="k_proj")
        self.k_proj.build(input_shape)
        self.v_proj = layers.Dense(self.hidden_dim, name="v_proj")
        self.v_proj.build(input_shape)
        self.o_proj = layers.Dense(self.hidden_dim, name="o_proj")
        self.o_proj.build(input_shape)
        self.built = True

    def call(self, hidden_states, position_embeddings):
        cos, sin = position_embeddings
        shape = ops.shape(hidden_states)
        batch_size = shape[0]
        seq_len = shape[1]

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ops.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = ops.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        cos = ops.expand_dims(ops.expand_dims(cos, 0), 0)
        sin = ops.expand_dims(ops.expand_dims(sin, 0), 0)
        q, k = apply_rotary_pos_emb_2d(q, k, cos, sin)

        attn_weights = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_output = ops.matmul(attn_weights, v)

        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(attn_output, (batch_size, seq_len, self.hidden_dim))
        attn_output = self.o_proj(attn_output)
        return attn_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM3ViTLayerScale(layers.Layer):
    """Per-channel learnable scaling applied after attention and MLP.

    Multiplies the input by a trainable vector initialized to a small
    constant, stabilizing training in deep ViT models.

    Args:
        hidden_dim (int): Channel dimension.
        init_value (float): Initial value for the scaling vector.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """

    def __init__(self, hidden_dim, init_value, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.init_value = init_value

    def build(self, input_shape):
        self.lambda1 = self.add_weight(
            name="lambda1",
            shape=(self.hidden_dim,),
            initializer=keras.initializers.Constant(self.init_value),
            trainable=True,
        )
        self.built = True

    def call(self, x):
        return x * self.lambda1

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "init_value": self.init_value,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM3ViTLayer(layers.Layer):
    """Single ViT transformer block with optional windowed attention.

    Pre-norm architecture: LayerNorm -> RoPE Attention -> LayerNorm -> MLP.
    Supports windowed attention with padding for local processing, and
    optional LayerScale for training stability.

    Args:
        hidden_dim (int): Hidden dimension.
        num_heads (int): Number of attention heads.
        mlp_dim (int): MLP intermediate dimension.
        window_size (int): Window size for windowed attention. ``0`` means
            global attention. Defaults to ``0``.
        image_size (int): Spatial size of the feature map for RoPE
            frequency computation. Defaults to ``72``.
        layer_norm_eps (float): Epsilon for layer normalization.
            Defaults to ``1e-6``.
        layer_scale_init (float or None): Initial value for
            LayerScale. ``None`` disables LayerScale. Defaults to ``None``.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        mlp_dim,
        window_size=0,
        image_size=72,
        layer_norm_eps=1e-6,
        layer_scale_init=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.window_size = window_size
        self.image_size = image_size
        self.layer_norm_eps = layer_norm_eps
        self.layer_scale_init = layer_scale_init

    def build(self, input_shape):
        seq_shape = (None, None, self.hidden_dim)

        self.layer_norm1 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm1"
        )
        self.layer_norm1.build(seq_shape)

        self.layer_norm2 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm2"
        )
        self.layer_norm2.build(seq_shape)

        self.attn = SAM3ViTRoPEAttention(
            self.hidden_dim, self.num_heads, name="attention"
        )
        self.attn.build(seq_shape)

        self.mlp_fc1 = layers.Dense(self.mlp_dim, name="mlp_fc1")
        self.mlp_fc1.build(seq_shape)
        self.mlp_fc2 = layers.Dense(self.hidden_dim, name="mlp_fc2")
        self.mlp_fc2.build((None, None, self.mlp_dim))

        if self.layer_scale_init is not None:
            self.layer_scale1 = SAM3ViTLayerScale(
                self.hidden_dim, self.layer_scale_init, name="layer_scale1"
            )
            self.layer_scale1.build(seq_shape)
            self.layer_scale2 = SAM3ViTLayerScale(
                self.hidden_dim, self.layer_scale_init, name="layer_scale2"
            )
            self.layer_scale2.build(seq_shape)

        self.built = True

    def _window_partition(self, x, window_size):
        shape = ops.shape(x)
        batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]

        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = ops.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])

        padded_h = height + pad_h
        padded_w = width + pad_w

        x = ops.reshape(
            x,
            (
                batch_size,
                padded_h // window_size,
                window_size,
                padded_w // window_size,
                window_size,
                channels,
            ),
        )
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        x = ops.reshape(x, (-1, window_size * window_size, channels))
        return x, (padded_h, padded_w)

    def _window_unpartition(self, windows, window_size, pad_hw, original_hw):
        padded_h, padded_w = pad_hw
        height, width = original_hw
        num_h = padded_h // window_size
        num_w = padded_w // window_size

        channels = ops.shape(windows)[-1]
        batch_size = ops.shape(windows)[0] // (num_h * num_w)

        x = ops.reshape(
            windows, (batch_size, num_h, num_w, window_size, window_size, channels)
        )
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        x = ops.reshape(x, (batch_size, padded_h, padded_w, channels))
        x = x[:, :height, :width, :]
        return x

    def call(self, hidden_states, cos, sin):
        shape = ops.shape(hidden_states)
        batch_size, height, width = shape[0], shape[1], shape[2]

        residual = hidden_states

        x = ops.reshape(hidden_states, (batch_size, height * width, self.hidden_dim))
        x = self.layer_norm1(x)

        if self.window_size > 0:
            x = ops.reshape(x, (batch_size, height, width, self.hidden_dim))
            x, pad_hw = self._window_partition(x, self.window_size)
        else:
            pass

        x = self.attn(x, (cos, sin))

        if self.window_size > 0:
            x = self._window_unpartition(x, self.window_size, pad_hw, (height, width))
            x = ops.reshape(x, (batch_size, height * width, self.hidden_dim))

        if self.layer_scale_init is not None:
            x = self.layer_scale1(x)

        x = ops.reshape(x, (batch_size, height, width, self.hidden_dim))
        hidden_states = residual + x

        residual = hidden_states
        x = ops.reshape(hidden_states, (batch_size, height * width, self.hidden_dim))
        x = self.layer_norm2(x)
        x = self.mlp_fc1(x)
        x = ops.nn.gelu(x, approximate=False)
        x = self.mlp_fc2(x)

        if self.layer_scale_init is not None:
            x = self.layer_scale2(x)

        x = ops.reshape(x, (batch_size, height, width, self.hidden_dim))
        hidden_states = residual + x

        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "window_size": self.window_size,
                "image_size": self.image_size,
                "layer_norm_eps": self.layer_norm_eps,
                "layer_scale_init": self.layer_scale_init,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM3MultiHeadAttention(layers.Layer):
    """Standard multi-head attention with optional key-value masking.

    Used in the DETR encoder, DETR decoder, geometry encoder, and mask
    decoder for self-attention and cross-attention operations.

    Args:
        hidden_dim (int): Input and output feature dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate applied to attention weights.
            Defaults to ``0.0``.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """

    def __init__(self, hidden_dim, num_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout_rate = dropout

    def build(self, input_shape):
        dim = self.hidden_dim
        self.q_proj = layers.Dense(dim, name="q_proj")
        self.q_proj.build((None, None, dim))
        self.k_proj = layers.Dense(dim, name="k_proj")
        self.k_proj.build((None, None, dim))
        self.v_proj = layers.Dense(dim, name="v_proj")
        self.v_proj.build((None, None, dim))
        self.o_proj = layers.Dense(dim, name="o_proj")
        self.o_proj.build((None, None, dim))
        self.attn_dropout = layers.Dropout(self.dropout_rate)
        self.built = True

    def call(self, query, key, value, attention_mask=None, training=None):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q_shape = ops.shape(q)
        k_shape = ops.shape(k)

        q = ops.reshape(q, (q_shape[0], q_shape[1], self.num_heads, self.head_dim))
        k = ops.reshape(k, (k_shape[0], k_shape[1], self.num_heads, self.head_dim))
        v = ops.reshape(v, (k_shape[0], k_shape[1], self.num_heads, self.head_dim))

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        attn_weights = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        attn_output = ops.matmul(attn_weights, v)
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(
            attn_output, (q_shape[0], q_shape[1], self.hidden_dim)
        )
        return self.o_proj(attn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout_rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM3GeometryEncoderLayer(layers.Layer):
    """Single transformer block for the geometry encoder.

    Pre-norm architecture with self-attention over prompt features,
    cross-attention to vision features, and a feed-forward MLP. Used
    to refine box prompt embeddings with vision context.

    Args:
        hidden_dim (int): Hidden dimension. Defaults to ``256``.
        num_heads (int): Number of attention heads.
            Defaults to ``8``.
        mlp_dim (int): MLP intermediate dimension.
            Defaults to ``2048``.
        dropout (float): Dropout rate. Defaults to ``0.1``.
        layer_norm_eps (float): Epsilon for layer normalization.
            Defaults to ``1e-6``.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """

    def __init__(
        self,
        hidden_dim=256,
        num_heads=8,
        mlp_dim=2048,
        dropout=0.1,
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout
        self.layer_norm_eps = layer_norm_eps

    def build(self, input_shape):
        dim = self.hidden_dim
        seq_shape = (None, None, dim)

        self.self_attn = SAM3MultiHeadAttention(
            dim, self.num_heads, self.dropout_rate, name="self_attn"
        )
        self.self_attn.build(seq_shape)
        self.layer_norm1 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm1"
        )
        self.layer_norm1.build(seq_shape)

        self.cross_attn = SAM3MultiHeadAttention(
            dim, self.num_heads, self.dropout_rate, name="cross_attn"
        )
        self.cross_attn.build(seq_shape)
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm2"
        )
        self.layer_norm2.build(seq_shape)

        self.fc1 = layers.Dense(self.mlp_dim, name="fc1")
        self.fc1.build(seq_shape)
        self.fc2 = layers.Dense(dim, name="fc2")
        self.fc2.build((None, None, self.mlp_dim))
        self.layer_norm3 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm3"
        )
        self.layer_norm3.build(seq_shape)

        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        self.dropout3 = layers.Dropout(self.dropout_rate)
        self.built = True

    def call(
        self,
        prompt_feats,
        vision_feats,
        vision_pos,
        prompt_mask=None,
        training=None,
    ):
        residual = prompt_feats
        x = self.layer_norm1(prompt_feats)
        x = self.self_attn(x, x, x, attention_mask=prompt_mask, training=training)
        prompt_feats = self.dropout1(x, training=training) + residual

        residual = prompt_feats
        x = self.layer_norm2(prompt_feats)
        k = vision_feats + vision_pos
        x = self.cross_attn(x, k, vision_feats, training=training)
        prompt_feats = self.dropout2(x, training=training) + residual

        residual = prompt_feats
        x = self.layer_norm3(prompt_feats)
        x = self.fc1(x)
        x = self.dropout3(x, training=training)
        x = ops.nn.relu(x)
        x = self.fc2(x)
        prompt_feats = self.dropout1(x, training=training) + residual

        return prompt_feats

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout_rate,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM3DecoderMLP(layers.Layer):
    """Multi-layer perceptron used in the DETR decoder heads.

    Stacks ``num_layers`` dense layers with ReLU activations on all
    but the last layer. Used for box regression, presence prediction,
    and reference point refinement.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden layer dimension.
        output_dim (int): Output dimension.
        num_layers (int): Number of dense layers. Defaults to ``2``.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

    def build(self, input_shape):
        dims = (
            [self.input_dim]
            + [self.hidden_dim] * (self.num_layers - 1)
            + [self.output_dim]
        )
        self.dense_layers = []
        for i in range(self.num_layers):
            dense = layers.Dense(dims[i + 1], name=f"dense_{i}")
            dense.build((None, dims[i]) if i == 0 else (None, None, dims[i]))
            self.dense_layers.append(dense)
        self.built = True

    def call(self, x):
        for i, layer in enumerate(self.dense_layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = ops.nn.relu(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM3BoxRPB(layers.Layer):
    """Box-conditioned relative position bias for vision cross-attention.

    Computes a spatially-aware attention bias based on the relative
    position of each spatial location to each reference box. The bias
    is added to attention logits in the DETR decoder's vision
    cross-attention layers.

    Args:
        hidden_dim (int): Feature dimension used to determine encoding
            size.
        num_heads (int): Number of attention heads for the
            output bias.
        spatial_h (int): Height of the spatial feature grid.
        spatial_w (int): Width of the spatial feature grid.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """

    def __init__(self, hidden_dim, num_heads, spatial_h, spatial_w, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w

    def build(self, input_shape):
        self.box_rpb_embed_x = SAM3DecoderMLP(
            2,
            self.hidden_dim,
            self.num_heads,
            num_layers=2,
            name="box_rpb_embed_x",
        )
        self.box_rpb_embed_x.build((None, None, None, 2))
        self.box_rpb_embed_y = SAM3DecoderMLP(
            2,
            self.hidden_dim,
            self.num_heads,
            num_layers=2,
            name="box_rpb_embed_y",
        )
        self.box_rpb_embed_y.build((None, None, None, 2))
        self.built = True

    def call(self, reference_boxes):
        cx = reference_boxes[..., 0:1]
        cy = reference_boxes[..., 1:2]
        w = reference_boxes[..., 2:3]
        h = reference_boxes[..., 3:4]
        x0 = cx - 0.5 * w
        y0 = cy - 0.5 * h
        x1 = cx + 0.5 * w
        y1 = cy + 0.5 * h

        coords_w = ops.cast(ops.arange(self.spatial_w), "float32") / self.spatial_w
        coords_h = ops.cast(ops.arange(self.spatial_h), "float32") / self.spatial_h

        coords_w = ops.reshape(coords_w, (1, 1, self.spatial_w, 1))
        coords_h = ops.reshape(coords_h, (1, 1, self.spatial_h, 1))
        x0 = ops.expand_dims(x0, 2)
        x1 = ops.expand_dims(x1, 2)
        y0 = ops.expand_dims(y0, 2)
        y1 = ops.expand_dims(y1, 2)

        deltas_x = ops.concatenate([coords_w - x0, coords_w - x1], axis=-1)
        deltas_y = ops.concatenate([coords_h - y0, coords_h - y1], axis=-1)

        log2_8 = ops.cast(math.log2(8.0), "float32")
        deltas_x = ops.sign(deltas_x) * ops.log2(ops.abs(deltas_x * 8.0) + 1.0) / log2_8
        deltas_y = ops.sign(deltas_y) * ops.log2(ops.abs(deltas_y * 8.0) + 1.0) / log2_8

        rpb_x = self.box_rpb_embed_x(deltas_x)
        rpb_y = self.box_rpb_embed_y(deltas_y)

        rpb = ops.expand_dims(rpb_y, 3) + ops.expand_dims(rpb_x, 2)

        hw_slices = [rpb[:, :, hi, :, :] for hi in range(self.spatial_h)]
        rpb_flat = ops.concatenate(hw_slices, axis=2)
        rpb_out = ops.transpose(rpb_flat, (0, 3, 1, 2))

        zeros = ops.zeros_like(rpb_out[:, :, :1, :])
        rpb_out = ops.concatenate([zeros, rpb_out], axis=2)
        return rpb_out

    def compute_output_spec(self, reference_boxes):
        batch = reference_boxes.shape[0]
        nq = reference_boxes.shape[1]
        q1 = nq + 1 if nq is not None else None
        hw = self.spatial_h * self.spatial_w
        return keras.KerasTensor(
            shape=(batch, self.num_heads, q1, hw),
            dtype=reference_boxes.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "spatial_h": self.spatial_h,
                "spatial_w": self.spatial_w,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM3GeometryEncoder(layers.Layer):
    """Encodes bounding box prompts into features for the DETR decoder.

    Combines direct box projection, sine position encoding, ROI-aligned
    visual features, and label embeddings into a unified prompt
    representation. These are refined through transformer layers with
    self-attention and cross-attention to flattened vision features.

    Args:
        hidden_dim (int): Hidden dimension. Defaults to ``256``.
        num_layers (int): Number of transformer refinement layers.
            Defaults to ``3``.
        num_heads (int): Number of attention heads.
            Defaults to ``8``.
        mlp_dim (int): MLP intermediate dimension.
            Defaults to ``2048``.
        dropout (float): Dropout rate. Defaults to ``0.1``.
        roi_size (int): ROI Align output spatial size. Defaults to ``7``.
        layer_norm_eps (float): Epsilon for layer normalization.
            Defaults to ``1e-6``.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """

    def __init__(
        self,
        hidden_dim=256,
        num_layers=3,
        num_heads=8,
        mlp_dim=2048,
        dropout=0.1,
        roi_size=7,
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers_val = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout
        self.roi_size = roi_size
        self.layer_norm_eps = layer_norm_eps

    def build(self, input_shape):
        dim = self.hidden_dim
        self.boxes_direct_project = layers.Dense(dim, name="boxes_direct_project")
        self.boxes_direct_project.build((None, None, 4))
        self.boxes_pos_enc_project = layers.Dense(dim, name="boxes_pos_enc_project")
        self.boxes_pos_enc_project.build((None, None, dim + 2))
        self.boxes_pool_project = layers.Conv2D(
            dim,
            kernel_size=self.roi_size,
            padding="valid",
            data_format="channels_last",
            name="boxes_pool_project",
        )
        self.boxes_pool_project.build((None, self.roi_size, self.roi_size, dim))
        self.label_embed = layers.Embedding(2, dim, name="label_embed")
        self.label_embed.build((None,))
        self.cls_embed = layers.Embedding(1, dim, name="cls_embed")
        self.cls_embed.build((None,))
        self.vision_layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="vision_layer_norm"
        )
        self.vision_layer_norm.build((None, None, dim))
        self.prompt_layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="prompt_layer_norm"
        )
        self.prompt_layer_norm.build((None, None, dim))
        self.output_layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="output_layer_norm"
        )
        self.output_layer_norm.build((None, None, dim))
        self.final_proj = layers.Dense(dim, name="final_proj")
        self.final_proj.build((None, None, dim))
        self.transformer_layers = []
        for i in range(self.num_layers_val):
            layer = SAM3GeometryEncoderLayer(
                hidden_dim=dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout_rate,
                layer_norm_eps=self.layer_norm_eps,
                name=f"layers_{i}",
            )
            layer.build((None, None, dim))
            self.transformer_layers.append(layer)
        self.built = True

    def _roi_align(
        self, vision_features, boxes_xyxy_denorm, data_format="channels_last"
    ):
        """ROI Align using pure Keras ops with bilinear interpolation.

        Args:
            vision_features: (B, H, W, C) or (B, C, H, W) feature map.
            boxes_xyxy_denorm: (B, num_boxes, 4) boxes in denormalized xyxy.

        Returns:
            pooled: (B*num_boxes, C, roi_size, roi_size) in NCHW.
        """
        if data_format == "channels_last":
            feats = ops.transpose(vision_features, (0, 3, 1, 2))
        else:
            feats = vision_features

        feat_h = ops.shape(feats)[2]
        feat_w = ops.shape(feats)[3]
        out_h = self.roi_size
        out_w = self.roi_size

        all_pooled = []
        for b in range(ops.shape(boxes_xyxy_denorm)[0]):
            feat = feats[b]
            boxes = boxes_xyxy_denorm[b]
            for n in range(ops.shape(boxes)[0]):
                x0 = boxes[n, 0]
                y0 = boxes[n, 1]
                x1 = boxes[n, 2]
                y1 = boxes[n, 3]

                step_h = (y1 - y0) / ops.cast(out_h, "float32")
                step_w = (x1 - x0) / ops.cast(out_w, "float32")
                center_y = y0 + step_h * (ops.cast(ops.arange(out_h), "float32") + 0.5)
                center_x = x0 + step_w * (ops.cast(ops.arange(out_w), "float32") + 0.5)

                gy = ops.reshape(center_y, (out_h, 1))
                gx = ops.reshape(center_x, (1, out_w))
                gy = ops.broadcast_to(gy, (out_h, out_w))
                gx = ops.broadcast_to(gx, (out_h, out_w))

                y_floor = ops.cast(ops.floor(gy), "int32")
                x_floor = ops.cast(ops.floor(gx), "int32")
                y_ceil = y_floor + 1
                x_ceil = x_floor + 1

                wy = gy - ops.cast(y_floor, "float32")
                wx = gx - ops.cast(x_floor, "float32")

                y_floor = ops.clip(y_floor, 0, feat_h - 1)
                y_ceil = ops.clip(y_ceil, 0, feat_h - 1)
                x_floor = ops.clip(x_floor, 0, feat_w - 1)
                x_ceil = ops.clip(x_ceil, 0, feat_w - 1)

                tl = feat[:, y_floor, x_floor]
                tr = feat[:, y_floor, x_ceil]
                bl = feat[:, y_ceil, x_floor]
                br = feat[:, y_ceil, x_ceil]

                w_tl = (1 - wy) * (1 - wx)
                w_tr = (1 - wy) * wx
                w_bl = wy * (1 - wx)
                w_br = wy * wx

                pooled_roi = tl * w_tl + tr * w_tr + bl * w_bl + br * w_br
                all_pooled.append(pooled_roi)

        return ops.stack(all_pooled, axis=0)

    def call(
        self,
        boxes,
        box_labels,
        box_mask,
        vision_features_flat,
        vision_pos_flat,
        vision_features_spatial=None,
        data_format="channels_last",
    ):
        """
        Args:
            boxes: (B, num_boxes, 4) cxcywh normalized [0,1]
            box_labels: (B, num_boxes) int
            box_mask: (B, num_boxes) float, 1=valid 0=padding
            vision_features_flat: (B, H*W, C) for cross-attention
            vision_pos_flat: (1, H*W, C) position encoding
            vision_features_spatial: (B, C, H, W) or (B, H, W, C) FPN features
                for ROI Align. If None, ROI Align pooling is skipped.
            data_format: "channels_first" or "channels_last".
        Returns:
            prompt_features: (B, num_boxes+1, C)
            prompt_mask: (B, num_boxes+1) float
        """
        batch_size = ops.shape(boxes)[0]
        num_boxes = ops.shape(boxes)[1]

        boxes_embed = self.boxes_direct_project(boxes)

        if vision_features_spatial is not None:
            if data_format == "channels_first":
                vis_nhwc = ops.transpose(vision_features_spatial, (0, 2, 3, 1))
            else:
                vis_nhwc = vision_features_spatial
            vis_nhwc = self.vision_layer_norm(vis_nhwc)

            cx = boxes[..., 0:1]
            cy = boxes[..., 1:2]
            w = boxes[..., 2:3]
            h = boxes[..., 3:4]
            x0 = cx - 0.5 * w
            y0 = cy - 0.5 * h
            x1 = cx + 0.5 * w
            y1 = cy + 0.5 * h
            boxes_xyxy = ops.concatenate([x0, y0, x1, y1], axis=-1)

            spatial_axis = (2, 3) if data_format == "channels_first" else (1, 2)
            feat_h = ops.shape(vision_features_spatial)[spatial_axis[0]]
            feat_w = ops.shape(vision_features_spatial)[spatial_axis[1]]
            scale = ops.convert_to_tensor(
                [feat_w, feat_h, feat_w, feat_h], dtype="float32"
            )
            scale = ops.reshape(scale, (1, 1, 4))
            boxes_xyxy_denorm = boxes_xyxy * ops.cast(scale, boxes_xyxy.dtype)

            pooled = self._roi_align(
                vision_features_spatial, boxes_xyxy_denorm, data_format
            )
            pooled_nhwc = ops.transpose(pooled, (0, 2, 3, 1))
            pooled_proj = self.boxes_pool_project(pooled_nhwc)
            pooled_proj = ops.reshape(
                pooled_proj, (batch_size, num_boxes, self.hidden_dim)
            )
            boxes_embed = boxes_embed + pooled_proj

        cx = boxes[..., 0]
        cy = boxes[..., 1]
        w_box = boxes[..., 2:3]
        h_box = boxes[..., 3:4]

        scale = 2.0 * math.pi
        num_feats = self.hidden_dim // 2
        dim_t = ops.cast(ops.arange(num_feats), "float32")
        dim_t = 10000.0 ** (2.0 * ops.floor(dim_t / 2) / num_feats)

        def _encode_1d(coord):
            c = coord * scale
            c = ops.expand_dims(c, -1) / dim_t
            c_sin = ops.sin(c[..., 0::2])
            c_cos = ops.cos(c[..., 1::2])
            return ops.reshape(
                ops.stack([c_sin, c_cos], axis=-1),
                ops.shape(c_sin)[:-1] + (num_feats,),
            )

        pos_x = _encode_1d(cx)
        pos_y = _encode_1d(cy)

        pos_with_hw = ops.concatenate([pos_y, pos_x, h_box, w_box], axis=-1)
        pos_encoded = self.boxes_pos_enc_project(pos_with_hw)

        label_emb = self.label_embed(box_labels)

        prompt_embeds = boxes_embed + pos_encoded + label_emb

        cls_token = self.cls_embed(ops.zeros((batch_size, 1), dtype="int32"))
        prompt_embeds = ops.concatenate([prompt_embeds, cls_token], axis=1)

        cls_mask = ops.ones((batch_size, 1), dtype="float32")
        prompt_mask_out = ops.concatenate([box_mask, cls_mask], axis=1)

        prompt_embeds = self.prompt_layer_norm(self.final_proj(prompt_embeds))

        for layer in self.transformer_layers:
            prompt_embeds = layer(prompt_embeds, vision_features_flat, vision_pos_flat)

        prompt_embeds = self.output_layer_norm(prompt_embeds)
        return prompt_embeds, prompt_mask_out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers_val,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout_rate,
                "roi_size": self.roi_size,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM3CLIPAttention(layers.Layer):
    """Multi-head self-attention for the CLIP text encoder.

    Standard scaled dot-product attention without rotary embeddings.
    Supports an optional additive attention mask for causal masking
    and padding.

    Args:
        hidden_dim (int): Input and output feature dimension.
        num_heads (int): Number of attention heads.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """

    def __init__(self, hidden_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

    def build(self, input_shape):
        dim = self.hidden_dim
        self.q_proj = layers.Dense(dim, name="q_proj")
        self.q_proj.build((None, None, dim))
        self.k_proj = layers.Dense(dim, name="k_proj")
        self.k_proj.build((None, None, dim))
        self.v_proj = layers.Dense(dim, name="v_proj")
        self.v_proj.build((None, None, dim))
        self.o_proj = layers.Dense(dim, name="o_proj")
        self.o_proj.build((None, None, dim))
        self.built = True

    def call(self, hidden_states, attention_mask=None):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q_shape = ops.shape(q)
        k_shape = ops.shape(k)
        q = ops.reshape(q, (q_shape[0], q_shape[1], self.num_heads, self.head_dim))
        k = ops.reshape(k, (k_shape[0], k_shape[1], self.num_heads, self.head_dim))
        v = ops.reshape(v, (k_shape[0], k_shape[1], self.num_heads, self.head_dim))

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        attn_weights = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = ops.softmax(attn_weights, axis=-1)

        attn_output = ops.matmul(attn_weights, v)
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(
            attn_output, (q_shape[0], q_shape[1], self.hidden_dim)
        )
        return self.o_proj(attn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM3CLIPPositionEmbedding(layers.Layer):
    """Learnable absolute position embedding for the CLIP text encoder.

    Adds a fixed-length position embedding to token embeddings using
    ``ops.arange`` for position IDs.

    Args:
        max_pos (int): Maximum sequence length.
        hidden_dim (int): Embedding dimension.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """

    def __init__(self, max_pos, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_pos = max_pos
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.position_embedding = layers.Embedding(
            self.max_pos, self.hidden_dim, name="position_embedding"
        )
        self.position_embedding.build((None, self.max_pos))
        self.built = True

    def call(self, hidden_states):
        position_ids = ops.arange(self.max_pos, dtype="int32")
        return hidden_states + self.position_embedding(position_ids)

    def get_config(self):
        config = super().get_config()
        config["max_pos"] = self.max_pos
        config["hidden_dim"] = self.hidden_dim
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM3CLIPCausalMask(layers.Layer):
    """Causal attention mask with padding for the CLIP text encoder.

    Combines an upper-triangular causal mask with a padding mask derived
    from the attention mask input. Outputs a ``(B, 1, seq, seq)`` additive
    mask suitable for attention logits.

    Args:
        seq_len (int): Sequence length.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    References:
        - SAM 3: https://arxiv.org/abs/2511.16719
    """

    def __init__(self, seq_len, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len

    def call(self, attention_mask):
        causal_mask = ops.triu(ops.full((self.seq_len, self.seq_len), -1e9), k=1)
        pad_mask = ops.cast(1.0 - ops.cast(attention_mask, "float32"), "float32") * (
            -1e9
        )
        pad_mask = ops.reshape(pad_mask, (-1, 1, 1, self.seq_len))
        return causal_mask + pad_mask

    def get_config(self):
        config = super().get_config()
        config["seq_len"] = self.seq_len
        return config
