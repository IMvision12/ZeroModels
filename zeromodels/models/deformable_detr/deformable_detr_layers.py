import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


def inverse_sigmoid(x, eps=1e-5):
    """Numerically safe ``logit``: ``log(x / (1 - x))`` with both terms clamped."""
    x = ops.clip(x, 0.0, 1.0)
    return ops.log(ops.maximum(x, eps) / ops.maximum(1.0 - x, eps))


def _bilinear_sample(value, grid, height, width):
    """``grid_sample(value, grid, bilinear, zeros padding, align_corners=False)``.

    Args:
        value: ``(N, C, H, W)``.
        grid: ``(N, Lq, P, 2)`` sampling locations in ``[-1, 1]`` as ``(x, y)``.

    Returns:
        ``(N, C, Lq, P)``.
    """
    if keras.config.backend() == "torch":
        import torch.nn.functional as F

        return F.grid_sample(
            value, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

    n = ops.shape(value)[0]
    channels = int(value.shape[1])
    num_queries = int(grid.shape[1])
    num_points = int(grid.shape[2])
    flat = ops.reshape(
        ops.transpose(value, (0, 2, 3, 1)), (n, height * width, channels)
    )

    # Unnormalize to pixel coordinates (align_corners=False).
    x = ((grid[..., 0] + 1.0) * width - 1.0) / 2.0
    y = ((grid[..., 1] + 1.0) * height - 1.0) / 2.0
    x0 = ops.floor(x)
    y0 = ops.floor(y)
    fx = x - x0
    fy = y - y0
    x0 = ops.cast(x0, "int32")
    y0 = ops.cast(y0, "int32")

    def corner(yi, xi, weight):
        valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        idx = ops.clip(yi, 0, height - 1) * width + ops.clip(xi, 0, width - 1)
        idx = ops.reshape(idx, (n, num_queries * num_points, 1))
        idx = ops.repeat(idx, channels, axis=-1)
        sampled = ops.take_along_axis(flat, idx, axis=1)
        sampled = ops.reshape(sampled, (n, num_queries, num_points, channels))
        weight = ops.where(valid, weight, ops.zeros_like(weight))
        return sampled * ops.expand_dims(ops.cast(weight, sampled.dtype), -1)

    out = (
        corner(y0, x0, (1 - fx) * (1 - fy))
        + corner(y0, x0 + 1, fx * (1 - fy))
        + corner(y0 + 1, x0, (1 - fx) * fy)
        + corner(y0 + 1, x0 + 1, fx * fy)
    )
    return ops.transpose(out, (0, 3, 1, 2))


def multi_scale_deformable_attention(
    value, spatial_shapes, sampling_locations, attention_weights
):
    """Multi-scale deformable attention core (the reference's pure-PyTorch path).

    Args:
        value: ``(B, S, num_heads, head_dim)``, the projected values of all
            levels flattened and concatenated (``S = sum(H * W)``).
        spatial_shapes: list of ``(H, W)`` per level (Python ints).
        sampling_locations: ``(B, Lq, num_heads, L, P, 2)`` in ``[0, 1]``.
        attention_weights: ``(B, Lq, num_heads, L, P)`` (softmaxed over L * P).

    Returns:
        ``(B, Lq, num_heads * head_dim)``.
    """
    b = ops.shape(value)[0]
    num_heads = int(value.shape[2])
    head_dim = int(value.shape[3])
    num_queries = int(sampling_locations.shape[1])
    num_levels = len(spatial_shapes)
    num_points = int(sampling_locations.shape[4])

    split_points = []
    total = 0
    for h, w in spatial_shapes[:-1]:
        total += h * w
        split_points.append(total)
    value_list = ops.split(value, split_points, axis=1) if split_points else [value]
    grids = 2.0 * sampling_locations - 1.0

    sampled_levels = []
    for level, (h, w) in enumerate(spatial_shapes):
        # (B, HW, heads, hd) -> (B * heads, hd, H, W)
        value_l = ops.reshape(
            ops.transpose(value_list[level], (0, 2, 3, 1)),
            (b * num_heads, head_dim, h, w),
        )
        # (B, Lq, heads, P, 2) -> (B * heads, Lq, P, 2)
        grid_l = ops.reshape(
            ops.transpose(grids[:, :, :, level], (0, 2, 1, 3, 4)),
            (b * num_heads, num_queries, num_points, 2),
        )
        sampled_levels.append(_bilinear_sample(value_l, grid_l, h, w))

    # (B * heads, hd, Lq, L, P) -> (B * heads, hd, Lq, L * P)
    sampled = ops.reshape(
        ops.stack(sampled_levels, axis=-2),
        (b * num_heads, head_dim, num_queries, num_levels * num_points),
    )
    weights = ops.reshape(
        ops.transpose(attention_weights, (0, 2, 1, 3, 4)),
        (b * num_heads, 1, num_queries, num_levels * num_points),
    )
    out = ops.sum(sampled * ops.cast(weights, sampled.dtype), axis=-1)
    out = ops.reshape(out, (b, num_heads * head_dim, num_queries))
    return ops.transpose(out, (0, 2, 1))


@keras.saving.register_keras_serializable(package="zeromodels")
class DeformableDetrMultiscaleDeformableAttention(layers.Layer):
    """Multi-scale deformable attention (Deformable DETR).

    Each query attends to ``n_points`` sampled locations per head per feature
    level, placed at learned offsets around its reference point and combined
    with learned (softmaxed) weights, instead of to every key.

    Args:
        d_model: Model width.
        num_heads: Attention heads.
        n_levels: Number of feature levels.
        n_points: Sampling points per head per level.

    Call args:
        hidden_states: ``(B, Lq, d_model)`` queries (position already added).
        value_states: ``(B, S, d_model)`` flattened multi-level features.
        reference_points: ``(B, Lq, n_levels, 2)`` points or
            ``(B, Lq, n_levels, 4)`` boxes, normalized.
        spatial_shapes: list of ``(H, W)`` per level.
    """

    def __init__(self, d_model, num_heads, n_levels, n_points, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.head_dim = d_model // num_heads
        self.sampling_offsets = layers.Dense(
            num_heads * n_levels * n_points * 2, name="sampling_offsets"
        )
        self.attention_weights = layers.Dense(
            num_heads * n_levels * n_points, name="attention_weights"
        )
        self.value_proj = layers.Dense(d_model, name="value_proj")
        self.output_proj = layers.Dense(d_model, name="output_proj")

    def build(self, input_shape=None):
        for layer in (
            self.sampling_offsets,
            self.attention_weights,
            self.value_proj,
            self.output_proj,
        ):
            layer.build((None, self.d_model))
        self.built = True

    def call(self, hidden_states, value_states, reference_points, spatial_shapes):
        b = ops.shape(hidden_states)[0]
        num_queries = int(hidden_states.shape[1])
        seq_len = int(value_states.shape[1])
        heads, levels, points = self.num_heads, self.n_levels, self.n_points

        value = ops.reshape(
            self.value_proj(value_states), (b, seq_len, heads, self.head_dim)
        )
        offsets = ops.reshape(
            self.sampling_offsets(hidden_states),
            (b, num_queries, heads, levels, points, 2),
        )
        weights = ops.reshape(
            self.attention_weights(hidden_states),
            (b, num_queries, heads, levels * points),
        )
        weights = ops.softmax(ops.cast(weights, "float32"), axis=-1)
        weights = ops.reshape(weights, (b, num_queries, heads, levels, points))

        offsets = ops.cast(offsets, "float32")
        reference_points = ops.cast(reference_points, "float32")
        if int(reference_points.shape[-1]) == 2:
            normalizer = ops.convert_to_tensor(
                [[w, h] for h, w in spatial_shapes], dtype="float32"
            )
            locations = (
                reference_points[:, :, None, :, None, :]
                + offsets / normalizer[None, None, None, :, None, :]
            )
        else:
            locations = (
                reference_points[:, :, None, :, None, :2]
                + offsets / points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        out = multi_scale_deformable_attention(
            value, spatial_shapes, ops.cast(locations, value.dtype), weights
        )
        return self.output_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "n_levels": self.n_levels,
                "n_points": self.n_points,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DeformableDetrSelfAttention(layers.Layer):
    """Multi-head self-attention over the object queries.

    The query position embedding is added to the queries and keys (not the
    values).
    """

    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = layers.Dense(d_model, name="q_proj")
        self.k_proj = layers.Dense(d_model, name="k_proj")
        self.v_proj = layers.Dense(d_model, name="v_proj")
        self.out_proj = layers.Dense(d_model, name="out_proj")

    def build(self, input_shape=None):
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            layer.build((None, self.d_model))
        self.built = True

    def call(self, hidden_states, position_embeddings):
        b = ops.shape(hidden_states)[0]
        length = int(hidden_states.shape[1])

        def split(x):
            return ops.transpose(
                ops.reshape(x, (b, length, self.num_heads, self.head_dim)),
                (0, 2, 1, 3),
            )

        query_key = hidden_states + position_embeddings
        out = fused_attention(
            split(self.q_proj(query_key)),
            split(self.k_proj(query_key)),
            split(self.v_proj(hidden_states)),
            self.head_dim**-0.5,
        )
        out = ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (b, length, self.d_model))
        return self.out_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads})
        return config


class _FeedForwardMixin:
    """Shared ReLU feed-forward block + post-norm residual helper."""

    def _build_ffn(self, d_model, ffn_dim, dropout_rate):
        self.fc1 = layers.Dense(ffn_dim, name="fc1")
        self.fc2 = layers.Dense(d_model, name="fc2")
        self.final_layer_norm = layers.LayerNormalization(
            epsilon=1e-5, name="final_layer_norm"
        )
        self.dropout = layers.Dropout(dropout_rate)

    def _feed_forward(self, hidden_states, training):
        ff = self.fc2(ops.relu(self.fc1(hidden_states)))
        return self.final_layer_norm(
            hidden_states + self.dropout(ff, training=training)
        )


@keras.saving.register_keras_serializable(package="zeromodels")
class DeformableDetrEncoderLayer(_FeedForwardMixin, layers.Layer):
    """One deformable encoder layer (post-norm).

    Multi-scale deformable self-attention over all feature-level tokens (the
    level-aware sine position is added to the queries), then a ReLU feed-forward.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        ffn_dim,
        n_levels,
        n_points,
        dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.n_levels = n_levels
        self.n_points = n_points
        self.dropout_rate = dropout_rate
        self.self_attn = DeformableDetrMultiscaleDeformableAttention(
            d_model, num_heads, n_levels, n_points, name="self_attn"
        )
        self.self_attn_layer_norm = layers.LayerNormalization(
            epsilon=1e-5, name="self_attn_layer_norm"
        )
        self._build_ffn(d_model, ffn_dim, dropout_rate)

    def build(self, input_shape=None):
        self.self_attn.build()
        self.self_attn_layer_norm.build((None, self.d_model))
        self.fc1.build((None, self.d_model))
        self.fc2.build((None, self.ffn_dim))
        self.final_layer_norm.build((None, self.d_model))
        self.built = True

    def call(
        self,
        hidden_states,
        position_embeddings,
        reference_points,
        spatial_shapes,
        training=None,
    ):
        attn = self.self_attn(
            hidden_states + position_embeddings,
            hidden_states,
            reference_points,
            spatial_shapes=spatial_shapes,
        )
        hidden_states = self.self_attn_layer_norm(
            hidden_states + self.dropout(attn, training=training)
        )
        return self._feed_forward(hidden_states, training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "ffn_dim": self.ffn_dim,
                "n_levels": self.n_levels,
                "n_points": self.n_points,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DeformableDetrDecoderLayer(_FeedForwardMixin, layers.Layer):
    """One Deformable DETR decoder layer (post-norm).

    Self-attention over the object queries, multi-scale deformable
    cross-attention into the encoder output around each query's reference point,
    then a ReLU feed-forward. The query position embedding is added to the
    queries of both attentions.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        ffn_dim,
        n_levels,
        n_points,
        dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.n_levels = n_levels
        self.n_points = n_points
        self.dropout_rate = dropout_rate
        self.self_attn = DeformableDetrSelfAttention(
            d_model, num_heads, name="self_attn"
        )
        self.self_attn_layer_norm = layers.LayerNormalization(
            epsilon=1e-5, name="self_attn_layer_norm"
        )
        self.encoder_attn = DeformableDetrMultiscaleDeformableAttention(
            d_model, num_heads, n_levels, n_points, name="encoder_attn"
        )
        self.encoder_attn_layer_norm = layers.LayerNormalization(
            epsilon=1e-5, name="encoder_attn_layer_norm"
        )
        self._build_ffn(d_model, ffn_dim, dropout_rate)

    def build(self, input_shape=None):
        self.self_attn.build()
        self.encoder_attn.build()
        for norm in (
            self.self_attn_layer_norm,
            self.encoder_attn_layer_norm,
            self.final_layer_norm,
        ):
            norm.build((None, self.d_model))
        self.fc1.build((None, self.d_model))
        self.fc2.build((None, self.ffn_dim))
        self.built = True

    def call(
        self,
        hidden_states,
        query_position,
        reference_points,
        memory,
        spatial_shapes,
        training=None,
    ):
        attn = self.self_attn(hidden_states, query_position)
        hidden_states = self.self_attn_layer_norm(
            hidden_states + self.dropout(attn, training=training)
        )
        attn = self.encoder_attn(
            hidden_states + query_position,
            memory,
            reference_points,
            spatial_shapes=spatial_shapes,
        )
        hidden_states = self.encoder_attn_layer_norm(
            hidden_states + self.dropout(attn, training=training)
        )
        return self._feed_forward(hidden_states, training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "ffn_dim": self.ffn_dim,
                "n_levels": self.n_levels,
                "n_points": self.n_points,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DeformableDetrMLP(layers.Layer):
    """Box prediction head: ``num_layers`` Dense layers with ReLU between."""

    def __init__(self, hidden_dim, output_dim, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        dims = [hidden_dim] * (num_layers - 1) + [output_dim]
        self.dense_layers = [
            layers.Dense(d, name=f"layers_{i}") for i, d in enumerate(dims)
        ]

    def build(self, input_shape=None):
        in_dim = self.hidden_dim
        for layer in self.dense_layers:
            layer.build((None, in_dim))
            in_dim = layer.units
        self.built = True

    def call(self, x):
        for i, layer in enumerate(self.dense_layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = ops.relu(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
            }
        )
        return config
