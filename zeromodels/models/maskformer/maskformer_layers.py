import math

import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


@keras.saving.register_keras_serializable(package="zeromodels")
class MaskFormerSinePositionEmbedding(layers.Layer):
    """Sinusoidal 2D positional embedding (DETR-style).

    Generates a fixed sine/cosine positional encoding for every spatial
    location of the input feature map.

    Reference:
    - [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

    Args:
        hidden_dim: Total embedding dimension. Half is allocated to row
            embeddings and half to column embeddings.
        temperature: Frequency scaling factor.
        normalize: Whether to normalize position coordinates to
            ``[0, 2*pi]``.
        eps: Small constant for the normalization denominator.

    Input shape: 4D tensor ``(B, H, W, C)``.
    Output shape: 4D tensor ``(B, H, W, hidden_dim)``.
    """

    def __init__(
        self,
        hidden_dim=256,
        temperature=10000,
        normalize=True,
        eps=1e-6,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.normalize = normalize
        self.eps = eps
        self.num_pos_feats = hidden_dim // 2
        self.data_format = data_format or keras.config.image_data_format()

    def call(self, inputs):
        shape = ops.shape(inputs)
        batch_size = shape[0]
        data_format = self.data_format
        if data_format == "channels_first":
            h = shape[2]
            w = shape[3]
        else:
            h = shape[1]
            w = shape[2]

        y_embed = ops.repeat(
            ops.expand_dims(ops.arange(1, h + 1, dtype="float32"), axis=1),
            w,
            axis=1,
        )
        x_embed = ops.repeat(
            ops.expand_dims(ops.arange(1, w + 1, dtype="float32"), axis=0),
            h,
            axis=0,
        )

        if self.normalize:
            y_embed = y_embed / (y_embed[-1:, :] + self.eps) * 2 * math.pi
            x_embed = x_embed / (x_embed[:, -1:] + self.eps) * 2 * math.pi

        dim_t = ops.arange(self.num_pos_feats, dtype="float32")
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = ops.expand_dims(x_embed, axis=-1) / dim_t
        pos_y = ops.expand_dims(y_embed, axis=-1) / dim_t

        pos_x_sin = ops.sin(pos_x[:, :, 0::2])
        pos_x_cos = ops.cos(pos_x[:, :, 1::2])
        pos_x = ops.reshape(
            ops.stack([pos_x_sin, pos_x_cos], axis=-1),
            [h, w, self.num_pos_feats],
        )

        pos_y_sin = ops.sin(pos_y[:, :, 0::2])
        pos_y_cos = ops.cos(pos_y[:, :, 1::2])
        pos_y = ops.reshape(
            ops.stack([pos_y_sin, pos_y_cos], axis=-1),
            [h, w, self.num_pos_feats],
        )

        pos = ops.concatenate([pos_y, pos_x], axis=-1)
        pos = ops.expand_dims(pos, axis=0)
        pos = ops.broadcast_to(pos, [batch_size, h, w, self.hidden_dim])
        if data_format == "channels_first":
            pos = ops.transpose(pos, [0, 3, 1, 2])
        return pos

    def get_config(self):
        c = super().get_config()
        c.update(
            {
                "hidden_dim": self.hidden_dim,
                "temperature": self.temperature,
                "normalize": self.normalize,
                "eps": self.eps,
                "data_format": self.data_format,
            }
        )
        return c


@keras.saving.register_keras_serializable(package="zeromodels")
class MaskFormerDetrAttention(layers.Layer):
    """DETR-style multi-head attention with separate q/k/v/out projections.

    Used in the MaskFormer transformer decoder for both self-attention
    (over object queries) and cross-attention (queries attending to the
    flattened image memory). Uses DETR-style projection naming so the
    state-dict transfer is direct.

    Args:
        hidden_dim: Total model dimension. Must be divisible by ``num_heads``.
        num_heads: Number of attention heads.
        dropout_rate: Dropout applied to the attention probabilities.

    Inputs (call):
        query: ``(B, Q, hidden_dim)``
        key: ``(B, K, hidden_dim)``
        value: ``(B, K, hidden_dim)``

    Output: ``(B, Q, hidden_dim)``.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads "
                f"({num_heads})."
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout_rate = dropout_rate

        self.q_proj = layers.Dense(hidden_dim, name="q_proj")
        self.k_proj = layers.Dense(hidden_dim, name="k_proj")
        self.v_proj = layers.Dense(hidden_dim, name="v_proj")
        self.o_proj = layers.Dense(hidden_dim, name="o_proj")
        self.attn_dropout = layers.Dropout(dropout_rate)

    def call(self, query, key, value, training=None):
        b = ops.shape(query)[0]
        q_len = ops.shape(query)[1]
        k_len = ops.shape(key)[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = ops.reshape(q, [b, q_len, self.num_heads, self.head_dim])
        k = ops.reshape(k, [b, k_len, self.num_heads, self.head_dim])
        v = ops.reshape(v, [b, k_len, self.num_heads, self.head_dim])

        q = ops.transpose(q, [0, 2, 1, 3])
        k = ops.transpose(k, [0, 2, 1, 3])
        v = ops.transpose(v, [0, 2, 1, 3])

        out = fused_attention(
            q, k, v, self.scale, dropout=self.attn_dropout, training=training
        )
        out = ops.transpose(out, [0, 2, 1, 3])
        out = ops.reshape(out, [b, q_len, self.hidden_dim])
        return self.o_proj(out)

    def get_config(self):
        c = super().get_config()
        c.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
            }
        )
        return c


@keras.saving.register_keras_serializable(package="zeromodels")
class MaskFormerExpandQueryEmbedding(layers.Layer):
    """Learned per-query embedding broadcast across the batch.

    Holds a ``(num_queries, hidden_dim)`` learned embedding (``weight``)
    and tiles it to ``(B, num_queries, hidden_dim)`` to produce the
    object-query positional embedding used in the decoder.

    Args:
        num_queries: Number of learned object queries.
        hidden_dim: Embedding dimension.

    Input (call): any tensor whose first dim is the batch (used to read ``B``).
    Output: ``(B, num_queries, hidden_dim)``.
    """

    def __init__(self, num_queries, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=(self.num_queries, self.hidden_dim),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, batch_ref):
        b = ops.shape(batch_ref)[0]
        w = ops.expand_dims(self.weight, axis=0)
        return ops.broadcast_to(w, [b, self.num_queries, self.hidden_dim])

    def get_config(self):
        c = super().get_config()
        c.update({"num_queries": self.num_queries, "hidden_dim": self.hidden_dim})
        return c
