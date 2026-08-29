import math

import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="zeromodels")
class TableTransformerExpandQueryEmbedding(layers.Layer):
    """Expands learned query embeddings to match the batch dimension.

    Wraps a standard `Embedding` layer and broadcasts its output along a new
    batch axis so that each sample in the batch receives the same set of learned
    object queries. Used to produce the positional part of the object queries
    fed into the Table Transformer decoder.

    Reference:
    - [PubTables-1M](https://arxiv.org/abs/2110.00061)
    - [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

    Args:
        num_queries: Integer, number of object queries (maximum detections per
            image).
        hidden_dim: Integer, embedding dimension for each query.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    Input Shape:
        Any tensor whose first dimension is the batch size. Only `batch_size` is
        read from the input; the content is unused.

    Output Shape:
        3D tensor: `(batch_size, num_queries, hidden_dim)`.
    """

    def __init__(self, num_queries, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.embedding = layers.Embedding(
            num_queries,
            hidden_dim,
            name="embedding",
        )

    def call(self, batch_ref):
        batch_size = ops.shape(batch_ref)[0]
        indices = ops.arange(self.num_queries)
        query_embed = self.embedding(indices)
        query_embed = ops.expand_dims(query_embed, axis=0)
        query_embed = ops.tile(query_embed, [batch_size, 1, 1])
        return query_embed

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_queries": self.num_queries,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class TableTransformerFlattenFeatures(layers.Layer):
    """Flattens spatial feature maps into a 1D token sequence.

    Reshapes a 4D spatial tensor into a 3D sequence tensor suitable for
    transformer input by collapsing the height and width dimensions into a
    single sequence dimension.

    Reference:
    - [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

    Args:
        hidden_dim: Integer, channel dimension of the input feature map. Used as
            the last dimension in the reshape target.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    Input Shape:
        4D tensor: `(batch_size, height, width, hidden_dim)`.

    Output Shape:
        3D tensor: `(batch_size, height * width, hidden_dim)`.
    """

    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def call(self, inputs):
        shape = ops.shape(inputs)
        data_format = keras.config.image_data_format()
        if data_format == "channels_first":
            x = ops.transpose(inputs, [0, 2, 3, 1])
            return ops.reshape(x, [shape[0], shape[2] * shape[3], self.hidden_dim])
        return ops.reshape(inputs, [shape[0], shape[1] * shape[2], self.hidden_dim])

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class TableTransformerPositionEmbeddingSine(layers.Layer):
    """Fixed sinusoidal 2D positional embedding for spatial feature maps.

    Generates non-learnable sine/cosine positional encodings that encode the row
    and column position of each spatial location. Half of the embedding
    dimension encodes the vertical position and the other half the horizontal
    position, using sinusoidal functions at geometrically spaced frequencies.
    Matches the reference Table Transformer / DETR sine position embedding for a
    full (unpadded) image, whose pixel mask cumulative sum reduces to
    `arange(1, size + 1)`.

    Reference:
    - [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

    Args:
        hidden_dim: Integer, total embedding dimension. Half is allocated to row
            embeddings and half to column embeddings. Defaults to `256`.
        temperature: Integer, temperature scaling factor for the sinusoidal
            frequencies. Defaults to `10000`.
        normalize: Boolean, whether to normalize position coordinates to the
            range `[0, 2*pi]` before computing the encoding. Defaults to `True`.
        eps: Float, small constant added during normalization to prevent
            division by zero. Defaults to `1e-6`.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    Input Shape:
        4D tensor: `(batch_size, height, width, channels)`. Only the spatial
        dimensions are used; the channel dimension is ignored.

    Output Shape:
        4D tensor: `(batch_size, height, width, hidden_dim)`.
    """

    def __init__(
        self,
        hidden_dim=256,
        temperature=10000,
        normalize=True,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.normalize = normalize
        self.eps = eps
        self.num_pos_feats = hidden_dim // 2

    def call(self, inputs):
        shape = ops.shape(inputs)
        batch_size = shape[0]
        data_format = keras.config.image_data_format()
        if data_format == "channels_first":
            h = shape[2]
            w = shape[3]
        else:
            h = shape[1]
            w = shape[2]

        y_embed = ops.cast(
            ops.repeat(
                ops.expand_dims(ops.arange(1, h + 1, dtype="float32"), axis=1),
                w,
                axis=1,
            ),
            dtype="float32",
        )
        x_embed = ops.cast(
            ops.repeat(
                ops.expand_dims(ops.arange(1, w + 1, dtype="float32"), axis=0),
                h,
                axis=0,
            ),
            dtype="float32",
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
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "temperature": self.temperature,
                "normalize": self.normalize,
                "eps": self.eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class TableTransformerMultiHeadAttention(layers.Layer):
    """Multi-head attention layer for the Table Transformer transformer.

    Implements scaled dot-product multi-head attention with separate query, key,
    and value projections followed by an output projection. The projection
    naming matches the reference layout (`q_proj`, `k_proj`, `v_proj`,
    `out_proj`) to simplify weight transfer from pretrained models. Used in both
    the encoder (self-attention) and decoder (self-attention and
    cross-attention) stages.

    Reference:
    - [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

    Args:
        hidden_dim: Integer, total model dimension. Must be divisible by
            `num_heads`.
        num_heads: Integer, number of parallel attention heads.
        dropout_rate: Float, dropout rate applied to the attention weight matrix.
            Defaults to `0.0`.
        block_prefix: String, name prefix for the internal dense layers
            (`q_proj`, `k_proj`, `v_proj`, `out_proj`). Defaults to `""`.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    Input Shape:
        Three 3D tensors:
        - `query`: `(batch_size, seq_len_q, hidden_dim)`
        - `key`:   `(batch_size, seq_len_k, hidden_dim)`
        - `value`: `(batch_size, seq_len_k, hidden_dim)`

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
        seq_len_q = ops.shape(query)[1]
        seq_len_k = ops.shape(key)[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = ops.reshape(q, [batch_size, seq_len_q, self.num_heads, self.head_dim])
        k = ops.reshape(k, [batch_size, seq_len_k, self.num_heads, self.head_dim])
        v = ops.reshape(v, [batch_size, seq_len_k, self.num_heads, self.head_dim])

        q = ops.transpose(q, [0, 2, 1, 3])
        k = ops.transpose(k, [0, 2, 1, 3])
        v = ops.transpose(v, [0, 2, 1, 3])

        attn_weights = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2])) * self.scale
        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        attn_output = ops.matmul(attn_weights, v)
        attn_output = ops.transpose(attn_output, [0, 2, 1, 3])
        attn_output = ops.reshape(attn_output, [batch_size, seq_len_q, self.hidden_dim])
        attn_output = self.out_proj(attn_output)

        return attn_output

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
