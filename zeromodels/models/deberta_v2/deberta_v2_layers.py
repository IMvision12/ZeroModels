import keras
from keras import layers, ops

MASK_NEG = -1e9


def make_log_bucket_position(relative_pos, bucket_size, max_position):
    rel = ops.cast(relative_pos, "float32")
    sign = ops.sign(rel)
    mid = bucket_size // 2
    cond = ops.logical_and(ops.less(rel, mid), ops.greater(rel, -mid))
    abs_pos = ops.where(cond, ops.cast(mid - 1, "float32"), ops.abs(rel))
    log_pos = (
        ops.ceil(
            ops.log(abs_pos / mid)
            / ops.log(ops.cast((max_position - 1) / mid, "float32"))
            * (mid - 1)
        )
        + mid
    )
    bucket_pos = ops.where(ops.less_equal(abs_pos, mid), rel, log_pos * sign)
    return ops.cast(bucket_pos, "int32")


@keras.saving.register_keras_serializable(package="zeromodels")
class DebertaV2RelativeEmbedding(layers.Layer):
    """Holds DeBERTa-v2's shared relative-position embedding table.

    Returns the full ``(2 * position_buckets, embed_dim)`` table (ignoring its
    input, which only connects it into the functional graph).
    """

    def __init__(self, num_positions, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_positions = num_positions
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.num_positions, self.embed_dim),
            initializer="uniform",
        )
        self.built = True

    def call(self, inputs):
        return self.embeddings

    def compute_output_shape(self, input_shape):
        return (self.num_positions, self.embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"num_positions": self.num_positions, "embed_dim": self.embed_dim}
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DebertaV2Embeddings(layers.Layer):
    """DeBERTa-v2 input embeddings: word embeddings only, LayerNorm, zero padding.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Embedding / model dimension.
        layer_norm_eps: Epsilon for the embedding LayerNorm.
        dropout: Dropout rate applied to the embeddings.
    """

    def __init__(
        self, vocab_size, embed_dim, layer_norm_eps=1e-7, dropout=0.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.layer_norm_eps = layer_norm_eps
        self.dropout_rate = dropout

        self.word_embeddings = layers.Embedding(
            vocab_size, embed_dim, name="word_embeddings"
        )
        self.layer_norm = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="LayerNorm"
        )
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, training=None):
        input_ids, attention_mask = inputs[0], inputs[1]
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.layer_norm(embeddings)
        mask = ops.cast(attention_mask, embeddings.dtype)
        embeddings = embeddings * ops.expand_dims(mask, -1)
        return self.dropout(embeddings, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "layer_norm_eps": self.layer_norm_eps,
                "dropout": self.dropout_rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DebertaV2ConvLayer(layers.Layer):
    """DeBERTa-v2 convolution layer (applied once after the first encoder layer).

    Convolves the embeddings, zeroes padding, applies ``conv_act``, then
    LayerNorm-s the sum with the first layer's output.

    Args:
        embed_dim: Model dimension.
        kernel_size: Conv kernel size.
        conv_act: Activation after the convolution.
        layer_norm_eps: Epsilon for the LayerNorm.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim,
        kernel_size,
        conv_act,
        layer_norm_eps=1e-7,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.conv_act = conv_act
        self.layer_norm_eps = layer_norm_eps
        self.dropout_rate = dropout

        self.conv = layers.Conv1D(
            embed_dim,
            kernel_size,
            padding="same",
            data_format="channels_last",
            name="conv",
        )
        self.layer_norm = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="LayerNorm"
        )
        self.activation = layers.Activation(conv_act)
        self.dropout = layers.Dropout(dropout)

    def build(self, input_shape):
        self.conv.build((None, None, self.embed_dim))
        self.layer_norm.build((None, None, self.embed_dim))
        self.built = True

    def compute_output_shape(self, input_shape, *args, **kwargs):
        return input_shape

    def call(self, conv_input, residual, attention_mask, training=None):
        out = self.conv(conv_input)
        mask = ops.expand_dims(ops.cast(attention_mask, out.dtype), -1)
        out = out * mask
        out = self.activation(self.dropout(out, training=training))
        out = self.layer_norm(residual + out)
        return out * mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "kernel_size": self.kernel_size,
                "conv_act": self.conv_act,
                "layer_norm_eps": self.layer_norm_eps,
                "dropout": self.dropout_rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DebertaV2DisentangledSelfAttention(layers.Layer):
    """DeBERTa-v2 disentangled self-attention (log buckets + shared att key).

    Uses separate query/key/value projections. With ``share_att_key=True`` the
    relative-position keys/queries reuse the content key/query projections;
    otherwise dedicated ``pos_key_proj`` / ``pos_query_proj`` projections are
    used (mirroring HF). Content-to-position (c2p) and position-to-content (p2c)
    terms are gathered per (log-bucketed) relative position and added to the
    content scores; the sum is scaled by ``1/sqrt(head_dim * (1 + #pos_att_type))``.

    Args:
        embed_dim: Model dimension. Must be divisible by ``num_heads``.
        num_heads: Number of attention heads.
        position_buckets: Half the relative-embedding table size (= att span).
        pos_att_type: Subset of ``["c2p", "p2c"]`` to enable.
        share_att_key: Whether the relative-position projections reuse the content
            key/query projections. Defaults to ``True``.
        attention_dropout: Dropout on the attention weights.
        block_prefix: Prefix for the projection layer names.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        position_buckets,
        pos_att_type=("p2c", "c2p"),
        share_att_key=True,
        attention_dropout=0.0,
        block_prefix=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.position_buckets = position_buckets
        self.pos_att_type = list(pos_att_type)
        self.share_att_key = share_att_key
        self.attention_dropout = attention_dropout
        self.block_prefix = block_prefix if block_prefix is not None else "attention"
        self.scale_factor = 1 + len(self.pos_att_type)

        prefix = f"{self.block_prefix}_"
        self.query_proj = layers.Dense(embed_dim, name=prefix + "query_proj")
        self.key_proj = layers.Dense(embed_dim, name=prefix + "key_proj")
        self.value_proj = layers.Dense(embed_dim, name=prefix + "value_proj")
        if not self.share_att_key:
            if "c2p" in self.pos_att_type:
                self.pos_key_proj = layers.Dense(
                    embed_dim, name=prefix + "pos_key_proj"
                )
            if "p2c" in self.pos_att_type:
                self.pos_query_proj = layers.Dense(
                    embed_dim, name=prefix + "pos_query_proj"
                )
        self.dropout = layers.Dropout(attention_dropout)

    def build(self, input_shape):
        dim = input_shape[-1]
        for proj in (self.query_proj, self.key_proj, self.value_proj):
            proj.build((None, dim))
        if not self.share_att_key:
            if "c2p" in self.pos_att_type:
                self.pos_key_proj.build((None, dim))
            if "p2c" in self.pos_att_type:
                self.pos_query_proj.build((None, dim))
        self.built = True

    def compute_output_shape(self, input_shape, *args, **kwargs):
        return input_shape

    def split_heads(self, x):
        batch = ops.shape(x)[0]
        length = ops.shape(x)[1]
        x = ops.reshape(x, (batch, length, self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def gather_rel(self, att, pos):
        idx = ops.expand_dims(ops.cast(pos, "int32"), 1)
        idx = idx + ops.zeros_like(ops.cast(att[..., :1], "int32"))
        return ops.take_along_axis(att, idx, axis=-1)

    def disentangled_att_bias(self, query, key, relative_pos, rel_embeddings):
        span = self.position_buckets
        score = 0
        if "c2p" in self.pos_att_type:
            key_proj = self.key_proj if self.share_att_key else self.pos_key_proj
            pos_key = self.split_heads(key_proj(rel_embeddings)[None])
            c2p = ops.matmul(query, ops.transpose(pos_key, (0, 1, 3, 2)))
            c2p_pos = ops.clip(relative_pos + span, 0, 2 * span - 1)
            score = score + self.gather_rel(c2p, c2p_pos)
        if "p2c" in self.pos_att_type:
            query_proj = self.query_proj if self.share_att_key else self.pos_query_proj
            pos_query = self.split_heads(query_proj(rel_embeddings)[None])
            p2c_pos = ops.clip(-relative_pos + span, 0, 2 * span - 1)
            p2c = ops.matmul(key, ops.transpose(pos_query, (0, 1, 3, 2)))
            p2c = self.gather_rel(p2c, p2c_pos)
            score = score + ops.transpose(p2c, (0, 1, 3, 2))
        return score

    def call(
        self, hidden_states, attention_mask, relative_pos, rel_embeddings, training=None
    ):
        query = self.split_heads(self.query_proj(hidden_states))
        key = self.split_heads(self.key_proj(hidden_states))
        value = self.split_heads(self.value_proj(hidden_states))

        scores = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2)))
        scores = scores + self.disentangled_att_bias(
            query, key, relative_pos, rel_embeddings
        )
        scale = ops.sqrt(ops.cast(self.head_dim * self.scale_factor, scores.dtype))
        scores = scores / scale

        mask = ops.cast(attention_mask, "bool")
        scores = ops.where(mask, scores, ops.cast(MASK_NEG, scores.dtype))
        probs = ops.softmax(scores, axis=-1)
        probs = self.dropout(probs, training=training)

        context = ops.matmul(probs, value)
        context = ops.transpose(context, (0, 2, 1, 3))
        batch = ops.shape(context)[0]
        return ops.reshape(context, (batch, -1, self.embed_dim))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "position_buckets": self.position_buckets,
                "pos_att_type": self.pos_att_type,
                "share_att_key": self.share_att_key,
                "attention_dropout": self.attention_dropout,
                "block_prefix": self.block_prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DebertaV2FlattenChoices(layers.Layer):
    """Merge the multiple-choice axis into the batch: ``(B, C, S) -> (B*C, S)``."""

    def call(self, inputs):
        return ops.reshape(inputs, (-1, ops.shape(inputs)[-1]))

    def compute_output_shape(self, input_shape):
        return (None, input_shape[-1])


@keras.saving.register_keras_serializable(package="zeromodels")
class DebertaV2UnflattenChoices(layers.Layer):
    """Inverse of :class:`DebertaV2FlattenChoices`: ``(B*C, 1) -> (B, C)``.

    Args:
        num_choices: Number of choices ``C`` to fold back out of the batch.
    """

    def __init__(self, num_choices, **kwargs):
        super().__init__(**kwargs)
        self.num_choices = num_choices

    def call(self, inputs):
        return ops.reshape(inputs, (-1, self.num_choices))

    def compute_output_shape(self, input_shape):
        return (None, self.num_choices)

    def get_config(self):
        config = super().get_config()
        config.update({"num_choices": self.num_choices})
        return config
