import keras
from keras import layers, ops

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="zeromodels")
class DebertaRelativeEmbedding(layers.Layer):
    """Holds DeBERTa's shared relative-position embedding table.

    Returns the full ``(2 * max_relative_positions, embed_dim)`` table (ignoring
    its input, which only connects it into the functional graph) so every
    encoder layer can share one set of relative embeddings.
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
class DebertaEmbeddings(layers.Layer):
    """Constructs DeBERTa's input embeddings.

    DeBERTa feeds only word embeddings into the encoder (position information is
    injected later through disentangled attention, so ``position_biased_input``
    is False, and there are no token-type embeddings). The embeddings are
    LayerNorm-ed, zeroed at padding positions, then dropped out.

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
class DebertaDisentangledSelfAttention(layers.Layer):
    """DeBERTa (v1) disentangled self-attention.

    Adds content->position (c2p) and position->content (p2c) relative-attention
    terms to the usual content->content scores. Content q/k/v come from a single
    bias-free ``in_proj`` (with separate ``q_bias`` / ``v_bias``); the relative
    embeddings are projected by ``pos_proj`` (c2p) and ``pos_q_proj`` (p2c) and
    gathered per relative position. All scores share the
    ``1/sqrt(head_dim * (1 + #pos_att_type))`` scale.

    The full ``(2 * max_relative_positions, embed_dim)`` relative-embedding table
    is used with a clamped index (equivalent to HF's slice-then-clamp) so the
    layer needs no dynamic-length slicing and stays backend-agnostic.

    Args:
        embed_dim: Model dimension. Must be divisible by ``num_heads``.
        num_heads: Number of attention heads.
        max_relative_positions: Half the relative-embedding table size.
        pos_att_type: Subset of ``["c2p", "p2c"]`` to enable.
        attention_dropout: Dropout on the attention weights.
        block_prefix: Prefix for the projection layer names (carries the encoder
            layer index so backbone weights get unique path suffixes).
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        max_relative_positions,
        pos_att_type=("c2p", "p2c"),
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
        self.max_relative_positions = max_relative_positions
        self.pos_att_type = list(pos_att_type)
        self.attention_dropout = attention_dropout
        self.block_prefix = block_prefix if block_prefix is not None else "attention"
        self.scale_factor = 1 + len(self.pos_att_type)

        prefix = f"{self.block_prefix}_"
        self.in_proj = layers.Dense(
            3 * embed_dim, use_bias=False, name=prefix + "in_proj"
        )
        if "c2p" in self.pos_att_type:
            self.pos_proj = layers.Dense(
                embed_dim, use_bias=False, name=prefix + "pos_proj"
            )
        if "p2c" in self.pos_att_type:
            self.pos_q_proj = layers.Dense(embed_dim, name=prefix + "pos_q_proj")
        self.dropout = layers.Dropout(attention_dropout)

    def build(self, input_shape):
        dim = input_shape[-1]
        self.in_proj.build((None, dim))
        if "c2p" in self.pos_att_type:
            self.pos_proj.build((None, dim))
        if "p2c" in self.pos_att_type:
            self.pos_q_proj.build((None, dim))
        self.q_bias = self.add_weight(
            name="q_bias", shape=(self.embed_dim,), initializer="zeros"
        )
        self.v_bias = self.add_weight(
            name="v_bias", shape=(self.embed_dim,), initializer="zeros"
        )
        self.built = True

    def compute_output_shape(self, input_shape, *args, **kwargs):
        return input_shape

    def split_heads(self, x):
        batch = ops.shape(x)[0]
        length = ops.shape(x)[1]
        x = ops.reshape(x, (batch, length, self.num_heads, -1))
        return ops.transpose(x, (0, 2, 1, 3))

    def gather_rel(self, att, pos):
        idx = ops.expand_dims(ops.cast(pos, "int32"), 1)
        idx = idx + ops.zeros_like(ops.cast(att[..., :1], "int32"))
        return ops.take_along_axis(att, idx, axis=-1)

    def disentangled_att_bias(self, query, key, relative_pos, rel_embeddings):
        span = self.max_relative_positions
        score = 0
        if "c2p" in self.pos_att_type:
            pos_key = self.split_heads(self.pos_proj(rel_embeddings)[None])
            c2p = ops.matmul(query, ops.transpose(pos_key, (0, 1, 3, 2)))
            c2p_pos = ops.clip(relative_pos + span, 0, 2 * span - 1)
            score = score + self.gather_rel(c2p, c2p_pos)
        if "p2c" in self.pos_att_type:
            pos_query = self.split_heads(self.pos_q_proj(rel_embeddings)[None])
            pos_query = pos_query / ops.sqrt(
                ops.cast(self.head_dim * self.scale_factor, pos_query.dtype)
            )
            p2c_pos = ops.clip(-relative_pos + span, 0, 2 * span - 1)
            p2c = ops.matmul(key, ops.transpose(pos_query, (0, 1, 3, 2)))
            p2c = self.gather_rel(p2c, p2c_pos)
            score = score + ops.transpose(p2c, (0, 1, 3, 2))
        return score

    def call(
        self, hidden_states, attention_mask, relative_pos, rel_embeddings, training=None
    ):
        qkv = self.in_proj(hidden_states)
        qkv = self.split_heads(qkv)  # (B, heads, L, 3*head_dim)
        query = qkv[..., : self.head_dim]
        key = qkv[..., self.head_dim : 2 * self.head_dim]
        value = qkv[..., 2 * self.head_dim :]

        query = query + ops.reshape(self.q_bias, (1, self.num_heads, 1, self.head_dim))
        value = value + ops.reshape(self.v_bias, (1, self.num_heads, 1, self.head_dim))

        scale = ops.sqrt(ops.cast(self.head_dim * self.scale_factor, query.dtype))
        query = query / scale
        scores = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2)))
        scores = scores + self.disentangled_att_bias(
            query, key, relative_pos, rel_embeddings
        )

        mask = ops.cast(attention_mask, "bool")
        scores = ops.where(mask, scores, ops.cast(MASK_NEG, scores.dtype))
        probs = ops.softmax(scores, axis=-1)
        probs = self.dropout(probs, training=training)

        context = ops.matmul(probs, value)  # (B, heads, L, head_dim)
        context = ops.transpose(context, (0, 2, 1, 3))
        batch = ops.shape(context)[0]
        return ops.reshape(context, (batch, -1, self.embed_dim))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "max_relative_positions": self.max_relative_positions,
                "pos_att_type": self.pos_att_type,
                "attention_dropout": self.attention_dropout,
                "block_prefix": self.block_prefix,
            }
        )
        return config
