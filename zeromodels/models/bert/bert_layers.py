import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


@keras.saving.register_keras_serializable(package="zeromodels")
class BertEmbeddings(layers.Layer):
    """Constructs BERT's input embeddings.

    Sums learned word, absolute-position, and token-type (segment) embeddings,
    then applies LayerNorm and dropout. Position ids are derived from the input
    length with ``cumsum(ones_like) - 1`` rather than ``arange`` so the layer
    stays shape-polymorphic across the TensorFlow / JAX / PyTorch backends.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Embedding / model dimension.
        max_position_embeddings: Size of the position-embedding table.
        type_vocab_size: Number of token-type (segment) ids.
        layer_norm_eps: Epsilon for the embedding LayerNorm.
        dropout: Dropout rate applied to the summed embeddings.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        max_position_embeddings,
        type_vocab_size,
        layer_norm_eps=1e-12,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.dropout_rate = dropout

        self.word_embeddings = layers.Embedding(
            vocab_size, embed_dim, name="word_embeddings"
        )
        self.position_embeddings = layers.Embedding(
            max_position_embeddings, embed_dim, name="position_embeddings"
        )
        self.token_type_embeddings = layers.Embedding(
            type_vocab_size, embed_dim, name="token_type_embeddings"
        )
        self.layer_norm = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="LayerNorm"
        )
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, training=None):
        input_ids, token_type_ids = inputs
        position_ids = ops.cumsum(ops.ones_like(input_ids), axis=1) - 1

        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "max_position_embeddings": self.max_position_embeddings,
                "type_vocab_size": self.type_vocab_size,
                "layer_norm_eps": self.layer_norm_eps,
                "dropout": self.dropout_rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class BertSelfAttention(layers.Layer):
    """BERT multi-head self-attention (the ``attention.self`` sub-block).

    Projects the input to query/key/value, computes scaled dot-product
    attention with an additive padding mask, and returns the concatenated
    context. The subsequent output projection, residual, and LayerNorm live in
    the encoder layer (matching Hugging Face's ``attention.output``).

    Args:
        embed_dim: Model dimension. Must be divisible by ``num_heads``.
        num_heads: Number of attention heads.
        attention_dropout: Dropout rate applied to the attention weights.
        block_prefix: Prefix for the q/k/v projection names. Carries the
            encoder-layer index so each layer's weights get a unique path
            suffix (required for backbone weight-sharing across task heads).
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
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
        self.attention_dropout = attention_dropout
        self.block_prefix = block_prefix if block_prefix is not None else "attention"
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        prefix = f"{self.block_prefix}_"
        self.query = layers.Dense(embed_dim, name=prefix + "query")
        self.key = layers.Dense(embed_dim, name=prefix + "key")
        self.value = layers.Dense(embed_dim, name=prefix + "value")
        self.dropout = layers.Dropout(attention_dropout)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.query.build((None, input_dim))
        self.key.build((None, input_dim))
        self.value.build((None, input_dim))
        self.built = True

    def compute_output_shape(self, input_shape, *args, **kwargs):
        return input_shape

    def transpose_for_scores(self, x):
        batch_size = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]
        x = ops.reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def call(self, hidden_states, attention_mask=None, training=None):
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))

        context = fused_attention(
            query,
            key,
            value,
            self.scale,
            attention_mask,
            dropout=self.dropout,
            training=training,
        )
        context = ops.transpose(context, (0, 2, 1, 3))
        batch_size = ops.shape(context)[0]
        return ops.reshape(context, (batch_size, -1, self.embed_dim))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "attention_dropout": self.attention_dropout,
                "block_prefix": self.block_prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class BertFlattenChoices(layers.Layer):
    """Merge the multiple-choice axis into the batch: ``(B, C, S) -> (B*C, S)``.

    Defining ``compute_output_shape`` keeps the dynamic reshape out of the
    functional-build trace, so it builds on every backend (the JAX backend
    rejects a symbolic ``(-1, None)`` reshape).
    """

    def call(self, inputs):
        return ops.reshape(inputs, (-1, ops.shape(inputs)[-1]))

    def compute_output_shape(self, input_shape):
        return (None, input_shape[-1])


@keras.saving.register_keras_serializable(package="zeromodels")
class BertUnflattenChoices(layers.Layer):
    """Inverse of :class:`BertFlattenChoices` for the scores: ``(B*C, 1) -> (B, C)``.

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
