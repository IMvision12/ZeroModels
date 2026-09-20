import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
    """Map raw relative positions to MPNet's bidirectional bucket ids.

    Half the buckets carry each direction; within a direction the first quarter is
    exact and the rest are log-spaced up to ``max_distance``. Mirrors HF
    ``MPNetEncoder.relative_position_bucket``, including its ``n = -relative_position``
    sign convention (the opposite of T5's).
    """
    n = -relative_position
    num_buckets //= 2
    ret = ops.cast(n < 0, "int32") * num_buckets
    n = ops.abs(n)

    max_exact = num_buckets // 2
    is_small = n < max_exact
    val_if_large = max_exact + ops.cast(
        ops.log(ops.cast(ops.maximum(n, 1), "float32") / max_exact)
        / ops.log(ops.cast(max_distance / max_exact, "float32"))
        * (num_buckets - max_exact),
        "int32",
    )
    val_if_large = ops.minimum(val_if_large, num_buckets - 1)
    return ret + ops.where(is_small, ops.cast(n, "int32"), val_if_large)


@keras.saving.register_keras_serializable(package="zeromodels")
class MPNetEmbeddings(layers.Layer):
    """Constructs MPNet's input embeddings.

    Sums learned word and absolute-position embeddings, then applies LayerNorm and
    dropout. MPNet has **no token-type embeddings**. Like RoBERTa, position ids are
    derived from the non-padding mask -- each non-pad token numbered sequentially from
    ``pad_token_id + 1``, pad tokens mapping to ``pad_token_id`` -- computed with a
    masked ``cumsum`` rather than ``arange`` so the layer stays shape-polymorphic
    across the TensorFlow / JAX / PyTorch backends.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Embedding / model dimension.
        max_position_embeddings: Size of the position-embedding table.
        pad_token_id: Padding token id; positions are offset by this value.
        layer_norm_eps: Epsilon for the embedding LayerNorm.
        dropout: Dropout rate applied to the summed embeddings.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        max_position_embeddings,
        pad_token_id=1,
        layer_norm_eps=1e-12,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.layer_norm_eps = layer_norm_eps
        self.dropout_rate = dropout

        self.word_embeddings = layers.Embedding(
            vocab_size, embed_dim, name="word_embeddings"
        )
        self.position_embeddings = layers.Embedding(
            max_position_embeddings, embed_dim, name="position_embeddings"
        )
        self.layer_norm = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="LayerNorm"
        )
        self.dropout = layers.Dropout(dropout)

    def call(self, input_ids, training=None):
        mask = ops.cast(ops.not_equal(input_ids, self.pad_token_id), input_ids.dtype)
        position_ids = ops.cumsum(mask, axis=1) * mask + self.pad_token_id

        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(
            position_ids
        )
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings, training=training)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape) + (self.embed_dim,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "max_position_embeddings": self.max_position_embeddings,
                "pad_token_id": self.pad_token_id,
                "layer_norm_eps": self.layer_norm_eps,
                "dropout": self.dropout_rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MPNetRelativeAttentionBias(layers.Layer):
    """MPNet's shared relative position bias, ``(1, num_heads, seq, seq)``.

    Holds the ``(num_buckets, num_heads)`` bias table. The bias depends only on the
    sequence length, so the encoder computes it **once** and every attention layer adds
    the same tensor -- matching HF, where ``MPNetEncoder.forward`` calls
    ``compute_position_bias`` before the layer loop.

    Positions are built with ``cumsum(ones_like) - 1`` instead of ``arange`` to stay
    shape-polymorphic (the reference uses a plain ``arange``, which is the same values;
    it does **not** use the pad-offset position ids from the embeddings).

    Args:
        num_buckets: Size of the bias table (``relative_attention_num_buckets``).
        num_heads: Number of attention heads (the table's second axis).
        max_distance: Largest relative distance given its own bucket.
    """

    def __init__(self, num_buckets, num_heads, max_distance=128, **kwargs):
        super().__init__(**kwargs)
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.max_distance = max_distance

    def build(self, input_shape):
        self.relative_attention_bias = self.add_weight(
            name="embeddings",
            shape=(self.num_buckets, self.num_heads),
            initializer="zeros",
        )
        self.built = True

    def call(self, input_ids):
        positions = ops.cumsum(ops.ones_like(input_ids), axis=1)[0] - 1
        relative_position = positions[None, :] - positions[:, None]
        bucket = relative_position_bucket(
            relative_position, self.num_buckets, self.max_distance
        )
        # (seq, seq, num_heads) -> (1, num_heads, seq, seq)
        values = ops.take(self.relative_attention_bias, bucket, axis=0)
        return ops.transpose(values, (2, 0, 1))[None]

    def compute_output_shape(self, input_shape):
        return (1, self.num_heads, input_shape[-1], input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_buckets": self.num_buckets,
                "num_heads": self.num_heads,
                "max_distance": self.max_distance,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MPNetSelfAttention(layers.Layer):
    """MPNet multi-head self-attention (HF's ``attention.attn`` sub-block).

    Projects the input to query/key/value, adds the shared relative position bias to
    the scores alongside the additive padding mask, and -- unlike BERT, where the
    output projection sits in ``attention.output`` -- applies its own ``o`` projection
    before returning. The residual and LayerNorm live in the encoder layer (HF's
    ``attention.LayerNorm``).

    Args:
        embed_dim: Model dimension. Must be divisible by ``num_heads``.
        num_heads: Number of attention heads.
        attention_dropout: Dropout rate applied to the attention weights.
        block_prefix: Prefix for the q/k/v/o projection names. Carries the
            encoder-layer index so each layer's weights get a unique path suffix
            (required for backbone weight-sharing across task heads).
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
        self.query = layers.Dense(embed_dim, name=prefix + "q")
        self.key = layers.Dense(embed_dim, name=prefix + "k")
        self.value = layers.Dense(embed_dim, name=prefix + "v")
        self.output_proj = layers.Dense(embed_dim, name=prefix + "o")
        self.dropout = layers.Dropout(attention_dropout)

    def build(self, input_shape):
        # Explicit build: this layer takes several tensors (states + mask + bias), which
        # TF cannot auto-build from a symbolic additive mask.
        hidden_shape = input_shape[0] if isinstance(input_shape, list) else input_shape
        input_dim = hidden_shape[-1]
        self.query.build((None, input_dim))
        self.key.build((None, input_dim))
        self.value.build((None, input_dim))
        self.output_proj.build((None, self.embed_dim))
        self.built = True

    def compute_output_shape(self, input_shape, *args, **kwargs):
        return input_shape[0] if isinstance(input_shape, list) else input_shape

    def transpose_for_scores(self, x):
        batch_size = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]
        x = ops.reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def call(
        self, hidden_states, attention_mask=None, position_bias=None, training=None
    ):
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))

        # The relative bias is part of the additive term, so fold it into the mask and
        # reuse the shared attention kernel.
        bias = attention_mask
        if position_bias is not None:
            bias = (
                position_bias
                if attention_mask is None
                else position_bias + attention_mask
            )

        context = fused_attention(
            query,
            key,
            value,
            self.scale,
            bias,
            dropout=self.dropout,
            training=training,
        )
        context = ops.transpose(context, (0, 2, 1, 3))
        batch_size = ops.shape(context)[0]
        context = ops.reshape(context, (batch_size, -1, self.embed_dim))
        return self.output_proj(context)

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
class MPNetFlattenChoices(layers.Layer):
    """Merge the multiple-choice axis into the batch: ``(B, C, S) -> (B*C, S)``.

    Defining ``compute_output_shape`` keeps the dynamic reshape out of the
    functional-build trace, so it builds on every backend (the JAX backend rejects a
    symbolic ``(-1, None)`` reshape).
    """

    def call(self, inputs):
        return ops.reshape(inputs, (-1, ops.shape(inputs)[-1]))

    def compute_output_shape(self, input_shape):
        return (None, input_shape[-1])


@keras.saving.register_keras_serializable(package="zeromodels")
class MPNetUnflattenChoices(layers.Layer):
    """Inverse of :class:`MPNetFlattenChoices` for the scores: ``(B*C, 1) -> (B, C)``.

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
