import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


def rotate_half(x):
    half = ops.shape(x)[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return ops.concatenate([-x2, x1], axis=-1)


@keras.saving.register_keras_serializable(package="zeromodels")
class ModernBertEmbeddings(layers.Layer):
    """Constructs ModernBERT's input embeddings.

    Looks up word embeddings and applies a (bias-free) LayerNorm and dropout.
    Unlike BERT there are no absolute-position or token-type embeddings, position
    is injected later via rotary embeddings inside the attention layers.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Embedding / model dimension.
        norm_eps: Epsilon for the embedding LayerNorm.
        dropout: Dropout rate applied to the embeddings.
    """

    def __init__(self, vocab_size, embed_dim, norm_eps=1e-5, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.norm_eps = norm_eps
        self.dropout_rate = dropout

        self.tok_embeddings = layers.Embedding(
            vocab_size, embed_dim, name="tok_embeddings"
        )
        self.norm = layers.LayerNormalization(
            epsilon=norm_eps, center=False, name="norm"
        )
        self.dropout = layers.Dropout(dropout)

    def call(self, input_ids, training=None):
        x = self.tok_embeddings(input_ids)
        x = self.norm(x)
        return self.dropout(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "norm_eps": self.norm_eps,
                "dropout": self.dropout_rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class ModernBertAttention(layers.Layer):
    """ModernBERT rotary multi-head self-attention.

    Projects the input with a single fused ``Wqkv`` (no bias), splits into
    query/key/value, applies rotary position embeddings (the per-layer ``cos`` /
    ``sin`` are computed in the backbone and passed in), computes scaled
    dot-product attention with an additive mask (a sliding-window mask on local
    layers, a plain padding mask on global layers), and projects the context
    back with ``Wo`` (no bias).

    Args:
        embed_dim: Model dimension. Must be divisible by ``num_heads``.
        num_heads: Number of attention heads.
        attention_dropout: Dropout rate applied to the attention weights.
        block_prefix: Prefix for the projection names. Carries the encoder-layer
            index so each layer's weights get a unique path suffix (required for
            backbone weight-sharing across task heads).
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
        self.block_prefix = block_prefix if block_prefix is not None else "attn"
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        prefix = f"{self.block_prefix}_"
        self.Wqkv = layers.Dense(3 * embed_dim, use_bias=False, name=prefix + "Wqkv")
        self.Wo = layers.Dense(embed_dim, use_bias=False, name=prefix + "Wo")
        self.dropout = layers.Dropout(attention_dropout)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.Wqkv.build((None, input_dim))
        self.Wo.build((None, self.embed_dim))
        self.built = True

    def compute_output_shape(self, input_shape, *args, **kwargs):
        return input_shape

    def call(
        self,
        hidden_states,
        attention_mask=None,
        rope_cos=None,
        rope_sin=None,
        training=None,
    ):
        batch_size = ops.shape(hidden_states)[0]
        seq_len = ops.shape(hidden_states)[1]

        qkv = self.Wqkv(hidden_states)
        qkv = ops.reshape(qkv, (batch_size, seq_len, 3, self.num_heads, self.head_dim))
        query = ops.transpose(qkv[:, :, 0], (0, 2, 1, 3))
        key = ops.transpose(qkv[:, :, 1], (0, 2, 1, 3))
        value = ops.transpose(qkv[:, :, 2], (0, 2, 1, 3))

        query = query * rope_cos + rotate_half(query) * rope_sin
        key = key * rope_cos + rotate_half(key) * rope_sin

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
        context = ops.reshape(context, (batch_size, -1, self.embed_dim))
        return self.Wo(context)

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
class ModernBertMLP(layers.Layer):
    """ModernBERT gated (GeGLU) feed-forward block.

    A single ``Wi`` projects to ``2 * mlp_dim`` (no bias), the halves form an
    input and a gate, the activation is applied to the input and multiplied by
    the gate, then ``Wo`` projects back to ``embed_dim`` (no bias).

    Args:
        embed_dim: Model dimension.
        mlp_dim: Feed-forward hidden dimension (``Wi`` outputs ``2 * mlp_dim``).
        hidden_act: Activation applied to the gated input.
        dropout: Dropout rate applied after the gate.
        block_prefix: Prefix for the projection names (carries the layer index).
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        hidden_act="gelu",
        dropout=0.0,
        block_prefix=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.hidden_act = hidden_act
        self.dropout_rate = dropout
        self.block_prefix = block_prefix if block_prefix is not None else "mlp"

        prefix = f"{self.block_prefix}_"
        self.Wi = layers.Dense(mlp_dim * 2, use_bias=False, name=prefix + "Wi")
        self.act = layers.Activation(hidden_act)
        self.Wo = layers.Dense(embed_dim, use_bias=False, name=prefix + "Wo")
        self.dropout = layers.Dropout(dropout)

    def build(self, input_shape):
        self.Wi.build((None, self.embed_dim))
        self.Wo.build((None, self.mlp_dim))
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, hidden_states, training=None):
        inp, gate = ops.split(self.Wi(hidden_states), 2, axis=-1)
        x = self.act(inp) * gate
        x = self.dropout(x, training=training)
        return self.Wo(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "hidden_act": self.hidden_act,
                "dropout": self.dropout_rate,
                "block_prefix": self.block_prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class ModernBertFlattenChoices(layers.Layer):
    """Merge the multiple-choice axis into the batch: ``(B, C, S) -> (B*C, S)``."""

    def call(self, inputs):
        return ops.reshape(inputs, (-1, ops.shape(inputs)[-1]))

    def compute_output_shape(self, input_shape):
        return (None, input_shape[-1])


@keras.saving.register_keras_serializable(package="zeromodels")
class ModernBertUnflattenChoices(layers.Layer):
    """Inverse of :class:`ModernBertFlattenChoices`: ``(B*C, 1) -> (B, C)``.

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
