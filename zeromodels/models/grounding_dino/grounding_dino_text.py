import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoTextEmbeddings(layers.Layer):
    """BERT embeddings: word + position + token-type, then LayerNorm."""

    def __init__(self, vocab_size, hidden_size, max_positions, eps=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_positions = max_positions
        self.eps = eps
        self.word_embeddings = layers.Embedding(
            vocab_size, hidden_size, name="word_embeddings"
        )
        self.position_embeddings = layers.Embedding(
            max_positions, hidden_size, name="position_embeddings"
        )
        self.token_type_embeddings = layers.Embedding(
            2, hidden_size, name="token_type_embeddings"
        )
        self.layer_norm = layers.LayerNormalization(epsilon=eps, name="LayerNorm")

    def call(self, input_ids, token_type_ids, position_ids):
        words = self.word_embeddings(input_ids)
        tokens = self.token_type_embeddings(token_type_ids)
        positions = self.position_embeddings(position_ids)
        return self.layer_norm(words + tokens + positions)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "max_positions": self.max_positions,
                "eps": self.eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoTextLayer(layers.Layer):
    """One BERT encoder layer (self-attention + FFN, post-norm)."""

    def __init__(self, hidden_size, num_heads, intermediate_size, eps=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.eps = eps
        self.head_dim = hidden_size // num_heads
        self.query = layers.Dense(hidden_size, name="query")
        self.key = layers.Dense(hidden_size, name="key")
        self.value = layers.Dense(hidden_size, name="value")
        self.attn_output = layers.Dense(hidden_size, name="attn_output")
        self.attn_norm = layers.LayerNormalization(epsilon=eps, name="attn_norm")
        self.intermediate = layers.Dense(intermediate_size, name="intermediate")
        self.output_dense = layers.Dense(hidden_size, name="output_dense")
        self.output_norm = layers.LayerNormalization(epsilon=eps, name="output_norm")

    def call(self, hidden_states, attention_mask=None):
        b = ops.shape(hidden_states)[0]
        s = int(hidden_states.shape[1])

        def split(t):
            return ops.transpose(
                ops.reshape(t, (b, s, self.num_heads, self.head_dim)), (0, 2, 1, 3)
            )

        q = split(self.query(hidden_states))
        k = split(self.key(hidden_states))
        v = split(self.value(hidden_states))
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) / (self.head_dim**0.5)
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), q.dtype)
        ctx = ops.matmul(attn, v)
        ctx = ops.reshape(ops.transpose(ctx, (0, 2, 1, 3)), (b, s, self.hidden_size))
        hidden_states = self.attn_norm(self.attn_output(ctx) + hidden_states)
        inter = ops.gelu(self.intermediate(hidden_states))
        return self.output_norm(self.output_dense(inter) + hidden_states)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "eps": self.eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoTextModel(layers.Layer):
    """BERT text encoder (embeddings + N layers) for Grounding DINO."""

    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_layers,
        num_heads,
        intermediate_size,
        max_positions,
        eps=1e-12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_positions = max_positions
        self.eps = eps
        self.embeddings = GroundingDinoTextEmbeddings(
            vocab_size, hidden_size, max_positions, eps, name="embeddings"
        )
        self.layer = [
            GroundingDinoTextLayer(
                hidden_size, num_heads, intermediate_size, eps, name=f"layer_{i}"
            )
            for i in range(num_layers)
        ]

    def call(self, input_ids, attention_mask, token_type_ids, position_ids):
        hidden = self.embeddings(input_ids, token_type_ids, position_ids)
        for layer in self.layer:
            hidden = layer(hidden, attention_mask=attention_mask)
        return hidden

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "max_positions": self.max_positions,
                "eps": self.eps,
            }
        )
        return config
