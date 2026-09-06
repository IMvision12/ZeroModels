import keras
from keras import ops

from zeromodels.base.base_attention import fused_attention


@keras.saving.register_keras_serializable(package="zeromodels")
class BartAttention(keras.layers.Layer):
    """Multi-head attention for BART self- and cross-attention.

    Q / K / V / output projections, all **with bias**. The query is scaled by
    ``1 / sqrt(head_dim)`` before the dot product. One layer serves both modes:

    * **Self-attention** (default): keys / values are projected from
      ``hidden_states``.
    * **Cross-attention**: pass the encoder output as ``key_value_states``;
      queries come from ``hidden_states``, keys / values from the encoder.

    The ``query`` / ``project`` / ``attend`` split lets the generation engine
    (:class:`~zeromodels.base.BaseSeq2SeqGeneration`) reuse the projections for a
    KV cache: static cross-attention K/V computed once from the encoder, and new
    self-attention K/V written per decode step.

    Args:
        proj_dim: Total projection dimension (``hidden_dim``); divisible by ``num_heads``.
        num_heads: Number of attention heads.
        name_prefix: Prepended to the inner ``Dense`` names so the source
            state-dict transfers by name (e.g. ``"decoder_layers_0_self_attn"``).
    """

    def __init__(self, proj_dim, num_heads, name_prefix=None, **kwargs):
        super().__init__(**kwargs)
        assert proj_dim % num_heads == 0, "proj_dim must be divisible by num_heads"
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.name_prefix = name_prefix
        self.head_dim = proj_dim // num_heads
        self.scale = self.head_dim**-0.5

        q_name = f"{name_prefix}_q_proj" if name_prefix else "q_proj"
        k_name = f"{name_prefix}_k_proj" if name_prefix else "k_proj"
        v_name = f"{name_prefix}_v_proj" if name_prefix else "v_proj"
        o_name = f"{name_prefix}_out_proj" if name_prefix else "out_proj"

        self.q_proj = keras.layers.Dense(proj_dim, use_bias=True, name=q_name)
        self.k_proj = keras.layers.Dense(proj_dim, use_bias=True, name=k_name)
        self.v_proj = keras.layers.Dense(proj_dim, use_bias=True, name=v_name)
        self.out_proj = keras.layers.Dense(proj_dim, use_bias=True, name=o_name)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.q_proj.build((None, input_dim))
        self.k_proj.build((None, input_dim))
        self.v_proj.build((None, input_dim))
        self.out_proj.build((None, self.proj_dim))
        self.built = True

    def split_heads(self, x):
        b = ops.shape(x)[0]
        t = ops.shape(x)[1]
        x = ops.reshape(x, (b, t, self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def query(self, hidden_states):
        """Scaled, head-split queries: ``(B, num_heads, T, head_dim)``."""
        return self.split_heads(self.q_proj(hidden_states) * self.scale)

    def project(self, kv):
        """Head-split keys + values from ``kv``, each ``(B, num_heads, T, head_dim)``."""
        return self.split_heads(self.k_proj(kv)), self.split_heads(self.v_proj(kv))

    def attend(self, q, k, v, attention_mask=None):
        """Scaled-dot-product attention over already-projected q/k/v + output proj."""
        out = fused_attention(q, k, v, 1.0, attention_mask)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (ops.shape(out)[0], -1, self.proj_dim))
        return self.out_proj(out)

    def call(self, hidden_states, key_value_states=None, attention_mask=None):
        kv = key_value_states if key_value_states is not None else hidden_states
        q = self.query(hidden_states)
        k, v = self.project(kv)
        return self.attend(q, k, v, attention_mask)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "proj_dim": self.proj_dim,
                "num_heads": self.num_heads,
                "name_prefix": self.name_prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class BartLearnedPositionalEmbedding(keras.layers.Layer):
    """Learned absolute position embedding with BART's +2 offset.

    A trainable ``(num_positions + 2, hidden_dim)`` table. BART reserves the
    first two rows (a historical padding-idx hack), so a sequence of length ``T``
    starting at ``past_length`` reads rows ``[offset + past_length : offset +
    past_length + T]`` with ``offset = 2``. The table is a learned checkpoint
    weight (unlike the sinusoidal M2M / Speech2Text variant).

    Args:
        num_positions: Maximum sequence length (``max_position_embeddings``).
        hidden_dim: Embedding dimension.
    """

    offset = 2

    def __init__(self, num_positions, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_positions = num_positions
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.weight = self.add_weight(
            shape=(self.num_positions + self.offset, self.hidden_dim),
            initializer="zeros",
            trainable=True,
            name="weight",
        )
        super().build(input_shape)

    def call(self, inputs):
        seq_len = ops.shape(inputs)[1]
        pe = self.weight[self.offset : self.offset + seq_len]
        return inputs + pe

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {"num_positions": self.num_positions, "hidden_dim": self.hidden_dim}
        )
        return config
