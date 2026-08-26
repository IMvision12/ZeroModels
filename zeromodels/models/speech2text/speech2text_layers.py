import keras
import numpy as np
from keras import ops

from zeromodels.base.base_attention import fused_attention


@keras.saving.register_keras_serializable(package="zeromodels")
class Speech2TextAttention(keras.layers.Layer):
    """Multi-head attention for Speech2Text self- and cross-attention.

    Reproduces ``Speech2TextAttention`` (a Bart-style attention) bit-for-bit.
    Each instance owns four ``Dense`` projections - Q, K, V, output - all
    **with bias** (unlike Whisper, which omits the K bias). Scaling by
    ``1 / sqrt(head_dim)`` is applied to the Q output before the scaled
    dot-product (``q *= scale`` then ``q @ k.T``).

    The same layer handles two attention modes via the optional
    ``key_value_states`` argument to ``call``:

    * **Self-attention** (default): ``key_value_states is None`` - keys and
      values are projected from ``hidden_states``.
    * **Cross-attention**: ``key_value_states`` is the encoder output -
      queries come from ``hidden_states``, keys / values from the encoder.

    A causal / padding mask broadcastable to ``(B, num_heads, T_q, T_kv)`` may
    be added to the pre-softmax scores via ``attention_mask``.

    Args:
        proj_dim: Total projection dimension (``hidden_dim``). Must be
            divisible by ``num_heads``.
        num_heads: Number of attention heads.
        name_prefix: Optional string prepended to the inner ``Dense`` layer
            names so the source state-dict transfers by name (e.g.
            ``"decoder_layers_0_self_attn_q_proj"``). When ``None``, the inner
            layers are named ``q_proj`` / ``k_proj`` / ``v_proj`` / ``out_proj``.
        **kwargs: Additional ``keras.layers.Layer`` keyword arguments.

    Input shape:
        - ``hidden_states``: ``(B, T_q, proj_dim)``.
        - ``key_value_states`` (optional): ``(B, T_kv, proj_dim)``.
        - ``attention_mask`` (optional): broadcastable to
          ``(B, num_heads, T_q, T_kv)``.

    Output shape:
        ``(B, T_q, proj_dim)``.
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

    def _split_heads(self, x):
        b = ops.shape(x)[0]
        t = ops.shape(x)[1]
        x = ops.reshape(x, (b, t, self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def query(self, hidden_states):
        """Scaled, head-split queries: ``(B, num_heads, T, head_dim)``."""
        return self._split_heads(self.q_proj(hidden_states) * self.scale)

    def project(self, kv):
        """Head-split keys + values from ``kv``: each ``(B, num_heads, T, head_dim)``.

        Used both for the functional forward and to (a) precompute the static
        cross-attention K/V from the encoder output and (b) compute a new token's
        self-attention K/V for the KV cache during generation.
        """
        return self._split_heads(self.k_proj(kv)), self._split_heads(self.v_proj(kv))

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
class Speech2TextSinusoidalPositionEmbedding(keras.layers.Layer):
    """Fixed sinusoidal position embedding (M2M / Speech2Text style).

    Builds a non-trainable ``(num_positions + 2, hidden_dim)`` sinusoid table
    (the classic "Attention Is All You Need" formulation: first-half sines,
    second-half cosines, timescales geometrically interpolated to ``10000``),
    with the ``padding_idx`` row zeroed. Position ids for a non-padded
    sequence start at ``padding_idx + 1``, so the rows added to the input are
    ``table[padding_idx + 1 : padding_idx + 1 + T]``.

    The weights are computed deterministically and are **not** part of the
    PyTorch state-dict (the reference registers them as a non-persistent
    buffer), so the converter never transfers them. Used by both the
    Speech2Text encoder and decoder.

    Args:
        num_positions: Maximum sequence length (``max_source_positions`` for
            the encoder, ``max_target_positions`` for the decoder).
        hidden_dim: Embedding dimension. Must be even.
        padding_idx: Padding token id whose embedding row is zeroed and which
            offsets the position ids. Defaults to ``1``.
        **kwargs: Additional ``keras.layers.Layer`` keyword arguments.

    Input shape:
        ``(B, T, hidden_dim)`` with ``T <= num_positions``.

    Output shape:
        Same as input.
    """

    def __init__(self, num_positions, hidden_dim, padding_idx=1, **kwargs):
        super().__init__(**kwargs)
        self.num_positions = num_positions
        self.hidden_dim = hidden_dim
        self.padding_idx = padding_idx
        self.offset = padding_idx + 1

    def build(self, input_shape):
        num_embeddings = self.num_positions + 2
        half = self.hidden_dim // 2
        log_timescale = np.log(10000.0) / (half - 1)
        inv_timescales = np.exp(-log_timescale * np.arange(half))
        positions = np.arange(num_embeddings)[:, None]
        scaled = positions * inv_timescales[None, :]
        embed = np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1).astype(
            np.float32
        )
        if self.hidden_dim % 2 == 1:
            embed = np.concatenate(
                [embed, np.zeros((num_embeddings, 1), dtype=np.float32)], axis=1
            )
        embed[self.padding_idx, :] = 0.0
        self.pos_embed = self.add_weight(
            shape=(num_embeddings, self.hidden_dim),
            initializer=keras.initializers.Constant(embed),
            trainable=False,
            name="weight",
        )
        super().build(input_shape)

    def call(self, inputs):
        seq_len = ops.shape(inputs)[1]
        pe = self.pos_embed[self.offset : self.offset + seq_len]
        return inputs + pe

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_positions": self.num_positions,
                "hidden_dim": self.hidden_dim,
                "padding_idx": self.padding_idx,
            }
        )
        return config
