import keras
import numpy as np
from keras import ops

from zeromodels.base.base_attention import fused_attention


def moonshine_rope_tables(rotary_dim, max_positions, base):
    half = rotary_dim // 2
    inv_freq = 1.0 / (
        base ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim)
    )
    positions = np.arange(max_positions, dtype=np.float32)[:, None]
    freqs = positions * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(emb).astype(np.float32)
    sin = np.sin(emb).astype(np.float32)
    cos = np.repeat(cos[:, :half], 2, axis=-1)
    sin = np.repeat(sin[:, :half], 2, axis=-1)
    return cos, sin


def rotate_half(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    stacked = ops.stack([-x2, x1], axis=-1)
    shape = ops.shape(x)
    return ops.reshape(stacked, shape)


def apply_rotary_pos_emb(t, cos, sin):
    rotary_dim = ops.shape(cos)[-1]
    t_rot = t[..., :rotary_dim]
    t_pass = t[..., rotary_dim:]
    t_embed = (t_rot * cos) + (rotate_half(t_rot) * sin)
    return ops.concatenate([t_embed, t_pass], axis=-1)


@keras.saving.register_keras_serializable(package="zeromodels")
class MoonshineRotaryEmbedding(keras.layers.Layer):
    """Partial rotary position embedding (GLM-style, interleaved) for Moonshine.

    Materializes non-trainable ``cos`` / ``sin`` tables of shape
    ``(max_positions, rotary_dim)`` from the default RoPE formulation, where
    ``rotary_dim = int(head_dim * partial_rotary_factor)`` rounded down to an
    even number. Only the first ``rotary_dim`` channels of each head are
    rotated (the remaining ``head_dim - rotary_dim`` pass through unchanged),
    matching ``MoonshineRotaryEmbedding`` / ``GlmRotaryEmbedding``.

    The interleaving convention follows GLM: cosine / sine are computed from
    ``cat(freqs, freqs)`` then ``repeat_interleave``-d by 2 so that even / odd
    channel pairs share a frequency, with ``rotate_half`` pairing ``(-x_odd,
    x_even)``. The layer exposes the sliced ``cos`` / ``sin`` for a given
    sequence length; the rotation itself happens inside
    :class:`MoonshineAttention` via :func:`apply_rotary_pos_emb`.

    Args:
        rotary_dim: Number of head channels to rotate (must be even).
        max_positions: Number of position rows to materialize.
        base: RoPE theta. Defaults to ``10000.0``.
        **kwargs: Additional ``keras.layers.Layer`` keyword arguments.
    """

    def __init__(self, rotary_dim, max_positions, base=10000.0, **kwargs):
        super().__init__(**kwargs)
        self.rotary_dim = rotary_dim
        self.max_positions = max_positions
        self.base = base

    def build(self, input_shape):
        cos, sin = moonshine_rope_tables(self.rotary_dim, self.max_positions, self.base)
        self.cos_table = self.add_weight(
            shape=(self.max_positions, self.rotary_dim),
            initializer=keras.initializers.Constant(cos),
            trainable=False,
            name="cos",
        )
        self.sin_table = self.add_weight(
            shape=(self.max_positions, self.rotary_dim),
            initializer=keras.initializers.Constant(sin),
            trainable=False,
            name="sin",
        )
        super().build(input_shape)

    def call(self, inputs):
        # The stored tables only hold ``max_positions`` rows (194 for the
        # released variants), but the encoder sequence grows with the audio
        # duration, so slicing them silently ran out past roughly 4.7 s and
        # attention raised a shape mismatch. Derive cos/sin for the length
        # actually being attended instead. The tables stay on the layer so
        # existing checkpoints keep loading, and the values are identical to
        # them for every position they do cover.
        seq_len = ops.shape(inputs)[1]
        inv_freq = 1.0 / (
            self.base
            ** (
                ops.cast(ops.arange(0, self.rotary_dim, 2), "float32") / self.rotary_dim
            )
        )
        positions = ops.cast(ops.arange(seq_len), "float32")[:, None]
        freqs = positions * inv_freq[None, :]
        cos = ops.repeat(ops.cos(freqs), 2, axis=-1)[None, None, :, :]
        sin = ops.repeat(ops.sin(freqs), 2, axis=-1)[None, None, :, :]
        return cos, sin

    def compute_output_shape(self, input_shape):
        return (
            (1, 1, input_shape[1], self.rotary_dim),
            (1, 1, input_shape[1], self.rotary_dim),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rotary_dim": self.rotary_dim,
                "max_positions": self.max_positions,
                "base": self.base,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MoonshineAttention(keras.layers.Layer):
    """Multi-head attention shared by Moonshine self-attention and cross-attention.

    Reproduces ``MoonshineAttention`` (a ``GlmAttention`` variant). Each
    instance owns four bias-free ``Dense`` projections: Q, K, V, output.
    Scaling by ``1 / sqrt(head_dim)`` is applied inside the scaled dot-product
    on the (unpadded) head dimension. Grouped-query attention is supported via
    ``num_kv_heads`` (repeated to ``num_heads`` before the dot product); the
    released tiny / base checkpoints use plain multi-head attention
    (``num_kv_heads == num_heads``).

    The same layer handles two modes through ``call``:

    * **Self-attention** (default): ``key_value_states is None``, keys and
      values are projected from ``hidden_states`` and rotary position
      embeddings (``cos`` / ``sin``) are applied to Q and K.
    * **Cross-attention**: ``key_value_states`` is the encoder output,
      queries come from ``hidden_states`` (the decoder input), keys / values
      from the encoder, and no rotary embedding is applied.

    A causal / padding mask broadcastable to ``(B, num_heads, T_q, T_kv)`` may
    be added to the pre-softmax scores via ``attention_mask``.

    Args:
        proj_dim: Total projection dimension (``hidden_dim``).
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads (GQA). Defaults to
            ``num_heads``.
        name_prefix: Optional string prepended to the inner ``Dense`` layer
            names to mirror the reference naming convention.
        **kwargs: Additional ``keras.layers.Layer`` keyword arguments.

    Input Shape:
        - ``hidden_states``: ``(B, T_q, proj_dim)``.
        - ``key_value_states`` (optional): ``(B, T_kv, proj_dim)``.

    Output Shape:
        ``(B, T_q, proj_dim)``.
    """

    def __init__(
        self, proj_dim, num_heads, num_kv_heads=None, name_prefix=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.name_prefix = name_prefix
        self.head_dim = proj_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.num_groups = self.num_heads // self.num_kv_heads

        q_name = f"{name_prefix}_q_proj" if name_prefix else "q_proj"
        k_name = f"{name_prefix}_k_proj" if name_prefix else "k_proj"
        v_name = f"{name_prefix}_v_proj" if name_prefix else "v_proj"
        o_name = f"{name_prefix}_o_proj" if name_prefix else "o_proj"

        self.q_proj = keras.layers.Dense(
            self.num_heads * self.head_dim, use_bias=False, name=q_name
        )
        self.k_proj = keras.layers.Dense(
            self.num_kv_heads * self.head_dim, use_bias=False, name=k_name
        )
        self.v_proj = keras.layers.Dense(
            self.num_kv_heads * self.head_dim, use_bias=False, name=v_name
        )
        self.o_proj = keras.layers.Dense(proj_dim, use_bias=False, name=o_name)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.q_proj.build((None, input_dim))
        self.k_proj.build((None, input_dim))
        self.v_proj.build((None, input_dim))
        self.o_proj.build((None, self.num_heads * self.head_dim))
        self.built = True

    def split_heads(self, x, num):
        b = ops.shape(x)[0]
        t = ops.shape(x)[1]
        x = ops.reshape(x, (b, t, num, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def repeat_kv(self, x):
        if self.num_groups == 1:
            return x
        b = ops.shape(x)[0]
        t = ops.shape(x)[2]
        x = x[:, :, None, :, :]
        x = ops.broadcast_to(
            x, (b, self.num_kv_heads, self.num_groups, t, self.head_dim)
        )
        return ops.reshape(x, (b, self.num_heads, t, self.head_dim))

    def query(self, hidden_states):
        return self.split_heads(self.q_proj(hidden_states), self.num_heads)

    def project(self, kv):
        return (
            self.split_heads(self.k_proj(kv), self.num_kv_heads),
            self.split_heads(self.v_proj(kv), self.num_kv_heads),
        )

    def attend(self, q, k, v, attention_mask=None):
        k = self.repeat_kv(k)
        v = self.repeat_kv(v)
        out = fused_attention(q, k, v, self.scale, attention_mask)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (ops.shape(out)[0], -1, self.num_heads * self.head_dim))
        return self.o_proj(out)

    def call(
        self,
        hidden_states,
        key_value_states=None,
        attention_mask=None,
        cos=None,
        sin=None,
    ):
        kv = key_value_states if key_value_states is not None else hidden_states
        q = self.query(hidden_states)
        k, v = self.project(kv)
        if cos is not None and key_value_states is None:
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)
        return self.attend(q, k, v, attention_mask)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "proj_dim": self.proj_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "name_prefix": self.name_prefix,
            }
        )
        return config
