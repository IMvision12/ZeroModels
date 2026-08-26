import keras
import numpy as np
from keras import ops

from zeromodels.base.base_attention import fused_attention


@keras.saving.register_keras_serializable(package="zeromodels")
class WhisperAttention(keras.layers.Layer):
    """Multi-head attention shared between Whisper self-attention and cross-attention.

    Reproduces ``WhisperAttention`` bit-for-bit. Each instance owns
    four ``Dense`` projections (Q, K, V, output) with biases on
    Q / V / output and no bias on K. Scaling by ``1 / sqrt(head_dim)``
    is applied to the Q output **before** the scaled dot-product
    (``q *= scale`` then ``q @ k.T``), matching the reference
    implementation.

    The same layer handles two attention modes via the optional
    ``key_value_states`` argument to ``call``:

    * **Self-attention** (default): ``key_value_states is None``, keys
      and values are projected from ``hidden_states``.
    * **Cross-attention**: ``key_value_states`` is the encoder output,
      queries come from ``hidden_states`` (the decoder input), keys and
      values come from the encoder.

    A causal / padding mask of any shape broadcastable to
    ``(B, num_heads, T_q, T_kv)`` may be added to the pre-softmax
    scores via ``attention_mask``.

    Args:
        proj_dim: Total projection dimension (``hidden_dim``). Must be
            divisible by ``num_heads``.
        num_heads: Number of attention heads.
        name_prefix: Optional string prepended to the inner ``Dense``
            layer names. Whisper uses this to mirror the reference naming
            convention (e.g. ``"encoder_layers_0_self_attn_q_proj"``).
            When ``None``, the inner layers are named ``q_proj``,
            ``k_proj``, ``v_proj``, ``out_proj``.
        **kwargs: Additional ``keras.layers.Layer`` keyword arguments.

    Input Shape:
        - ``hidden_states``: ``(B, T_q, proj_dim)``.
        - ``key_value_states`` (optional): ``(B, T_kv, proj_dim)``.
        - ``attention_mask`` (optional): broadcastable to
          ``(B, num_heads, T_q, T_kv)``.

    Output Shape:
        ``(B, T_q, proj_dim)``.
    """

    def __init__(self, proj_dim, num_heads, name_prefix=None, **kwargs):
        super().__init__(**kwargs)
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.name_prefix = name_prefix
        self.head_dim = proj_dim // num_heads
        self.scale = self.head_dim**-0.5
        assert proj_dim % num_heads == 0, "proj_dim must be divisible by num_heads"

        q_name = f"{name_prefix}_q_proj" if name_prefix else "q_proj"
        k_name = f"{name_prefix}_k_proj" if name_prefix else "k_proj"
        v_name = f"{name_prefix}_v_proj" if name_prefix else "v_proj"
        o_name = f"{name_prefix}_out_proj" if name_prefix else "out_proj"

        self.q_proj = keras.layers.Dense(proj_dim, use_bias=True, name=q_name)
        self.k_proj = keras.layers.Dense(proj_dim, use_bias=False, name=k_name)
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
        return self._split_heads(self.q_proj(hidden_states) * self.scale)

    def project(self, kv):
        return self._split_heads(self.k_proj(kv)), self._split_heads(self.v_proj(kv))

    def attend(self, q, k, v, attention_mask=None):
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
class WhisperSinusoidalPositionEmbedding(keras.layers.Layer):
    """Fixed sinusoidal position embedding for the Whisper encoder.

    Builds a non-trainable ``(max_source_positions, hidden_dim)`` embedding
    table from the original "Attention Is All You Need" sinusoid
    formulation: the first half of the channels carry sines, the second
    half carry cosines, with timescales geometrically interpolated from
    ``1`` to ``10000``. The first ``T`` rows are sliced and added to
    the input, where ``T`` is the encoder sequence length (post
    stride-2 conv stem, so ``T == 1500`` for a full 30-second chunk).

    Used only by :func:`whisper_encoder`, the decoder uses
    :class:`WhisperLearnedPositionEmbedding` instead.

    Args:
        max_source_positions: Number of position rows to materialize.
            Always ``1500`` for Whisper (= 30 s of 16 kHz audio with
            320-sample stride after the conv stem).
        hidden_dim: Embedding dimension. Must be even (split into a sine
            half and a cosine half).
        **kwargs: Additional ``keras.layers.Layer`` keyword arguments.

    Input Shape:
        ``(B, T, hidden_dim)`` with ``T <= max_source_positions``.

    Output Shape:
        Same as input.
    """

    def __init__(self, max_source_positions, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_source_positions = max_source_positions
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        half = self.hidden_dim // 2
        log_timescale = np.log(10000.0) / (half - 1)
        inv_timescales = np.exp(-log_timescale * np.arange(half))
        positions = np.arange(self.max_source_positions)[:, None]
        scaled = positions * inv_timescales[None, :]
        embed = np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1).astype(
            np.float32
        )
        self.pos_embed = self.add_weight(
            shape=(self.max_source_positions, self.hidden_dim),
            initializer=keras.initializers.Constant(embed),
            trainable=False,
            name="weight",
        )
        super().build(input_shape)

    def call(self, inputs):
        seq_len = ops.shape(inputs)[1]
        pe = self.pos_embed[:seq_len]
        return inputs + pe

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_source_positions": self.max_source_positions,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class WhisperLearnedPositionEmbedding(keras.layers.Layer):
    """Trainable position embedding table for the Whisper decoder.

    Reproduces the ``nn.Embedding(max_target_positions, hidden_dim)`` used
    in the Whisper decoder: a ``(max_target_positions, hidden_dim)`` weight
    is initialized to zero and learned during training. At call time,
    rows ``[start : start + T]`` are added to the token embeddings,
    where ``start = past_key_values_length``. With the current
    no-cache implementation ``start`` is always ``0``; the parameter is
    kept in the signature for API symmetry with the reference KV-cache path.

    Args:
        max_target_positions: Number of position rows in the table.
            Always ``448`` for Whisper, the maximum supported decoded
            sequence length including the prompt prefix.
        hidden_dim: Embedding dimension.
        **kwargs: Additional ``keras.layers.Layer`` keyword arguments.

    Input Shape:
        ``(B, T, hidden_dim)``.

    Output Shape:
        Same as input.
    """

    def __init__(self, max_target_positions, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_target_positions = max_target_positions
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            shape=(self.max_target_positions, self.hidden_dim),
            initializer="zeros",
            trainable=True,
            name="weight",
        )
        super().build(input_shape)

    def call(self, inputs, past_key_values_length=0):
        seq_len = ops.shape(inputs)[1]
        start = past_key_values_length
        pe = self.pos_embed[start : start + seq_len]
        return inputs + pe

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_target_positions": self.max_target_positions,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class WhisperLayerWeights(keras.layers.Layer):
    """Learnable softmax weights over all encoder hidden states.

    Reproduces ``WhisperForAudioClassification``'s
    ``use_weighted_layer_sum`` path: given a list of
    ``num_layers + 1`` encoder hidden states (post-embedding through
    final-LN), holds a learnable ``layer_weights`` vector, softmaxes
    it, and computes the weighted sum across the layer axis.

    Args:
        num_layers: Number of hidden states being combined (typically
            ``encoder_num_layers + 1``).
        **kwargs: Additional ``keras.layers.Layer`` keyword arguments.

    Input Shape:
        List of ``num_layers`` tensors, each ``(B, T, hidden_dim)``.

    Output Shape:
        ``(B, T, hidden_dim)``.
    """

    def __init__(self, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers

    def build(self, input_shape):
        self.layer_weights = self.add_weight(
            shape=(self.num_layers,),
            initializer=keras.initializers.Constant(1.0 / self.num_layers),
            trainable=True,
            name="layer_weights",
        )
        super().build(input_shape)

    def call(self, inputs):
        stacked = ops.stack(inputs, axis=1)  # (B, num_layers, T, hidden_dim)
        weights = ops.softmax(self.layer_weights, axis=-1)
        weights = ops.reshape(weights, (1, self.num_layers, 1, 1))
        return ops.sum(stacked * weights, axis=1)

    def get_config(self):
        config = super().get_config()
        config["num_layers"] = self.num_layers
        return config
