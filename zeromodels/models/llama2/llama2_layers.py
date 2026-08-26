import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


def rotate_half(x):
    half = ops.shape(x)[-1] // 2
    return ops.concatenate([-x[..., half:], x[..., :half]], axis=-1)


@keras.saving.register_keras_serializable(package="zeromodels")
class Llama2RMSNorm(layers.Layer):
    """Root-mean-square layer norm (Llama style).

    Normalizes the last axis by its RMS in float32 (for numerical stability),
    casts back to the input dtype, then scales by a learned per-channel weight.
    No mean subtraction, no bias. Shape-preserving: ``(..., dim) -> (..., dim)``.

    Args:
        eps: Variance epsilon added before the reciprocal square root.
            Defaults to ``1e-5`` (the Llama 2 checkpoints' value).
    """

    def __init__(self, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight", shape=(input_shape[-1],), initializer="ones", trainable=True
        )
        self.built = True

    def call(self, x):
        dtype = x.dtype
        x = ops.cast(x, "float32")
        variance = ops.mean(ops.square(x), axis=-1, keepdims=True)
        x = x * ops.rsqrt(variance + self.eps)
        return self.weight * ops.cast(x, dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Llama2MLP(layers.Layer):
    """SwiGLU feed-forward block: ``down(silu(gate(x)) * up(x))``.

    Two parallel bias-free projections to ``mlp_dim``: a SiLU-gated ``gate``
    and a linear ``up``: are multiplied elementwise, then projected back to
    ``embed_dim`` by ``down``. Shape-preserving on the last axis.

    Args:
        embed_dim: Model / residual-stream width (input and output dim).
        mlp_dim: Hidden expansion width of the ``gate`` / ``up`` projections.
    """

    def __init__(self, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.gate = layers.Dense(mlp_dim, use_bias=False, name="gate")
        self.up = layers.Dense(mlp_dim, use_bias=False, name="up")
        self.down = layers.Dense(embed_dim, use_bias=False, name="down")

    def call(self, x):
        return self.down(ops.silu(self.gate(x)) * self.up(x))

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "mlp_dim": self.mlp_dim})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Llama2Attention(layers.Layer):
    """Llama 2 grouped-query causal self-attention.

    Bias-free ``query`` / ``key`` / ``value`` / ``output_proj`` projections with
    half-rotation rotary position embeddings applied to the per-head query and
    key. When ``num_kv_heads < num_heads`` (the 70B's GQA; the 7B/13B are plain
    MHA with ``num_kv_heads == num_heads``) the K/V heads are repeated to match
    the query heads. A KV cache can be threaded through ``past_key_value`` for
    incremental decoding.

    Args:
        embed_dim: Model width (output dim of ``output_proj``).
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (``<= num_heads``).
        head_dim: Per-head dim.

    Call args:
        hidden_states: ``(batch, q_len, embed_dim)``.
        cos, sin: rotary tables ``(batch, q_len, head_dim)``.
        attention_mask: additive mask broadcastable to
            ``(batch, 1, q_len, kv_len)`` (``0`` keep / large-negative block),
            or ``None``.
        past_key_value: optional ``(past_k, past_v)``, each
            ``(batch, num_kv_heads, past_len, head_dim)``.
        use_cache: when ``True``, also return the updated ``(key, value)``.

    Returns:
        Output ``(batch, q_len, embed_dim)``, or ``(output, (key, value))``
        when ``use_cache`` is set.
    """

    def __init__(self, embed_dim, num_heads, num_kv_heads, head_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim**-0.5
        self.query = layers.Dense(num_heads * head_dim, use_bias=False, name="query")
        self.key = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="key")
        self.value = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="value")
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")

    def call(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        b = ops.shape(hidden_states)[0]
        q_len = ops.shape(hidden_states)[1]
        q = ops.reshape(
            self.query(hidden_states), (b, q_len, self.num_heads, self.head_dim)
        )
        k = ops.reshape(
            self.key(hidden_states), (b, q_len, self.num_kv_heads, self.head_dim)
        )
        v = ops.reshape(
            self.value(hidden_states), (b, q_len, self.num_kv_heads, self.head_dim)
        )
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = ops.concatenate([past_k, k], axis=2)
            v = ops.concatenate([past_v, v], axis=2)
        new_kv = (k, v) if use_cache else None

        if self.num_kv_groups > 1:
            k = ops.repeat(k, self.num_kv_groups, axis=1)
            v = ops.repeat(v, self.num_kv_groups, axis=1)

        out = fused_attention(q, k, v, self.scaling, attention_mask)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, q_len, self.num_heads * self.head_dim)
        )
        out = self.output_proj(out)
        return (out, new_kv) if use_cache else out

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        # Single-token attention against a fixed-size KV cache written in place
        # at ``write_pos``; the constant cache shape keeps the decode loop
        # compilable. ``key_mask`` (additive, (.., max_len)) blocks the
        # still-empty cache slots.
        b = ops.shape(hidden_states)[0]
        q = ops.reshape(
            self.query(hidden_states), (b, 1, self.num_heads, self.head_dim)
        )
        k = ops.reshape(
            self.key(hidden_states), (b, 1, self.num_kv_heads, self.head_dim)
        )
        v = ops.reshape(
            self.value(hidden_states), (b, 1, self.num_kv_heads, self.head_dim)
        )
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))
        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        # rope runs in float32; match the cache dtype before writing
        k = ops.cast(k, cache_k.dtype)
        v = ops.cast(v, cache_v.dtype)
        cache_k = ops.slice_update(cache_k, (0, 0, write_pos, 0), k)
        cache_v = ops.slice_update(cache_v, (0, 0, write_pos, 0), v)
        kk, vv = cache_k, cache_v
        if self.num_kv_groups > 1:
            kk = ops.repeat(kk, self.num_kv_groups, axis=1)
            vv = ops.repeat(vv, self.num_kv_groups, axis=1)
        out = fused_attention(q, kk, vv, self.scaling, key_mask)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, 1, self.num_heads * self.head_dim)
        )
        return self.output_proj(out), cache_k, cache_v

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Llama2DecoderLayer(layers.Layer):
    """One Llama 2 transformer block: pre-norm attention, then pre-norm SwiGLU.

    Computes ``h = x + attention(attention_norm(x))`` followed by
    ``h = h + mlp(mlp_norm(h))``: RMSNorm pre-normalization with residual
    adds. The rotary tables, mask, and KV cache pass straight through to the
    attention.

    Args:
        embed_dim: Model / residual-stream width.
        mlp_dim: SwiGLU hidden width.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Per-head dim.
        norm_eps: Epsilon shared by both RMSNorms.

    Call args:
        hidden_states, cos, sin, attention_mask, past_key_value, use_cache: as
            in :class:`Llama2Attention`.

    Returns:
        The block output, or ``(output, (key, value))`` when ``use_cache`` is
        set.
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.attention_norm = Llama2RMSNorm(eps=norm_eps, name="attention_norm")
        self.attention = Llama2Attention(
            embed_dim, num_heads, num_kv_heads, head_dim, name="attention"
        )
        self.mlp_norm = Llama2RMSNorm(eps=norm_eps, name="mlp_norm")
        self.mlp = Llama2MLP(embed_dim, mlp_dim, name="mlp")

    def call(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        attn_out = self.attention(
            hidden_states,
            cos,
            sin,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        new_kv = None
        if use_cache:
            attn_out, new_kv = attn_out
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return (hidden_states, new_kv) if use_cache else hidden_states

    def compute_output_spec(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        # The residual stream keeps its shape; an explicit spec stops the
        # functional builder from tracing the dynamic-shape reshapes / rope in
        # attention (which resolve only at run time).
        return keras.KerasTensor(hidden_states.shape, dtype=self.compute_dtype)

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        residual = hidden_states
        x = self.attention_norm(hidden_states)
        attn_out, cache_k, cache_v = self.attention.decode_step(
            x, cos, sin, cache_k, cache_v, write_pos, key_mask
        )
        hidden_states = residual + attn_out
        residual = hidden_states
        x = self.mlp_norm(hidden_states)
        hidden_states = residual + self.mlp(x)
        return hidden_states, cache_k, cache_v

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "norm_eps": self.norm_eps,
            }
        )
        return config
