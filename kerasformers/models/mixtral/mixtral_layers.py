import keras
from keras import layers, ops

from kerasformers.base.base_attention import fused_attention


def rotate_half(x):
    half = ops.shape(x)[-1] // 2
    return ops.concatenate([-x[..., half:], x[..., :half]], axis=-1)


@keras.saving.register_keras_serializable(package="kerasformers")
class MixtralRMSNorm(layers.Layer):
    """Root-mean-square layer norm (Mixtral style).

    Normalizes the last axis by its RMS in float32 (for numerical stability),
    casts back to the input dtype, then scales by a learned per-channel weight.
    No mean subtraction, no bias. Shape-preserving: ``(..., dim) -> (..., dim)``.

    Args:
        eps: Variance epsilon added before the reciprocal square root.
            Defaults to ``1e-5``.
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


@keras.saving.register_keras_serializable(package="kerasformers")
class MixtralExperts(layers.Layer):
    """Mixtral fused expert bank (dense evaluation).

    Holds the per-expert SwiGLU parameters in Hugging Face's fused layout:
    ``gate_up_proj`` ``(E, 2I, H)`` with contiguous gate/up halves along the
    output dim, ``down_proj`` ``(E, H, I)``, no biases, and, given per-token
    per-expert routing weights ``(T, E)`` (zero for non-selected experts),
    computes every expert and combines the *outputs* by those weights.
    Backend-agnostic ``einsum``; mathematically identical to sparse top-2
    routing with compute O(num_experts).

    Args:
        num_experts: Number of experts ``E``.
        embed_dim: Model width ``H``.
        mlp_dim: Per-expert hidden width ``I``.
    """

    def __init__(self, num_experts, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim

    def build(self, input_shape):
        e, h, i = self.num_experts, self.embed_dim, self.mlp_dim
        self.gate_up_proj = self.add_weight(
            name="gate_up_proj",
            shape=(e, 2 * i, h),
            initializer="zeros",
            trainable=True,
        )
        self.down_proj = self.add_weight(
            name="down_proj", shape=(e, h, i), initializer="zeros", trainable=True
        )
        self.built = True

    def call(self, hidden_states, routing_weights):
        gate_up = ops.einsum("th,eoh->teo", hidden_states, self.gate_up_proj)
        gate = gate_up[..., : self.mlp_dim]
        up = gate_up[..., self.mlp_dim :]
        expert_out = ops.einsum(
            "tei,ehi->teh", ops.silu(gate) * up, self.down_proj
        )  # (T, E, H)
        return ops.einsum("te,teh->th", routing_weights, expert_out)  # (T, H)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class MixtralMoE(layers.Layer):
    """Mixtral sparse MoE block: softmax top-k router + :class:`MixtralExperts`.

    The bias-free router scores every expert; the scores are softmaxed over
    *all* experts in float32, the top-``num_experts_per_tok`` (2) are kept and
    renormalized to sum to one, scattered into a dense ``(T, E)`` weight
    matrix, and applied to the expert bank's outputs.

    Args:
        num_experts: Number of experts.
        num_experts_per_tok: Top-k experts routed per token (Mixtral: 2).
        embed_dim: Model width.
        mlp_dim: Per-expert hidden width.
    """

    def __init__(self, num_experts, num_experts_per_tok, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.experts = MixtralExperts(num_experts, embed_dim, mlp_dim, name="experts")

    def build(self, input_shape):
        self.gate_weight = self.add_weight(
            name="gate_weight",
            shape=(self.num_experts, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def call(self, hidden_states):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        x = ops.reshape(hidden_states, (-1, self.embed_dim))  # (T, H)
        logits = ops.matmul(x, ops.transpose(self.gate_weight))  # (T, E)
        probs = ops.softmax(ops.cast(logits, "float32"), axis=-1)
        top_vals, top_idx = ops.top_k(probs, self.num_experts_per_tok)
        top_vals = top_vals / ops.sum(top_vals, axis=-1, keepdims=True)
        one_hot = ops.one_hot(top_idx, self.num_experts)  # (T, k, E)
        routing = ops.cast(
            ops.sum(one_hot * top_vals[..., None], axis=1), x.dtype
        )  # (T, E)
        out = self.experts(x, routing)
        return ops.reshape(out, (b, s, self.embed_dim))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class MixtralAttention(layers.Layer):
    """Mixtral grouped-query causal self-attention (identical math to Mistral).

    Bias-free ``query`` / ``key`` / ``value`` / ``output_proj`` projections
    with half-rotation rotary position embeddings and K/V head repetition for
    GQA. A KV cache can be threaded through ``past_key_value``.

    Args:
        embed_dim: Model width (output dim of ``output_proj``).
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (GQA).
        head_dim: Per-head dim.

    Call args:
        hidden_states: ``(batch, q_len, embed_dim)``.
        cos, sin: rotary tables ``(batch, q_len, head_dim)``.
        attention_mask: additive mask broadcastable to
            ``(batch, 1, q_len, kv_len)``, or ``None``.
        past_key_value: optional ``(past_k, past_v)``.
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
        # at ``write_pos``. ``key_mask`` (additive, (.., max_len)) blocks the
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


@keras.saving.register_keras_serializable(package="kerasformers")
class MixtralDecoderLayer(layers.Layer):
    """One Mixtral block: pre-norm attention, then pre-norm sparse MoE.

    Computes ``h = x + attention(attention_norm(x))`` followed by
    ``h = h + mlp(mlp_norm(h))`` where ``mlp`` is the top-2-routed
    :class:`MixtralMoE`.

    Args:
        embed_dim: Model / residual-stream width.
        mlp_dim: Per-expert hidden width.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (GQA).
        head_dim: Per-head dim.
        num_experts: Expert count (8).
        num_experts_per_tok: Top-k experts per token (2).
        norm_eps: Epsilon shared by both RMSNorms.

    Call args:
        hidden_states, cos, sin, attention_mask, past_key_value, use_cache: as
            in :class:`MixtralAttention`.

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
        num_experts,
        num_experts_per_tok,
        norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_eps = norm_eps
        self.attention_norm = MixtralRMSNorm(eps=norm_eps, name="attention_norm")
        self.attention = MixtralAttention(
            embed_dim, num_heads, num_kv_heads, head_dim, name="attention"
        )
        self.mlp_norm = MixtralRMSNorm(eps=norm_eps, name="mlp_norm")
        self.mlp = MixtralMoE(
            num_experts, num_experts_per_tok, embed_dim, mlp_dim, name="mlp"
        )

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

    def compute_output_spec(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        return keras.KerasTensor(hidden_states.shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "norm_eps": self.norm_eps,
            }
        )
        return config
