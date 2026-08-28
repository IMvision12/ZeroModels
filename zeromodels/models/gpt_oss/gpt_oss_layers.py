import keras
from keras import layers, ops


def rotate_half(x):
    half = ops.shape(x)[-1] // 2
    return ops.concatenate([-x[..., half:], x[..., :half]], axis=-1)


@keras.saving.register_keras_serializable(package="zeromodels")
class GptOssRMSNorm(layers.Layer):
    """Root-mean-square layer norm (computed in float32), scaled by a learned weight."""

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
class GptOssExperts(layers.Layer):
    """GPT-OSS mixture-of-experts feed-forward bank (dense evaluation).

    Holds the packed per-expert parameters in Hugging Face's layout
    (``gate_up_proj`` ``(E, H, 2I)``, ``down_proj`` ``(E, I, H)`` + biases) and,
    given per-token expert weights ``(T, E)`` (zero for non-selected experts),
    computes every expert and combines them by those weights: mathematically
    identical to the sparse top-k routing but written with backend-agnostic
    ``einsum``. The expert activation is GPT-OSS's clamped gated-SiLU on the
    interleaved gate/up halves.

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
        self.alpha = 1.702
        self.limit = 7.0

    def build(self, input_shape):
        e, h, i = self.num_experts, self.embed_dim, self.mlp_dim
        self.gate_up_proj = self.add_weight(
            name="gate_up_proj",
            shape=(e, h, 2 * i),
            initializer="zeros",
            trainable=True,
        )
        self.gate_up_proj_bias = self.add_weight(
            name="gate_up_proj_bias",
            shape=(e, 2 * i),
            initializer="zeros",
            trainable=True,
        )
        self.down_proj = self.add_weight(
            name="down_proj", shape=(e, i, h), initializer="zeros", trainable=True
        )
        self.down_proj_bias = self.add_weight(
            name="down_proj_bias", shape=(e, h), initializer="zeros", trainable=True
        )
        self.built = True

    def call(self, hidden_states, routing_weights):
        gate_up = (
            ops.einsum("th,ehi->tei", hidden_states, self.gate_up_proj)
            + self.gate_up_proj_bias
        )
        gate_up = ops.reshape(gate_up, (-1, self.num_experts, self.mlp_dim, 2))
        gate = gate_up[..., 0]
        up = gate_up[..., 1]
        gate = ops.minimum(gate, self.limit)
        up = ops.clip(up, -self.limit, self.limit)
        glu = gate * ops.sigmoid(gate * self.alpha)
        gated = (up + 1.0) * glu  # (T, E, I)
        expert_out = (
            ops.einsum("tei,eih->teh", gated, self.down_proj) + self.down_proj_bias
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


@keras.saving.register_keras_serializable(package="zeromodels")
class GptOssMLP(layers.Layer):
    """GPT-OSS sparse MoE block: top-k router + :class:`GptOssExperts`.

    The router scores every expert, the top-``num_experts_per_tok`` are kept and
    softmax-normalized (over the selected experts), scattered into a dense
    ``(T, E)`` weight matrix, and applied by the expert bank.

    Args:
        num_experts: Number of experts.
        num_experts_per_tok: Top-k experts routed per token.
        embed_dim: Model width.
        mlp_dim: Per-expert hidden width.
    """

    def __init__(
        self,
        num_experts,
        num_experts_per_tok,
        embed_dim,
        mlp_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.router = layers.Dense(num_experts, use_bias=True, name="router")
        # Always builds the float bank; a ZmQuantizer swaps in GptOssMXFP4Experts at
        # load time for an mxfp4 checkpoint (the model stays quantization-agnostic).
        self.experts = GptOssExperts(num_experts, embed_dim, mlp_dim, name="experts")

    def build(self, input_shape):
        self.router.build(input_shape)
        self.experts.build(input_shape)
        self.built = True

    def call(self, hidden_states):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        x = ops.reshape(hidden_states, (-1, self.embed_dim))  # (T, H)
        router_logits = self.router(x)  # (T, E)
        top_vals, top_idx = ops.top_k(router_logits, self.num_experts_per_tok)
        routing = ops.softmax(top_vals, axis=-1)  # (T, k)
        one_hot = ops.one_hot(
            top_idx, self.num_experts, dtype=routing.dtype
        )  # (T, k, E)
        full_weights = ops.sum(one_hot * routing[..., None], axis=1)  # (T, E)
        out = self.experts(x, full_weights)  # (T, H)
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


@keras.saving.register_keras_serializable(package="zeromodels")
class GptOssAttention(layers.Layer):
    """GPT-OSS grouped-query attention with learned per-head attention sinks.

    Bias-carrying q/k/v/o projections, rotary positions, GQA (K/V heads repeated
    to match the query heads), and a learned per-head "sink" logit appended to the
    attention scores before softmax (then dropped), letting a head attend to
    "nothing". The additive mask (causal, full or sliding-window) is supplied by
    the caller; a KV cache can be threaded through ``past_key_value``.

    Args:
        embed_dim: Model width.
        num_heads / num_kv_heads: Query / key-value head counts (GQA).
        head_dim: Per-head dim.
        attention_bias: Whether q/k/v/o carry a bias (GPT-OSS: True).
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        attention_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.attention_bias = attention_bias
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim**-0.5
        self.q_proj = layers.Dense(
            num_heads * head_dim, use_bias=attention_bias, name="q_proj"
        )
        self.k_proj = layers.Dense(
            num_kv_heads * head_dim, use_bias=attention_bias, name="k_proj"
        )
        self.v_proj = layers.Dense(
            num_kv_heads * head_dim, use_bias=attention_bias, name="v_proj"
        )
        self.o_proj = layers.Dense(embed_dim, use_bias=attention_bias, name="o_proj")

    def build(self, input_shape):
        self.sinks = self.add_weight(
            name="sinks", shape=(self.num_heads,), initializer="zeros", trainable=True
        )
        self.built = True

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
            self.q_proj(hidden_states), (b, q_len, self.num_heads, self.head_dim)
        )
        k = ops.reshape(
            self.k_proj(hidden_states), (b, q_len, self.num_kv_heads, self.head_dim)
        )
        v = ops.reshape(
            self.v_proj(hidden_states), (b, q_len, self.num_kv_heads, self.head_dim)
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

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scaling
        if attention_mask is not None:
            attn = attn + attention_mask
        sink_col = ops.zeros_like(attn[..., :1]) + ops.reshape(
            self.sinks, (1, self.num_heads, 1, 1)
        )
        combined = ops.concatenate([attn, sink_col], axis=-1)
        probs = ops.softmax(ops.cast(combined, "float32"), axis=-1)
        scores = ops.cast(probs[..., :-1], v.dtype)
        out = ops.matmul(scores, v)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, q_len, self.num_heads * self.head_dim)
        )
        out = self.o_proj(out)
        return (out, new_kv) if use_cache else out

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        # Single-token sink attention against a fixed-size KV cache. ``key_mask``
        # (additive, (.., max_len)) is the full or sliding-window mask for this layer
        # and also blocks the still-empty cache slots.
        b = ops.shape(hidden_states)[0]
        q = ops.reshape(
            self.q_proj(hidden_states), (b, 1, self.num_heads, self.head_dim)
        )
        k = ops.reshape(
            self.k_proj(hidden_states), (b, 1, self.num_kv_heads, self.head_dim)
        )
        v = ops.reshape(
            self.v_proj(hidden_states), (b, 1, self.num_kv_heads, self.head_dim)
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
        attn = ops.matmul(q, ops.transpose(kk, (0, 1, 3, 2))) * self.scaling
        attn = attn + key_mask
        sink_col = ops.zeros_like(attn[..., :1]) + ops.reshape(
            self.sinks, (1, self.num_heads, 1, 1)
        )
        combined = ops.concatenate([attn, sink_col], axis=-1)
        probs = ops.softmax(ops.cast(combined, "float32"), axis=-1)
        scores = ops.cast(probs[..., :-1], vv.dtype)
        out = ops.matmul(scores, vv)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, 1, self.num_heads * self.head_dim)
        )
        return self.o_proj(out), cache_k, cache_v

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "attention_bias": self.attention_bias,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GptOssDecoderLayer(layers.Layer):
    """One GPT-OSS block: pre-norm sink attention, then pre-norm MoE MLP."""

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
        attention_bias=True,
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
        self.attention_bias = attention_bias
        self.input_layernorm = GptOssRMSNorm(eps=norm_eps, name="input_layernorm")
        self.self_attn = GptOssAttention(
            embed_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            attention_bias,
            name="self_attn",
        )
        self.post_attention_layernorm = GptOssRMSNorm(
            eps=norm_eps, name="post_attention_layernorm"
        )
        self.mlp = GptOssMLP(
            num_experts,
            num_experts_per_tok,
            embed_dim,
            mlp_dim,
            name="mlp",
        )

    def build(self, input_shape):
        self.input_layernorm.build(input_shape)
        self.self_attn.build(input_shape)
        self.post_attention_layernorm.build(input_shape)
        self.mlp.build(input_shape)
        self.built = True

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
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
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
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return (hidden_states, new_kv) if use_cache else hidden_states

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        attn_out, cache_k, cache_v = self.self_attn.decode_step(
            x, cos, sin, cache_k, cache_v, write_pos, key_mask
        )
        hidden_states = residual + attn_out
        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
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
                "attention_bias": self.attention_bias,
            }
        )
        return config
