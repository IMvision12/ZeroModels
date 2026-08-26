import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


def rotate_half(x):
    half = ops.shape(x)[-1] // 2
    return ops.concatenate([-x[..., half:], x[..., :half]], axis=-1)


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen2MoeRMSNorm(layers.Layer):
    """Root-mean-square layer norm (Llama/Qwen style, ones-init weight).

    Args:
        eps: Variance epsilon. Defaults to ``1e-6``.
    """

    def __init__(self, eps=1e-6, **kwargs):
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
class Qwen2MoeMLP(layers.Layer):
    """SwiGLU feed-forward: ``down(silu(gate(x)) * up(x))``.

    Used both as the dense-layer MLP and as the MoE shared expert.

    Args:
        embed_dim: Output width.
        mlp_dim: Hidden width.
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
class Qwen2MoeExperts(layers.Layer):
    """Dense bank of SwiGLU experts evaluated for every token via einsum.

    Weights are stored fused as the HF layout: ``gate_up_proj`` ``(E, 2I, H)``
    (gate stacked over up) and ``down_proj`` ``(E, H, I)``. Routing weights
    are a dense ``(T, E)`` matrix (zero for unrouted experts), so the result
    equals sparse top-k dispatch.

    Args:
        num_experts: Number of routed experts ``E``.
        embed_dim: Model width ``H``.
        mlp_dim: Per-expert hidden width ``I`` (``moe_intermediate_size``).
    """

    def __init__(self, num_experts, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim

    def build(self, input_shape):
        self.gate_up_proj = self.add_weight(
            name="gate_up_proj",
            shape=(self.num_experts, 2 * self.mlp_dim, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.down_proj = self.add_weight(
            name="down_proj",
            shape=(self.num_experts, self.embed_dim, self.mlp_dim),
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def call(self, hidden_states, routing_weights):
        gate_up = ops.einsum("th,eoh->teo", hidden_states, self.gate_up_proj)
        gate = gate_up[..., : self.mlp_dim]
        up = gate_up[..., self.mlp_dim :]
        expert_out = ops.einsum("tei,ehi->teh", ops.silu(gate) * up, self.down_proj)
        return ops.einsum("te,teh->th", routing_weights, expert_out)

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
class Qwen2MoeSparseBlock(layers.Layer):
    """Qwen2-MoE sparse block: softmax top-k router + sigmoid-gated shared expert.

    The bias-free router softmaxes over all experts in float32, keeps the
    top-``num_experts_per_tok`` (optionally renormalized to sum to one), and
    applies them to the expert bank. A full-width shared expert runs on every
    token, scaled by ``sigmoid(shared_expert_gate(x))``, and is added to the
    routed output.

    Args:
        num_experts / num_experts_per_tok: Routing shape.
        embed_dim: Model width.
        moe_mlp_dim: Per-routed-expert hidden width (``moe_intermediate_size``).
        shared_mlp_dim: Shared-expert hidden width.
        norm_topk_prob: Renormalize the selected weights to sum to one.
    """

    def __init__(
        self,
        num_experts,
        num_experts_per_tok,
        embed_dim,
        moe_mlp_dim,
        shared_mlp_dim,
        norm_topk_prob=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.embed_dim = embed_dim
        self.moe_mlp_dim = moe_mlp_dim
        self.shared_mlp_dim = shared_mlp_dim
        self.norm_topk_prob = norm_topk_prob
        self.experts = Qwen2MoeExperts(
            num_experts, embed_dim, moe_mlp_dim, name="experts"
        )
        self.shared_expert = Qwen2MoeMLP(
            embed_dim, shared_mlp_dim, name="shared_expert"
        )
        self.shared_expert_gate = layers.Dense(
            1, use_bias=False, name="shared_expert_gate"
        )

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
        x = ops.reshape(hidden_states, (-1, self.embed_dim))
        logits = ops.matmul(x, ops.transpose(self.gate_weight))
        probs = ops.softmax(ops.cast(logits, "float32"), axis=-1)
        top_vals, top_idx = ops.top_k(probs, self.num_experts_per_tok)
        if self.norm_topk_prob:
            top_vals = top_vals / ops.sum(top_vals, axis=-1, keepdims=True)
        top_vals = ops.cast(top_vals, x.dtype)
        one_hot = ops.one_hot(top_idx, self.num_experts, dtype=x.dtype)
        routing = ops.sum(one_hot * top_vals[..., None], axis=1)
        routed = self.experts(x, routing)
        shared = ops.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        out = routed + shared
        return ops.reshape(out, (b, s, self.embed_dim))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "embed_dim": self.embed_dim,
                "moe_mlp_dim": self.moe_mlp_dim,
                "shared_mlp_dim": self.shared_mlp_dim,
                "norm_topk_prob": self.norm_topk_prob,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen2MoeAttention(layers.Layer):
    """Grouped-query causal self-attention with biased QKV (Qwen2-MoE).

    ``query`` / ``key`` / ``value`` are biased projections; ``output_proj`` is
    bias-free. Half-rotation rotary embeddings on Q and K, K/V head repetition
    for GQA, and an optional ``past_key_value`` KV cache.

    Args:
        embed_dim: Model width.
        num_heads / num_kv_heads / head_dim: Attention geometry.
    """

    def __init__(self, embed_dim, num_heads, num_kv_heads, head_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or embed_dim // num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.query = layers.Dense(
            num_heads * self.head_dim, use_bias=True, name="query"
        )
        self.key = layers.Dense(num_kv_heads * self.head_dim, use_bias=True, name="key")
        self.value = layers.Dense(
            num_kv_heads * self.head_dim, use_bias=True, name="value"
        )
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")

    def split_heads(self, x, num_heads):
        b = ops.shape(x)[0]
        s = ops.shape(x)[1]
        return ops.transpose(
            ops.reshape(x, (b, s, num_heads, self.head_dim)), (0, 2, 1, 3)
        )

    def call(self, hidden_states, cos, sin, attention_mask=None, use_cache=False):
        b = ops.shape(hidden_states)[0]
        q_len = ops.shape(hidden_states)[1]
        query = self.split_heads(self.query(hidden_states), self.num_heads)
        key = self.split_heads(self.key(hidden_states), self.num_kv_heads)
        value = self.split_heads(self.value(hidden_states), self.num_kv_heads)
        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        query = query * cos + rotate_half(query) * sin
        key = key * cos + rotate_half(key) * sin
        new_kv = (key, value) if use_cache else None
        if self.num_kv_groups > 1:
            key = ops.repeat(key, self.num_kv_groups, axis=1)
            value = ops.repeat(value, self.num_kv_groups, axis=1)
        out = fused_attention(query, key, value, self.scaling, attention_mask)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, q_len, self.num_heads * self.head_dim)
        )
        out = self.output_proj(out)
        return (out, new_kv) if use_cache else out

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        b = ops.shape(hidden_states)[0]
        query = self.split_heads(self.query(hidden_states), self.num_heads)
        key = self.split_heads(self.key(hidden_states), self.num_kv_heads)
        value = self.split_heads(self.value(hidden_states), self.num_kv_heads)
        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        query = query * cos + rotate_half(query) * sin
        key = key * cos + rotate_half(key) * sin
        # rope runs in float32; match the cache dtype before writing
        key = ops.cast(key, cache_k.dtype)
        value = ops.cast(value, cache_v.dtype)
        cache_k = ops.slice_update(cache_k, (0, 0, write_pos, 0), key)
        cache_v = ops.slice_update(cache_v, (0, 0, write_pos, 0), value)
        kk, vv = cache_k, cache_v
        if self.num_kv_groups > 1:
            kk = ops.repeat(kk, self.num_kv_groups, axis=1)
            vv = ops.repeat(vv, self.num_kv_groups, axis=1)
        out = fused_attention(query, kk, vv, self.scaling, key_mask)
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
class Qwen2MoeDecoderLayer(layers.Layer):
    """One Qwen2-MoE block: pre-norm GQA attention, then pre-norm MLP/MoE.

    ``use_moe`` selects the sparse block (router + experts + shared expert) or
    a plain dense SwiGLU MLP (the ``mlp_only_layers`` / ``decoder_sparse_step``
    schedule decides which, in the model).

    Args:
        embed_dim / num_heads / num_kv_heads / head_dim: Attention geometry.
        use_moe: Sparse MoE block (True) or dense MLP.
        mlp_dim: Dense-MLP hidden width (``intermediate_size``).
        num_experts / num_experts_per_tok / moe_mlp_dim / shared_mlp_dim /
        norm_topk_prob: MoE shape (used when ``use_moe``).
        norm_eps: RMSNorm epsilon.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        use_moe,
        mlp_dim,
        num_experts,
        num_experts_per_tok,
        moe_mlp_dim,
        shared_mlp_dim,
        norm_topk_prob=False,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or embed_dim // num_heads
        self.use_moe = use_moe
        self.mlp_dim = mlp_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_mlp_dim = moe_mlp_dim
        self.shared_mlp_dim = shared_mlp_dim
        self.norm_topk_prob = norm_topk_prob
        self.norm_eps = norm_eps
        self.attention_norm = Qwen2MoeRMSNorm(eps=norm_eps, name="attention_norm")
        self.attention = Qwen2MoeAttention(
            embed_dim, num_heads, num_kv_heads, head_dim=self.head_dim, name="attention"
        )
        self.mlp_norm = Qwen2MoeRMSNorm(eps=norm_eps, name="mlp_norm")
        if use_moe:
            self.mlp = Qwen2MoeSparseBlock(
                num_experts,
                num_experts_per_tok,
                embed_dim,
                moe_mlp_dim,
                shared_mlp_dim,
                norm_topk_prob,
                name="mlp",
            )
        else:
            self.mlp = Qwen2MoeMLP(embed_dim, mlp_dim, name="mlp")

    def call(self, hidden_states, cos, sin, attention_mask=None, use_cache=False):
        residual = hidden_states
        attn_out = self.attention(
            self.attention_norm(hidden_states),
            cos,
            sin,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )
        new_kv = None
        if use_cache:
            attn_out, new_kv = attn_out
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = residual + self.mlp(self.mlp_norm(hidden_states))
        return (hidden_states, new_kv) if use_cache else hidden_states

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        residual = hidden_states
        attn_out, cache_k, cache_v = self.attention.decode_step(
            self.attention_norm(hidden_states),
            cos,
            sin,
            cache_k,
            cache_v,
            write_pos,
            key_mask,
        )
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = residual + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states, cache_k, cache_v

    def compute_output_spec(
        self, hidden_states, cos, sin, attention_mask=None, use_cache=False
    ):
        return keras.KerasTensor(hidden_states.shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "use_moe": self.use_moe,
                "mlp_dim": self.mlp_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "moe_mlp_dim": self.moe_mlp_dim,
                "shared_mlp_dim": self.shared_mlp_dim,
                "norm_topk_prob": self.norm_topk_prob,
                "norm_eps": self.norm_eps,
            }
        )
        return config
