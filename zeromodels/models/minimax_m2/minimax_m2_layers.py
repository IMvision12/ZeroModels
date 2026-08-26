import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


def rotate_half(x):
    half = ops.shape(x)[-1] // 2
    return ops.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_partial_rope(x, cos, sin):
    # Rotate the first ``cos.shape[-1]`` head channels; pass the rest through.
    rot = cos.shape[-1]
    if rot == x.shape[-1]:
        return x * cos + rotate_half(x) * sin
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    x_rot = x_rot * cos + rotate_half(x_rot) * sin
    return ops.concatenate([x_rot, x_pass], axis=-1)


@keras.saving.register_keras_serializable(package="zeromodels")
class MiniMaxM2RMSNorm(layers.Layer):
    """Root-mean-square norm (plain learned weight, ones init).

    Args:
        eps: Variance epsilon. Defaults to ``1e-6`` (the MiniMax-M2
            checkpoints' value).
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
class MiniMaxM2Experts(layers.Layer):
    """Dense bank of SwiGLU experts evaluated for every token via einsum.

    HF layout: fused ``gate_up_proj`` ``(E, 2I, H)`` and ``down_proj``
    ``(E, H, I)``; dense ``(T, E)`` routing weights make this equal to sparse
    top-k dispatch.

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
class MiniMaxM2MoE(layers.Layer):
    """MiniMax-M2 sparse MoE block: sigmoid scoring with a selection bias.

    The router scores every expert with a float32 **sigmoid** (not softmax).
    Expert *choice* adds the learned ``e_score_correction_bias`` to the
    scores (DeepSeek-V3-style aux-free balancing), but the routing weights
    gathered for the chosen experts are the unbiased sigmoid scores,
    renormalized to sum to one.

    Args:
        num_experts: Number of experts (256).
        num_experts_per_tok: Experts routed per token (8).
        embed_dim: Model width.
        mlp_dim: Per-expert hidden width.
    """

    def __init__(self, num_experts, num_experts_per_tok, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.experts = MiniMaxM2Experts(num_experts, embed_dim, mlp_dim, name="experts")

    def build(self, input_shape):
        self.gate_weight = self.add_weight(
            name="gate_weight",
            shape=(self.num_experts, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.e_score_correction_bias = self.add_weight(
            name="e_score_correction_bias",
            shape=(self.num_experts,),
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def call(self, hidden_states):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        x = ops.reshape(hidden_states, (-1, self.embed_dim))
        logits = ops.matmul(x, ops.transpose(self.gate_weight))
        scores = ops.sigmoid(ops.cast(logits, "float32"))
        biased = scores + ops.cast(self.e_score_correction_bias, "float32")[None]
        _, top_idx = ops.top_k(biased, self.num_experts_per_tok)
        top_vals = ops.take_along_axis(scores, top_idx, axis=-1)
        top_vals = top_vals / ops.sum(top_vals, axis=-1, keepdims=True)
        one_hot = ops.one_hot(top_idx, self.num_experts)
        routing = ops.cast(ops.sum(one_hot * top_vals[..., None], axis=1), x.dtype)
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


@keras.saving.register_keras_serializable(package="zeromodels")
class MiniMaxM2Attention(layers.Layer):
    """MiniMax-M2 grouped-query attention with full-width QK RMSNorm.

    Bias-free ``query`` / ``key`` / ``value`` / ``output_proj``;
    ``query_norm`` / ``key_norm`` are RMSNorms over the *whole* projected
    width (all heads jointly), applied before the head split, then rotary
    embeddings on the first ``cos.shape[-1]`` head channels.

    Args:
        embed_dim: Model width.
        num_heads / num_kv_heads / head_dim: Attention geometry.
        norm_eps: QK-norm epsilon.
    """

    def __init__(
        self, embed_dim, num_heads, num_kv_heads, head_dim, norm_eps=1e-6, **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim**-0.5
        self.query = layers.Dense(num_heads * head_dim, use_bias=False, name="query")
        self.key = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="key")
        self.value = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="value")
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")
        self.query_norm = MiniMaxM2RMSNorm(eps=norm_eps, name="query_norm")
        self.key_norm = MiniMaxM2RMSNorm(eps=norm_eps, name="key_norm")

    def project_qkv(self, hidden_states):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        q = ops.reshape(
            self.query_norm(self.query(hidden_states)),
            (b, s, self.num_heads, self.head_dim),
        )
        k = ops.reshape(
            self.key_norm(self.key(hidden_states)),
            (b, s, self.num_kv_heads, self.head_dim),
        )
        v = ops.reshape(
            self.value(hidden_states), (b, s, self.num_kv_heads, self.head_dim)
        )
        return (
            ops.transpose(q, (0, 2, 1, 3)),
            ops.transpose(k, (0, 2, 1, 3)),
            ops.transpose(v, (0, 2, 1, 3)),
        )

    def call(self, hidden_states, cos, sin, attention_mask=None, use_cache=False):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        q, k, v = self.project_qkv(hidden_states)
        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        q = apply_partial_rope(q, cos, sin)
        k = apply_partial_rope(k, cos, sin)
        new_kv = (k, v) if use_cache else None
        if self.num_kv_groups > 1:
            k = ops.repeat(k, self.num_kv_groups, axis=1)
            v = ops.repeat(v, self.num_kv_groups, axis=1)
        out = fused_attention(q, k, v, self.scaling, attention_mask)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, s, self.num_heads * self.head_dim)
        )
        out = self.output_proj(out)
        return (out, new_kv) if use_cache else out

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        b = ops.shape(hidden_states)[0]
        q, k, v = self.project_qkv(hidden_states)
        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        q = apply_partial_rope(q, cos, sin)
        k = apply_partial_rope(k, cos, sin)
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
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MiniMaxM2DecoderLayer(layers.Layer):
    """One MiniMax-M2 block: pre-norm attention, then pre-norm sparse MoE.

    Args:
        embed_dim / mlp_dim / num_heads / num_kv_heads / head_dim: Dims.
        num_experts / num_experts_per_tok: MoE shape (256 / 8).
        norm_eps: RMSNorm epsilon.
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
        norm_eps=1e-6,
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
        self.attention_norm = MiniMaxM2RMSNorm(eps=norm_eps, name="attention_norm")
        self.attention = MiniMaxM2Attention(
            embed_dim, num_heads, num_kv_heads, head_dim, norm_eps, name="attention"
        )
        self.mlp_norm = MiniMaxM2RMSNorm(eps=norm_eps, name="mlp_norm")
        self.mlp = MiniMaxM2MoE(
            num_experts, num_experts_per_tok, embed_dim, mlp_dim, name="mlp"
        )

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
