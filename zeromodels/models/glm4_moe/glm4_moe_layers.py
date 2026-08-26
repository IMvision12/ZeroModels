import keras
import numpy as np
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


def rotate_half(x):
    # NeoX (non-interleaved) rotate: split into halves and swap with a sign.
    half = ops.shape(x)[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return ops.concatenate([-x2, x1], axis=-1)


def apply_glm_rope(x, cos, sin, rotary_dim):
    # Partial NeoX rope: rotate the first ``rotary_dim`` channels (cos / sin are
    # ``cat((freqs, freqs))``) and pass the rest through.
    dtype = x.dtype
    x_rot = ops.cast(x[..., :rotary_dim], "float32")
    x_pass = x[..., rotary_dim:]
    cos = ops.cast(cos, "float32")
    sin = ops.cast(sin, "float32")
    out = x_rot * cos + rotate_half(x_rot) * sin
    return ops.concatenate([ops.cast(out, dtype), x_pass], axis=-1)


@keras.saving.register_keras_serializable(package="zeromodels")
class Glm4MoeRMSNorm(layers.Layer):
    """Root-mean-square norm (plain learned weight, ones init).

    Args:
        eps: Variance epsilon.
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
class Glm4MoeMLP(layers.Layer):
    """SwiGLU feed-forward: ``down(silu(gate(x)) * up(x))`` (separate gate/up).

    Used for the leading dense layers and the shared expert.

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
class Glm4MoeExperts(layers.Layer):
    """Dense bank of SwiGLU experts via einsum (fused HF layout).

    ``gate_up_proj`` ``(E, 2I, H)`` (gate stacked over up), ``down_proj``
    ``(E, H, I)``; the dense ``(T, E)`` routing matrix equals sparse dispatch.

    Args:
        num_experts / embed_dim / mlp_dim: Bank shape.
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
class Glm4MoeMoE(layers.Layer):
    """GLM-4.5 MoE block: sigmoid grouped-topk router + shared expert.

    Scores are float32 sigmoids; expert *choice* adds the learned
    ``e_score_correction_bias``, groups are ranked by the sum of their top-2
    biased scores and only ``topk_group`` groups stay eligible; the gathered
    weights are the *unbiased* sigmoid scores, renormalized when
    ``norm_topk_prob`` and scaled by ``routed_scaling_factor``. A shared-expert
    SwiGLU of the block input is added.

    Args:
        num_experts / num_experts_per_tok: Routing shape.
        embed_dim / mlp_dim: Expert dims (``moe_intermediate_size``).
        shared_mlp_dim: Shared-expert hidden width.
        n_group / topk_group / norm_topk_prob / routed_scaling_factor: Routing.
    """

    def __init__(
        self,
        num_experts,
        num_experts_per_tok,
        embed_dim,
        mlp_dim,
        shared_mlp_dim,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.shared_mlp_dim = shared_mlp_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.experts = Glm4MoeExperts(num_experts, embed_dim, mlp_dim, name="experts")
        self.shared_experts = Glm4MoeMLP(
            embed_dim, shared_mlp_dim, name="shared_experts"
        )

    def build(self, input_shape):
        self.gate_weight = self.add_weight(
            name="gate_weight",
            shape=(self.num_experts, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        # Kept in float32 (transformers' `_keep_in_fp32_modules_strict`): this bias
        # decides group / expert selection, so bf16 rounding could flip the top-k.
        # `dtype` overrides the bf16 build policy; the load path stores/reads it fp32.
        self.e_score_correction_bias = self.add_weight(
            name="e_score_correction_bias",
            shape=(self.num_experts,),
            initializer="zeros",
            trainable=True,
            dtype="float32",
        )
        self.built = True

    def call(self, hidden_states):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        x = ops.reshape(hidden_states, (-1, self.embed_dim))
        shared_out = self.shared_experts(x)
        logits = ops.matmul(
            ops.cast(x, "float32"), ops.transpose(ops.cast(self.gate_weight, "float32"))
        )
        scores = ops.sigmoid(logits)  # (T, E) fp32
        biased = scores + ops.cast(self.e_score_correction_bias, "float32")[None]
        grouped = ops.reshape(
            biased, (-1, self.n_group, self.num_experts // self.n_group)
        )
        group_scores = ops.sum(ops.top_k(grouped, 2)[0], axis=-1)  # (T, G)
        _, group_idx = ops.top_k(group_scores, self.topk_group)
        group_mask = ops.max(
            ops.one_hot(group_idx, self.n_group, dtype="float32"), axis=1
        )  # (T, G)
        score_mask = ops.repeat(group_mask, self.num_experts // self.n_group, axis=-1)
        choice = ops.where(score_mask > 0, biased, -np.inf)
        _, top_idx = ops.top_k(choice, self.num_experts_per_tok)
        top_vals = ops.take_along_axis(scores, top_idx, axis=-1)
        if self.norm_topk_prob:
            top_vals = top_vals / (ops.sum(top_vals, axis=-1, keepdims=True) + 1e-20)
        top_vals = top_vals * self.routed_scaling_factor
        one_hot = ops.one_hot(top_idx, self.num_experts, dtype="float32")
        routing = ops.cast(ops.sum(one_hot * top_vals[..., None], axis=1), x.dtype)
        out = self.experts(x, routing) + shared_out
        return ops.reshape(out, (b, s, self.embed_dim))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "shared_mlp_dim": self.shared_mlp_dim,
                "n_group": self.n_group,
                "topk_group": self.topk_group,
                "norm_topk_prob": self.norm_topk_prob,
                "routed_scaling_factor": self.routed_scaling_factor,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Glm4MoeAttention(layers.Layer):
    """GLM-4.5 grouped-query attention with partial NeoX rope + optional QK-norm.

    The ``query`` / ``key`` / ``value`` projections carry an optional bias
    (``attention_bias``); ``output_proj`` is bias-free. When ``use_qk_norm`` the
    per-head query and key are RMSNorm'd (over ``head_dim``) before rotary.

    Args:
        embed_dim: Model width.
        num_heads / num_kv_heads / head_dim: Attention geometry.
        rotary_dim: Channels of each head that receive rotary embeddings.
        use_qk_norm: Apply the per-head RMSNorm on q/k.
        norm_eps: QK-norm epsilon.
        attention_bias: Whether the q/k/v projections carry bias.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        use_qk_norm=False,
        norm_eps=1e-5,
        attention_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.use_qk_norm = use_qk_norm
        self.norm_eps = norm_eps
        self.attention_bias = attention_bias
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim**-0.5
        self.query = layers.Dense(
            num_heads * head_dim, use_bias=attention_bias, name="query"
        )
        self.key = layers.Dense(
            num_kv_heads * head_dim, use_bias=attention_bias, name="key"
        )
        self.value = layers.Dense(
            num_kv_heads * head_dim, use_bias=attention_bias, name="value"
        )
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")
        if use_qk_norm:
            self.query_norm = Glm4MoeRMSNorm(eps=norm_eps, name="query_norm")
            self.key_norm = Glm4MoeRMSNorm(eps=norm_eps, name="key_norm")

    def project(self, hidden_states):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        q = ops.reshape(
            self.query(hidden_states), (b, s, self.num_heads, self.head_dim)
        )
        k = ops.reshape(
            self.key(hidden_states), (b, s, self.num_kv_heads, self.head_dim)
        )
        v = ops.reshape(
            self.value(hidden_states), (b, s, self.num_kv_heads, self.head_dim)
        )
        if self.use_qk_norm:
            q = self.query_norm(q)
            k = self.key_norm(k)
        return (
            ops.transpose(q, (0, 2, 1, 3)),
            ops.transpose(k, (0, 2, 1, 3)),
            ops.transpose(v, (0, 2, 1, 3)),
        )

    def attend(self, q, k, v, attention_mask):
        if self.num_kv_groups > 1:
            k = ops.repeat(k, self.num_kv_groups, axis=1)
            v = ops.repeat(v, self.num_kv_groups, axis=1)
        return fused_attention(q, k, v, self.scaling, attention_mask)

    def call(self, hidden_states, cos, sin, attention_mask=None, use_cache=False):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        q, k, v = self.project(hidden_states)
        cos_e = ops.expand_dims(cos, axis=1)
        sin_e = ops.expand_dims(sin, axis=1)
        q = apply_glm_rope(q, cos_e, sin_e, self.rotary_dim)
        k = apply_glm_rope(k, cos_e, sin_e, self.rotary_dim)
        new_kv = (k, v) if use_cache else None
        out = self.attend(q, k, v, attention_mask)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, s, self.num_heads * self.head_dim)
        )
        out = self.output_proj(out)
        return (out, new_kv) if use_cache else out

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        b = ops.shape(hidden_states)[0]
        q, k, v = self.project(hidden_states)
        cos_e = ops.expand_dims(cos, axis=1)
        sin_e = ops.expand_dims(sin, axis=1)
        q = apply_glm_rope(q, cos_e, sin_e, self.rotary_dim)
        k = apply_glm_rope(k, cos_e, sin_e, self.rotary_dim)
        # rope runs in float32; match the cache dtype before writing
        k = ops.cast(k, cache_k.dtype)
        v = ops.cast(v, cache_v.dtype)
        cache_k = ops.slice_update(cache_k, (0, 0, write_pos, 0), k)
        cache_v = ops.slice_update(cache_v, (0, 0, write_pos, 0), v)
        out = self.attend(q, cache_k, cache_v, key_mask)
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
                "rotary_dim": self.rotary_dim,
                "use_qk_norm": self.use_qk_norm,
                "norm_eps": self.norm_eps,
                "attention_bias": self.attention_bias,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Glm4MoeDecoderLayer(layers.Layer):
    """One GLM-4.5 block: pre-norm attention then pre-norm dense MLP or MoE.

    Args:
        embed_dim / num_heads / num_kv_heads / head_dim / rotary_dim: Attention.
        use_qk_norm / attention_bias: Attention options.
        use_moe: MoE block (True) or dense MLP (first ``first_k_dense`` layers).
        mlp_dim: Dense-MLP hidden width (``intermediate_size``).
        moe_mlp_dim / shared_mlp_dim / num_experts / num_experts_per_tok /
        n_group / topk_group / norm_topk_prob / routed_scaling_factor: MoE shape.
        norm_eps: RMSNorm epsilon.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        use_moe,
        mlp_dim,
        moe_mlp_dim,
        shared_mlp_dim,
        num_experts,
        num_experts_per_tok,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        use_qk_norm=False,
        attention_bias=False,
        norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.use_moe = use_moe
        self.mlp_dim = mlp_dim
        self.moe_mlp_dim = moe_mlp_dim
        self.shared_mlp_dim = shared_mlp_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.use_qk_norm = use_qk_norm
        self.attention_bias = attention_bias
        self.norm_eps = norm_eps

        self.input_layernorm = Glm4MoeRMSNorm(eps=norm_eps, name="input_layernorm")
        self.attention = Glm4MoeAttention(
            embed_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            use_qk_norm=use_qk_norm,
            norm_eps=norm_eps,
            attention_bias=attention_bias,
            name="attention",
        )
        self.post_attention_layernorm = Glm4MoeRMSNorm(
            eps=norm_eps, name="post_attention_layernorm"
        )
        if use_moe:
            self.mlp = Glm4MoeMoE(
                num_experts,
                num_experts_per_tok,
                embed_dim,
                moe_mlp_dim,
                shared_mlp_dim,
                n_group,
                topk_group,
                norm_topk_prob,
                routed_scaling_factor,
                name="mlp",
            )
        else:
            self.mlp = Glm4MoeMLP(embed_dim, mlp_dim, name="mlp")

    def call(self, hidden_states, cos, sin, attention_mask=None, use_cache=False):
        residual = hidden_states
        attn_out = self.attention(
            self.input_layernorm(hidden_states),
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
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return (hidden_states, new_kv) if use_cache else hidden_states

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        residual = hidden_states
        attn_out, cache_k, cache_v = self.attention.decode_step(
            self.input_layernorm(hidden_states),
            cos,
            sin,
            cache_k,
            cache_v,
            write_pos,
            key_mask,
        )
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
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
                "rotary_dim": self.rotary_dim,
                "use_moe": self.use_moe,
                "mlp_dim": self.mlp_dim,
                "moe_mlp_dim": self.moe_mlp_dim,
                "shared_mlp_dim": self.shared_mlp_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "n_group": self.n_group,
                "topk_group": self.topk_group,
                "norm_topk_prob": self.norm_topk_prob,
                "routed_scaling_factor": self.routed_scaling_factor,
                "use_qk_norm": self.use_qk_norm,
                "attention_bias": self.attention_bias,
                "norm_eps": self.norm_eps,
            }
        )
        return config
