import math

import keras
import numpy as np
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


def yarn_inv_freq(
    dim, base, factor, original_max_position_embeddings, beta_fast=32, beta_slow=1
):
    """YaRN inverse frequencies (the HF `_compute_yarn_parameters` recipe)."""

    def find_correction_dim(num_rotations):
        return (
            dim
            * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))
        ) / (2 * math.log(base))

    low = max(math.floor(find_correction_dim(beta_fast)), 0)
    high = min(math.ceil(find_correction_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2, dtype="float32") - low) / (high - low), 0, 1)
    pos_freqs = base ** (np.arange(0, dim, 2, dtype="float32") / dim)
    inv_extra = 1.0 / pos_freqs
    inv_inter = 1.0 / (factor * pos_freqs)
    return inv_inter * ramp + inv_extra * (1 - ramp)


def yarn_get_mscale(scale=1.0, mscale=1.0):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def apply_interleaved_rope(x, cos, sin):
    """DeepSeek interleaved rope: pairs ``(x[2i], x[2i+1])`` rotate by one angle.

    ``cos`` / ``sin`` carry one entry per pair ``(..., dim // 2)``. The output
    is laid out de-interleaved (evens then odds): bit-identical attention to
    the reference complex formulation since q and k transform consistently.
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return ops.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


@keras.saving.register_keras_serializable(package="zeromodels")
class Glm4MoeLiteRMSNorm(layers.Layer):
    """Root-mean-square norm (plain learned weight, ones init).

    Args:
        eps: Variance epsilon (1e-6).
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
class Glm4MoeLiteMLP(layers.Layer):
    """SwiGLU feed-forward: ``down(silu(gate(x)) * up(x))``.

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
class Glm4MoeLiteExperts(layers.Layer):
    """Dense bank of SwiGLU experts via einsum (fused HF layout).

    ``gate_up_proj`` ``(E, 2I, H)`` (gate stacked over up), ``down_proj``
    ``(E, H, I)``; dense ``(T, E)`` routing weights equal sparse dispatch.

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
class Glm4MoeLiteMoE(layers.Layer):
    """DeepSeek-V3 MoE block: sigmoid "noaux" router + shared experts.

    Scores are float32 sigmoids; expert *choice* adds the learned
    ``e_score_correction_bias`` (kept float32), groups are ranked by the sum of their top-2
    biased scores and only the ``topk_group`` best groups stay eligible; the
    gathered weights are the unbiased sigmoid scores, renormalized when
    ``norm_topk_prob`` and scaled by ``routed_scaling_factor``. A
    shared-expert SwiGLU of the block input is added.

    Args:
        num_experts / num_experts_per_tok: Routing shape (256 / 8).
        embed_dim / mlp_dim: Expert dims (``moe_intermediate_size``).
        shared_mlp_dim: Shared-expert hidden width.
        n_group / topk_group: Group-limited routing geometry (8 / 4).
        norm_topk_prob: Renormalize the selected weights.
        routed_scaling_factor: Routed-weight multiplier (2.5).
    """

    def __init__(
        self,
        num_experts,
        num_experts_per_tok,
        embed_dim,
        mlp_dim,
        shared_mlp_dim,
        n_group=8,
        topk_group=4,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
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
        self.experts = Glm4MoeLiteExperts(
            num_experts, embed_dim, mlp_dim, name="experts"
        )
        self.shared_experts = Glm4MoeLiteMLP(
            embed_dim, shared_mlp_dim, name="shared_experts"
        )

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
            # GLM-4.7-Flash keeps the router selection bias float32 (its bf16 + f32
            # checkpoint; transformers' _keep_in_fp32_modules_strict): bf16 rounding
            # here could flip the group / expert top-k. Overrides the build policy.
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
class Glm4MoeLiteAttention(layers.Layer):
    """Multi-head Latent Attention (MLA).

    Queries go through an optional low-rank bottleneck
    (``query_a`` -> RMSNorm -> ``query_b``; plain ``query`` when
    ``q_lora_rank`` is None, e.g. V2-Lite); keys/values are jointly
    compressed by ``kv_a`` into ``kv_lora_rank`` latents plus a single
    *shared* rope key, normalized, and expanded by ``kv_b`` into per-head
    ``(qk_nope_head_dim | v_head_dim)``. Interleaved rope rotates the
    per-head query rope slice and the shared key rope slice (broadcast to
    all heads). Scores use ``softmax_scale`` (qk_head_dim^-0.5 times the
    optional yarn mscale^2 correction).

    Args:
        embed_dim / num_heads: Model geometry.
        q_lora_rank: Query bottleneck width (None disables it).
        kv_lora_rank: KV latent width (512).
        qk_nope_head_dim / qk_rope_head_dim / v_head_dim: Per-head splits.
        softmax_scale: Attention score scale.
        norm_eps: Epsilon of the two MLA bottleneck norms. The reference builds
            them with the RMSNorm class default and never with the model's
            ``rms_norm_eps``, so this stays 1e-6 even for checkpoints whose other
            norms use 1e-5 (Kimi K2.5, GLM-5).
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        softmax_scale,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.softmax_scale = softmax_scale
        self.norm_eps = norm_eps
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        if q_lora_rank is None:
            self.query = layers.Dense(
                num_heads * self.qk_head_dim, use_bias=False, name="query"
            )
        else:
            self.query_a = layers.Dense(q_lora_rank, use_bias=False, name="query_a")
            self.query_a_norm = Glm4MoeLiteRMSNorm(eps=norm_eps, name="query_a_norm")
            self.query_b = layers.Dense(
                num_heads * self.qk_head_dim, use_bias=False, name="query_b"
            )
        self.kv_a = layers.Dense(
            kv_lora_rank + qk_rope_head_dim, use_bias=False, name="kv_a"
        )
        self.kv_a_norm = Glm4MoeLiteRMSNorm(eps=norm_eps, name="kv_a_norm")
        self.kv_b = layers.Dense(
            num_heads * (qk_nope_head_dim + v_head_dim), use_bias=False, name="kv_b"
        )
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")

    def project_qkv(self, hidden_states, cos, sin):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        if self.q_lora_rank is None:
            q = self.query(hidden_states)
        else:
            q = self.query_b(self.query_a_norm(self.query_a(hidden_states)))
        q = ops.transpose(
            ops.reshape(q, (b, s, self.num_heads, self.qk_head_dim)), (0, 2, 1, 3)
        )
        q_nope = q[..., : self.qk_nope_head_dim]
        q_rope = q[..., self.qk_nope_head_dim :]

        compressed = self.kv_a(hidden_states)
        kv_latent = compressed[..., : self.kv_lora_rank]
        k_rope = compressed[..., self.kv_lora_rank :][:, None, :, :]  # (B,1,S,rd)
        kv = self.kv_b(self.kv_a_norm(kv_latent))
        kv = ops.transpose(
            ops.reshape(
                kv, (b, s, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            ),
            (0, 2, 1, 3),
        )
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]

        cos_e = ops.expand_dims(cos, axis=1)
        sin_e = ops.expand_dims(sin, axis=1)
        q_rope = apply_interleaved_rope(q_rope, cos_e, sin_e)
        k_rope = apply_interleaved_rope(k_rope, cos_e, sin_e)
        k_rope = ops.broadcast_to(k_rope, (b, self.num_heads, s, self.qk_rope_head_dim))
        q = ops.concatenate([q_nope, q_rope], axis=-1)
        k = ops.concatenate([k_nope, k_rope], axis=-1)
        return q, k, v

    def attend(self, q, k, v, attention_mask):
        b = ops.shape(q)[0]
        s = ops.shape(q)[2]
        out = fused_attention(q, k, v, self.softmax_scale, attention_mask)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, s, self.num_heads * self.v_head_dim)
        )
        return self.output_proj(out)

    def call(self, hidden_states, cos, sin, attention_mask=None, use_cache=False):
        q, k, v = self.project_qkv(hidden_states, cos, sin)
        out = self.attend(q, k, v, attention_mask)
        return (out, (k, v)) if use_cache else out

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        q, k, v = self.project_qkv(hidden_states, cos, sin)
        # rope runs in float32; match the cache dtype before writing
        k = ops.cast(k, cache_k.dtype)
        v = ops.cast(v, cache_v.dtype)
        cache_k = ops.slice_update(cache_k, (0, 0, write_pos, 0), k)
        cache_v = ops.slice_update(cache_v, (0, 0, write_pos, 0), v)
        out = fused_attention(q, cache_k, cache_v, self.softmax_scale, key_mask)
        b = ops.shape(q)[0]
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, 1, self.num_heads * self.v_head_dim)
        )
        return self.output_proj(out), cache_k, cache_v

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "q_lora_rank": self.q_lora_rank,
                "kv_lora_rank": self.kv_lora_rank,
                "qk_nope_head_dim": self.qk_nope_head_dim,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "v_head_dim": self.v_head_dim,
                "softmax_scale": self.softmax_scale,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Glm4MoeLiteDecoderLayer(layers.Layer):
    """One DeepSeek-V2 block: pre-norm MLA, then pre-norm dense MLP or MoE.

    Args:
        embed_dim / num_heads + MLA dims: see :class:`Glm4MoeLiteAttention`.
        use_moe: MoE block (True) or dense MLP (first ``first_k_dense`` layers).
        mlp_dim: Dense-MLP hidden width (``intermediate_size``).
        moe_mlp_dim / shared_mlp_dim / num_experts / num_experts_per_tok /
        n_group / topk_group / norm_topk_prob / routed_scaling_factor: MoE shape.
        softmax_scale: Attention score scale.
        norm_eps: RMSNorm epsilon.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        softmax_scale,
        use_moe,
        mlp_dim,
        moe_mlp_dim,
        shared_mlp_dim,
        num_experts,
        num_experts_per_tok,
        n_group=8,
        topk_group=4,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.softmax_scale = softmax_scale
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
        self.norm_eps = norm_eps

        self.attention_norm = Glm4MoeLiteRMSNorm(eps=norm_eps, name="attention_norm")
        # norm_eps is deliberately not forwarded: the MLA bottleneck norms keep
        # the 1e-6 default (see Glm4MoeLiteAttention), unlike the block norms.
        self.attention = Glm4MoeLiteAttention(
            embed_dim,
            num_heads,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            softmax_scale,
            name="attention",
        )
        self.mlp_norm = Glm4MoeLiteRMSNorm(eps=norm_eps, name="mlp_norm")
        if use_moe:
            self.mlp = Glm4MoeLiteMoE(
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
            self.mlp = Glm4MoeLiteMLP(embed_dim, mlp_dim, name="mlp")

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
                "q_lora_rank": self.q_lora_rank,
                "kv_lora_rank": self.kv_lora_rank,
                "qk_nope_head_dim": self.qk_nope_head_dim,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "v_head_dim": self.v_head_dim,
                "softmax_scale": self.softmax_scale,
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
                "norm_eps": self.norm_eps,
            }
        )
        return config
