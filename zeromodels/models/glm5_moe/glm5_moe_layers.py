import keras
import numpy as np
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention

MASK_NEG = -1e9


def apply_rope(x, cos, sin, unsqueeze_axis):
    """Interleaved rope: pairs ``(x[2i], x[2i+1])`` rotate by one angle.

    Inherited from DeepSeek, and used by both the MLA attention and the DSA
    indexer -- *not* the NeoX split-half rotation the rest of the GLM family
    uses. ``cos`` / ``sin`` are ``(B, S, D)`` (``cat(freqs, freqs)``), so only
    the first half carries the per-pair angle; they are unsqueezed at
    ``unsqueeze_axis`` (1 for ``[B, H, S, D]``, 2 for ``[B, S, H, D]``). The
    output is laid out de-interleaved (evens then odds) -- attention is
    unchanged since q and k transform consistently.
    """
    half = int(cos.shape[-1]) // 2
    cos = ops.expand_dims(cos[..., :half], unsqueeze_axis)
    sin = ops.expand_dims(sin[..., :half], unsqueeze_axis)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return ops.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


@keras.saving.register_keras_serializable(package="zeromodels")
class Glm5MoeRMSNorm(layers.Layer):
    """Root-mean-square norm (learned weight, ones init)."""

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
class Glm5MoeMLP(layers.Layer):
    """SwiGLU feed-forward: ``down(silu(gate(x)) * up(x))``."""

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
class Glm5MoeExperts(layers.Layer):
    """Dense bank of SwiGLU experts via einsum (fused HF layout)."""

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
class Glm5MoeMoE(layers.Layer):
    """DeepSeekMoE block: float32 sigmoid "noaux" router + shared experts.

    Scores are float32 sigmoids; expert choice adds the learned
    ``e_score_correction_bias``, groups are ranked by their top-2 biased scores
    and only the ``topk_group`` best groups stay eligible (GLM-5 uses
    ``n_group=1``/``topk_group=1`` so this degenerates to plain top-k); gathered
    weights are the unbiased sigmoids, renormalized when ``norm_topk_prob`` and
    scaled by ``routed_scaling_factor``, plus a shared-expert SwiGLU.
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
        self.experts = Glm5MoeExperts(num_experts, embed_dim, mlp_dim, name="experts")
        self.shared_experts = Glm5MoeMLP(
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
        scores = ops.sigmoid(logits)
        biased = scores + ops.cast(self.e_score_correction_bias, "float32")[None]
        grouped = ops.reshape(
            biased, (-1, self.n_group, self.num_experts // self.n_group)
        )
        group_scores = ops.sum(ops.top_k(grouped, 2)[0], axis=-1)
        _, group_idx = ops.top_k(group_scores, self.topk_group)
        group_mask = ops.max(
            ops.one_hot(group_idx, self.n_group, dtype="float32"), axis=1
        )
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
class Glm5MoeIndexer(layers.Layer):
    """DSA (DeepSeek Sparse Attention) indexer: scores tokens and returns the
    top-``index_topk`` key indices per query.

    Lightweight projections separate from MLA: ``wq_b`` (from the query LoRA
    residual), ``wk`` + a ``k_norm`` LayerNorm, and a per-head ``weights_proj``.
    Uses the same NeoX rope as the main attention on the ``qk_rope_head_dim``
    slice. Mirrors the bf16 reference (no Hadamard/FP8): score is
    ``Σ_h weight_h · relu(softmax_scale · q_h·kᵀ)``. When ``index_topk >= T``
    every token is selected, so the resulting mask is a no-op.
    """

    def __init__(
        self,
        hidden_size,
        n_heads,
        head_dim,
        qk_rope_head_dim,
        index_topk,
        q_lora_rank,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.softmax_scale = head_dim**-0.5
        self.wq_b = layers.Dense(n_heads * head_dim, use_bias=False, name="wq_b")
        self.wk = layers.Dense(head_dim, use_bias=False, name="wk")
        self.k_norm = layers.LayerNormalization(epsilon=1e-6, name="k_norm")
        self.weights_proj = layers.Dense(n_heads, use_bias=False, name="weights_proj")

    def call(self, hidden_states, q_resid, cos, sin, attention_mask):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        rd = self.qk_rope_head_dim

        q = ops.reshape(self.wq_b(q_resid), (b, s, self.n_heads, self.head_dim))
        q_pe, q_nope = q[..., :rd], q[..., rd:]
        q_pe = apply_rope(q_pe, cos, sin, unsqueeze_axis=2)  # BSHD
        q = ops.concatenate([q_pe, q_nope], axis=-1)

        k = self.k_norm(self.wk(hidden_states))  # (B, S, head_dim)
        k_pe, k_nope = k[..., :rd], k[..., rd:]
        k_pe = apply_rope(k_pe[:, :, None, :], cos, sin, unsqueeze_axis=2)[:, :, 0, :]
        k = ops.concatenate([k_pe, k_nope], axis=-1)

        weights = self.weights_proj(ops.cast(hidden_states, "float32")) * (
            self.n_heads**-0.5
        )  # (B, S, H)
        scores = (
            ops.einsum("bshd,btd->bsht", ops.cast(q, "float32"), ops.cast(k, "float32"))
            * self.softmax_scale
        )
        scores = ops.relu(scores)
        index_scores = ops.einsum("bsht,bsh->bst", scores, weights)  # (B, S, T)
        if attention_mask is not None:
            index_scores = index_scores + ops.cast(attention_mask, "float32")
        topk = min(self.index_topk, int(index_scores.shape[-1]))
        return ops.top_k(index_scores, topk)[1]  # (B, S, topk)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "n_heads": self.n_heads,
                "head_dim": self.head_dim,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "index_topk": self.index_topk,
                "q_lora_rank": self.q_lora_rank,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Glm5MoeAttention(layers.Layer):
    """Multi-head Latent Attention (MLA, DeepSeek-V3 style) + DSA indexer.

    Query: ``x -> q_a_proj -> RMSNorm -> q_b_proj -> split(nope, pe) -> rope(pe)``.
    KV:    ``x -> kv_a_proj -> split(latent, k_pe) -> RMSNorm(latent) -> kv_b_proj
            -> split(k_nope, v)``; ``rope(k_pe)`` is shared across heads. RoPE is
    interleaved (DeepSeek style), not the NeoX split-half rotation. The DSA
    indexer selects top-k keys, applied as an additive ``-inf`` mask combined
    with the causal mask before attention.

    ``norm_eps`` is the epsilon of the two MLA bottleneck norms only. The
    reference builds them with the RMSNorm class default, never with the model's
    ``rms_norm_eps`` (1e-5 for GLM-5), so it stays 1e-6 here.
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
        index_n_heads,
        index_head_dim,
        index_topk,
        attention_bias=False,
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
        self.attention_bias = attention_bias
        self.norm_eps = norm_eps
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.softmax_scale = self.qk_head_dim**-0.5

        self.q_a_proj = layers.Dense(
            q_lora_rank, use_bias=attention_bias, name="q_a_proj"
        )
        self.q_a_norm = Glm5MoeRMSNorm(eps=norm_eps, name="q_a_norm")
        self.q_b_proj = layers.Dense(
            num_heads * self.qk_head_dim, use_bias=False, name="q_b_proj"
        )
        self.kv_a_proj = layers.Dense(
            kv_lora_rank + qk_rope_head_dim, use_bias=attention_bias, name="kv_a_proj"
        )
        self.kv_a_norm = Glm5MoeRMSNorm(eps=norm_eps, name="kv_a_norm")
        self.kv_b_proj = layers.Dense(
            num_heads * (qk_nope_head_dim + v_head_dim),
            use_bias=False,
            name="kv_b_proj",
        )
        self.o_proj = layers.Dense(embed_dim, use_bias=attention_bias, name="o_proj")
        self.indexer = Glm5MoeIndexer(
            embed_dim,
            index_n_heads,
            index_head_dim,
            qk_rope_head_dim,
            index_topk,
            q_lora_rank,
            name="indexer",
        )

    def project_qkv(self, hidden_states, cos, sin):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        q_resid = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_resid)
        q = ops.transpose(
            ops.reshape(q, (b, s, self.num_heads, self.qk_head_dim)), (0, 2, 1, 3)
        )
        q_nope = q[..., : self.qk_nope_head_dim]
        q_pe = q[..., self.qk_nope_head_dim :]

        compressed = self.kv_a_proj(hidden_states)
        kv_latent = compressed[..., : self.kv_lora_rank]
        k_pe = compressed[..., self.kv_lora_rank :][:, None, :, :]  # (B,1,S,rd)
        kv = self.kv_b_proj(self.kv_a_norm(kv_latent))
        kv = ops.transpose(
            ops.reshape(
                kv, (b, s, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            ),
            (0, 2, 1, 3),
        )
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]

        q_pe = apply_rope(q_pe, cos, sin, unsqueeze_axis=1)  # BHSD
        k_pe = apply_rope(k_pe, cos, sin, unsqueeze_axis=1)
        k_pe = ops.broadcast_to(k_pe, (b, self.num_heads, s, self.qk_rope_head_dim))
        q = ops.concatenate([q_nope, q_pe], axis=-1)
        k = ops.concatenate([k_nope, k_pe], axis=-1)
        return q, k, v, q_resid

    def dsa_mask(self, hidden_states, q_resid, cos, sin, attention_mask):
        """Build the combined DSA + causal additive mask ``(B, 1, S, T)``."""
        b = int(hidden_states.shape[0])
        s = int(hidden_states.shape[1])
        causal_3d = attention_mask[:, 0]  # (B, S, T)
        topk_indices = self.indexer(hidden_states, q_resid, cos, sin, causal_3d)
        total_len = int(causal_3d.shape[-1])
        k = int(topk_indices.shape[-1])
        b_idx = ops.broadcast_to(ops.arange(b)[:, None, None], (b, s, k))
        s_idx = ops.broadcast_to(ops.arange(s)[None, :, None], (b, s, k))
        idx = ops.reshape(ops.stack([b_idx, s_idx, topk_indices], axis=-1), (-1, 3))
        init = ops.full((b, s, total_len), MASK_NEG, dtype="float32")
        index_mask = ops.scatter_update(init, idx, ops.zeros((b * s * k,), "float32"))
        return ops.expand_dims(index_mask, 1) + ops.cast(attention_mask, "float32")

    def call(self, hidden_states, cos, sin, attention_mask=None, use_cache=False):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        q, k, v, q_resid = self.project_qkv(hidden_states, cos, sin)
        combined = self.dsa_mask(hidden_states, q_resid, cos, sin, attention_mask)
        out = fused_attention(q, k, v, self.softmax_scale, combined)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, s, self.num_heads * self.v_head_dim)
        )
        out = self.o_proj(out)
        return (out, (k, v)) if use_cache else out

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        # MLA single-token decode; the DSA indexer is skipped (no-op while the
        # cached length <= index_topk -- exact in that regime).
        q, k, v, _ = self.project_qkv(hidden_states, cos, sin)
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
        return self.o_proj(out), cache_k, cache_v

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
                "index_n_heads": self.indexer.n_heads,
                "index_head_dim": self.indexer.head_dim,
                "index_topk": self.indexer.index_topk,
                "attention_bias": self.attention_bias,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Glm5MoeDecoderLayer(layers.Layer):
    """Pre-norm MLA+DSA block, then pre-norm dense MLP or DeepSeekMoE."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        index_n_heads,
        index_head_dim,
        index_topk,
        use_moe,
        mlp_dim,
        moe_mlp_dim,
        shared_mlp_dim,
        num_experts,
        num_experts_per_tok,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        attention_bias=False,
        norm_eps=1e-5,
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
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
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
        self.attention_bias = attention_bias
        self.norm_eps = norm_eps

        self.attention_norm = Glm5MoeRMSNorm(eps=norm_eps, name="attention_norm")
        # norm_eps is deliberately not forwarded: the MLA bottleneck norms keep
        # the 1e-6 default (see Glm5MoeAttention), unlike the block norms.
        self.attention = Glm5MoeAttention(
            embed_dim,
            num_heads,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            index_n_heads,
            index_head_dim,
            index_topk,
            attention_bias,
            name="attention",
        )
        self.mlp_norm = Glm5MoeRMSNorm(eps=norm_eps, name="mlp_norm")
        if use_moe:
            self.mlp = Glm5MoeMoE(
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
            self.mlp = Glm5MoeMLP(embed_dim, mlp_dim, name="mlp")

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
                "index_n_heads": self.index_n_heads,
                "index_head_dim": self.index_head_dim,
                "index_topk": self.index_topk,
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
                "attention_bias": self.attention_bias,
                "norm_eps": self.norm_eps,
            }
        )
        return config
