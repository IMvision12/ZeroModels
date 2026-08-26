import keras
from keras import layers, ops

from zeromodels.base import BaseGeneration, BaseModel, CausalMask, TiedHead

from .deepseek_v3_config import DEEPSEEK_V3_CONFIG, DEEPSEEK_V3_WEIGHTS_URLS
from .deepseek_v3_layers import (
    DeepseekV3DecoderLayer,
    DeepseekV3RMSNorm,
    yarn_get_mscale,
    yarn_inv_freq,
)

MASK_NEG = -1e9


def deepseek_v3_rope_tables(position_ids, inv_freq, attention_scaling, compute_dtype):
    inv_freq = ops.convert_to_tensor(inv_freq)
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    return (
        ops.cast(ops.cos(freqs) * attention_scaling, compute_dtype),
        ops.cast(ops.sin(freqs) * attention_scaling, compute_dtype),
    )


def deepseek_v3_backbone_features(
    input_ids,
    attention_mask,
    *,
    token_embedding,
    decoder_layers,
    final_norm,
    causal_mask,
    inv_freq,
    attention_scaling,
    compute_dtype,
):
    hidden = token_embedding(input_ids)
    # DeepSeek uses plain arange positions (independent of the padding mask).
    position_ids = ops.cumsum(ops.ones_like(input_ids), axis=-1) - 1
    cos, sin = deepseek_v3_rope_tables(
        position_ids, inv_freq, attention_scaling, compute_dtype
    )
    mask = causal_mask(input_ids, attention_mask)
    for layer in decoder_layers:
        hidden = layer(hidden, cos, sin, attention_mask=mask)
    return final_norm(hidden)


@keras.saving.register_keras_serializable(package="zeromodels")
class DeepseekV3Model(BaseModel):
    """DeepSeek-V3 / R1 MoE decoder (MLA + aux-loss-free DeepSeekMoE).

    Multi-head Latent Attention compresses keys/values into ``kv_lora_rank``
    latents with a single shared interleaved-rope key (the yarn mscale^2
    correction is folded into the softmax scale); the MoE layers route with
    float32 sigmoids plus the learned ``e_score_correction_bias`` (group
    top-2-sum selection over ``n_group``/``topk_group``), renormalize the
    unbiased weights, scale by ``routed_scaling_factor``, and add a
    shared-expert SwiGLU. The first ``first_k_dense`` layers are dense.
    A functional model; returns ``last_hidden_state``: use
    :class:`DeepseekV3TextGenerate` for logits / text.

    Args:
        vocab_size / embed_dim / num_layers / num_heads: Model geometry.
        mlp_dim: Dense-layer SwiGLU width (``intermediate_size``).
        moe_mlp_dim: Per-expert width (``moe_intermediate_size``).
        num_experts / num_experts_per_tok / n_shared_experts: MoE shape.
        n_group / topk_group / norm_topk_prob / routed_scaling_factor: Routing.
        first_k_dense: Leading dense layers.
        q_lora_rank: Query bottleneck (None on V2-Lite).
        kv_lora_rank / qk_nope_head_dim / qk_rope_head_dim / v_head_dim: MLA.
        rope_theta: Base frequency.
        rope_scaling: The HF ``rope_scaling`` dict (yarn) or None.
        norm_eps: RMSNorm epsilon.
        max_position_embeddings: Used by the yarn attention-factor default.
        tie_embeddings: Whether :class:`DeepseekV3TextGenerate` ties the LM head.
    """

    HF_MODEL_TYPE = "deepseek_v3"
    BASE_MODEL_CONFIG = DEEPSEEK_V3_CONFIG
    BASE_WEIGHT_CONFIG = DEEPSEEK_V3_WEIGHTS_URLS
    output_logits = False

    def __init__(
        self,
        vocab_size=129280,
        embed_dim=7168,
        num_layers=61,
        num_heads=128,
        mlp_dim=18432,
        moe_mlp_dim=2048,
        num_experts=256,
        num_experts_per_tok=8,
        n_shared_experts=1,
        n_group=8,
        topk_group=4,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        first_k_dense=3,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rope_theta=10000.0,
        rope_scaling=None,
        norm_eps=1e-6,
        max_position_embeddings=163840,
        tie_embeddings=False,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        rope_scaling = dict(rope_scaling) if rope_scaling else None
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        softmax_scale = qk_head_dim**-0.5
        # V3 folds the yarn mscale^2 correction into the softmax scale.
        scaling_cfg = rope_scaling or {}
        rope_type = scaling_cfg.get("rope_type", scaling_cfg.get("type", "default"))
        if rope_type != "default":
            mscale_all_dim = scaling_cfg.get("mscale_all_dim", 0)
            factor = scaling_cfg.get("factor")
            if factor is None and scaling_cfg.get("original_max_position_embeddings"):
                factor = (
                    max_position_embeddings
                    / scaling_cfg["original_max_position_embeddings"]
                )
            if mscale_all_dim and factor:
                mscale = yarn_get_mscale(factor, mscale_all_dim)
                softmax_scale = softmax_scale * mscale * mscale

        inv_freq, attention_scaling = self.build_rope(
            qk_rope_head_dim, rope_theta, rope_scaling, max_position_embeddings
        )

        token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        decoder_layers = [
            DeepseekV3DecoderLayer(
                embed_dim,
                num_heads,
                q_lora_rank,
                kv_lora_rank,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
                softmax_scale,
                use_moe=i >= first_k_dense,
                mlp_dim=mlp_dim,
                moe_mlp_dim=moe_mlp_dim,
                shared_mlp_dim=moe_mlp_dim * n_shared_experts,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                n_group=n_group,
                topk_group=topk_group,
                norm_topk_prob=norm_topk_prob,
                routed_scaling_factor=routed_scaling_factor,
                norm_eps=norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        final_norm = DeepseekV3RMSNorm(eps=norm_eps, name="final_norm")
        causal_mask = CausalMask(name="causal_mask")
        lm_head = None
        if self.output_logits and not tie_embeddings:
            lm_head = layers.Dense(vocab_size, use_bias=False, name="lm_head")

        inputs = {
            "input_ids": layers.Input(shape=(None,), dtype="int32", name="input_ids"),
            "attention_mask": layers.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
        }
        hidden = deepseek_v3_backbone_features(
            inputs["input_ids"],
            inputs["attention_mask"],
            token_embedding=token_embedding,
            decoder_layers=decoder_layers,
            final_norm=final_norm,
            causal_mask=causal_mask,
            inv_freq=inv_freq,
            attention_scaling=attention_scaling,
            compute_dtype=token_embedding.compute_dtype,
        )
        outputs = {"last_hidden_state": hidden}
        if self.output_logits:
            outputs["logits"] = (
                lm_head(hidden)
                if lm_head is not None
                else TiedHead(token_embedding, name="lm_head")(hidden)
            )

        super().__init__(
            inputs=inputs, outputs=outputs, name=name or type(self).__name__, **kwargs
        )

        self.token_embedding = token_embedding
        self.decoder_layers = decoder_layers
        self.final_norm = final_norm
        self.causal_mask_layer = causal_mask
        self.lm_head = lm_head
        self.inv_freq = inv_freq
        self.attention_scaling = attention_scaling
        self.qk_head_dim = qk_head_dim
        self.softmax_scale = softmax_scale
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.moe_mlp_dim = moe_mlp_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense = first_k_dense
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.norm_eps = norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.tie_embeddings = tie_embeddings

    @staticmethod
    def build_rope(rotary_dim, rope_theta, rope_scaling, max_position_embeddings):
        """Host-side inverse frequencies + the yarn cos/sin attention factor."""
        import numpy as np

        scaling = rope_scaling or {}
        rope_type = scaling.get("rope_type", scaling.get("type", "default"))
        if rope_type == "yarn":
            factor = scaling.get("factor")
            original = scaling["original_max_position_embeddings"]
            if factor is None:
                factor = max_position_embeddings / original
            inv_freq = yarn_inv_freq(
                rotary_dim,
                rope_theta,
                factor,
                original,
                scaling.get("beta_fast") or 32,
                scaling.get("beta_slow") or 1,
            )
            attention_factor = scaling.get("attention_factor")
            if attention_factor is None:
                mscale = scaling.get("mscale")
                mscale_all_dim = scaling.get("mscale_all_dim")
                if mscale and mscale_all_dim:
                    attention_factor = float(
                        yarn_get_mscale(factor, mscale)
                        / yarn_get_mscale(factor, mscale_all_dim)
                    )
                else:
                    attention_factor = yarn_get_mscale(factor)
            return inv_freq, float(attention_factor)
        inv_freq = 1.0 / (
            rope_theta ** (np.arange(0, rotary_dim, 2, dtype="float32") / rotary_dim)
        )
        return inv_freq, 1.0

    def rope_tables(self, position_ids):
        return deepseek_v3_rope_tables(
            position_ids, self.inv_freq, self.attention_scaling, self.compute_dtype
        )

    def causal_mask(self, seq, attention_mask=None):
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        mask = ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            mask = mask + (1.0 - am)[:, None, None, :] * MASK_NEG
        return mask

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "vocab_size": hf_config["vocab_size"],
            "embed_dim": hf_config["hidden_size"],
            "num_layers": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "mlp_dim": hf_config["intermediate_size"],
            "moe_mlp_dim": hf_config.get("moe_intermediate_size", 2048),
            "num_experts": hf_config.get("n_routed_experts", 256),
            "num_experts_per_tok": hf_config.get("num_experts_per_tok", 8),
            "n_shared_experts": hf_config.get("n_shared_experts", 1),
            "n_group": hf_config.get("n_group") or 8,
            "topk_group": hf_config.get("topk_group") or 4,
            "norm_topk_prob": bool(hf_config.get("norm_topk_prob", True)),
            "routed_scaling_factor": hf_config.get("routed_scaling_factor", 2.5),
            "first_k_dense": hf_config.get("first_k_dense_replace", 0),
            "q_lora_rank": hf_config.get("q_lora_rank"),
            "kv_lora_rank": hf_config.get("kv_lora_rank", 512),
            "qk_nope_head_dim": hf_config.get("qk_nope_head_dim", 128),
            "qk_rope_head_dim": hf_config.get("qk_rope_head_dim", 64),
            "v_head_dim": hf_config.get("v_head_dim", 128),
            "rope_theta": hf_config.get("rope_theta", 10000.0),
            "rope_scaling": hf_config.get("rope_scaling")
            or (hf_config.get("rope_parameters") or None),
            "norm_eps": hf_config.get("rms_norm_eps", 1e-6),
            "max_position_embeddings": hf_config.get("max_position_embeddings", 163840),
            "tie_embeddings": bool(hf_config.get("tie_word_embeddings") or False),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_deepseek_v3_hf_to_keras import transfer_deepseek_v3_weights

        transfer_deepseek_v3_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "moe_mlp_dim": self.moe_mlp_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "n_shared_experts": self.n_shared_experts,
                "n_group": self.n_group,
                "topk_group": self.topk_group,
                "norm_topk_prob": self.norm_topk_prob,
                "routed_scaling_factor": self.routed_scaling_factor,
                "first_k_dense": self.first_k_dense,
                "q_lora_rank": self.q_lora_rank,
                "kv_lora_rank": self.kv_lora_rank,
                "qk_nope_head_dim": self.qk_nope_head_dim,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "v_head_dim": self.v_head_dim,
                "rope_theta": self.rope_theta,
                "rope_scaling": self.rope_scaling,
                "norm_eps": self.norm_eps,
                "max_position_embeddings": self.max_position_embeddings,
                "tie_embeddings": self.tie_embeddings,
                "name": self.name,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DeepseekV3TextGenerate(DeepseekV3Model, BaseGeneration):
    """DeepSeek-V3 with an LM head + fast ``.generate()``.

    The MLA cache stores expanded per-head keys and values as a per-layer
    ``(k, v)`` tuple: their head dims differ (k: nope+rope = 192,
    v: ``v_head_dim`` = 128), so they cannot share one stacked buffer.
    """

    # DeepSeek-V3 end-of-sentence id (1). Explicit generate() args override.
    eos_token_id = (1,)
    output_logits = True

    def project(self, hidden):
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def build_cache(self, token_ids, padding_mask, max_len):
        batch = int(token_ids.shape[0])
        seq = int(token_ids.shape[1])
        hidden = self.token_embedding(ops.cast(token_ids, "int32"))
        position_ids = ops.broadcast_to(ops.arange(seq), (batch, seq))
        cos, sin = self.rope_tables(position_ids)
        causal = self.causal_mask(seq, padding_mask)
        caches = []
        for layer in self.decoder_layers:
            hidden, (k, v) = layer(
                hidden, cos, sin, attention_mask=causal, use_cache=True
            )
            ck = ops.slice_update(
                ops.zeros(
                    (batch, self.num_heads, max_len, self.qk_head_dim), dtype=k.dtype
                ),
                (0, 0, 0, 0),
                k,
            )
            cv = ops.slice_update(
                ops.zeros(
                    (batch, self.num_heads, max_len, self.v_head_dim), dtype=v.dtype
                ),
                (0, 0, 0, 0),
                v,
            )
            caches.append((ck, cv))
        logits = self.project(self.final_norm(hidden)[:, -1, :])
        return tuple(caches), logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        batch = int(token_ids.shape[0])
        max_len = int(cache[0][0].shape[2])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos, sin = self.rope_tables(positions)
        key_mask = ops.cast(
            ops.where(ops.arange(max_len) <= pos, 0.0, MASK_NEG), "float32"
        )[None, None, None, :]
        h = self.token_embedding(token_ids)
        new_cache = []
        for i, layer in enumerate(self.decoder_layers):
            h, ck, cv = layer.decode_step(
                h, cos, sin, cache[i][0], cache[i][1], pos, key_mask
            )
            new_cache.append((ck, cv))
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, tuple(new_cache)
