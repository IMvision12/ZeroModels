import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, BaseModel, CausalMask, TiedHead
from kerasformers.base.base_mixin import inference_scope

from .glm5_moe_config import GLM5_MOE_CONFIG, GLM5_MOE_WEIGHTS_URLS
from .glm5_moe_layers import Glm5MoeDecoderLayer, Glm5MoeRMSNorm

MASK_NEG = -1e9


def glm5_moe_rope_tables(position_ids, qk_rope_head_dim, rope_theta, compute_dtype):
    # Interleaved rope over qk_rope_head_dim. cos/sin keep the cat((freqs, freqs))
    # layout the reference builds; apply_rope reads the first half.
    rd = qk_rope_head_dim
    inv_freq = 1.0 / ops.power(rope_theta, ops.arange(0, rd, 2, dtype="float32") / rd)
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    emb = ops.concatenate([freqs, freqs], axis=-1)
    return ops.cast(ops.cos(emb), compute_dtype), ops.cast(ops.sin(emb), compute_dtype)


def glm5_moe_backbone_features(
    input_ids,
    attention_mask,
    *,
    token_embedding,
    decoder_layers,
    final_norm,
    causal_mask,
    qk_rope_head_dim,
    rope_theta,
    compute_dtype,
):
    hidden = token_embedding(input_ids)
    # Plain arange positions (0, 1, 2, ...), matching the original forward.
    position_ids = ops.cumsum(ops.ones_like(input_ids), axis=-1) - 1
    cos, sin = glm5_moe_rope_tables(
        position_ids, qk_rope_head_dim, rope_theta, compute_dtype
    )
    mask = causal_mask(input_ids, attention_mask)
    for layer in decoder_layers:
        hidden = layer(hidden, cos, sin, attention_mask=mask)
    return final_norm(hidden)


@keras.saving.register_keras_serializable(package="kerasformers")
class Glm5MoeModel(BaseModel):
    """GLM-5 (glm_moe_dsa) decoder backbone (no LM head).

    Pre-norm decoder with Multi-head Latent Attention (DeepSeek-V3 style) plus a
    DeepSeek Sparse Attention (DSA) indexer, NeoX partial rope on the
    ``qk_rope_head_dim`` slice, and DeepSeekMoE routing (float32 sigmoid scores +
    ``e_score_correction_bias``, shared expert, first ``first_k_dense`` layers
    dense). GLM-5/5.1/5.2 share this forward; 5.2's IndexShare is an
    inference-time optimization not applied here. The checkpoint's trailing MTP
    layer (index ``num_layers``) is ignored. A functional model; returns
    ``last_hidden_state``: use :class:`Glm5MoeTextGenerate` for logits / text.
    """

    HF_MODEL_TYPE = "glm_moe_dsa"
    BASE_MODEL_CONFIG = GLM5_MOE_CONFIG
    BASE_WEIGHT_CONFIG = GLM5_MOE_WEIGHTS_URLS
    output_logits = False

    def __init__(
        self,
        vocab_size=154880,
        embed_dim=6144,
        num_layers=78,
        num_heads=64,
        mlp_dim=12288,
        moe_mlp_dim=2048,
        num_experts=256,
        num_experts_per_tok=8,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        first_k_dense=3,
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        norm_eps=1e-5,
        rope_theta=1000000.0,
        attention_bias=False,
        tie_embeddings=False,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)

        token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        decoder_layers = [
            Glm5MoeDecoderLayer(
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
                attention_bias=attention_bias,
                norm_eps=norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        final_norm = Glm5MoeRMSNorm(eps=norm_eps, name="final_norm")
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
        hidden = glm5_moe_backbone_features(
            inputs["input_ids"],
            inputs["attention_mask"],
            token_embedding=token_embedding,
            decoder_layers=decoder_layers,
            final_norm=final_norm,
            causal_mask=causal_mask,
            qk_rope_head_dim=qk_rope_head_dim,
            rope_theta=rope_theta,
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
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.tie_embeddings = tie_embeddings

        # The MLA/DSA/MoE sublayers do not all materialize during Keras' symbolic
        # auto-build on every backend; a concrete dummy forward builds every weight
        # so ``from_weights`` (which loads before any forward) has a complete model.
        with inference_scope():
            self(
                {
                    "input_ids": ops.zeros((1, 4), dtype="int32"),
                    "attention_mask": ops.ones((1, 4), dtype="int32"),
                }
            )

    def rope_tables(self, position_ids):
        return glm5_moe_rope_tables(
            position_ids, self.qk_rope_head_dim, self.rope_theta, self.compute_dtype
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
        # GlmMoeDsa hardcodes the dense/sparse split via mlp_layer_types (3 dense
        # + rest sparse), ignoring first_k_dense_replace -- derive from it.
        mlp_types = hf_config.get("mlp_layer_types")
        first_k_dense = (
            sum(t == "dense" for t in mlp_types)
            if mlp_types
            else hf_config.get("first_k_dense_replace", 3)
        )
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
            "n_group": hf_config.get("n_group") or 1,
            "topk_group": hf_config.get("topk_group") or 1,
            "norm_topk_prob": bool(hf_config.get("norm_topk_prob", True)),
            "routed_scaling_factor": hf_config.get("routed_scaling_factor", 2.5),
            "first_k_dense": first_k_dense,
            "q_lora_rank": hf_config.get("q_lora_rank", 2048),
            "kv_lora_rank": hf_config.get("kv_lora_rank", 512),
            "qk_nope_head_dim": hf_config.get("qk_nope_head_dim", 192),
            "qk_rope_head_dim": hf_config.get("qk_rope_head_dim", 64),
            "v_head_dim": hf_config.get("v_head_dim", 256),
            "index_n_heads": hf_config.get("index_n_heads", 32),
            "index_head_dim": hf_config.get("index_head_dim", 128),
            "index_topk": hf_config.get("index_topk", 2048),
            "norm_eps": hf_config.get("rms_norm_eps", 1e-5),
            "rope_theta": hf_config.get("rope_theta", 1000000.0),
            "attention_bias": bool(hf_config.get("attention_bias") or False),
            "tie_embeddings": bool(hf_config.get("tie_word_embeddings") or False),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_glm5_moe_hf_to_keras import transfer_glm5_moe_weights

        transfer_glm5_moe_weights(keras_model, hf_state_dict)

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
                "index_n_heads": self.index_n_heads,
                "index_head_dim": self.index_head_dim,
                "index_topk": self.index_topk,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "attention_bias": self.attention_bias,
                "tie_embeddings": self.tie_embeddings,
                "name": self.name,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Glm5MoeTextGenerate(Glm5MoeModel, BaseGeneration):
    """GLM-5 with an LM head + fast cached ``.generate()``.

    Decode uses MLA k/v caching; the DSA indexer is skipped at decode time (it is
    a no-op while the cached length stays <= ``index_topk``, so generation is
    exact in that regime; beyond it the cached path is denser than the reference
    -- DSA-pruned decode is a future optimization).

    The cache is a per-layer ``(k, v)`` tuple rather than one stacked buffer: the
    key head dim is ``qk_nope + qk_rope`` while the value head dim is
    ``v_head_dim``, and nothing requires those to agree (they happen to both be
    256 on the released GLM-5 configs)."""

    eos_token_id = (154820, 154827, 154829)
    output_logits = True

    def project(self, hidden):
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def build_cache(self, token_ids, padding_mask, max_len):
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        nh = self.num_heads
        position_ids = ops.broadcast_to(ops.arange(prompt_len), (batch, prompt_len))
        cos_p, sin_p = self.rope_tables(position_ids)
        causal = self.causal_mask(prompt_len, padding_mask)
        hidden = self.token_embedding(ops.cast(token_ids, "int32"))
        caches = []
        for layer in self.decoder_layers:
            hidden, (k, v) = layer(
                hidden, cos_p, sin_p, attention_mask=causal, use_cache=True
            )
            ck = ops.slice_update(
                ops.zeros((batch, nh, max_len, qk_head_dim), dtype=k.dtype),
                (0, 0, 0, 0),
                k,
            )
            cv = ops.slice_update(
                ops.zeros((batch, nh, max_len, self.v_head_dim), dtype=v.dtype),
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
        cos_t, sin_t = self.rope_tables(positions)
        key_mask = ops.cast(
            ops.where(ops.arange(max_len) <= pos, 0.0, MASK_NEG), "float32"
        )[None, None, None, :]
        h = self.token_embedding(token_ids)
        new_cache = []
        for i, layer in enumerate(self.decoder_layers):
            h, ck, cv = layer.decode_step(
                h, cos_t, sin_t, cache[i][0], cache[i][1], pos, key_mask
            )
            new_cache.append((ck, cv))
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, tuple(new_cache)
