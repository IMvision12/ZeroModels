import keras
from keras import layers, ops

from kerasformers.base import BaseGeneration, BaseModel, CausalMask, TiedHead

from .qwen2_moe_config import Qwen2MoeConfig
from .qwen2_moe_layers import Qwen2MoeDecoderLayer, Qwen2MoeRMSNorm

MASK_NEG = -1e9


def qwen2_moe_rope_tables(position_ids, head_dim, rope_theta, compute_dtype):
    inv_freq = 1.0 / ops.power(
        rope_theta, ops.arange(0, head_dim, 2, dtype="float32") / head_dim
    )
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    emb = ops.concatenate([freqs, freqs], axis=-1)
    return ops.cast(ops.cos(emb), compute_dtype), ops.cast(ops.sin(emb), compute_dtype)


def qwen2_moe_backbone_features(
    input_ids,
    attention_mask,
    *,
    token_embedding,
    decoder_layers,
    final_norm,
    causal_mask,
    head_dim,
    rope_theta,
    compute_dtype,
):
    hidden = token_embedding(input_ids)
    position_ids = ops.where(
        attention_mask == 0, 1, ops.cumsum(attention_mask, axis=-1) - 1
    )
    cos, sin = qwen2_moe_rope_tables(position_ids, head_dim, rope_theta, compute_dtype)
    mask = causal_mask(input_ids, attention_mask)
    for layer in decoder_layers:
        hidden = layer(hidden, cos, sin, attention_mask=mask)
    return final_norm(hidden)


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2MoeModel(BaseModel):
    """Qwen2-MoE sparse decoder (e.g. Qwen1.5-MoE-A2.7B, Qwen2-57B-A14B).

    Standard Qwen2 GQA (biased QKV) with a Qwen-MoE block on the sparse
    layers: a float32-softmax top-k router over ``num_experts`` routed
    experts (optionally renormalized) plus a full-width shared expert scaled
    by ``sigmoid(shared_expert_gate(x))``. Layers selected by
    ``decoder_sparse_step`` / ``mlp_only_layers`` are MoE; the rest are dense
    SwiGLU. A functional model; returns ``last_hidden_state``; use
    :class:`Qwen2MoeTextGenerate` for logits.

    Args:
        vocab_size / embed_dim / num_layers / num_heads / num_kv_heads /
        head_dim: Model geometry.
        mlp_dim: Dense-layer SwiGLU width (``intermediate_size``).
        num_experts / num_experts_per_tok: Routing shape.
        moe_mlp_dim: Per-routed-expert width (``moe_intermediate_size``).
        shared_mlp_dim: Shared-expert width (``shared_expert_intermediate_size``).
        norm_topk_prob: Renormalize the top-k routing weights.
        decoder_sparse_step: A layer is MoE when ``(i + 1) % step == 0``.
        mlp_only_layers: Layer indices forced to dense MLP.
        rope_theta: Rotary base frequency.
        norm_eps: RMSNorm epsilon.
        tie_embeddings: Whether :class:`Qwen2MoeTextGenerate` ties the LM head.
    """

    HF_MODEL_TYPE = "qwen2_moe"
    default_load_dtype = "bfloat16"
    config_class = Qwen2MoeConfig
    output_logits = False

    def __init__(
        self,
        vocab_size=151936,
        embed_dim=2048,
        num_layers=24,
        num_heads=16,
        num_kv_heads=16,
        head_dim=None,
        mlp_dim=5632,
        num_experts=60,
        num_experts_per_tok=4,
        moe_mlp_dim=1408,
        shared_mlp_dim=5632,
        norm_topk_prob=False,
        decoder_sparse_step=1,
        mlp_only_layers=(),
        rope_theta=1000000.0,
        norm_eps=1e-6,
        tie_embeddings=False,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        head_dim = head_dim or embed_dim // num_heads
        mlp_only_layers = tuple(mlp_only_layers)

        def is_moe(i):
            return (
                i not in mlp_only_layers
                and num_experts > 0
                and (i + 1) % decoder_sparse_step == 0
            )

        token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        decoder_layers = [
            Qwen2MoeDecoderLayer(
                embed_dim,
                num_heads,
                num_kv_heads,
                head_dim,
                use_moe=is_moe(i),
                mlp_dim=mlp_dim,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                moe_mlp_dim=moe_mlp_dim,
                shared_mlp_dim=shared_mlp_dim,
                norm_topk_prob=norm_topk_prob,
                norm_eps=norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        final_norm = Qwen2MoeRMSNorm(eps=norm_eps, name="final_norm")
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
        hidden = qwen2_moe_backbone_features(
            inputs["input_ids"],
            inputs["attention_mask"],
            token_embedding=token_embedding,
            decoder_layers=decoder_layers,
            final_norm=final_norm,
            causal_mask=causal_mask,
            head_dim=head_dim,
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
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.mlp_dim = mlp_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_mlp_dim = moe_mlp_dim
        self.shared_mlp_dim = shared_mlp_dim
        self.norm_topk_prob = norm_topk_prob
        self.decoder_sparse_step = decoder_sparse_step
        self.mlp_only_layers = mlp_only_layers
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        self.tie_embeddings = tie_embeddings

    def is_moe_layer(self, i):
        return (
            i not in self.mlp_only_layers
            and self.num_experts > 0
            and (i + 1) % self.decoder_sparse_step == 0
        )

    def rope_tables(self, position_ids):
        return qwen2_moe_rope_tables(
            position_ids, self.head_dim, self.rope_theta, self.compute_dtype
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
            "num_kv_heads": hf_config.get(
                "num_key_value_heads", hf_config["num_attention_heads"]
            ),
            "head_dim": hf_config.get("head_dim"),
            "mlp_dim": hf_config["intermediate_size"],
            "num_experts": hf_config.get("num_experts", 60),
            "num_experts_per_tok": hf_config.get("num_experts_per_tok", 4),
            "moe_mlp_dim": hf_config.get("moe_intermediate_size", 1408),
            "shared_mlp_dim": hf_config.get(
                "shared_expert_intermediate_size", hf_config["intermediate_size"]
            ),
            "norm_topk_prob": bool(hf_config.get("norm_topk_prob", False)),
            "decoder_sparse_step": hf_config.get("decoder_sparse_step", 1),
            "mlp_only_layers": tuple(hf_config.get("mlp_only_layers") or ()),
            "rope_theta": hf_config.get("rope_theta", 1000000.0),
            "norm_eps": hf_config.get("rms_norm_eps", 1e-6),
            "tie_embeddings": bool(hf_config.get("tie_word_embeddings") or False),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_qwen2_moe_hf_to_keras import transfer_qwen2_moe_weights

        transfer_qwen2_moe_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "mlp_dim": self.mlp_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "moe_mlp_dim": self.moe_mlp_dim,
                "shared_mlp_dim": self.shared_mlp_dim,
                "norm_topk_prob": self.norm_topk_prob,
                "decoder_sparse_step": self.decoder_sparse_step,
                "mlp_only_layers": self.mlp_only_layers,
                "rope_theta": self.rope_theta,
                "norm_eps": self.norm_eps,
                "tie_embeddings": self.tie_embeddings,
                "name": self.name,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2MoeTextGenerate(Qwen2MoeModel, BaseGeneration):
    """Qwen2-MoE with an LM head + fast ``.generate()`` (text -> text).

    Adds an optional bias-free ``lm_head`` (tied to the token embedding when
    ``tie_embeddings``) and the fixed-cache compiled decode loop.
    """

    eos_token_id = (151645, 151643)
    output_logits = True

    def project(self, hidden):
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def build_cache(self, token_ids, padding_mask, max_len):
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        hd, nkv = self.head_dim, self.num_kv_heads
        if padding_mask is not None:
            am = ops.cast(padding_mask, "int32")
            position_ids = ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
        else:
            position_ids = ops.broadcast_to(ops.arange(prompt_len), (batch, prompt_len))
        cos_p, sin_p = self.rope_tables(position_ids)
        causal = self.causal_mask(prompt_len, padding_mask)
        hidden = self.token_embedding(ops.cast(token_ids, "int32"))
        layer_caches = []
        for layer in self.decoder_layers:
            hidden, (k, v) = layer(
                hidden, cos_p, sin_p, attention_mask=causal, use_cache=True
            )
            ck = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=k.dtype), (0, 0, 0, 0), k
            )
            cv = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=v.dtype), (0, 0, 0, 0), v
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.final_norm(hidden)[:, -1, :])
        return cache, logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        batch = int(token_ids.shape[0])
        max_len = int(cache.shape[4])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos_t, sin_t = self.rope_tables(positions)
        key_mask = ops.cast(
            ops.where(ops.arange(max_len) <= pos, 0.0, MASK_NEG), "float32"
        )[None, None, None, :]
        h = self.token_embedding(token_ids)
        layer_caches = []
        for i, layer in enumerate(self.decoder_layers):
            h, ck, cv = layer.decode_step(
                h, cos_t, sin_t, cache[:, i, 0], cache[:, i, 1], pos, key_mask
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, cache
