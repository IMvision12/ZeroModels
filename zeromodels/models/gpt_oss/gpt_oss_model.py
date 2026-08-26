import math

import keras
from keras import layers, ops

from zeromodels.base import BaseGeneration, BaseModel, CausalMask, TiedHead
from zeromodels.base.base_mixin import inference_scope

from .gpt_oss_config import GptOssConfig
from .gpt_oss_layers import GptOssDecoderLayer, GptOssRMSNorm

MASK_NEG = -1e9


def yarn_inv_freq(head_dim, base, factor, beta_fast, beta_slow, orig_max, truncate):
    """YaRN-scaled inverse frequencies + cos/sin scaling (port of HF's YaRN init).

    Returns ``(inv_freq, attention_scaling)`` where ``inv_freq`` is a
    ``(head_dim // 2,)`` tensor and ``attention_scaling`` (mscale) multiplies the
    rotary cos/sin. All inputs are config constants, so this is recomputed cheaply
    each forward (no stored state).
    """
    dim = head_dim
    pos_freqs = ops.power(
        float(base), ops.arange(0, dim, 2, dtype="float32") / dim
    )  # (dim/2,)
    inv_extrap = 1.0 / pos_freqs
    inv_interp = 1.0 / (factor * pos_freqs)

    def correction_dim(num_rotations):
        return (dim * math.log(orig_max / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = correction_dim(beta_fast)
    high = correction_dim(beta_slow)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    low = max(low, 0.0)
    high = min(high, dim - 1.0)
    if low == high:
        high += 0.001

    ramp = ops.clip(
        (ops.arange(dim // 2, dtype="float32") - low) / (high - low), 0.0, 1.0
    )
    extrapolation_factor = 1.0 - ramp
    inv_freq = (
        inv_interp * (1.0 - extrapolation_factor) + inv_extrap * extrapolation_factor
    )
    attention_scaling = 0.1 * math.log(factor) + 1.0 if factor > 1.0 else 1.0
    return inv_freq, attention_scaling


def gpt_oss_rope_tables(
    position_ids,
    head_dim,
    rope_theta,
    rope_factor,
    rope_beta_fast,
    rope_beta_slow,
    rope_original_max_pos,
    rope_truncate,
):
    inv_freq, scaling = yarn_inv_freq(
        head_dim,
        rope_theta,
        rope_factor,
        rope_beta_fast,
        rope_beta_slow,
        rope_original_max_pos,
        rope_truncate,
    )
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    emb = ops.concatenate([freqs, freqs], axis=-1)
    return ops.cos(emb) * scaling, ops.sin(emb) * scaling


def gpt_oss_is_sliding(layer_idx):
    # HF: "sliding_attention" if (i + 1) % 2 else "full_attention" -> even i slides
    return bool((layer_idx + 1) % 2)


def gpt_oss_backbone_features(
    input_ids,
    attention_mask,
    *,
    token_embedding,
    decoder_layers,
    final_norm,
    full_mask_layer,
    sliding_mask_layer,
    rope_kwargs,
):
    hidden = token_embedding(input_ids)
    position_ids = ops.where(
        attention_mask == 0, 1, ops.cumsum(attention_mask, axis=-1) - 1
    )
    cos, sin = gpt_oss_rope_tables(position_ids, **rope_kwargs)
    full_mask = full_mask_layer(input_ids, attention_mask)
    sliding_mask = sliding_mask_layer(input_ids, attention_mask)
    for i, layer in enumerate(decoder_layers):
        mask = sliding_mask if gpt_oss_is_sliding(i) else full_mask
        hidden = layer(hidden, cos, sin, attention_mask=mask)
    return final_norm(hidden)


@keras.saving.register_keras_serializable(package="zeromodels")
class GptOssModel(BaseModel):
    """GPT-OSS mixture-of-experts decoder-only transformer backbone (no LM head).

    ``token_embedding -> num_layers x GptOssDecoderLayer -> final RMSNorm``, with
    grouped-query attention + learned per-head attention sinks, alternating
    sliding-window / full causal attention, YaRN-scaled rotary positions, and a
    top-k-routed mixture-of-experts feed-forward per layer. (The router picks
    top-k experts per token; this port evaluates *all* experts densely and
    combines them by the routing weights: mathematically identical to sparse
    top-k routing, but compute is O(num_experts).) A functional model; returns
    ``last_hidden_state``: use :class:`GptOssTextGenerate` for logits / text.

    Args:
        vocab_size, embed_dim, mlp_dim, num_layers, num_heads,
        num_kv_heads, head_dim: standard decoder dimensions (``mlp_dim``
            is the *per-expert* hidden width).
        num_experts, num_experts_per_tok: MoE expert count and top-k.
        sliding_window: window size of the sliding-attention layers (even layers).
        norm_eps: RMSNorm epsilon.
        rope_theta, rope_factor, rope_beta_fast, rope_beta_slow, rope_truncate,
        rope_original_max_pos: YaRN rotary parameters.
        attention_bias: whether q/k/v/o carry a bias (GPT-OSS: True).
        tie_embeddings: whether :class:`GptOssTextGenerate` ties the LM head.
    """

    HF_MODEL_TYPE = "gpt_oss"
    config_class = GptOssConfig

    # TODO : Auto Detection of weights from hub
    # OpenAI ships GPT-OSS as bf16 dense + MXFP4 experts, and the hosted
    # zeromodels weights match. Default from_weights() to bf16 so the loaded
    # model keeps that native ~2 bytes/param footprint (pass load_dtype="float32"
    # to upcast). The experts stay uint8 MXFP4 regardless of this policy.
    default_load_dtype = "bfloat16"
    output_logits = False

    def __init__(
        self,
        vocab_size=201088,
        embed_dim=2880,
        mlp_dim=2880,
        num_layers=24,
        num_heads=64,
        num_kv_heads=8,
        head_dim=64,
        num_experts=32,
        num_experts_per_tok=4,
        sliding_window=128,
        norm_eps=1e-5,
        rope_theta=150000.0,
        rope_factor=32.0,
        rope_beta_fast=32.0,
        rope_beta_slow=1.0,
        rope_truncate=False,
        rope_original_max_pos=4096,
        attention_bias=True,
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
            GptOssDecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                head_dim,
                num_experts,
                num_experts_per_tok,
                norm_eps,
                attention_bias,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        final_norm = GptOssRMSNorm(eps=norm_eps, name="final_norm")
        full_mask_layer = CausalMask(name="causal_mask")
        sliding_mask_layer = CausalMask(
            sliding_window=sliding_window, name="sliding_mask"
        )
        lm_head = None
        if self.output_logits and not tie_embeddings:
            lm_head = layers.Dense(vocab_size, use_bias=False, name="lm_head")

        inputs = {
            "input_ids": layers.Input(shape=(None,), dtype="int32", name="input_ids"),
            "attention_mask": layers.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
        }
        rope_kwargs = {
            "head_dim": head_dim,
            "rope_theta": rope_theta,
            "rope_factor": rope_factor,
            "rope_beta_fast": rope_beta_fast,
            "rope_beta_slow": rope_beta_slow,
            "rope_original_max_pos": rope_original_max_pos,
            "rope_truncate": rope_truncate,
        }
        hidden = gpt_oss_backbone_features(
            inputs["input_ids"],
            inputs["attention_mask"],
            token_embedding=token_embedding,
            decoder_layers=decoder_layers,
            final_norm=final_norm,
            full_mask_layer=full_mask_layer,
            sliding_mask_layer=sliding_mask_layer,
            rope_kwargs=rope_kwargs,
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
        self.full_mask_layer = full_mask_layer
        self.sliding_mask_layer = sliding_mask_layer
        self.lm_head = lm_head
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.sliding_window = sliding_window
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.rope_factor = rope_factor
        self.rope_beta_fast = rope_beta_fast
        self.rope_beta_slow = rope_beta_slow
        self.rope_truncate = rope_truncate
        self.rope_original_max_pos = rope_original_max_pos
        self.attention_bias = attention_bias
        self.tie_embeddings = tie_embeddings

        # The learned attention sinks abort Keras' symbolic auto-build trace of the
        # attention on some backends, leaving it unbuilt after __init__; a concrete
        # dummy forward materializes every weight so ``from_weights`` (which loads
        # before any forward) has a complete model to populate.
        with inference_scope():
            self(
                {
                    "input_ids": ops.zeros((1, 4), dtype="int32"),
                    "attention_mask": ops.ones((1, 4), dtype="int32"),
                }
            )

    def rope(self, position_ids):
        return gpt_oss_rope_tables(
            position_ids,
            self.head_dim,
            self.rope_theta,
            self.rope_factor,
            self.rope_beta_fast,
            self.rope_beta_slow,
            self.rope_original_max_pos,
            self.rope_truncate,
        )

    def is_sliding(self, layer_idx):
        return gpt_oss_is_sliding(layer_idx)

    @classmethod
    def config_from_hf(cls, hf_config):
        rope = hf_config.get("rope_parameters") or hf_config.get("rope_scaling") or {}
        return {
            "vocab_size": hf_config["vocab_size"],
            "embed_dim": hf_config["hidden_size"],
            "mlp_dim": hf_config["intermediate_size"],
            "num_layers": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "num_kv_heads": hf_config["num_key_value_heads"],
            "head_dim": hf_config.get("head_dim")
            or hf_config["hidden_size"] // hf_config["num_attention_heads"],
            "num_experts": hf_config["num_local_experts"],
            "num_experts_per_tok": hf_config["num_experts_per_tok"],
            "sliding_window": hf_config.get("sliding_window", 128),
            "norm_eps": hf_config.get("rms_norm_eps", 1e-5),
            "rope_theta": rope.get("rope_theta", hf_config.get("rope_theta", 150000.0)),
            "rope_factor": rope.get("factor", 32.0),
            "rope_beta_fast": rope.get("beta_fast", 32.0),
            "rope_beta_slow": rope.get("beta_slow", 1.0),
            "rope_truncate": rope.get("truncate", False),
            "rope_original_max_pos": rope.get("original_max_position_embeddings", 4096),
            "attention_bias": hf_config.get("attention_bias", True),
            "tie_embeddings": hf_config.get("tie_word_embeddings", False),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_gpt_oss_hf_to_keras import transfer_gpt_oss_weights

        transfer_gpt_oss_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "sliding_window": self.sliding_window,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "rope_factor": self.rope_factor,
                "rope_beta_fast": self.rope_beta_fast,
                "rope_beta_slow": self.rope_beta_slow,
                "rope_truncate": self.rope_truncate,
                "rope_original_max_pos": self.rope_original_max_pos,
                "attention_bias": self.attention_bias,
                "tie_embeddings": self.tie_embeddings,
                "name": self.name,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GptOssTextGenerate(GptOssModel, BaseGeneration):
    """GPT-OSS backbone + a language-model head and fast ``.generate()``.

    Adds a bias-free ``lm_head`` (GPT-OSS does not tie embeddings). The forward
    graph returns ``logits`` ``(batch, seq, vocab_size)`` and
    ``last_hidden_state``. Fast generation comes from
    :class:`~zeromodels.base.BaseGeneration`, fulfilled here by ``build_cache``
    (parallel prefill into a fixed KV cache) and ``call_with_cache`` (one compiled
    decode step): both respect the per-layer sliding window (full / sliding key
    masks) and the learned attention sinks. Constructor ``Args`` are inherited from
    :class:`GptOssModel`.
    """

    # GPT-OSS <|return|> stop id. Explicit generate() args override this.
    eos_token_id = (200002,)
    output_logits = True

    def project(self, hidden):
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def build_cache(self, token_ids, padding_mask, max_len):
        # Parallel prefill into a fixed (B, num_layers, 2, num_kv_heads, max_len,
        # head_dim) cache, with the per-layer full / sliding-window causal mask.
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        hd, nkv = self.head_dim, self.num_kv_heads
        if padding_mask is not None:
            am = ops.cast(padding_mask, "int32")
            position_ids = ops.where(am == 0, 1, ops.cumsum(am, axis=-1) - 1)
        else:
            position_ids = ops.broadcast_to(ops.arange(prompt_len), (batch, prompt_len))
        cos, sin = self.rope(position_ids)
        qi = ops.arange(prompt_len)[:, None]
        ki = ops.arange(prompt_len)[None, :]
        causal = ki <= qi
        full_mask = ops.cast(ops.where(causal, 0.0, MASK_NEG), "float32")[None, None]
        sliding_keep = ops.logical_and(causal, ki > qi - self.sliding_window)
        sliding_mask = ops.cast(ops.where(sliding_keep, 0.0, MASK_NEG), "float32")[
            None, None
        ]
        if padding_mask is not None:
            pad = (1.0 - ops.cast(am, "float32"))[:, None, None, :] * MASK_NEG
            full_mask = full_mask + pad
            sliding_mask = sliding_mask + pad
        hidden = self.token_embedding(token_ids)
        layer_caches = []
        for i, layer in enumerate(self.decoder_layers):
            mask = sliding_mask if self.is_sliding(i) else full_mask
            hidden, (k, v) = layer(
                hidden, cos, sin, attention_mask=mask, use_cache=True
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
        # One decode step; full layers see slots [0, pos], sliding layers see the
        # window (pos - sliding_window, pos]. Empty cache slots are masked too.
        batch = int(token_ids.shape[0])
        max_len = int(cache.shape[4])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos, sin = self.rope(positions)
        ar = ops.arange(max_len)
        full_km = ops.cast(ops.where(ar <= pos, 0.0, MASK_NEG), "float32")[
            None, None, None, :
        ]
        sliding_km = ops.cast(
            ops.where(
                ops.logical_and(ar <= pos, ar > pos - self.sliding_window),
                0.0,
                MASK_NEG,
            ),
            "float32",
        )[None, None, None, :]
        h = self.token_embedding(token_ids)
        layer_caches = []
        for i, layer in enumerate(self.decoder_layers):
            km = sliding_km if self.is_sliding(i) else full_km
            h, ck, cv = layer.decode_step(
                h, cos, sin, cache[:, i, 0], cache[:, i, 1], pos, km
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.final_norm(h))[:, 0, :]
        return logits, cache
