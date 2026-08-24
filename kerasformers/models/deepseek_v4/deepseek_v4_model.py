import math

import keras
import numpy as np
from keras import layers, ops

from kerasformers.base import BaseGeneration, BaseModel, CausalMask, TiedHead
from kerasformers.base.base_mixin import inference_scope

from .deepseek_v4_config import DEEPSEEK_V4_CONFIG, DEEPSEEK_V4_WEIGHTS_URLS
from .deepseek_v4_layers import (
    MASK_NEG,
    DeepseekV4DecoderLayer,
    DeepseekV4HyperHead,
    DeepseekV4RMSNorm,
    apply_v4_rope,
)

COMPRESS_RATIO_TO_LAYER_TYPE = {
    0: "sliding_attention",
    4: "compressed_sparse_attention",
    128: "heavily_compressed_attention",
}


def deepseek_v4_rope(position_ids, inv_freq, compute_dtype):
    inv_freq = ops.convert_to_tensor(inv_freq)
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    return (
        ops.cast(ops.cos(freqs), compute_dtype),
        ops.cast(ops.sin(freqs), compute_dtype),
    )


def deepseek_v4_backbone_features(
    input_ids,
    attention_mask,
    *,
    token_embedding,
    decoder_layers,
    hc_head,
    final_norm,
    sliding_mask_layer,
    main_inv_freq,
    compress_inv_freq,
    hc_mult,
    compute_dtype,
):
    hidden = token_embedding(input_ids)
    streams = ops.repeat(hidden[:, :, None, :], hc_mult, axis=2)
    position_ids = ops.cumsum(ops.ones_like(input_ids), axis=-1) - 1
    cos, sin = deepseek_v4_rope(position_ids, main_inv_freq, compute_dtype)
    cos_c, sin_c = deepseek_v4_rope(position_ids, compress_inv_freq, compute_dtype)
    sliding_mask = sliding_mask_layer(input_ids, attention_mask)
    for layer in decoder_layers:
        streams = layer(
            streams,
            cos if layer.layer_type == "sliding_attention" else cos_c,
            sin if layer.layer_type == "sliding_attention" else sin_c,
            cos_c,
            sin_c,
            position_ids=position_ids,
            sliding_mask=sliding_mask,
            input_ids=input_ids,
        )
    return final_norm(hc_head(streams))


def yarn_inv_freq(dim, base, factor, original_max, beta_fast=32, beta_slow=1):
    def find_correction_dim(num_rotations):
        return (dim * math.log(original_max / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = max(math.floor(find_correction_dim(beta_fast)), 0)
    high = min(math.ceil(find_correction_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2, dtype="float32") - low) / (high - low), 0, 1)
    pos_freqs = base ** (np.arange(0, dim, 2, dtype="float32") / dim)
    return (1.0 / (factor * pos_freqs)) * ramp + (1.0 / pos_freqs) * (1 - ramp)


# Both DeepSeek-V4 families (Flash and Pro) share this yarn rope scaling for the
# compress rope, so it is the DeepseekV4Model default and the per-variant configs
# need not repeat it. ``config_from_hf`` still passes the checkpoint's own value.
DEFAULT_ROPE_SCALING = {
    "type": "yarn",
    "factor": 16,
    "beta_fast": 32,
    "beta_slow": 1,
    "original_max_position_embeddings": 65536,
}


@keras.saving.register_keras_serializable(package="kerasformers")
class DeepseekV4Model(BaseModel):
    """DeepSeek-V4 decoder (V4-Flash / V4-Pro).

    Shared-KV MQA with per-head attention sinks and a 128-token sliding
    window on every layer; CSA layers add Lightning-Indexer-selected
    compressed KV entries (4-token two-series windows) and HCA layers add all
    causal heavily-compressed entries (128-token windows). The residual is
    ``hc_mult`` parallel streams mixed through Sinkhorn-projected
    Manifold-Constrained Hyper-Connections; every layer ends in a
    sqrt-softplus MoE with a shared expert (the first ``hash_moe`` layers
    select experts from the frozen ``tid2eid`` token-hash table). Sliding
    layers use plain rope at ``rope_theta``; compressor layers and the
    compressed entries use yarn rope at ``compress_rope_theta``. Returns raw
    features; use :class:`DeepseekV4TextGenerate` for logits / text.

    Args:
        vocab_size / embed_dim / num_layers / num_heads / head_dim: Geometry.
        q_lora_rank / qk_rope_head_dim / o_groups / o_lora_rank: Attention.
        layer_types: Per-layer attention flavor.
        mlp_layer_types: Per-layer ``"hash_moe"`` / ``"moe"``.
        num_experts / num_experts_per_tok / moe_mlp_dim /
        routed_scaling_factor / swiglu_limit: MoE shape.
        sliding_window: Raw-KV window (128).
        compress_rate_csa / compress_rate_hca: Compressor windows (4 / 128).
        index_n_heads / index_head_dim / index_topk: Indexer geometry.
        hc_mult / hc_sinkhorn_iters / hc_eps: Hyper-connection geometry.
        rope_theta / compress_rope_theta: Dual rope bases (1e4 / 1.6e5).
        rope_scaling: Yarn dict applied to the compress rope (the reference
            forces its cos/sin ``attention_factor`` to 1.0).
        norm_eps: RMSNorm epsilon.
        tie_embeddings: Whether :class:`DeepseekV4TextGenerate` ties the LM head.
    """

    HF_MODEL_TYPE = "deepseek_v4"
    BASE_MODEL_CONFIG = DEEPSEEK_V4_CONFIG
    BASE_WEIGHT_CONFIG = DEEPSEEK_V4_WEIGHTS_URLS
    output_logits = False

    def __init__(
        self,
        vocab_size=129280,
        embed_dim=4096,
        num_layers=43,
        num_heads=64,
        head_dim=512,
        q_lora_rank=1024,
        qk_rope_head_dim=64,
        o_groups=8,
        o_lora_rank=1024,
        layer_types=None,
        mlp_layer_types=None,
        num_experts=256,
        num_experts_per_tok=6,
        moe_mlp_dim=2048,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        sliding_window=128,
        compress_rate_csa=4,
        compress_rate_hca=128,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_scaling=DEFAULT_ROPE_SCALING,
        norm_eps=1e-6,
        tie_embeddings=False,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        if layer_types is None:
            interleave = [
                "compressed_sparse_attention"
                if i % 2
                else "heavily_compressed_attention"
                for i in range(max(num_layers - 2, 0))
            ]
            layer_types = ["heavily_compressed_attention"] * min(
                num_layers, 2
            ) + interleave
        if mlp_layer_types is None:
            mlp_layer_types = ["hash_moe"] * min(num_layers, 3) + ["moe"] * max(
                0, num_layers - 3
            )
        layer_types = tuple(layer_types)
        mlp_layer_types = tuple(mlp_layer_types)
        rope_scaling = dict(rope_scaling) if rope_scaling else None

        rd = qk_rope_head_dim
        main_inv_freq = 1.0 / (
            rope_theta ** (np.arange(0, rd, 2, dtype="float32") / rd)
        )
        scaling = rope_scaling or {}
        if scaling.get("rope_type", scaling.get("type", "default")) == "yarn":
            compress_inv_freq = yarn_inv_freq(
                rd,
                compress_rope_theta,
                scaling.get("factor", 16),
                scaling.get("original_max_position_embeddings", 65536),
                scaling.get("beta_fast") or 32,
                scaling.get("beta_slow") or 1,
            )
        else:
            compress_inv_freq = 1.0 / (
                compress_rope_theta ** (np.arange(0, rd, 2, dtype="float32") / rd)
            )

        token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        decoder_layers = [
            DeepseekV4DecoderLayer(
                embed_dim,
                num_heads,
                head_dim,
                q_lora_rank,
                qk_rope_head_dim,
                o_groups,
                o_lora_rank,
                layer_types[i],
                sliding_window,
                compress_rate_hca
                if layer_types[i] == "heavily_compressed_attention"
                else compress_rate_csa,
                index_n_heads,
                index_head_dim,
                index_topk,
                num_experts,
                num_experts_per_tok,
                moe_mlp_dim,
                routed_scaling_factor,
                mlp_layer_types[i] == "hash_moe",
                vocab_size,
                swiglu_limit,
                hc_mult,
                hc_sinkhorn_iters,
                hc_eps,
                compress_inv_freq=compress_inv_freq,
                norm_eps=norm_eps,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        hc_head = DeepseekV4HyperHead(hc_mult, embed_dim, hc_eps, name="hc_head")
        final_norm = DeepseekV4RMSNorm(eps=norm_eps, name="final_norm")
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
        hidden = deepseek_v4_backbone_features(
            inputs["input_ids"],
            inputs["attention_mask"],
            token_embedding=token_embedding,
            decoder_layers=decoder_layers,
            hc_head=hc_head,
            final_norm=final_norm,
            sliding_mask_layer=sliding_mask_layer,
            main_inv_freq=main_inv_freq,
            compress_inv_freq=compress_inv_freq,
            hc_mult=hc_mult,
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
        self.hc_head = hc_head
        self.final_norm = final_norm
        self.sliding_mask_layer = sliding_mask_layer
        self.lm_head = lm_head
        self.main_inv_freq = main_inv_freq
        self.compress_inv_freq = compress_inv_freq
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.layer_types = layer_types
        self.mlp_layer_types = mlp_layer_types
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_mlp_dim = moe_mlp_dim
        self.routed_scaling_factor = routed_scaling_factor
        self.swiglu_limit = swiglu_limit
        self.sliding_window = sliding_window
        self.compress_rate_csa = compress_rate_csa
        self.compress_rate_hca = compress_rate_hca
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.rope_theta = rope_theta
        self.compress_rope_theta = compress_rope_theta
        self.rope_scaling = rope_scaling
        self.norm_eps = norm_eps
        self.tie_embeddings = tie_embeddings

        # V4's attention (sinks + compressor branch) trips Keras' symbolic
        # auto-build trace on some backends, leaving sublayers unbuilt after
        # __init__; a concrete dummy forward materializes every weight so
        # from_weights (which loads before any forward) has a complete model.
        with inference_scope():
            self(
                {
                    "input_ids": ops.zeros((1, 4), dtype="int32"),
                    "attention_mask": ops.ones((1, 4), dtype="int32"),
                }
            )

    def rope_tables_main(self, position_ids):
        inv_freq = ops.convert_to_tensor(self.main_inv_freq)
        freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
        return (
            ops.cast(ops.cos(freqs), self.compute_dtype),
            ops.cast(ops.sin(freqs), self.compute_dtype),
        )

    def rope_tables_compress(self, position_ids):
        # The reference forces the compress yarn attention_factor to 1.0.
        inv_freq = ops.convert_to_tensor(self.compress_inv_freq)
        freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
        return (
            ops.cast(ops.cos(freqs), self.compute_dtype),
            ops.cast(ops.sin(freqs), self.compute_dtype),
        )

    def sliding_causal_mask(self, seq, attention_mask=None):
        qi = ops.arange(seq)[:, None]
        ki = ops.arange(seq)[None, :]
        keep = ops.logical_and(ki <= qi, (qi - ki) < self.sliding_window)
        mask = ops.cast(ops.where(keep, 0.0, MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            mask = mask + (1.0 - am)[:, None, None, :] * MASK_NEG
        return mask

    @classmethod
    def config_from_hf(cls, hf_config):
        n = hf_config["num_hidden_layers"]
        layer_types = hf_config.get("layer_types")
        if layer_types is None and hf_config.get("compress_ratios") is not None:
            layer_types = [
                COMPRESS_RATIO_TO_LAYER_TYPE[r]
                for r in hf_config["compress_ratios"][:n]
            ]
        mlp_layer_types = hf_config.get("mlp_layer_types")
        if mlp_layer_types is None:
            n_hash = hf_config.get("num_hash_layers", 3)
            mlp_layer_types = ["hash_moe"] * min(n, n_hash) + ["moe"] * max(
                0, n - n_hash
            )
        rates = hf_config.get("compress_rates") or {}
        return {
            "vocab_size": hf_config["vocab_size"],
            "embed_dim": hf_config["hidden_size"],
            "num_layers": n,
            "num_heads": hf_config["num_attention_heads"],
            "head_dim": hf_config.get("head_dim", 512),
            "q_lora_rank": hf_config.get("q_lora_rank", 1024),
            "qk_rope_head_dim": hf_config.get("qk_rope_head_dim", 64),
            "o_groups": hf_config.get("o_groups", 8),
            "o_lora_rank": hf_config.get("o_lora_rank", 1024),
            "layer_types": tuple(layer_types[:n]) if layer_types else None,
            "mlp_layer_types": tuple(mlp_layer_types[:n]),
            "num_experts": hf_config.get("n_routed_experts", 256),
            "num_experts_per_tok": hf_config.get("num_experts_per_tok", 6),
            "moe_mlp_dim": hf_config.get("moe_intermediate_size", 2048),
            "routed_scaling_factor": hf_config.get("routed_scaling_factor", 1.5),
            "swiglu_limit": hf_config.get("swiglu_limit", 10.0),
            "sliding_window": hf_config.get("sliding_window", 128),
            "compress_rate_csa": rates.get(
                "compressed_sparse_attention",
                hf_config.get("compress_rate_csa", 4),
            ),
            "compress_rate_hca": rates.get(
                "heavily_compressed_attention",
                hf_config.get("compress_rate_hca", 128),
            ),
            "index_n_heads": hf_config.get("index_n_heads", 64),
            "index_head_dim": hf_config.get("index_head_dim", 128),
            "index_topk": hf_config.get("index_topk", 512),
            "hc_mult": hf_config.get("hc_mult", 4),
            "hc_sinkhorn_iters": hf_config.get("hc_sinkhorn_iters", 20),
            "hc_eps": hf_config.get("hc_eps", 1e-6),
            "rope_theta": hf_config.get("rope_theta", 10000.0),
            "compress_rope_theta": hf_config.get("compress_rope_theta", 160000.0),
            "rope_scaling": hf_config.get("rope_scaling")
            or (hf_config.get("rope_parameters") or {}).get("compress"),
            "norm_eps": hf_config.get("rms_norm_eps", 1e-6),
            "tie_embeddings": bool(hf_config.get("tie_word_embeddings") or False),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_deepseek_v4_hf_to_keras import transfer_deepseek_v4_weights

        transfer_deepseek_v4_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "q_lora_rank": self.q_lora_rank,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "o_groups": self.o_groups,
                "o_lora_rank": self.o_lora_rank,
                "layer_types": self.layer_types,
                "mlp_layer_types": self.mlp_layer_types,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "moe_mlp_dim": self.moe_mlp_dim,
                "routed_scaling_factor": self.routed_scaling_factor,
                "swiglu_limit": self.swiglu_limit,
                "sliding_window": self.sliding_window,
                "compress_rate_csa": self.compress_rate_csa,
                "compress_rate_hca": self.compress_rate_hca,
                "index_n_heads": self.index_n_heads,
                "index_head_dim": self.index_head_dim,
                "index_topk": self.index_topk,
                "hc_mult": self.hc_mult,
                "hc_sinkhorn_iters": self.hc_sinkhorn_iters,
                "hc_eps": self.hc_eps,
                "rope_theta": self.rope_theta,
                "compress_rope_theta": self.compress_rope_theta,
                "rope_scaling": self.rope_scaling,
                "norm_eps": self.norm_eps,
                "tie_embeddings": self.tie_embeddings,
                "name": self.name,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class DeepseekV4TextGenerate(DeepseekV4Model, BaseGeneration):
    """DeepSeek-V4 with an LM head + fast ``.generate()``.

    The streaming state per layer is ``(kv,)`` for sliding layers and
    ``(kv, comp_kv, comp_gate, comp_entries[, idx_kv, idx_gate,
    idx_entries])`` for compressor layers: raw compressor projections are
    cached for every position, and whenever a window boundary closes during
    decode (``(pos + 1) % rate == 0``) the new compressed entry is computed
    from the cached projections and written into the fixed-size entries
    buffer, so the loop stays constant-shape.
    """

    # DeepSeek-V4 end-of-sentence id (1). Explicit generate() args override.
    eos_token_id = (1,)
    output_logits = True

    def project(self, hidden):
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return ops.matmul(hidden, ops.transpose(self.token_embedding.embeddings))

    def entry_capacity(self, rate, max_len):
        return max(max_len // rate, 1)

    def build_cache(self, token_ids, padding_mask, max_len):
        batch = int(token_ids.shape[0])
        seq = int(token_ids.shape[1])
        input_ids = ops.cast(token_ids, "int32")
        hidden = self.token_embedding(input_ids)
        streams = ops.repeat(hidden[:, :, None, :], self.hc_mult, axis=2)
        position_ids = ops.broadcast_to(ops.arange(seq), (batch, seq))
        cos, sin = self.rope_tables_main(position_ids)
        cos_c, sin_c = self.rope_tables_compress(position_ids)
        sliding_mask = self.sliding_causal_mask(seq, padding_mask)
        caches = []
        for layer in self.decoder_layers:
            streams, pieces = layer(
                streams,
                cos if layer.layer_type == "sliding_attention" else cos_c,
                sin if layer.layer_type == "sliding_attention" else sin_c,
                cos_c,
                sin_c,
                position_ids=position_ids,
                sliding_mask=sliding_mask,
                input_ids=input_ids,
                use_cache=True,
            )
            kv = pieces["kv"]
            ckv = ops.slice_update(
                ops.zeros((batch, 1, max_len, self.head_dim), dtype=kv.dtype),
                (0, 0, 0, 0),
                kv,
            )
            if layer.layer_type == "sliding_attention":
                caches.append((ckv,))
                continue
            rate = layer.compress_rate
            cap = self.entry_capacity(rate, max_len)
            piece = [ckv]
            for raw_key, ent_key, ent_dim in (
                ("comp_kv", "comp_entries", self.head_dim),
                ("idx_kv", "idx_entries", self.index_head_dim),
            ):
                gate_key = raw_key.replace("kv", "gate")
                if raw_key not in pieces:
                    break
                raw_kv = pieces[raw_key]
                raw_gate = pieces[gate_key]
                width = int(raw_kv.shape[-1])
                rkv = ops.slice_update(
                    ops.zeros((batch, max_len, width), dtype=raw_kv.dtype),
                    (0, 0, 0),
                    raw_kv,
                )
                rgate = ops.slice_update(
                    ops.zeros((batch, max_len, width), dtype=raw_gate.dtype),
                    (0, 0, 0),
                    raw_gate,
                )
                ent = ops.zeros((batch, cap, ent_dim), dtype=kv.dtype)
                if ent_key in pieces:
                    ent = ops.slice_update(ent, (0, 0, 0), pieces[ent_key])
                piece.extend([rkv, rgate, ent])
            caches.append(tuple(piece))
        logits = self.project(self.final_norm(self.hc_head(streams))[:, -1, :])
        return tuple(caches), logits

    def decode_entry_update(self, compressor, raw_kv, raw_gate, entries, pos, rate):
        """Close a compressor window at a decode boundary.

        When ``(pos + 1) % rate == 0``, window ``w = (pos + 1) // rate - 1``
        is compressed from the cached raw projections (with the previous
        window's Ca slice for the two-series flavor) and written to slot
        ``w`` of the fixed entries buffer; otherwise the buffer is returned
        unchanged.
        """
        batch = int(raw_kv.shape[0])
        width = int(raw_kv.shape[-1])
        hd = compressor.head_dim
        threshold = (pos + 1) // rate
        closed = ops.equal((pos + 1) % rate, 0)
        w = ops.maximum(threshold - 1, 0)
        start = ops.maximum(pos + 1 - rate, 0)
        chunk_kv = ops.slice(raw_kv, (0, start, 0), (batch, rate, width))[:, None]
        chunk_gate = (
            ops.slice(raw_gate, (0, start, 0), (batch, rate, width))[:, None]
            + compressor.position_bias
        )
        prior_kv = None
        prior_gate = None
        if compressor.two_series:
            start2 = ops.maximum(pos + 1 - 2 * rate, 0)
            p_kv = ops.slice(raw_kv, (0, start2, 0), (batch, rate, width))[..., :hd]
            p_gate = ops.slice(raw_gate, (0, start2, 0), (batch, rate, width))[..., :hd]
            has_prior = ops.cast(w > 0, p_kv.dtype)
            prior_kv = p_kv * has_prior
            prior_gate = ops.where(w > 0, p_gate, MASK_NEG)
        entry = compressor.compress_chunk(chunk_kv, chunk_gate, prior_kv, prior_gate)
        epos = ops.broadcast_to(ops.reshape(pos + 1 - rate, (1, 1)), (batch, 1))
        ecos, esin = self.rope_tables_compress(epos)
        entry = apply_v4_rope(entry[:, None], ecos, esin)[:, 0]
        updated = ops.slice_update(entries, (0, w, 0), entry)
        return ops.where(closed, updated, entries), threshold

    def call_with_cache(self, token_ids, cache, cache_update_index):
        batch = int(token_ids.shape[0])
        max_len = int(cache[0][0].shape[2])
        pos = cache_update_index
        positions = ops.broadcast_to(ops.reshape(pos, (1, 1)), (batch, 1))
        cos, sin = self.rope_tables_main(positions)
        cos_c, sin_c = self.rope_tables_compress(positions)
        key_idx = ops.arange(max_len)
        raw_keep = ops.logical_and(
            key_idx <= pos, (pos - key_idx) < self.sliding_window
        )
        raw_mask = ops.cast(ops.where(raw_keep, 0.0, MASK_NEG), "float32")[
            None, None, None, :
        ]
        input_ids = ops.cast(token_ids, "int32")
        h = self.token_embedding(input_ids)
        streams = ops.repeat(h[:, :, None, :], self.hc_mult, axis=2)
        new_cache = []
        for i, layer in enumerate(self.decoder_layers):
            attention = layer.attention
            lcos, lsin = (
                (cos, sin)
                if layer.layer_type == "sliding_attention"
                else (cos_c, sin_c)
            )
            post, comb, collapsed = layer.attn_hc(streams)
            x = layer.attention_norm(collapsed)
            q, q_residual = attention.project_q(x, lcos, lsin)
            kv_new = attention.project_kv(x, lcos, lsin)
            # rope runs in float32; match the cache dtype before writing
            kv_new = ops.cast(kv_new, cache[i][0].dtype)
            ckv = ops.slice_update(cache[i][0], (0, 0, pos, 0), kv_new)
            kv_all = ckv
            mask = raw_mask
            piece = [ckv]
            if layer.layer_type != "sliding_attention":
                rate = layer.compress_rate
                kv_raw_new, gate_raw_new = attention.compressor(x)
                rkv = ops.slice_update(cache[i][1], (0, pos, 0), kv_raw_new)
                rgate = ops.slice_update(cache[i][2], (0, pos, 0), gate_raw_new)
                entries, threshold = self.decode_entry_update(
                    attention.compressor, rkv, rgate, cache[i][3], pos, rate
                )
                piece.extend([rkv, rgate, entries])
                cap = int(entries.shape[1])
                visible = (
                    ops.arange(cap, dtype="int32")[None, None, :]
                    < ops.cast(threshold, "int32")[..., None, None]
                )
                visible = ops.broadcast_to(visible, (batch, 1, cap))
                if layer.layer_type == "heavily_compressed_attention":
                    block_bias = ops.where(visible, 0.0, MASK_NEG)[:, None]
                else:
                    ikv_new, igate_new = attention.index_compressor(x)
                    irkv = ops.slice_update(cache[i][4], (0, pos, 0), ikv_new)
                    irgate = ops.slice_update(cache[i][5], (0, pos, 0), igate_new)
                    idx_entries, _ = self.decode_entry_update(
                        attention.index_compressor, irkv, irgate, cache[i][6], pos, rate
                    )
                    piece.extend([irkv, irgate, idx_entries])
                    block_bias = attention.index_block_bias(
                        x, q_residual, idx_entries, visible, cos_c, sin_c
                    )
                kv_all = ops.concatenate([kv_all, entries[:, None]], axis=2)
                mask = ops.concatenate(
                    [mask, ops.cast(block_bias, mask.dtype)], axis=-1
                )
            attn = attention.sink_attention(q, kv_all, mask)
            attn_out = attention.project_out(attn, lcos, lsin)
            streams = layer.mix(streams, post, comb, attn_out)
            post, comb, collapsed = layer.ffn_hc(streams)
            mlp_out = layer.mlp(layer.mlp_norm(collapsed), input_ids=input_ids)
            streams = layer.mix(streams, post, comb, mlp_out)
            new_cache.append(tuple(piece))
        logits = self.project(self.final_norm(self.hc_head(streams)))[:, 0, :]
        return logits, tuple(new_cache)
