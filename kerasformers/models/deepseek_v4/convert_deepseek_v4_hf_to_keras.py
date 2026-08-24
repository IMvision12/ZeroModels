import re

import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

WEIGHT_NAME_MAPPING = {
    "token_embedding.embeddings": "embed_tokens.weight",
    "final_norm.weight": "norm.weight",
    "decoder_layer_": "layers.",
    "attention.index_compressor.kv_norm": "self_attn.compressor.indexer.kv_norm",
    "attention.index_compressor.kv_proj": "self_attn.compressor.indexer.kv_proj",
    "attention.index_compressor.gate_proj": "self_attn.compressor.indexer.gate_proj",
    "attention.index_compressor.position_bias": (
        "self_attn.compressor.indexer.position_bias"
    ),
    "attention.index_query": "self_attn.compressor.indexer.q_b_proj",
    "attention.index_weights": "self_attn.compressor.indexer.scorer.weights_proj",
    "attention.compressor.": "self_attn.compressor.",
    "attention.query_a_norm": "self_attn.q_a_norm",
    "attention.query_a": "self_attn.q_a_proj",
    "attention.query_b": "self_attn.q_b_proj",
    "attention.kv_norm": "self_attn.kv_norm",
    "attention.kv": "self_attn.kv_proj",
    "attention.output_a": "self_attn.o_a_proj.weight",
    "attention.output_b": "self_attn.o_b_proj",
    "attention.sinks": "self_attn.sinks",
    "attention_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    "mlp.shared_experts.gate": "mlp.shared_experts.gate_proj",
    "mlp.shared_experts.up": "mlp.shared_experts.up_proj",
    "mlp.shared_experts.down": "mlp.shared_experts.down_proj",
    "mlp.gate_weight": "mlp.gate.weight",
    "mlp.e_score_correction_bias": "mlp.gate.e_score_correction_bias",
    "mlp.tid2eid": "mlp.gate.tid2eid",
    "kernel": "weight",
}

DIRECT_SUFFIXES = (
    ".experts.gate_up_proj",
    ".experts.down_proj",
    "mlp.gate.weight",
    "e_score_correction_bias",
    "tid2eid",
    ".sinks",
    "o_a_proj.weight",
    ".position_bias",
    "attn_hc.fn",
    "attn_hc.base",
    "attn_hc.scale",
    "ffn_hc.fn",
    "ffn_hc.base",
    "ffn_hc.scale",
    "hc_head.hc_fn",
    "hc_head.hc_base",
    "hc_head.hc_scale",
)


def dequantize_fp8(state):
    """Dequantize block-FP8 tensors. The V4 hub checkpoints name the
    per-128x128-block scales ``<leaf>.scale`` beside ``<leaf>.weight``."""
    scales = {
        k
        for k in state
        if k.endswith(".scale") and k[: -len("scale")] + "weight" in state
    }
    scales |= {k for k in state if k.endswith(".weight_scale_inv")}
    if not scales:
        return state
    out = {}
    for key, value in state.items():
        if key in scales:
            continue
        scale_key = None
        if key.endswith(".weight"):
            for cand in (key[: -len("weight")] + "scale", key + "_scale_inv"):
                if cand in scales:
                    scale_key = cand
                    break
        if scale_key is not None:
            if hasattr(value, "float"):
                value = value.float().cpu().numpy()
            scale = state[scale_key]
            if hasattr(scale, "float"):
                scale = scale.float().cpu().numpy()
            value = np.asarray(value, dtype="float32")
            scale = np.asarray(scale, dtype="float32")
            scale_full = np.repeat(np.repeat(scale, 128, axis=0), 128, axis=1)
            value = value * scale_full[: value.shape[0], : value.shape[1]]
        out[key] = value
    return out


def normalize_keys(hf_state_dict):
    """Canonicalize both layouts to bare current-transformers names.

    The hub checkpoints use the original DeepSeek naming (``embed.weight``,
    ``layers.N.attn.wq_a`` / ``wkv`` / ``wo_a``, flat ``hc_attn_fn`` params,
    ``ffn.experts.N.w1/w2/w3``, ``ffn.gate.bias`` / ``tid2eid``, indexer
    leaves under ``attn.indexer.*``, ``mtp.*`` heads); in-memory state dicts
    already use ``model.*``-prefixed module names with fused experts.
    """
    out = {}
    for key, value in hf_state_dict.items():
        if key.startswith("mtp."):
            continue
        if key.startswith("model."):
            key = key[len("model.") :]
        if key == "embed.weight":
            key = "embed_tokens.weight"
        elif key == "head.weight":
            key = "lm_head.weight"
        elif key == "hc_head_fn":
            key = "hc_head.hc_fn"
        elif key == "hc_head_base":
            key = "hc_head.hc_base"
        elif key == "hc_head_scale":
            key = "hc_head.hc_scale"
        key = re.sub(r"^layers\.(\d+)\.attn_norm\.", r"layers.\1.input_layernorm.", key)
        key = re.sub(
            r"^layers\.(\d+)\.ffn_norm\.", r"layers.\1.post_attention_layernorm.", key
        )
        key = re.sub(r"^layers\.(\d+)\.hc_attn_fn$", r"layers.\1.attn_hc.fn", key)
        key = re.sub(r"^layers\.(\d+)\.hc_attn_base$", r"layers.\1.attn_hc.base", key)
        key = re.sub(r"^layers\.(\d+)\.hc_attn_scale$", r"layers.\1.attn_hc.scale", key)
        key = re.sub(r"^layers\.(\d+)\.hc_ffn_fn$", r"layers.\1.ffn_hc.fn", key)
        key = re.sub(r"^layers\.(\d+)\.hc_ffn_base$", r"layers.\1.ffn_hc.base", key)
        key = re.sub(r"^layers\.(\d+)\.hc_ffn_scale$", r"layers.\1.ffn_hc.scale", key)
        key = re.sub(r"^layers\.(\d+)\.attn\.", r"layers.\1.self_attn.", key)
        key = re.sub(r"^layers\.(\d+)\.ffn\.", r"layers.\1.mlp.", key)
        key = key.replace(".self_attn.attn_sink", ".self_attn.sinks")
        key = key.replace(
            ".self_attn.indexer.compressor.norm.",
            ".self_attn.compressor.indexer.kv_norm.",
        )
        key = key.replace(
            ".self_attn.indexer.compressor.ape",
            ".self_attn.compressor.indexer.position_bias",
        )
        key = key.replace(
            ".self_attn.indexer.compressor.", ".self_attn.compressor.indexer."
        )
        key = key.replace(".self_attn.indexer.", ".self_attn.compressor.indexer.")
        key = key.replace(
            ".compressor.indexer.weights_proj.",
            ".compressor.indexer.scorer.weights_proj.",
        )
        key = key.replace(
            ".self_attn.compressor.norm.", ".self_attn.compressor.kv_norm."
        )
        key = key.replace(
            ".self_attn.compressor.ape", ".self_attn.compressor.position_bias"
        )
        key = key.replace(".wq_a.", ".q_a_proj.")
        key = key.replace(".wq_b.", ".q_b_proj.")
        key = key.replace(".wkv.", ".kv_proj.")
        key = key.replace(".wgate.", ".gate_proj.")
        key = key.replace(".wo_a.", ".o_a_proj.")
        key = key.replace(".wo_b.", ".o_b_proj.")
        key = key.replace(".self_attn.q_norm.", ".self_attn.q_a_norm.")
        key = key.replace(".mlp.gate.bias", ".mlp.gate.e_score_correction_bias")
        key = key.replace(".mlp.shared_experts.w1.", ".mlp.shared_experts.gate_proj.")
        key = key.replace(".mlp.shared_experts.w2.", ".mlp.shared_experts.down_proj.")
        key = key.replace(".mlp.shared_experts.w3.", ".mlp.shared_experts.up_proj.")
        out[key] = value
    return out


def fuse_expert_weights(state):
    """Fuse hub per-expert ``mlp.experts.N.w1/w2/w3`` into the fused layout."""
    pat = re.compile(r"^(layers\.\d+)\.mlp\.experts\.(\d+)\.(w[123])\.weight$")
    if not any(pat.match(k) for k in state):
        return state
    out = {}
    w1, w3, w2 = {}, {}, {}
    for key, value in state.items():
        match = pat.match(key)
        if match:
            layer, expert, which = match.group(1), int(match.group(2)), match.group(3)
            {"w1": w1, "w3": w3, "w2": w2}[which].setdefault(layer, {})[expert] = value
        else:
            out[key] = value
    for layer in w1:
        experts = sorted(w1[layer])
        gate_up = np.stack(
            [
                np.concatenate(
                    [np.asarray(w1[layer][e]), np.asarray(w3[layer][e])], axis=0
                )
                for e in experts
            ],
            axis=0,
        )
        down = np.stack([np.asarray(w2[layer][e]) for e in experts], axis=0)
        out[f"{layer}.mlp.experts.gate_up_proj"] = gate_up
        out[f"{layer}.mlp.experts.down_proj"] = down
    return out


def transfer_deepseek_v4_weights(keras_model, hf_state_dict):
    state = fuse_expert_weights(normalize_keys(dequantize_fp8(hf_state_dict)))
    if not keras_model.built or not keras_model.weights:
        keras_model({"input_ids": np.zeros((1, 4), dtype="int64")})
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        name = weight.path.replace("/", ".")
        for old, new in WEIGHT_NAME_MAPPING.items():
            name = name.replace(old, new)
        if name not in state:
            raise WeightMappingError(weight.path, name)
        value = state[name]
        if any(name.endswith(sfx) or sfx in name for sfx in DIRECT_SUFFIXES):
            value = np.asarray(value)
            if name.endswith("tid2eid"):
                value = value.astype("int32")
            weight.assign(value)
        else:
            transfer_weights(weight.path, weight, value)
