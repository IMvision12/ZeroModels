import re

import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

WEIGHT_NAME_MAPPING = {
    "token_embedding.embeddings": "model.embed_tokens.weight",
    "final_norm.weight": "model.norm.weight",
    "decoder_layer_": "model.layers.",
    "attention.query_norm": "self_attn.q_norm",
    "attention.key_norm": "self_attn.k_norm",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "attention_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    "mlp.gate_weight": "mlp.gate.weight",
    "mlp.e_score_correction_bias": "mlp.e_score_correction_bias",
    "kernel": "weight",
}


def dequantize_fp8(hf_state_dict):
    """Dequantize DeepSeek-style block-FP8 tensors in place.

    The MiniMax-M2 hub checkpoint stores most matrices as ``float8_e4m3fn``
    with a per-128x128-block ``*.weight_scale_inv``; multiply each block by
    its scale and drop the scale keys.
    """
    scales = {k for k in hf_state_dict if k.endswith(".weight_scale_inv")}
    if not scales:
        return hf_state_dict
    out = {}
    for key, value in hf_state_dict.items():
        if key in scales:
            continue
        scale_key = key.replace(".weight", ".weight_scale_inv")
        if scale_key in hf_state_dict:
            if hasattr(value, "float"):
                value = value.float().cpu().numpy()
            scale = hf_state_dict[scale_key]
            if hasattr(scale, "float"):
                scale = scale.float().cpu().numpy()
            value = np.asarray(value, dtype="float32")
            scale = np.asarray(scale, dtype="float32")
            scale_full = np.repeat(np.repeat(scale, 128, axis=0), 128, axis=1)
            value = value * scale_full[: value.shape[0], : value.shape[1]]
        out[key] = value
    return out


def fuse_expert_weights(hf_state_dict):
    """Canonicalize hub per-expert MoE weights (w1/w2/w3) to the fused layout."""
    if not any("block_sparse_moe" in key for key in hf_state_dict):
        return hf_state_dict
    out = {}
    w1, w3, w2 = {}, {}, {}
    pat = re.compile(
        r"^(model\.layers\.\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight$"
    )
    for key, value in hf_state_dict.items():
        match = pat.match(key)
        if match:
            layer, expert, which = match.group(1), int(match.group(2)), match.group(3)
            {"w1": w1, "w3": w3, "w2": w2}[which].setdefault(layer, {})[expert] = value
        elif ".block_sparse_moe.gate." in key:
            out[key.replace(".block_sparse_moe.gate.", ".mlp.gate.")] = value
        elif key.endswith(".block_sparse_moe.e_score_correction_bias"):
            out[
                key.replace(
                    ".block_sparse_moe.e_score_correction_bias",
                    ".mlp.e_score_correction_bias",
                )
            ] = value
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


def transfer_minimax_m2_weights(keras_model, hf_state_dict):
    state = fuse_expert_weights(dequantize_fp8(hf_state_dict))
    if not keras_model.built or not keras_model.weights:
        ids = np.array([[0, 1, 2, 3]], dtype="int32")
        keras_model({"input_ids": ids, "attention_mask": np.ones_like(ids)})
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        # Functional model weight paths are flat (no model-name root to strip).
        name = weight.path.replace("/", ".")
        for old, new in WEIGHT_NAME_MAPPING.items():
            name = name.replace(old, new)
        if name not in state:
            raise WeightMappingError(weight.path, name)
        if (
            ".experts.gate_up_proj" in name
            or ".experts.down_proj" in name
            or name.endswith("mlp.gate.weight")
            or name.endswith("e_score_correction_bias")
        ):
            weight.assign(np.asarray(state[name]))
        else:
            transfer_weights(weight.path, weight, state[name])
