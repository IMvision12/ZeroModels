import re

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights

WEIGHT_NAME_MAPPING = {
    "token_embedding.embeddings": "model.embed_tokens.weight",
    "final_norm.weight": "model.norm.weight",
    "decoder_layer_": "model.layers.",
    "attention.query_a_norm": "self_attn.q_a_layernorm",
    "attention.query_a": "self_attn.q_a_proj",
    "attention.query_b": "self_attn.q_b_proj",
    "attention.query": "self_attn.q_proj",
    "attention.kv_a_norm": "self_attn.kv_a_layernorm",
    "attention.kv_a": "self_attn.kv_a_proj_with_mqa",
    "attention.kv_b": "self_attn.kv_b_proj",
    "attention.output_proj": "self_attn.o_proj",
    "attention_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    "mlp.shared_experts.gate": "mlp.shared_experts.gate_proj",
    "mlp.shared_experts.up": "mlp.shared_experts.up_proj",
    "mlp.shared_experts.down": "mlp.shared_experts.down_proj",
    "mlp.gate.kernel": "mlp.gate_proj.weight",
    "mlp.up.kernel": "mlp.up_proj.weight",
    "mlp.down.kernel": "mlp.down_proj.weight",
    "mlp.gate_weight": "mlp.gate.weight",
    "kernel": "weight",
}


def fuse_expert_weights(hf_state_dict):
    pat = re.compile(
        r"^(model\.layers\.\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
    )
    if not any(pat.match(k) for k in hf_state_dict):
        return hf_state_dict
    out = {}
    gate, up, down = {}, {}, {}
    for key, value in hf_state_dict.items():
        match = pat.match(key)
        if match:
            layer, expert, which = match.group(1), int(match.group(2)), match.group(3)
            {"gate_proj": gate, "up_proj": up, "down_proj": down}[which].setdefault(
                layer, {}
            )[expert] = value
        else:
            out[key] = value
    for layer in gate:
        experts = sorted(gate[layer])
        gate_up = np.stack(
            [
                np.concatenate(
                    [np.asarray(gate[layer][e]), np.asarray(up[layer][e])], axis=0
                )
                for e in experts
            ],
            axis=0,
        )
        down_w = np.stack([np.asarray(down[layer][e]) for e in experts], axis=0)
        out[f"{layer}.mlp.experts.gate_up_proj"] = gate_up
        out[f"{layer}.mlp.experts.down_proj"] = down_w
    return out


def transfer_deepseek_v2_weights(keras_model, hf_state_dict):
    state = fuse_expert_weights(hf_state_dict)
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
        ):
            weight.assign(np.asarray(state[name]))
        else:
            transfer_weights(weight.path, weight, state[name])
