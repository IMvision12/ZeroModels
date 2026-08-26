import re

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights

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
    # MoE: shared expert + router (specific rules before the dense gate rule).
    "mlp.shared_expert.gate": "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up": "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down": "mlp.shared_expert.down_proj",
    "mlp.shared_expert_gate.kernel": "mlp.shared_expert_gate.weight",
    "mlp.gate_weight": "mlp.gate.weight",
    # Dense MLP (mlp_only_layers); explicit ``.kernel`` so it can't corrupt the
    # router's ``mlp.gate.weight``.
    "mlp.gate.kernel": "mlp.gate_proj.weight",
    "mlp.up.kernel": "mlp.up_proj.weight",
    "mlp.down.kernel": "mlp.down_proj.weight",
    "linear_attn.conv_weight": "linear_attn.conv1d.weight",
    "kernel": "weight",
}


def split_gated_deltanet_in_proj(layer, qkvz, ba):
    nk, nv = layer.num_k_heads, layer.num_v_heads
    hk, hv = layer.head_k_dim, layer.head_v_dim
    ratio = nv // nk
    hidden = qkvz.shape[1]
    group = 2 * hk + 2 * ratio * hv
    qkvz = qkvz.reshape(nk, group, hidden)
    q = qkvz[:, :hk].reshape(nk * hk, hidden)
    k = qkvz[:, hk : 2 * hk].reshape(nk * hk, hidden)
    v = qkvz[:, 2 * hk : 2 * hk + ratio * hv].reshape(nv * hv, hidden)
    z = qkvz[:, 2 * hk + ratio * hv :].reshape(nv * hv, hidden)
    ba = ba.reshape(nk, 2 * ratio, hidden)
    b = ba[:, :ratio].reshape(nv, hidden)
    a = ba[:, ratio:].reshape(nv, hidden)
    return np.concatenate([q, k, v], axis=0), z, b, a


def fuse_expert_weights(hf_state_dict):
    """Fuse hub per-expert ``mlp.experts.N.{gate,up,down}_proj`` into the fused
    ``mlp.experts.gate_up_proj`` (E, 2I, H) / ``down_proj`` (E, H, I)."""
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


def transfer_qwen3_next_weights(keras_model, hf_state_dict):
    state = fuse_expert_weights(hf_state_dict)
    if not keras_model.built or not keras_model.weights:
        ids = np.array([[0, 1, 2, 3]], dtype="int32")
        keras_model({"input_ids": ids, "attention_mask": np.ones_like(ids)})

    handled = set()
    for i, decoder_layer in enumerate(keras_model.decoder_layers):
        if getattr(decoder_layer, "layer_type", None) == "full_attention":
            continue
        gdn = decoder_layer.linear_attn
        prefix = f"model.layers.{i}.linear_attn"
        qkvz = np.asarray(state[f"{prefix}.in_proj_qkvz.weight"])
        ba = np.asarray(state[f"{prefix}.in_proj_ba.weight"])
        qkv_w, z_w, b_w, a_w = split_gated_deltanet_in_proj(gdn, qkvz, ba)
        for dense, packed in (
            (gdn.in_proj_qkv, qkv_w),
            (gdn.in_proj_z, z_w),
            (gdn.in_proj_b, b_w),
            (gdn.in_proj_a, a_w),
        ):
            transfer_weights(dense.kernel.path, dense.kernel, packed)
            handled.add(dense.kernel.path)

    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        if weight.path in handled:
            continue
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
        elif "conv_weight" in weight.path:
            weight.assign(np.asarray(state[name]).squeeze(1))
        else:
            transfer_weights(weight.path, weight, state[name])
