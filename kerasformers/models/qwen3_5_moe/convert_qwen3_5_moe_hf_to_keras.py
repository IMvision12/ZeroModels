import re

import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

# Keras weight-path substring -> HF name. Text keys are normalized so
# ``model.language_model.*`` becomes ``model.*`` (see the key rewrite below); vision
# keys keep ``visual.*``. Order matters: the specific MoE / vision rules run before the
# generic ``kernel``/``gamma``/``beta`` fallbacks.
WEIGHT_NAME_MAPPING = {
    "token_embedding.embeddings": "model.embed_tokens.weight",
    "language_model.final_norm.weight": "model.norm.weight",
    "language_model.": "model.",
    "decoder_layer_": "layers.",
    # Full-attention (gated GQA + QK-norm).
    "attention.query_norm": "self_attn.q_norm",
    "attention.key_norm": "self_attn.k_norm",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "attention_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    # MoE: shared expert + router (specific rules before any generic fallback).
    "mlp.shared_expert.gate": "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up": "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down": "mlp.shared_expert.down_proj",
    "mlp.shared_expert_gate.kernel": "mlp.shared_expert_gate.weight",
    "mlp.gate_weight": "mlp.gate.weight",
    "linear_attn.conv_weight": "linear_attn.conv1d.weight",
    # Vision tower.
    "visual.pos_embed": "visual.pos_embed.weight",
    "blocks_": "blocks.",
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
}

EXPERT_RE = re.compile(
    r"^(model\.layers\.\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


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


def fuse_expert_weights(state):
    """Fuse per-expert ``mlp.experts.N.{gate,up,down}_proj`` into the fused
    ``mlp.experts.gate_up_proj`` (E, 2I, H) / ``down_proj`` (E, H, I)."""
    if not any(EXPERT_RE.match(k) for k in state):
        return state
    out, gate, up, down = {}, {}, {}, {}
    for key, value in state.items():
        match = EXPERT_RE.match(key)
        if match:
            layer, expert, which = match.group(1), int(match.group(2)), match.group(3)
            {"gate_proj": gate, "up_proj": up, "down_proj": down}[which].setdefault(
                layer, {}
            )[expert] = value
        else:
            out[key] = value
    for layer in gate:
        experts = sorted(gate[layer])
        out[f"{layer}.mlp.experts.gate_up_proj"] = np.stack(
            [
                np.concatenate(
                    [np.asarray(gate[layer][e]), np.asarray(up[layer][e])], axis=0
                )
                for e in experts
            ],
            axis=0,
        )
        out[f"{layer}.mlp.experts.down_proj"] = np.stack(
            [np.asarray(down[layer][e]) for e in experts], axis=0
        )
    return out


def transfer_qwen3_5_moe_weights(keras_model, hf_state_dict):
    """Load an HF Qwen3.5-MoE (``Qwen3_5MoeForConditionalGeneration``) state dict into
    a freshly built Keras model in place.

    Combines the Qwen3-Next MoE text rules (fuse per-expert experts, split the fused
    Gated-DeltaNet ``in_proj_qkvz`` / ``in_proj_ba``, squeeze the conv1d weight) with
    the Qwen3-VL vision rules (learned ``pos_embed`` assigned directly, Conv3d patch
    embed reshaped for the Keras ``Dense``). HF ``model.language_model.*`` keys are
    rewritten to ``model.*`` and ``model.visual.*`` to ``visual.*`` first.
    """
    if not keras_model.built or not keras_model.weights:
        m = keras_model.spatial_merge_size
        h = w = 2 * m
        n_merged = (h * w) // (m * m)
        keras_model(
            {
                "input_ids": np.array(
                    [[0] + [keras_model.image_token_id] * n_merged + [1]], dtype="int64"
                ),
                "pixel_values": np.zeros(
                    (h * w, keras_model.patch_dim), dtype="float32"
                ),
                "image_grid_thw": np.array([[1, h, w]], dtype=np.int64),
            }
        )

    # Normalize the VLM key namespaces, then fuse the per-expert MoE weights.
    state = {}
    for k, v in hf_state_dict.items():
        if k.startswith("model.visual."):
            k = k[len("model.") :]
        elif k.startswith("model.language_model."):
            k = "model." + k[len("model.language_model.") :]
        state[k] = v
    state = fuse_expert_weights(state)

    # Gated-DeltaNet: split the fused HF in_proj into the 4 Keras projections.
    handled = set()
    for i, decoder_layer in enumerate(keras_model.language_model.decoder_layers):
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
            weight.assign(np.asarray(state[name]))  # fused / router: direct copy
        elif "conv_weight" in weight.path:
            weight.assign(np.asarray(state[name]).squeeze(1))  # (D,1,K) -> (D,K)
        elif weight.path.endswith("pos_embed"):
            weight.assign(np.asarray(state[name]))
        elif "patch_embed" in weight.path and weight.path.endswith("kernel"):
            tw = np.asarray(state[name])
            transfer_weights(weight.path, weight, tw.reshape(tw.shape[0], -1))
        else:
            transfer_weights(weight.path, weight, state[name])
