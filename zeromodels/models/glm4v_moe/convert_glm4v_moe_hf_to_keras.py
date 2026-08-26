import re

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights

VISION_MAPPING = {
    "visual.patch_embed.proj": "model.visual.patch_embed.proj",
    "visual.embeddings.position_embedding.embeddings": (
        "model.visual.embeddings.position_embedding.weight"
    ),
    "visual.post_conv_layernorm": "model.visual.post_conv_layernorm",
    "visual.post_layernorm": "model.visual.post_layernorm",
    "visual.downsample": "model.visual.downsample",
    "visual.merger.proj": "model.visual.merger.proj",
    "visual.merger.post_projection_norm": "model.visual.merger.post_projection_norm",
    "visual.merger.gate": "model.visual.merger.gate_proj",
    "visual.merger.up": "model.visual.merger.up_proj",
    "visual.merger.down": "model.visual.merger.down_proj",
    "visual.blocks_": "model.visual.blocks.",
    "attn.qkv": "attn.qkv",
    "attn.proj": "attn.proj",
    "mlp.gate": "mlp.gate_proj",
    "mlp.up": "mlp.up_proj",
    "mlp.down": "mlp.down_proj",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}

TEXT_MAPPING = {
    "token_embedding.embeddings": "model.language_model.embed_tokens.weight",
    "language_model.final_norm.weight": "model.language_model.norm.weight",
    "language_model.decoder_layer_": "model.language_model.layers.",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "mlp.shared_experts.gate": "mlp.shared_experts.gate_proj",
    "mlp.shared_experts.up": "mlp.shared_experts.up_proj",
    "mlp.shared_experts.down": "mlp.shared_experts.down_proj",
    "mlp.gate.kernel": "mlp.gate_proj.weight",
    "mlp.up.kernel": "mlp.up_proj.weight",
    "mlp.down.kernel": "mlp.down_proj.weight",
    "mlp.gate_weight": "mlp.gate.weight",
    "mlp.e_score_correction_bias": "mlp.gate.e_score_correction_bias",
    "kernel": "weight",
}

LAYER_PREFIX = "model.language_model.layers"


def dequantize_fp8(hf_state_dict):
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


def drop_mtp_keys(hf_state_dict, num_layers):
    out = {}
    for key, value in hf_state_dict.items():
        match = re.match(rf"^{re.escape(LAYER_PREFIX)}\.(\d+)\.", key)
        if match and int(match.group(1)) >= num_layers:
            continue
        if re.search(r"\.(eh_proj|enorm|hnorm|shared_head)\.", key):
            continue
        out[key] = value
    return out


def fuse_expert_weights(hf_state_dict):
    pat = re.compile(
        rf"^({re.escape(LAYER_PREFIX)}\.\d+)\.mlp\.experts\.(\d+)\."
        r"(gate_proj|up_proj|down_proj)\.weight$"
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


def transfer_glm4v_moe_weights(keras_model, hf_state_dict):
    state = fuse_expert_weights(
        dequantize_fp8(drop_mtp_keys(hf_state_dict, keras_model.num_layers))
    )
    if not keras_model.built or not keras_model.weights:
        size = keras_model.image_size
        orig = size // keras_model.patch_size
        n_tok = (orig // keras_model.spatial_merge_size) ** 2
        patch_dim = (
            keras_model.in_channels
            * keras_model.temporal_patch_size
            * keras_model.patch_size
            * keras_model.patch_size
        )
        keras_model(
            {
                "input_ids": np.array(
                    [[0] + [keras_model.image_token_id] * n_tok + [1]], dtype="int64"
                ),
                "pixel_values": np.zeros((orig * orig, patch_dim), dtype="float32"),
                "image_grid_thw": np.array([[1, orig, orig]], dtype="int64"),
            }
        )
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        # Functional model weight paths are flat (no model-name root to strip).
        name = weight.path.replace("/", ".")
        if name.startswith("visual."):
            mapping = VISION_MAPPING
        elif name.startswith("language_model.") or name.startswith("token_embedding."):
            mapping = TEXT_MAPPING
        else:
            mapping = {"kernel": "weight"}
        for old, new in mapping.items():
            name = name.replace(old, new)
        if name not in state:
            raise WeightMappingError(weight.path, name)
        value = state[name]
        if name.endswith("patch_embed.proj.weight"):
            arr = np.asarray(value)
            weight.assign(arr.reshape(arr.shape[0], -1).T)
        elif name.endswith("downsample.weight"):
            weight.assign(np.transpose(np.asarray(value), (2, 3, 1, 0)))
        elif (
            ".experts.gate_up_proj" in name
            or ".experts.down_proj" in name
            or name.endswith("mlp.gate.weight")
            or name.endswith("e_score_correction_bias")
        ):
            weight.assign(np.asarray(value))
        else:
            transfer_weights(weight.path, weight, value)
