import re

import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

# Text decoder + projector + lm_head (keras paths "decoder_layer_*",
# "projector_*", the bare "token_embedding" / "final_norm", "lm_head").
TEXT_MAPPING = {
    "token_embedding.embeddings": "language_model.embed_tokens.weight",
    "final_norm.weight": "language_model.norm.weight",
    "decoder_layer_": "language_model.layers.",
    "attention.index_query_norm": "self_attn.indexer.q_norm",
    "attention.index_key_norm": "self_attn.indexer.k_norm",
    "attention.index_query": "self_attn.indexer.q_proj",
    "attention.index_key": "self_attn.indexer.k_proj",
    "attention.query_norm": "self_attn.q_norm",
    "attention.key_norm": "self_attn.k_norm",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
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
    "mlp.e_score_correction_bias": "mlp.gate.e_score_correction_bias",
    "projector_merge_linear_1": "multi_modal_projector.merge_linear_1",
    "projector_merge_linear_2": "multi_modal_projector.merge_linear_2",
    "projector_linear_1": "multi_modal_projector.linear_1",
    "projector_linear_2": "multi_modal_projector.linear_2",
    "kernel": "weight",
}

# The CLIP-style vision tower (keras paths "vision_tower.*").
VISION_MAPPING = {
    "vision_tower.patch_embed.kernel": "vision_tower.embeddings.proj.weight",
    "vision_tower.pre_norm": "vision_tower.pre_layrnorm",
    "vision_tower.blocks_": "vision_tower.layers.",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.out_proj",
    "fc1": "mlp.fc1",
    "fc2": "mlp.fc2",
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
}


def fuse_hub_experts(state):
    """Fuse the hub's per-expert ``block_sparse_moe.experts.N.w1/w2/w3`` into
    ``mlp.experts.gate_up_proj`` (E, 2I, H) / ``mlp.experts.down_proj``."""
    out = {}
    w1, w3, w2 = {}, {}, {}
    pat = re.compile(
        r"^(language_model\.layers\.\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight$"
    )
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


def normalize_keys(hf_state_dict):
    """Canonicalize both checkpoint layouts.

    The hub checkpoint uses the original MiniMax names
    (``language_model.model.*``, ``vision_tower.vision_model.*``,
    ``patch_merge_mlp.*``, ``block_sparse_moe.*`` with per-expert w1/w2/w3 and
    ``index_*`` attention leaves, plus skipped ``mtp.*`` heads); in-memory
    state dicts from current transformers use ``model.*`` prefixes, fused
    experts, and fused dense/shared ``gate_up_proj`` (split here, since the
    keras dense MLPs keep separate gate/up).
    """
    out = {}
    for key, value in hf_state_dict.items():
        if key.startswith("mtp.") or ".mtp." in key:
            continue
        if key.startswith("model."):
            key = key[len("model.") :]
        if key.startswith("language_model.lm_head."):
            key = key.replace("language_model.lm_head.", "lm_head.")
        elif key.startswith("language_model.model."):
            key = key.replace("language_model.model.", "language_model.")
        key = key.replace(
            "vision_tower.vision_model.embeddings.patch_embedding.",
            "vision_tower.embeddings.proj.",
        )
        key = key.replace(
            "vision_tower.vision_model.encoder.layers.", "vision_tower.layers."
        )
        key = key.replace(
            "vision_tower.vision_model.pre_layrnorm.", "vision_tower.pre_layrnorm."
        )
        key = key.replace(
            "patch_merge_mlp.linear_1.", "multi_modal_projector.merge_linear_1."
        )
        key = key.replace(
            "patch_merge_mlp.linear_2.", "multi_modal_projector.merge_linear_2."
        )
        key = key.replace(".block_sparse_moe.shared_experts.", ".mlp.shared_experts.")
        key = key.replace(".block_sparse_moe.gate.weight", ".mlp.gate.weight")
        key = key.replace(
            ".block_sparse_moe.e_score_correction_bias",
            ".mlp.gate.e_score_correction_bias",
        )
        key = key.replace(".block_sparse_moe.experts.", ".mlp.experts__hub__.")
        key = key.replace(".self_attn.index_q_proj.", ".self_attn.indexer.q_proj.")
        key = key.replace(".self_attn.index_k_proj.", ".self_attn.indexer.k_proj.")
        key = key.replace(".self_attn.index_q_norm.", ".self_attn.indexer.q_norm.")
        key = key.replace(".self_attn.index_k_norm.", ".self_attn.indexer.k_norm.")
        out[key] = value

    if any(".mlp.experts__hub__." in k for k in out):
        out = fuse_hub_experts(
            {
                k.replace(".mlp.experts__hub__.", ".block_sparse_moe.experts."): v
                for k, v in out.items()
            }
        )

    # Split fused dense / shared-expert gate_up (current-transformers layout).
    split = {}
    for key, value in out.items():
        if key.endswith("mlp.gate_up_proj.weight") or key.endswith(
            "shared_experts.gate_up_proj.weight"
        ):
            value = np.asarray(value)
            half = value.shape[0] // 2
            split[key.replace("gate_up_proj", "gate_proj")] = value[:half]
            split[key.replace("gate_up_proj", "up_proj")] = value[half:]
        else:
            split[key] = value
    return split


def transfer_minimax_m3_vl_weights(keras_model, hf_state_dict):
    state = normalize_keys(hf_state_dict)
    if not keras_model.built or not keras_model.weights:
        size = keras_model.patch_size * keras_model.spatial_merge_size
        grid = (1, keras_model.spatial_merge_size, keras_model.spatial_merge_size)
        n_tokens = 1
        patch_dim = 3 * keras_model.temporal_patch_size * keras_model.patch_size**2
        keras_model(
            {
                "input_ids": np.array(
                    [[1] + [keras_model.image_token_id] * n_tokens + [2]],
                    dtype="int64",
                ),
                "pixel_values": np.zeros(
                    (grid[1] * grid[2], patch_dim), dtype="float32"
                ),
                "image_grid_thw": np.asarray([grid], dtype="int64"),
            }
        )
        del size
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        # Functional model weight paths are flat (no model-name root to strip).
        name = weight.path.replace("/", ".")
        mapping = VISION_MAPPING if name.startswith("vision_tower.") else TEXT_MAPPING
        for old, new in mapping.items():
            name = name.replace(old, new)
        if name not in state:
            raise WeightMappingError(weight.path, name)
        value = state[name]
        if name == "vision_tower.embeddings.proj.weight":
            # Conv3d (E, C, T, P, P) -> Dense kernel (C*T*P*P, E).
            value = np.asarray(value)
            weight.assign(np.transpose(value.reshape(value.shape[0], -1)))
        elif (
            ".experts.gate_up_proj" in name
            or ".experts.down_proj" in name
            or name.endswith("mlp.gate.weight")
            or name.endswith("e_score_correction_bias")
        ):
            weight.assign(np.asarray(value))
        else:
            transfer_weights(weight.path, weight, value)
