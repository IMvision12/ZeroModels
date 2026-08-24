import re

import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

# The text decoder and lm_head (keras paths starting "language_model." /
# "lm_head" / the bare "token_embedding", which builds under the outer model's
# name scope because the fusion path calls it directly). Backbone-parametric:
# the QK-norm rules cover Qwen3 / Qwen3-MoE towers, and the MoE router / dense
# SwiGLU rules are disjoint so one map serves qwen2, qwen3 and qwen3_moe.
TEXT_MAPPING = {
    "token_embedding.embeddings": "language_model.embed_tokens.weight",
    "language_model.final_norm.weight": "language_model.norm.weight",
    "decoder_layer_": "layers.",
    # Qwen3 / Qwen3-MoE per-head QK norms: specific, before the q/k proj rules
    # so "attention.query" cannot clip "attention.query_norm".
    "attention.query_norm": "self_attn.q_norm",
    "attention.key_norm": "self_attn.k_norm",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "attention_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    # MoE router (specific, before the dense gate rule).
    "mlp.gate_weight": "mlp.gate.weight",
    # Dense SwiGLU (explicit ".kernel" so it cannot corrupt the MoE router's
    # "mlp.gate.weight" or the fused expert banks).
    "mlp.gate.kernel": "mlp.gate_proj.weight",
    "mlp.up.kernel": "mlp.up_proj.weight",
    "mlp.down.kernel": "mlp.down_proj.weight",
    "kernel": "weight",
}

# The InternViT tower (keras paths starting "vision_tower.").
VISION_MAPPING = {
    "blocks_": "encoder.layer.",
    "embeddings.patch_embed": "embeddings.patch_embeddings.projection",
    "attention.query_norm": "attention.q_norm",
    "attention.key_norm": "attention.k_norm",
    "attention.query": "attention.q_proj",
    "attention.key": "attention.k_proj",
    "attention.value": "attention.v_proj",
    "attention.output_proj": "attention.projection_layer",
    "vision_norm": "layernorm",
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
}

# The pixel-shuffle projector (keras paths starting "multi_modal_projector.").
PROJECTOR_MAPPING = {
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
}


def normalize_keys(hf_state_dict):
    # Canonicalize both checkpoint layouts to bare "vision_tower.* /
    # language_model.* / multi_modal_projector.* / lm_head.weight":
    # new (transformers >= 5) prefixes everything except lm_head with
    # "model."; the hub checkpoints (4.52-era) nest the text decoder as
    # "language_model.model.*" and the head as "language_model.lm_head".
    out = {}
    for key, value in hf_state_dict.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        if key.startswith("language_model.model."):
            key = "language_model." + key[len("language_model.model.") :]
        elif key == "language_model.lm_head.weight":
            key = "lm_head.weight"
        out[key] = value
    return out


def fuse_expert_weights(state):
    """Fuse hub per-expert ``...mlp.experts.N.{gate,up,down}_proj`` (Qwen3-MoE text
    towers) into the fused ``mlp.experts.gate_up_proj`` (E, 2I, H) / ``down_proj``
    (E, H, I) banks the reused ``Qwen3MoeSparseBlock`` expects. In-memory state
    dicts that already carry the fused tensors pass through untouched. Operates on
    post-``normalize_keys`` names, so the layer prefix is ``language_model.layers.N``
    (the leading ``.*`` in the pattern also accepts the bare ``model.layers.N``).
    """
    pat = re.compile(
        r"^(.*layers\.\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
    )
    if not any(pat.match(k) for k in state):
        return state
    out = {}
    gate, up, down = {}, {}, {}
    for key, value in state.items():
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


def transfer_internvl_weights(keras_model, hf_state_dict):
    state = fuse_expert_weights(normalize_keys(hf_state_dict))
    if not keras_model.built or not keras_model.weights:
        size = keras_model.image_size
        n_tokens = int(
            (size // keras_model.patch_size * keras_model.downsample_ratio) ** 2
        )
        ids = np.array(
            [[0] + [keras_model.image_token_id] * n_tokens + [1]], dtype="int32"
        )
        keras_model(
            {
                "input_ids": ids,
                "attention_mask": np.ones_like(ids),
                "pixel_values": np.zeros((1, size, size, 3), dtype="float32"),
            }
        )
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        # Functional model weight paths are flat (no model-name root to strip); the
        # vision_tower / language_model sub-layer names lead the path and are kept.
        name = weight.path.replace("/", ".")
        if name.startswith("vision_tower."):
            mapping = VISION_MAPPING
        elif name.startswith("multi_modal_projector."):
            mapping = PROJECTOR_MAPPING
        else:
            mapping = TEXT_MAPPING
        for old, new in mapping.items():
            name = name.replace(old, new)
        if name not in state:
            raise WeightMappingError(weight.path, name)
        if name.endswith("patch_embeddings.projection.weight"):
            # Conv2D patch embed: HF (out, in, kh, kw) -> Keras (kh, kw, in, out).
            weight.assign(np.transpose(np.asarray(state[name]), (2, 3, 1, 0)))
        elif (
            ".experts.gate_up_proj" in name
            or ".experts.down_proj" in name
            or name.endswith("mlp.gate.weight")
        ):
            # Fused Qwen3-MoE expert banks + the (E, H) router: direct copy.
            weight.assign(np.asarray(state[name]))
        else:
            transfer_weights(weight.path, weight, state[name])
