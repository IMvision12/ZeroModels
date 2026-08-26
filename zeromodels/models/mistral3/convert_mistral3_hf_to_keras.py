import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights

# The text decoder and lm_head (keras paths starting "language_model." /
# "lm_head" / the bare "token_embedding", which builds under the outer model's
# name scope because the fusion path calls it directly).
TEXT_MAPPING = {
    "token_embedding.embeddings": "language_model.embed_tokens.weight",
    "language_model.final_norm.weight": "language_model.norm.weight",
    "decoder_layer_": "layers.",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "attention_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    "mlp.gate": "mlp.gate_proj",
    "mlp.up": "mlp.up_proj",
    "mlp.down": "mlp.down_proj",
    "kernel": "weight",
}

# The Pixtral tower (keras paths starting "vision_tower.").
VISION_MAPPING = {
    "blocks_": "transformer.layers.",
    "attention.query": "attention.q_proj",
    "attention.key": "attention.k_proj",
    "attention.value": "attention.v_proj",
    "attention.output_proj": "attention.o_proj",
    "feed_forward.gate": "feed_forward.gate_proj",
    "feed_forward.up": "feed_forward.up_proj",
    "feed_forward.down": "feed_forward.down_proj",
    "kernel": "weight",
}

# The patch-merging projector (keras paths starting "multi_modal_projector.").
PROJECTOR_MAPPING = {
    "kernel": "weight",
}


def normalize_keys(hf_state_dict):
    # Canonicalize both layouts to bare "vision_tower.* / language_model.* /
    # multi_modal_projector.* / lm_head.weight": new (transformers >= 5)
    # prefixes everything except lm_head with "model."; the hub checkpoints
    # nest the text decoder as "language_model.model.*" and the head as
    # "language_model.lm_head".
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


def transfer_mistral3_weights(keras_model, hf_state_dict):
    state = normalize_keys(hf_state_dict)
    if not keras_model.built or not keras_model.weights:
        merged = keras_model.patch_size * keras_model.spatial_merge_size
        side = 2 * merged
        n_tokens = (side // merged) ** 2
        ids = np.array(
            [[0] + [keras_model.image_token_id] * n_tokens + [1]], dtype="int32"
        )
        keras_model(
            {
                "input_ids": ids,
                "attention_mask": np.ones_like(ids),
                "pixel_values": np.zeros((1, side, side, 3), dtype="float32"),
                "image_sizes": np.array([[side, side]], dtype="int32"),
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
        if name.endswith("vision_tower.patch_conv.weight"):
            # Conv2D patch embed: HF (out, in, kh, kw) -> Keras (kh, kw, in, out).
            weight.assign(np.transpose(np.asarray(state[name]), (2, 3, 1, 0)))
        else:
            transfer_weights(weight.path, weight, state[name])
