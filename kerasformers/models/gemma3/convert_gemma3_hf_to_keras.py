import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

TEXT_MAPPING = {
    "token_embedding.embeddings": "language_model.embed_tokens.weight",
    "final_norm.weight": "language_model.norm.weight",
    "decoder_layer_": "language_model.layers.",
    "attention.query_norm": "self_attn.q_norm",
    "attention.key_norm": "self_attn.k_norm",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "post_attention_norm": "post_attention_layernorm",
    "pre_feedforward_norm": "pre_feedforward_layernorm",
    "post_feedforward_norm": "post_feedforward_layernorm",
    "attention_norm": "input_layernorm",
    "mlp.gate": "mlp.gate_proj",
    "mlp.up": "mlp.up_proj",
    "mlp.down": "mlp.down_proj",
    "kernel": "weight",
}

VISION_MAPPING = {
    "vision_tower.patch_embed": "vision_tower.vision_model.embeddings.patch_embedding",
    "vision_tower.position_embedding.embeddings": (
        "vision_tower.vision_model.embeddings.position_embedding.weight"
    ),
    "vision_tower.blocks_": "vision_tower.vision_model.encoder.layers.",
    "vision_tower.post_layernorm": "vision_tower.vision_model.post_layernorm",
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

PROJECTOR_MAPPING = {
    "kernel": "weight",
}


def normalize_keys(hf_state_dict):
    keys = list(hf_state_dict.keys())
    has_lm = any("language_model." in key for key in keys)
    out = {}
    for key, value in hf_state_dict.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        if not has_lm:
            if not key.startswith(("lm_head.",)):
                key = "language_model." + key
        elif key.startswith("language_model.model."):
            key = "language_model." + key[len("language_model.model.") :]
        out[key] = value
    return out


def transfer_gemma3_weights(keras_model, hf_state_dict):
    state = normalize_keys(hf_state_dict)
    if not keras_model.built or not keras_model.weights:
        feed = {"input_ids": np.array([[0, 1, 2, 3]], dtype="int64")}
        if keras_model.vision_tower is not None:
            n = keras_model.mm_tokens_per_image
            feed = {
                "input_ids": np.array(
                    [[0] + [keras_model.image_token_id] * n + [1]], dtype="int64"
                ),
                "pixel_values": np.zeros(
                    (1, keras_model.image_size, keras_model.image_size, 3),
                    dtype="float32",
                ),
            }
        keras_model(feed)
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        # Functional model weight paths are flat (no model-name root to strip).
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
        if name.endswith("patch_embedding.weight"):
            weight.assign(np.transpose(np.asarray(state[name]), (2, 3, 1, 0)))
        elif name.endswith("mm_input_projection_weight"):
            weight.assign(np.asarray(state[name]))
        else:
            transfer_weights(weight.path, weight, state[name])
