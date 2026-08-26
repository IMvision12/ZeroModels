import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights

WEIGHT_NAME_MAPPING = {
    "token_embedding.embeddings": "model.embed_tokens.weight",
    "final_norm.weight": "model.norm.weight",
    "decoder_layer_": "model.layers.",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "attention_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    "shared_expert.gate": "shared_expert.gate_proj",
    "shared_expert.up": "shared_expert.up_proj",
    "shared_expert.down": "shared_expert.down_proj",
    "feed_forward.gate": "feed_forward.gate_proj",
    "feed_forward.up": "feed_forward.up_proj",
    "feed_forward.down": "feed_forward.down_proj",
    "kernel": "weight",
}


def strip_language_model_prefix(hf_state_dict):
    # Llama4ForConditionalGeneration checkpoints carry the text decoder under
    # "language_model."; text-only state dicts don't. Normalize to the bare
    # "model." / "lm_head." keys the mapping targets.
    prefix = "language_model."
    if not any(key.startswith(prefix) for key in hf_state_dict):
        return hf_state_dict
    return {
        (key[len(prefix) :] if key.startswith(prefix) else key): value
        for key, value in hf_state_dict.items()
    }


def transfer_llama4_weights(keras_model, hf_state_dict):
    hf_state_dict = strip_language_model_prefix(hf_state_dict)
    if not keras_model.built or not keras_model.weights:
        ids = np.array([[0, 1, 2, 3]], dtype="int32")
        keras_model({"input_ids": ids, "attention_mask": np.ones_like(ids)})
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        # Functional model weight paths are flat (no model-name root to strip).
        name = weight.path.replace("/", ".")
        for old, new in WEIGHT_NAME_MAPPING.items():
            name = name.replace(old, new)
        if name not in hf_state_dict:
            raise WeightMappingError(weight.path, name)
        if ".experts.gate_up_proj" in name or ".experts.down_proj" in name:
            # Fused per-expert banks, stored (E, H, 2I) / (E, I, H): direct
            # copy, no Dense kernel transpose.
            weight.assign(np.asarray(hf_state_dict[name]))
        else:
            transfer_weights(weight.path, weight, hf_state_dict[name])
