import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights

WEIGHT_NAME_MAPPING = {
    "token_embedding.embeddings": "model.embed_tokens.weight",
    "language_model.final_norm.weight": "model.norm.weight",
    "language_model.": "model.",
    "decoder_layer_": "layers.",
    "attention.query_norm": "self_attn.q_norm",
    "attention.key_norm": "self_attn.k_norm",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "attention_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    "mlp.gate": "mlp.gate_proj",
    "mlp.up": "mlp.up_proj",
    "mlp.down": "mlp.down_proj",
    "deepstack_merger_": "deepstack_merger_list.",
    "visual.pos_embed": "visual.pos_embed.weight",
    "blocks_": "blocks.",
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
}


def transfer_qwen3_vl_weights(keras_model, hf_state_dict):
    """Load an HF Qwen3-VL state dict into a (freshly built) Keras model in place.

    Like Qwen2-VL but for the Qwen3 text decoder (QK-norm) plus the Qwen3-VL
    vision extras: the learned ``pos_embed`` is assigned directly and the Conv3d
    patch embed is reshaped to feed the Keras ``Dense``.
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

    state = {}
    for k, v in hf_state_dict.items():
        if k.startswith("model.visual."):
            k = k[len("model.") :]
        elif k.startswith("model.language_model."):
            k = "model." + k[len("model.language_model.") :]
        state[k] = v

    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        # Functional model weight paths are flat (no model-name root to strip).
        name = weight.path.replace("/", ".")
        for old, new in WEIGHT_NAME_MAPPING.items():
            name = name.replace(old, new)
        if name not in state:
            raise WeightMappingError(weight.path, name)
        torch_weight = state[name]
        if weight.path.endswith("pos_embed"):
            weight.assign(np.asarray(torch_weight))
        elif "patch_embed" in weight.path and weight.path.endswith("kernel"):
            tw = np.asarray(torch_weight)
            transfer_weights(weight.path, weight, tw.reshape(tw.shape[0], -1))
        else:
            transfer_weights(weight.path, weight, torch_weight)
