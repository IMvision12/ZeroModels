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
    "mlp.gate_up": "mlp.gate_up_proj",
    "mlp.down": "mlp.down_proj",
    "kernel": "weight",
}


def transfer_glm4v_weights(keras_model, hf_state_dict):
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
        if name not in hf_state_dict:
            raise WeightMappingError(weight.path, name)
        value = hf_state_dict[name]
        if name.endswith("patch_embed.proj.weight"):
            # Conv3d (embed, in*t*p*p) -> Dense kernel (in*t*p*p, embed).
            arr = np.asarray(value)
            weight.assign(arr.reshape(arr.shape[0], -1).T)
        elif name.endswith("downsample.weight"):
            # Conv2d (out, in, kh, kw) -> keras channels_last (kh, kw, in, out).
            weight.assign(np.transpose(np.asarray(value), (2, 3, 1, 0)))
        else:
            transfer_weights(weight.path, weight, value)
