import re
from typing import Dict

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from zeromodels.conversion.weight_split_util import split_model_weights
from zeromodels.conversion.weight_transfer_util import (
    compare_keras_torch_names,
    transfer_weights,
)

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "blocks.": "vision_encoder.blocks.",
    "text.model.encoder.layers.": "text_encoder.transformer.resblocks.",
    "text.model.token.embedding": "text_encoder.token_embedding",
    "text.model.final.layernorm": "text_encoder.ln_final",
    "layer.norm1": "ln_1",
    "layer.norm2": "ln_2",
    "self.attn.out.proj": "attn.out_proj",
    "mlp.fc1": "mlp.c_fc",
    "mlp.fc2": "mlp.c_proj",
    "conv1": "vision_encoder.patch_embed.proj",
    "final.layernorm": "vision_encoder.norm",
    "layernorm.1": "norm1",
    "layernorm.2": "norm2",
    "dense.1": "mlp.fc1",
    "dense.2": "mlp.fc2",
    # SwiGLU FFN leaves (DINOv2-giant / g14 backbone). The raw checkpoint keeps
    # the original DINOv2 names w12 (fused gate+value) / w3, not the HF-refactored
    # weights_in / weights_out.
    "weights.in": "mlp.w12",
    "weights.out": "mlp.w3",
    "layerscale.1": "ls1",
    "layerscale.2": "ls2",
    "cls.token.cls.token": "vision_encoder.cls_token",
    "register.tokens.register.tokens": "vision_encoder.register_tokens",
    "pos.embed.pos.embed": "vision_encoder.pos_embed",
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
    "embeddings": "weight",
}

QKV_SLICE = {"q": 0, "k": 1, "v": 2}


def to_numpy(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def apply_name_mapping(name: str) -> str:
    for old, new in WEIGHT_NAME_MAPPING.items():
        name = name.replace(old, new)
    return name


def interpolate_pos_embed(keras_weight, hf_pe):
    import torch

    target_num_patches = keras_weight.shape[1] - 1
    pe = (
        hf_pe
        if isinstance(hf_pe, torch.Tensor)
        else torch.from_numpy(np.asarray(hf_pe))
    )
    cls_pe, spatial_pe = pe[:, :1], pe[:, 1:]
    src = int(round(spatial_pe.shape[1] ** 0.5))
    tgt = int(round(target_num_patches**0.5))
    if src == tgt:
        keras_weight.assign(pe.numpy())
        return
    dim = spatial_pe.shape[-1]
    spatial_pe = spatial_pe.reshape(1, src, src, dim).permute(0, 3, 1, 2)
    spatial_pe = torch.nn.functional.interpolate(
        spatial_pe.float(),
        size=(tgt, tgt),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    spatial_pe = spatial_pe.permute(0, 2, 3, 1).reshape(1, tgt * tgt, dim)
    keras_weight.assign(torch.cat([cls_pe, spatial_pe], dim=1).numpy())


def transfer_tipsv2_weights(keras_model, hf_state_dict: Dict[str, np.ndarray]) -> None:
    trainable, non_trainable = split_model_weights(keras_model)
    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring TIPSv2 weights"
    ):
        path = keras_weight.path
        segment = path.split("/")[-2]
        var = path.split("/")[-1]

        if segment == "text_model_position_embedding":
            continue

        m = re.match(
            r"text_model_encoder_layers_(\d+)_self_attn_(q|k|v)_proj$", segment
        )
        if m:
            idx, which = m.group(1), m.group(2)
            suffix = "weight" if var == "kernel" else "bias"
            key = f"text_encoder.transformer.resblocks.{idx}.attn.in_proj_{suffix}"
            if key not in hf_state_dict:
                raise WeightMappingError(keras_weight_name, key)
            full = to_numpy(hf_state_dict[key])
            chunk = full.shape[0] // 3
            start = QKV_SLICE[which] * chunk
            piece = full[start : start + chunk]
            transfer_weights(keras_weight_name, keras_weight, piece)
            continue

        if segment.endswith(("layerscale_1", "layerscale_2")):
            hf_key = apply_name_mapping(segment) + ".gamma"
        else:
            hf_key = apply_name_mapping(f"{segment}_{var}")
        if hf_key not in hf_state_dict:
            raise WeightMappingError(keras_weight_name, hf_key)
        hf_weight = hf_state_dict[hf_key]

        if segment == "pos_embed":
            interpolate_pos_embed(keras_weight, hf_weight)
            continue

        if not compare_keras_torch_names(
            keras_weight_name, keras_weight, hf_key, hf_weight
        ):
            raise WeightShapeMismatchError(
                keras_weight_name, keras_weight.shape, hf_key, hf_weight.shape
            )
        transfer_weights(keras_weight_name, keras_weight, hf_weight)
