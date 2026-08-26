from typing import Dict

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "blocks.": "vision_encoder.blocks.",
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
    "depth.head.linear": "depth_head.depth_head",
    "segmentation.head.linear": "segmentation_head.segmentation_head",
    "depth.": "depth_head.",
    "segmentation.": "segmentation_head.",
    "fusion.": "fusion_blocks.",
    "main.conv": "main_unit.conv",
    "residual.conv": "residual_unit.conv",
    "out.conv": "out_conv",
    "reassemble.readout.": "reassemble.readout_projects.",
    "reassemble.proj.": "reassemble.out_projections.",
    "reassemble.resize.": "reassemble.resize_layers.",
    "head.conv.": "head.convs.",
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
}


def to_numpy(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def apply_name_mapping(name: str) -> str:
    for old, new in WEIGHT_NAME_MAPPING.items():
        name = name.replace(old, new)
    return name


def assign_weight(keras_weight, torch_weight):
    tw = to_numpy(torch_weight)
    n = len(keras_weight.shape)
    if n == 4:
        tw = np.transpose(tw, (2, 3, 1, 0))
    elif n == 2:
        tw = np.transpose(tw, (1, 0))
    keras_weight.assign(tw)


def transfer_tipsv2_dpt_weights(
    keras_model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    for keras_weight in tqdm(
        keras_model.weights, desc="Transferring TIPSv2-DPT weights"
    ):
        path = keras_weight.path
        segment = path.split("/")[-2]
        var = path.split("/")[-1]

        if segment == "depth_bin_regressor":
            continue

        if segment == "conv1":
            suffix = "weight" if var == "kernel" else "bias"
            hf_key = f"vision_encoder.patch_embed.proj.{suffix}"
        elif segment.endswith(("layerscale_1", "layerscale_2")):
            hf_key = apply_name_mapping(segment) + ".gamma"
        else:
            hf_key = apply_name_mapping(f"{segment}_{var}")

        if hf_key not in hf_state_dict:
            raise WeightMappingError(path, hf_key)
        assign_weight(keras_weight, hf_state_dict[hf_key])
