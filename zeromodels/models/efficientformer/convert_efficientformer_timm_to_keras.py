import gc
import re
from typing import Dict

import keras
import numpy as np
from tqdm import tqdm

from zeromodels.conversion import verify_cls_model_equivalence
from zeromodels.conversion.exceptions import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from zeromodels.conversion.hf_download_utils import download_hf_state_dict
from zeromodels.conversion.weight_split_util import split_model_weights
from zeromodels.conversion.weight_transfer_util import (
    compare_keras_torch_names,
    transfer_weights,
)
from zeromodels.models.efficientformer import EfficientFormerImageClassify

# Architecture presets, moved here from efficientformer_config.py: the package config no
# longer carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer timm weights into.
EFFICIENTFORMER_MODEL_CONFIG = {
    "efficientformer_l1": {
        "depths": [3, 2, 6, 4],
        "embed_dim": [48, 96, 224, 448],
        "num_vit": 1,
        "image_size": 224,
        "num_classes": 1000,
    },
    "efficientformer_l3": {
        "depths": [4, 4, 12, 6],
        "embed_dim": [64, 128, 320, 512],
        "num_vit": 4,
        "image_size": 224,
        "num_classes": 1000,
    },
    "efficientformer_l7": {
        "depths": [6, 6, 18, 8],
        "embed_dim": [96, 192, 384, 768],
        "num_vit": 8,
        "image_size": 224,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (zm_config.json); the github release urls have been removed.
EFFICIENTFORMER_VARIANTS = {
    "efficientformer_l1_snap_dist_in1k": {
        "model": "efficientformer_l1",
        "timm_id": "efficientformer_l1.snap_dist_in1k",
    },
    "efficientformer_l3_snap_dist_in1k": {
        "model": "efficientformer_l3",
        "timm_id": "efficientformer_l3.snap_dist_in1k",
    },
    "efficientformer_l7_snap_dist_in1k": {
        "model": "efficientformer_l7",
        "timm_id": "efficientformer_l7.snap_dist_in1k",
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "stem.conv1": "stem.conv1",
    "stem.norm1": "stem.norm1",
    "stem.conv2": "stem.conv2",
    "stem.norm2": "stem.norm2",
    "downsample.conv": "downsample.conv",
    "downsample.norm": "downsample.norm",
    "pool.pool": "token_mixer.pool",
    "mlp.conv.1": "mlp.fc1",
    "mlp.norm.1": "mlp.norm1",
    "mlp.conv.2": "mlp.fc2",
    "mlp.norm.2": "mlp.norm2",
    "mlp.dense.1": "mlp.fc1",
    "mlp.dense.2": "mlp.fc2",
    "attn.qkv": "token_mixer.qkv",
    "attn.proj": "token_mixer.proj",
    "norm1": "norm1",
    "norm2": "norm2",
    "final.norm": "norm",
    "kernel": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "head.": "head.",
    "head.dist.": "head_dist.",
}


def build_block_index_remap(last_stage_depth: int, num_vit: int) -> Dict[str, str]:
    if num_vit == 0:
        return {}
    first_vit_keras = max(0, last_stage_depth - num_vit)
    remap: Dict[str, str] = {}
    for keras_idx in range(last_stage_depth - 1, first_vit_keras - 1, -1):
        remap[f"stages.3.blocks.{keras_idx}"] = f"stages.3.blocks.{keras_idx + 1}"
    return remap


def transfer_efficientformer_weights(
    keras_model,
    state_dict: Dict[str, np.ndarray],
) -> None:
    block_remap = build_block_index_remap(
        last_stage_depth=keras_model.depths[-1],
        num_vit=keras_model.num_vit,
    )
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_weight_name = keras_weight_name
        torch_weight_name = re.sub(r"_variable(_\d+)?$", "_gamma", torch_weight_name)

        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if ".gamma" in torch_weight_name and ".ls" not in torch_weight_name:
            torch_weight_name = torch_weight_name.replace(".gamma", ".weight")

        for keras_block, torch_block in block_remap.items():
            if keras_block in torch_weight_name:
                torch_weight_name = torch_weight_name.replace(keras_block, torch_block)
                break

        if "attn.attention.biases" in torch_weight_name:
            torch_weight_name = torch_weight_name.replace(
                ".attn.attention.biases", ".token_mixer.attention_biases"
            )

        if "attention_bias_idxs" in torch_weight_name:
            continue

        if torch_weight_name not in state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = state_dict[torch_weight_name]

        if "attention_biases" in keras_weight_name:
            keras_weight.assign(
                torch_weight.numpy() if hasattr(torch_weight, "numpy") else torch_weight
            )
            continue

        if not compare_keras_torch_names(
            keras_weight_name, keras_weight, torch_weight_name, torch_weight
        ):
            raise WeightShapeMismatchError(
                keras_weight_name,
                keras_weight.shape,
                torch_weight_name,
                torch_weight.shape,
            )

        transfer_weights(keras_weight_name, keras_weight, torch_weight)


if __name__ == "__main__":
    import timm

    for variant, meta in EFFICIENTFORMER_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = EfficientFormerImageClassify(
            **EFFICIENTFORMER_MODEL_CONFIG[meta["model"]]
        )
        transfer_efficientformer_weights(keras_model, state)

        torch_model = timm.create_model(timm_id, pretrained=True).eval()
        results = verify_cls_model_equivalence(
            model_a=torch_model,
            model_b=keras_model,
            input_shape=keras_model.input_shape[1:],
            output_specs={"num_classes": keras_model.output_shape[-1]},
            comparison_type="torch_to_keras",
            run_performance=False,
            atol=1e-4,
            rtol=1e-4,
        )
        if not results["standard_input"]:
            raise ValueError(
                "Model equivalence test failed - model outputs do not match for standard input"
            )

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state, torch_model
        keras.backend.clear_session()
        gc.collect()
