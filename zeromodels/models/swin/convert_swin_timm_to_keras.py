import gc
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
    transfer_attention_weights,
    transfer_weights,
)
from zeromodels.models.swin import SwinImageClassify

# Architecture presets, moved here from swin_config.py: the package config no
# longer carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer timm weights into.
SWIN_MODEL_CONFIG = {
    "swin_tiny": {
        "window_size": 7,
        "embed_dim": 96,
        "depths": (2, 2, 6, 2),
        "num_heads": (3, 6, 12, 24),
        "pretrain_size": 224,
        "image_size": 224,
        "num_classes": 1000,
    },
    "swin_tiny_in22k": {
        "window_size": 7,
        "embed_dim": 96,
        "depths": (2, 2, 6, 2),
        "num_heads": (3, 6, 12, 24),
        "pretrain_size": 224,
        "image_size": 224,
        "num_classes": 21841,
    },
    "swin_small": {
        "window_size": 7,
        "embed_dim": 96,
        "depths": (2, 2, 18, 2),
        "num_heads": (3, 6, 12, 24),
        "pretrain_size": 224,
        "image_size": 224,
        "num_classes": 1000,
    },
    "swin_small_in22k": {
        "window_size": 7,
        "embed_dim": 96,
        "depths": (2, 2, 18, 2),
        "num_heads": (3, 6, 12, 24),
        "pretrain_size": 224,
        "image_size": 224,
        "num_classes": 21841,
    },
    "swin_base_224": {
        "window_size": 7,
        "embed_dim": 128,
        "depths": (2, 2, 18, 2),
        "num_heads": (4, 8, 16, 32),
        "pretrain_size": 224,
        "image_size": 224,
        "num_classes": 1000,
    },
    "swin_base_224_in22k": {
        "window_size": 7,
        "embed_dim": 128,
        "depths": (2, 2, 18, 2),
        "num_heads": (4, 8, 16, 32),
        "pretrain_size": 224,
        "image_size": 224,
        "num_classes": 21841,
    },
    "swin_base_384": {
        "window_size": 12,
        "embed_dim": 128,
        "depths": (2, 2, 18, 2),
        "num_heads": (4, 8, 16, 32),
        "pretrain_size": 384,
        "image_size": 384,
        "num_classes": 1000,
    },
    "swin_base_384_in22k": {
        "window_size": 12,
        "embed_dim": 128,
        "depths": (2, 2, 18, 2),
        "num_heads": (4, 8, 16, 32),
        "pretrain_size": 384,
        "image_size": 384,
        "num_classes": 21841,
    },
    "swin_large_224": {
        "window_size": 7,
        "embed_dim": 192,
        "depths": (2, 2, 18, 2),
        "num_heads": (6, 12, 24, 48),
        "pretrain_size": 224,
        "image_size": 224,
        "num_classes": 1000,
    },
    "swin_large_224_in22k": {
        "window_size": 7,
        "embed_dim": 192,
        "depths": (2, 2, 18, 2),
        "num_heads": (6, 12, 24, 48),
        "pretrain_size": 224,
        "image_size": 224,
        "num_classes": 21841,
    },
    "swin_large_384": {
        "window_size": 12,
        "embed_dim": 192,
        "depths": (2, 2, 18, 2),
        "num_heads": (6, 12, 24, 48),
        "pretrain_size": 384,
        "image_size": 384,
        "num_classes": 1000,
    },
    "swin_large_384_in22k": {
        "window_size": 12,
        "embed_dim": 192,
        "depths": (2, 2, 18, 2),
        "num_heads": (6, 12, 24, 48),
        "pretrain_size": 384,
        "image_size": 384,
        "num_classes": 21841,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (zm_config.json); the github release urls have been removed.
SWIN_VARIANTS = {
    "swin_tiny_patch4_window7_224_ms_in1k": {
        "model": "swin_tiny",
        "timm_id": "swin_tiny_patch4_window7_224.ms_in1k",
    },
    "swin_tiny_patch4_window7_224_ms_in22k": {
        "model": "swin_tiny_in22k",
        "timm_id": "swin_tiny_patch4_window7_224.ms_in22k",
    },
    "swin_small_patch4_window7_224_ms_in1k": {
        "model": "swin_small",
        "timm_id": "swin_small_patch4_window7_224.ms_in1k",
    },
    "swin_small_patch4_window7_224_ms_in22k": {
        "model": "swin_small_in22k",
        "timm_id": "swin_small_patch4_window7_224.ms_in22k",
    },
    "swin_small_patch4_window7_224_ms_in22k_ft_in1k": {
        "model": "swin_small",
        "timm_id": "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
    },
    "swin_base_patch4_window7_224_ms_in1k": {
        "model": "swin_base_224",
        "timm_id": "swin_base_patch4_window7_224.ms_in1k",
    },
    "swin_base_patch4_window7_224_ms_in22k": {
        "model": "swin_base_224_in22k",
        "timm_id": "swin_base_patch4_window7_224.ms_in22k",
    },
    "swin_base_patch4_window7_224_ms_in22k_ft_in1k": {
        "model": "swin_base_224",
        "timm_id": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    },
    "swin_base_patch4_window12_384_ms_in1k": {
        "model": "swin_base_384",
        "timm_id": "swin_base_patch4_window12_384.ms_in1k",
    },
    "swin_base_patch4_window12_384_ms_in22k": {
        "model": "swin_base_384_in22k",
        "timm_id": "swin_base_patch4_window12_384.ms_in22k",
    },
    "swin_base_patch4_window12_384_ms_in22k_ft_in1k": {
        "model": "swin_base_384",
        "timm_id": "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
    },
    "swin_large_patch4_window7_224_ms_in22k": {
        "model": "swin_large_224_in22k",
        "timm_id": "swin_large_patch4_window7_224.ms_in22k",
    },
    "swin_large_patch4_window7_224_ms_in22k_ft_in1k": {
        "model": "swin_large_224",
        "timm_id": "swin_large_patch4_window7_224.ms_in22k_ft_in1k",
    },
    "swin_large_patch4_window12_384_ms_in22k": {
        "model": "swin_large_384_in22k",
        "timm_id": "swin_large_patch4_window12_384.ms_in22k",
    },
    "swin_large_patch4_window12_384_ms_in22k_ft_in1k": {
        "model": "swin_large_384",
        "timm_id": "swin_large_patch4_window12_384.ms_in22k_ft_in1k",
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "stem.conv": "patch_embed.proj",
    "stem.norm": "patch_embed.norm",
    "layernorm.1": "norm1",
    "layernorm.2": "norm2",
    "dense.1": "fc1",
    "dense.2": "fc2",
    "pm.layernorm": "norm",
    "pm.dense": "reduction",
    "final.norm": "norm",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "head.fc",
}


def transfer_swin_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if "relative.position.bias.table" in torch_weight_name:
            layer_name = keras_weight.path.split("/")[-1]
            layer_name = layer_name.replace("_", ".").replace(
                "relative.position.bias.table", "relative_position_bias_table"
            )
            keras_weight.assign(state_dict[layer_name])
            continue

        if "window.attention" in torch_weight_name:
            transfer_attention_weights(keras_weight_name, keras_weight, state_dict)
            continue

        if torch_weight_name not in state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = state_dict[torch_weight_name]
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

    for variant, meta in SWIN_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = SwinImageClassify(
            **SWIN_MODEL_CONFIG[meta["model"]], include_normalization=False
        )
        transfer_swin_weights(keras_model, state)

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
