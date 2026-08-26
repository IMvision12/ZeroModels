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
from zeromodels.models.efficientnetv2 import EfficientNetV2ImageClassify

# Architecture presets, moved here from efficientnetv2_config.py: the package config no
# longer carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer timm weights into.
EFFICIENTNETV2_MODEL_CONFIG = {
    "efficientnetv2_s": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.0,
        "default_size": 300,
        "block_arch": "EfficientNetV2S",
        "head_filters": 1280,
        "image_size": 300,
        "num_classes": 1000,
    },
    "efficientnetv2_s_in21k": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.0,
        "default_size": 300,
        "block_arch": "EfficientNetV2S",
        "head_filters": 1280,
        "image_size": 300,
        "num_classes": 21843,
    },
    "efficientnetv2_m": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.0,
        "default_size": 384,
        "block_arch": "EfficientNetV2M",
        "head_filters": 1280,
        "image_size": 384,
        "num_classes": 1000,
    },
    "efficientnetv2_m_in21k": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.0,
        "default_size": 384,
        "block_arch": "EfficientNetV2M",
        "head_filters": 1280,
        "image_size": 384,
        "num_classes": 21843,
    },
    "efficientnetv2_l": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.0,
        "default_size": 384,
        "block_arch": "EfficientNetV2L",
        "head_filters": 1280,
        "image_size": 384,
        "num_classes": 1000,
    },
    "efficientnetv2_l_in21k": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.0,
        "default_size": 384,
        "block_arch": "EfficientNetV2L",
        "head_filters": 1280,
        "image_size": 384,
        "num_classes": 21843,
    },
    "efficientnetv2_xl": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.0,
        "default_size": 384,
        "block_arch": "EfficientNetV2XL",
        "head_filters": 1280,
        "image_size": 384,
        "num_classes": 1000,
    },
    "efficientnetv2_xl_in21k": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.0,
        "default_size": 384,
        "block_arch": "EfficientNetV2XL",
        "head_filters": 1280,
        "image_size": 384,
        "num_classes": 21843,
    },
    "efficientnetv2_b0": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.0,
        "default_size": 192,
        "block_arch": "EfficientNetV2B",
        "head_filters": 1280,
        "image_size": 192,
        "num_classes": 1000,
    },
    "efficientnetv2_b1": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.1,
        "default_size": 192,
        "block_arch": "EfficientNetV2B",
        "head_filters": 1280,
        "image_size": 192,
        "num_classes": 1000,
    },
    "efficientnetv2_b2": {
        "width_coefficient": 1.1,
        "depth_coefficient": 1.2,
        "default_size": 208,
        "block_arch": "EfficientNetV2B",
        "head_filters": 1408,
        "image_size": 208,
        "num_classes": 1000,
    },
    "efficientnetv2_b3": {
        "width_coefficient": 1.2,
        "depth_coefficient": 1.4,
        "default_size": 240,
        "block_arch": "EfficientNetV2B",
        "head_filters": 1536,
        "image_size": 240,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (zm_config.json); the github release urls have been removed.
EFFICIENTNETV2_VARIANTS = {
    "tf_efficientnetv2_s_in1k": {
        "model": "efficientnetv2_s",
        "timm_id": "tf_efficientnetv2_s.in1k",
    },
    "tf_efficientnetv2_s_in21k": {
        "model": "efficientnetv2_s_in21k",
        "timm_id": "tf_efficientnetv2_s.in21k",
    },
    "tf_efficientnetv2_s_in21k_ft_in1k": {
        "model": "efficientnetv2_s",
        "timm_id": "tf_efficientnetv2_s.in21k_ft_in1k",
    },
    "tf_efficientnetv2_m_in1k": {
        "model": "efficientnetv2_m",
        "timm_id": "tf_efficientnetv2_m.in1k",
    },
    "tf_efficientnetv2_m_in21k": {
        "model": "efficientnetv2_m_in21k",
        "timm_id": "tf_efficientnetv2_m.in21k",
    },
    "tf_efficientnetv2_m_in21k_ft_in1k": {
        "model": "efficientnetv2_m",
        "timm_id": "tf_efficientnetv2_m.in21k_ft_in1k",
    },
    "tf_efficientnetv2_l_in1k": {
        "model": "efficientnetv2_l",
        "timm_id": "tf_efficientnetv2_l.in1k",
    },
    "tf_efficientnetv2_l_in21k": {
        "model": "efficientnetv2_l_in21k",
        "timm_id": "tf_efficientnetv2_l.in21k",
    },
    "tf_efficientnetv2_l_in21k_ft_in1k": {
        "model": "efficientnetv2_l",
        "timm_id": "tf_efficientnetv2_l.in21k_ft_in1k",
    },
    "tf_efficientnetv2_xl_in21k": {
        "model": "efficientnetv2_xl_in21k",
        "timm_id": "tf_efficientnetv2_xl.in21k",
    },
    "tf_efficientnetv2_xl_in21k_ft_in1k": {
        "model": "efficientnetv2_xl",
        "timm_id": "tf_efficientnetv2_xl.in21k_ft_in1k",
    },
    "tf_efficientnetv2_b0_in1k": {
        "model": "efficientnetv2_b0",
        "timm_id": "tf_efficientnetv2_b0.in1k",
    },
    "tf_efficientnetv2_b1_in1k": {
        "model": "efficientnetv2_b1",
        "timm_id": "tf_efficientnetv2_b1.in1k",
    },
    "tf_efficientnetv2_b2_in1k": {
        "model": "efficientnetv2_b2",
        "timm_id": "tf_efficientnetv2_b2.in1k",
    },
    "tf_efficientnetv2_b3_in1k": {
        "model": "efficientnetv2_b3",
        "timm_id": "tf_efficientnetv2_b3.in1k",
    },
    "tf_efficientnetv2_b3_in21k_ft_in1k": {
        "model": "efficientnetv2_b3",
        "timm_id": "tf_efficientnetv2_b3.in21k_ft_in1k",
    },
}

_BLOCK0_REMAP = {}
for j in range(8):
    prefix = f"blocks.0.{j}"
    _BLOCK0_REMAP[f"{prefix}.conv_pwl"] = f"{prefix}.conv"
    _BLOCK0_REMAP[f"{prefix}.bn2"] = f"{prefix}.bn1"

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_kernel": ".weight",
    "_gamma": ".weight",
    "_beta": ".bias",
    "_bias": ".bias",
    "_moving_mean": ".running_mean",
    "_moving_variance": ".running_var",
    "FMBconv1": "conv_exp",
    "FMBconv2": "conv_pwl",
    "MBconv1": "conv_pw",
    "MBdwconv": "conv_dw",
    "MBconv2": "conv_pwl",
    "batchnorm1": "bn1",
    "batchnorm2": "bn2",
    "batchnorm3": "bn3",
    "se_": "se.",
    "predictions": "classifier",
    **_BLOCK0_REMAP,
}


def transfer_efficientnetv2_weights(
    keras_model, state_dict: Dict[str, np.ndarray]
) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_weight_name = keras_weight_name
        torch_weight_name = re.sub(
            r"blocks_(\d+)_(\d+)_", r"blocks.\1.\2.", torch_weight_name
        )
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

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

    for variant, meta in EFFICIENTNETV2_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = EfficientNetV2ImageClassify(
            **EFFICIENTNETV2_MODEL_CONFIG[meta["model"]], include_normalization=False
        )
        transfer_efficientnetv2_weights(keras_model, state)

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
