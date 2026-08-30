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
    transfer_weights,
)
from zeromodels.models.efficientnet_lite import EfficientNetLiteImageClassify

# Architecture presets, moved here from efficientnet_lite_config.py: the package config no
# longer carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer timm weights into.
EFFICIENTNET_LITE_MODEL_CONFIG = {
    "efficientnet_lite_b0": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.0,
        "dropout_rate": 0.2,
        "default_size": 224,
        "image_size": 224,
        "num_classes": 1000,
    },
    "efficientnet_lite_b1": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.1,
        "dropout_rate": 0.2,
        "default_size": 240,
        "image_size": 240,
        "num_classes": 1000,
    },
    "efficientnet_lite_b2": {
        "width_coefficient": 1.1,
        "depth_coefficient": 1.2,
        "dropout_rate": 0.3,
        "default_size": 260,
        "image_size": 260,
        "num_classes": 1000,
    },
    "efficientnet_lite_b3": {
        "width_coefficient": 1.2,
        "depth_coefficient": 1.4,
        "dropout_rate": 0.3,
        "default_size": 300,
        "image_size": 300,
        "num_classes": 1000,
    },
    "efficientnet_lite_b4": {
        "width_coefficient": 1.4,
        "depth_coefficient": 1.8,
        "dropout_rate": 0.3,
        "default_size": 380,
        "image_size": 380,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (zm_config.json); the github release urls have been removed.
EFFICIENTNET_LITE_VARIANTS = {
    "tf_efficientnet_lite0_in1k": {
        "model": "efficientnet_lite_b0",
        "timm_id": "tf_efficientnet_lite0.in1k",
    },
    "tf_efficientnet_lite1_in1k": {
        "model": "efficientnet_lite_b1",
        "timm_id": "tf_efficientnet_lite1.in1k",
    },
    "tf_efficientnet_lite2_in1k": {
        "model": "efficientnet_lite_b2",
        "timm_id": "tf_efficientnet_lite2.in1k",
    },
    "tf_efficientnet_lite3_in1k": {
        "model": "efficientnet_lite_b3",
        "timm_id": "tf_efficientnet_lite3.in1k",
    },
    "tf_efficientnet_lite4_in1k": {
        "model": "efficientnet_lite_b4",
        "timm_id": "tf_efficientnet_lite4.in1k",
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "conv.stem": "conv_stem",
    "conv2d.1": "conv_pw",
    "dwconv2d": "conv_dw",
    "conv2d.2": "conv_pwl",
    "batchnorm.1": "bn1",
    "batchnorm.2": "bn2",
    "batchnorm.3": "bn3",
    "blocks.0.0.bn2": "blocks.0.0.bn1",
    "blocks.0.0.bn3": "blocks.0.0.bn2",
    "blocks.0.0.conv_pwl": "blocks.0.0.conv_pw",
    "conv.head": "conv_head",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "classifier",
}


def transfer_efficientnet_lite_weights(
    keras_model, state_dict: Dict[str, np.ndarray]
) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_weight_name = keras_weight_name
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

    for variant, meta in EFFICIENTNET_LITE_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = EfficientNetLiteImageClassify(
            **EFFICIENTNET_LITE_MODEL_CONFIG[meta["model"]]
        )
        transfer_efficientnet_lite_weights(keras_model, state)

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
