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
from zeromodels.models.densenet import DenseNetImageClassify

# Architecture presets, moved here from densenet_config.py: the package config no
# longer carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer timm weights into.
DENSENET_MODEL_CONFIG = {
    "densenet121": {
        "depths": [6, 12, 24, 16],
        "growth_rate": 32,
        "initial_filter": 64,
        "image_size": 224,
        "num_classes": 1000,
    },
    "densenet161": {
        "depths": [6, 12, 36, 24],
        "growth_rate": 48,
        "initial_filter": 96,
        "image_size": 224,
        "num_classes": 1000,
    },
    "densenet169": {
        "depths": [6, 12, 32, 32],
        "growth_rate": 32,
        "initial_filter": 64,
        "image_size": 224,
        "num_classes": 1000,
    },
    "densenet201": {
        "depths": [6, 12, 48, 32],
        "growth_rate": 32,
        "initial_filter": 64,
        "image_size": 224,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (zm_config.json); the github release urls have been removed.
DENSENET_VARIANTS = {
    "densenet121_tv_in1k": {
        "model": "densenet121",
        "timm_id": "densenet121.tv_in1k",
    },
    "densenet161_tv_in1k": {
        "model": "densenet161",
        "timm_id": "densenet161.tv_in1k",
    },
    "densenet169_tv_in1k": {
        "model": "densenet169",
        "timm_id": "densenet169.tv_in1k",
    },
    "densenet201_tv_in1k": {
        "model": "densenet201",
        "timm_id": "densenet201.tv_in1k",
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "batchnorm_1": "norm1",
    "batchnorm_2": "norm2",
    "conv2d_1": "conv1",
    "conv2d_2": "conv2",
    "transition_batchnorm": "norm",
    "transition_conv2d": "conv",
    "stem_conv": "features_conv0",
    "stem_norm": "features_norm0",
    "stem_pool": "features_pool0",
    "dense_block": "features_denseblock",
    "transition_block": "features_transition",
    "final_batchnorm": "features_norm5",
    "_": ".",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "classifier",
}


def transfer_densenet_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
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

    for variant, meta in DENSENET_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = DenseNetImageClassify(**DENSENET_MODEL_CONFIG[meta["model"]])
        transfer_densenet_weights(keras_model, state)

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
