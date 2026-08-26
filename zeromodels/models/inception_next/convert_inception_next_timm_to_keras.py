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
from zeromodels.models.inception_next import InceptionNextImageClassify

# Architecture presets, moved here from inception_next_config.py: the package config no
# longer carries arch (models load by Hub repo id / kf_config). Only this converter
# builds an untrained model to transfer timm weights into.
INCEPTION_NEXT_MODEL_CONFIG = {
    "inception_next_atto": {
        "depths": [2, 2, 6, 2],
        "num_filters": [40, 80, 160, 320],
        "mlp_ratios": [4, 4, 4, 3],
        "band_kernel_size": 9,
        "branch_ratio": 0.25,
        "image_size": 224,
        "num_classes": 1000,
    },
    "inception_next_tiny": {
        "depths": [3, 3, 9, 3],
        "num_filters": [96, 192, 384, 768],
        "mlp_ratios": [4, 4, 4, 3],
        "band_kernel_size": 11,
        "branch_ratio": 0.125,
        "image_size": 224,
        "num_classes": 1000,
    },
    "inception_next_small": {
        "depths": [3, 3, 27, 3],
        "num_filters": [96, 192, 384, 768],
        "mlp_ratios": [4, 4, 4, 3],
        "band_kernel_size": 11,
        "branch_ratio": 0.125,
        "image_size": 224,
        "num_classes": 1000,
    },
    "inception_next_base": {
        "depths": [3, 3, 27, 3],
        "num_filters": [128, 256, 512, 1024],
        "mlp_ratios": [4, 4, 4, 3],
        "band_kernel_size": 11,
        "branch_ratio": 0.125,
        "image_size": 224,
        "num_classes": 1000,
    },
    "inception_next_base_384": {
        "depths": [3, 3, 27, 3],
        "num_filters": [128, 256, 512, 1024],
        "mlp_ratios": [4, 4, 4, 3],
        "band_kernel_size": 11,
        "branch_ratio": 0.125,
        "image_size": 384,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (kf_config.json); the github release urls have been removed.
INCEPTION_NEXT_VARIANTS = {
    "inception_next_atto_sail_in1k": {
        "model": "inception_next_atto",
        "timm_id": "inception_next_atto.sail_in1k",
    },
    "inception_next_tiny_sail_in1k": {
        "model": "inception_next_tiny",
        "timm_id": "inception_next_tiny.sail_in1k",
    },
    "inception_next_small_sail_in1k": {
        "model": "inception_next_small",
        "timm_id": "inception_next_small.sail_in1k",
    },
    "inception_next_base_sail_in1k": {
        "model": "inception_next_base",
        "timm_id": "inception_next_base.sail_in1k",
    },
    "inception_next_base_sail_in1k_384": {
        "model": "inception_next_base_384",
        "timm_id": "inception_next_base.sail_in1k_384",
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "stem.conv": "stem.0",
    "stem.batchnorm": "stem.1",
    "token.mixer": "token_mixer",
    "dwconv.hw": "dwconv_hw",
    "dwconv.w": "dwconv_w",
    "dwconv.h": "dwconv_h",
    "batchnorm": "norm",
    "conv1": "mlp.fc1",
    "conv2": "mlp.fc2",
    "downsample.conv": "downsample.1",
    "downsample.norm": "downsample.0",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "head.fc": "head.fc1",
    "predictions": "head.fc2",
}


def transfer_inception_next_weights(
    keras_model, state_dict: Dict[str, np.ndarray]
) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)
        torch_weight_name = re.sub(
            r"weight.variable(?:.\d+)?", "gamma", torch_weight_name
        )

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

    for variant, meta in INCEPTION_NEXT_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = InceptionNextImageClassify(
            **INCEPTION_NEXT_MODEL_CONFIG[meta["model"]], include_normalization=False
        )
        transfer_inception_next_weights(keras_model, state)

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
