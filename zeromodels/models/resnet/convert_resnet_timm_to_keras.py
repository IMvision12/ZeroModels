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
from zeromodels.models.resnet import ResNetImageClassify

# Architecture presets, moved here from resnet_config.py: the package config no
# longer carries arch (models load by Hub repo id / kf_config). Only this converter
# builds an untrained model to transfer timm weights into.
RESNET_MODEL_CONFIG = {
    "resnet50": {
        "depths": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
    },
    "resnet101": {
        "depths": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
    },
    "resnet152": {
        "depths": [3, 8, 36, 3],
        "filters": [64, 128, 256, 512],
    },
}

# Hosted variants -> (base model arch key, timm id). Weights load by Hub repo id
# (kf_config.json); the github release urls have been removed. ``timm_id`` is the
# conversion source used by the converter + the ``hf:timm/...`` path.
RESNET_VARIANTS = {
    "resnet50_tv_in1k": {"model": "resnet50", "timm_id": "resnet50.tv_in1k"},
    "resnet50_a1_in1k": {"model": "resnet50", "timm_id": "resnet50.a1_in1k"},
    "resnet50_gluon_in1k": {"model": "resnet50", "timm_id": "resnet50.gluon_in1k"},
    "resnet101_tv_in1k": {"model": "resnet101", "timm_id": "resnet101.tv_in1k"},
    "resnet101_a1_in1k": {"model": "resnet101", "timm_id": "resnet101.a1_in1k"},
    "resnet101_gluon_in1k": {"model": "resnet101", "timm_id": "resnet101.gluon_in1k"},
    "resnet152_tv_in1k": {"model": "resnet152", "timm_id": "resnet152.tv_in1k"},
    "resnet152_a1_in1k": {"model": "resnet152", "timm_id": "resnet152.a1_in1k"},
    "resnet152_gluon_in1k": {"model": "resnet152", "timm_id": "resnet152.gluon_in1k"},
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "resnet_layer": "layer",
    "_": ".",
    "downsample.conv": "downsample.0",
    "downsample.batchnorm": "downsample.1",
    "batchnorm1": "bn1",
    "batchnorm2": "bn2",
    "batchnorm3": "bn3",
    "dense1": "fc1",
    "dense2": "fc2",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "fc",
}


def transfer_resnet_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
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

    for variant, meta in RESNET_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = ResNetImageClassify(
            **RESNET_MODEL_CONFIG[meta["model"]], include_normalization=False
        )
        transfer_resnet_weights(keras_model, state)

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
