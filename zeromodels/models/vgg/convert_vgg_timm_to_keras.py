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
from zeromodels.models.vgg import VGGImageClassify

# Architecture presets, moved here from vgg_config.py: the package config no
# longer carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer timm weights into.
VGG_MODEL_CONFIG = {
    "vgg11": {
        "num_filters": [
            64,
            "M",
            128,
            "M",
            256,
            256,
            "M",
            512,
            512,
            "M",
            512,
            512,
            "M",
        ],
        "batch_norm": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "vgg11_bn": {
        "num_filters": [
            64,
            "M",
            128,
            "M",
            256,
            256,
            "M",
            512,
            512,
            "M",
            512,
            512,
            "M",
        ],
        "batch_norm": True,
        "image_size": 224,
        "num_classes": 1000,
    },
    "vgg13": {
        "num_filters": [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            "M",
            512,
            512,
            "M",
            512,
            512,
            "M",
        ],
        "batch_norm": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "vgg13_bn": {
        "num_filters": [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            "M",
            512,
            512,
            "M",
            512,
            512,
            "M",
        ],
        "batch_norm": True,
        "image_size": 224,
        "num_classes": 1000,
    },
    "vgg16": {
        "num_filters": [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            "M",
        ],
        "batch_norm": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "vgg16_bn": {
        "num_filters": [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            "M",
        ],
        "batch_norm": True,
        "image_size": 224,
        "num_classes": 1000,
    },
    "vgg19": {
        "num_filters": [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            512,
            "M",
        ],
        "batch_norm": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "vgg19_bn": {
        "num_filters": [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            512,
            "M",
        ],
        "batch_norm": True,
        "image_size": 224,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (zm_config.json); the github release urls have been removed.
VGG_VARIANTS = {
    "vgg11_tv_in1k": {
        "model": "vgg11",
        "timm_id": "vgg11.tv_in1k",
    },
    "vgg11_bn_tv_in1k": {
        "model": "vgg11_bn",
        "timm_id": "vgg11_bn.tv_in1k",
    },
    "vgg13_tv_in1k": {
        "model": "vgg13",
        "timm_id": "vgg13.tv_in1k",
    },
    "vgg13_bn_tv_in1k": {
        "model": "vgg13_bn",
        "timm_id": "vgg13_bn.tv_in1k",
    },
    "vgg16_tv_in1k": {
        "model": "vgg16",
        "timm_id": "vgg16.tv_in1k",
    },
    "vgg16_bn_tv_in1k": {
        "model": "vgg16_bn",
        "timm_id": "vgg16_bn.tv_in1k",
    },
    "vgg19_tv_in1k": {
        "model": "vgg19",
        "timm_id": "vgg19.tv_in1k",
    },
    "vgg19_bn_tv_in1k": {
        "model": "vgg19_bn",
        "timm_id": "vgg19_bn.tv_in1k",
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "conv.fc1": "pre_logits.fc1",
    "conv.fc2": "pre_logits.fc2",
    "batchnorm": "features",
    "conv2d": "features",
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "head.fc",
}


def transfer_vgg_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
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

    for variant, meta in VGG_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = VGGImageClassify(
            **VGG_MODEL_CONFIG[meta["model"]], include_normalization=False
        )
        transfer_vgg_weights(keras_model, state)

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
