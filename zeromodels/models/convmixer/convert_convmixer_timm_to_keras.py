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
from zeromodels.models.convmixer import ConvMixerImageClassify

# Architecture presets, moved here from convmixer_config.py: the package config no
# longer carries arch (models load by Hub repo id / kf_config). Only this converter
# builds an untrained model to transfer timm weights into.
CONVMIXER_MODEL_CONFIG = {
    "convmixer_1536_20": {
        "embed_dim": 1536,
        "depth": 20,
        "patch_size": 7,
        "kernel_size": 9,
        "activation": "gelu",
        "image_size": 224,
        "num_classes": 1000,
    },
    "convmixer_768_32": {
        "embed_dim": 768,
        "depth": 32,
        "patch_size": 7,
        "kernel_size": 7,
        "activation": "relu",
        "image_size": 224,
        "num_classes": 1000,
    },
    "convmixer_1024_20_ks9_p14": {
        "embed_dim": 1024,
        "depth": 20,
        "patch_size": 14,
        "kernel_size": 9,
        "activation": "gelu",
        "image_size": 224,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (kf_config.json). Architecture presets now live in the converter.
CONVMIXER_VARIANTS = {
    "convmixer_1536_20_in1k": {
        "model": "convmixer_1536_20",
        "timm_id": "convmixer_1536_20.in1k",
    },
    "convmixer_768_32_in1k": {
        "model": "convmixer_768_32",
        "timm_id": "convmixer_768_32.in1k",
    },
    "convmixer_1024_20_ks9_p14_in1k": {
        "model": "convmixer_1024_20_ks9_p14",
        "timm_id": "convmixer_1024_20_ks9_p14.in1k",
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "stem.conv2d": "stem.0",
    "stem.batchnorm": "stem.2",
    "mixer.block.": "blocks.",
    ".depthwise": ".0.fn.0",
    ".batchnorm.1": ".0.fn.2",
    ".add": ".0",
    ".conv2d": ".1",
    ".batchnorm.2": ".3",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "head",
}


def transfer_convmixer_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
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

    for variant, meta in CONVMIXER_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = ConvMixerImageClassify(
            **CONVMIXER_MODEL_CONFIG[meta["model"]], include_normalization=False
        )
        transfer_convmixer_weights(keras_model, state)

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
