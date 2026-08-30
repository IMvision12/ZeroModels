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
from zeromodels.models.poolformer import PoolFormerImageClassify

# Architecture presets, moved here from poolformer_config.py: the package config no
# longer carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer timm weights into.
POOLFORMER_MODEL_CONFIG = {
    "poolformer_s12": {
        "embed_dim": (64, 128, 320, 512),
        "depths": (2, 2, 6, 2),
        "init_scale": 1e-5,
        "image_size": 224,
        "num_classes": 1000,
    },
    "poolformer_s24": {
        "embed_dim": (64, 128, 320, 512),
        "depths": (4, 4, 12, 4),
        "init_scale": 1e-5,
        "image_size": 224,
        "num_classes": 1000,
    },
    "poolformer_s36": {
        "embed_dim": (64, 128, 320, 512),
        "depths": (6, 6, 18, 6),
        "init_scale": 1e-6,
        "image_size": 224,
        "num_classes": 1000,
    },
    "poolformer_m36": {
        "embed_dim": (96, 192, 384, 768),
        "depths": (6, 6, 18, 6),
        "init_scale": 1e-6,
        "image_size": 224,
        "num_classes": 1000,
    },
    "poolformer_m48": {
        "embed_dim": (96, 192, 384, 768),
        "depths": (8, 8, 24, 8),
        "init_scale": 1e-6,
        "image_size": 224,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (zm_config.json); the github release urls have been removed.
POOLFORMER_VARIANTS = {
    "poolformer_s12_sail_in1k": {
        "model": "poolformer_s12",
        "timm_id": "poolformer_s12.sail_in1k",
    },
    "poolformer_s24_sail_in1k": {
        "model": "poolformer_s24",
        "timm_id": "poolformer_s24.sail_in1k",
    },
    "poolformer_s36_sail_in1k": {
        "model": "poolformer_s36",
        "timm_id": "poolformer_s36.sail_in1k",
    },
    "poolformer_m36_sail_in1k": {
        "model": "poolformer_m36",
        "timm_id": "poolformer_m36.sail_in1k",
    },
    "poolformer_m48_sail_in1k": {
        "model": "poolformer_m48",
        "timm_id": "poolformer_m48.sail_in1k",
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "stage": "stages",
    "block": "blocks",
    "conv.1": "fc1",
    "conv.2": "fc2",
    "groupnorm.1": "norm1",
    "groupnorm.2": "norm2",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
    "predictions": "head.fc",
    "layernorm": "head.norm",
}


def transfer_poolformer_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)
        torch_weight_name = re.sub(
            r"layerscale\.(\d+)\.variable(?:\.\d+)?$",
            r"layer_scale\1.scale",
            torch_weight_name,
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

    for variant, meta in POOLFORMER_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = PoolFormerImageClassify(**POOLFORMER_MODEL_CONFIG[meta["model"]])
        transfer_poolformer_weights(keras_model, state)

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
