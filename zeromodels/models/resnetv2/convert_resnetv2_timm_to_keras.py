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
from zeromodels.models.resnetv2 import ResNetV2ImageClassify

# Architecture presets, moved here from resnetv2_config.py: the package config no
# longer carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer timm weights into.
RESNETV2_MODEL_CONFIG = {
    "resnetv2_50x1_in21k": {
        "depths": [3, 4, 6, 3],
        "width_factor": 1,
        "image_size": 224,
        "num_classes": 21843,
    },
    "resnetv2_50x1_in1k": {
        "depths": [3, 4, 6, 3],
        "width_factor": 1,
        "image_size": 448,
        "num_classes": 1000,
    },
    "resnetv2_50x3_in21k": {
        "depths": [3, 4, 6, 3],
        "width_factor": 3,
        "image_size": 224,
        "num_classes": 21843,
    },
    "resnetv2_50x3_in1k": {
        "depths": [3, 4, 6, 3],
        "width_factor": 3,
        "image_size": 448,
        "num_classes": 1000,
    },
    "resnetv2_101x1_in21k": {
        "depths": [3, 4, 23, 3],
        "width_factor": 1,
        "image_size": 224,
        "num_classes": 21843,
    },
    "resnetv2_101x1_in1k": {
        "depths": [3, 4, 23, 3],
        "width_factor": 1,
        "image_size": 448,
        "num_classes": 1000,
    },
    "resnetv2_101x3_in21k": {
        "depths": [3, 4, 23, 3],
        "width_factor": 3,
        "image_size": 224,
        "num_classes": 21843,
    },
    "resnetv2_101x3_in1k": {
        "depths": [3, 4, 23, 3],
        "width_factor": 3,
        "image_size": 448,
        "num_classes": 1000,
    },
    "resnetv2_152x2_in21k": {
        "depths": [3, 8, 36, 3],
        "width_factor": 2,
        "image_size": 224,
        "num_classes": 21843,
    },
    "resnetv2_152x2_in1k": {
        "depths": [3, 8, 36, 3],
        "width_factor": 2,
        "image_size": 448,
        "num_classes": 1000,
    },
    "resnetv2_152x4_in21k": {
        "depths": [3, 8, 36, 3],
        "width_factor": 4,
        "image_size": 224,
        "num_classes": 21843,
    },
    "resnetv2_152x4_in1k": {
        "depths": [3, 8, 36, 3],
        "width_factor": 4,
        "image_size": 480,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (zm_config.json); the github release urls have been removed.
RESNETV2_VARIANTS = {
    "resnetv2_50x1_bit_goog_in21k": {
        "model": "resnetv2_50x1_in21k",
        "timm_id": "resnetv2_50x1_bit.goog_in21k",
    },
    "resnetv2_50x1_bit_goog_in21k_ft_in1k": {
        "model": "resnetv2_50x1_in1k",
        "timm_id": "resnetv2_50x1_bit.goog_in21k_ft_in1k",
    },
    "resnetv2_50x3_bit_goog_in21k": {
        "model": "resnetv2_50x3_in21k",
        "timm_id": "resnetv2_50x3_bit.goog_in21k",
    },
    "resnetv2_50x3_bit_goog_in21k_ft_in1k": {
        "model": "resnetv2_50x3_in1k",
        "timm_id": "resnetv2_50x3_bit.goog_in21k_ft_in1k",
    },
    "resnetv2_101x1_bit_goog_in21k": {
        "model": "resnetv2_101x1_in21k",
        "timm_id": "resnetv2_101x1_bit.goog_in21k",
    },
    "resnetv2_101x1_bit_goog_in21k_ft_in1k": {
        "model": "resnetv2_101x1_in1k",
        "timm_id": "resnetv2_101x1_bit.goog_in21k_ft_in1k",
    },
    "resnetv2_101x3_bit_goog_in21k": {
        "model": "resnetv2_101x3_in21k",
        "timm_id": "resnetv2_101x3_bit.goog_in21k",
    },
    "resnetv2_101x3_bit_goog_in21k_ft_in1k": {
        "model": "resnetv2_101x3_in1k",
        "timm_id": "resnetv2_101x3_bit.goog_in21k_ft_in1k",
    },
    "resnetv2_152x2_bit_goog_in21k": {
        "model": "resnetv2_152x2_in21k",
        "timm_id": "resnetv2_152x2_bit.goog_in21k",
    },
    "resnetv2_152x2_bit_goog_in21k_ft_in1k": {
        "model": "resnetv2_152x2_in1k",
        "timm_id": "resnetv2_152x2_bit.goog_in21k_ft_in1k",
    },
    "resnetv2_152x4_bit_goog_in21k": {
        "model": "resnetv2_152x4_in21k",
        "timm_id": "resnetv2_152x4_bit.goog_in21k",
    },
    "resnetv2_152x4_bit_goog_in21k_ft_in1k": {
        "model": "resnetv2_152x4_in1k",
        "timm_id": "resnetv2_152x4_bit.goog_in21k_ft_in1k",
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "groupnorm.1": "norm1",
    "groupnorm.2": "norm2",
    "groupnorm.3": "norm3",
    "groupnorm": "norm",
    "conv.1": "conv1",
    "conv.2": "conv2",
    "conv.3": "conv3",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "head.fc",
}


def transfer_resnetv2_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if torch_weight_name == "head.fc.weight":
            if torch_weight_name not in state_dict:
                raise WeightMappingError(keras_weight_name, torch_weight_name)
            w = np.asarray(state_dict[torch_weight_name])
            keras_weight.assign(w.squeeze().T)
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

    for variant, meta in RESNETV2_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = ResNetV2ImageClassify(**RESNETV2_MODEL_CONFIG[meta["model"]])
        transfer_resnetv2_weights(keras_model, state)

        torch_model = timm.create_model(timm_id, pretrained=True).eval()
        results = verify_cls_model_equivalence(
            model_a=torch_model,
            model_b=keras_model,
            input_shape=keras_model.input_shape[1:],
            output_specs={"num_classes": keras_model.output_shape[-1]},
            comparison_type="torch_to_keras",
            run_performance=False,
            atol=1e-3,
            rtol=1e-3,
        )
        if not results["standard_input"]:
            raise ValueError(
                "Model equivalence test failed - model outputs do not match for standard input"
            )

        total_params = sum(int(np.prod(w.shape)) for w in keras_model.weights)
        total_gb = (total_params * 4) / (1024**3)
        if total_gb > 1.7:
            out_path = f"{variant}.weights.json"
            keras_model.save_weights(out_path, max_shard_size=1.7)
            print(f"  Saved -> {out_path} (sharded, ~{total_gb:.2f} GB)")
        else:
            out_path = f"{variant}.weights.h5"
            keras_model.save_weights(out_path)
            print(f"  Saved -> {out_path} (~{total_gb:.2f} GB)")

        del keras_model, state, torch_model
        keras.backend.clear_session()
        gc.collect()
