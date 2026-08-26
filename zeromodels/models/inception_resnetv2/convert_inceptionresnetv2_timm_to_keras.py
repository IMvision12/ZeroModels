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
from zeromodels.models.inception_resnetv2 import InceptionResNetV2ImageClassify

# Architecture presets, moved here from inception_resnetv2_config.py: the package config no
# longer carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer timm weights into.
INCEPTION_RESNETV2_MODEL_CONFIG = {
    "inception_resnet_v2": {
        "image_size": 299,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (zm_config.json); the github release urls have been removed.
INCEPTION_RESNETV2_VARIANTS = {
    "inception_resnet_v2_tf_in1k": {
        "model": "inception_resnet_v2",
        "timm_id": "inception_resnet_v2.tf_in1k",
    },
    "inception_resnet_v2_tf_ens_adv_in1k": {
        "model": "inception_resnet_v2",
        "timm_id": "inception_resnet_v2.tf_ens_adv_in1k",
    },
}


def base_mappings() -> Dict[str, str]:
    return {
        "_conv": ".conv",
        "_batchnorm": ".bn",
        "_kernel": ".weight",
        "_gamma": ".weight",
        "_beta": ".bias",
        "_bias": ".bias",
        "_moving_mean": ".running_mean",
        "_moving_variance": ".running_var",
        "mixed_5b_": "mixed_5b.",
        "mixed_6a_": "mixed_6a.",
        "mixed_7a_": "mixed_7a.",
        "repeats_1_": "repeat_1.",
        "repeats_2_": "repeat_2.",
        "branch1_0": "branch1.0",
        "branch1_1": "branch1.1",
        "branch1_2": "branch1.2",
        "branch2_0": "branch2.0",
        "branch2_1": "branch2.1",
        "branch2_2": "branch2.2",
        "branch3_1": "branch3.1",
        "branch0_0": "branch0.0",
        "branch0_1": "branch0.1",
        "block8_": "block8.",
        "predictions": "classif",
    }


def generate_repeat_mappings() -> Dict[str, str]:
    mappings: Dict[str, str] = {}

    for i in range(10):
        mappings[f"repeat_{i}_"] = f"repeat.{i}."
        mappings[f"repeat_{i}"] = f"repeat.{i}"

    for i in range(20):
        base = f"repeat.1.{i}"
        keras_base = f"repeat_1.{i}"
        mappings[f"{base}_branch1.0"] = f"{keras_base}.branch1.0"
        mappings[f"{base}_branch1.1"] = f"{keras_base}.branch1.1"
        mappings[f"{base}_branch1.2"] = f"{keras_base}.branch1.2"
        mappings[f"{base}_branch0"] = f"{keras_base}.branch0"
        mappings[f"{base}.conv2d"] = f"{keras_base}.conv2d"

    for i in range(9):
        base = f"repeat.2.{i}"
        keras_base = f"repeat_2.{i}"
        mappings[f"{base}_branch1.0"] = f"{keras_base}.branch1.0"
        mappings[f"{base}_branch1.1"] = f"{keras_base}.branch1.1"
        mappings[f"{base}_branch1.2"] = f"{keras_base}.branch1.2"
        mappings[f"{base}_branch0"] = f"{keras_base}.branch0"
        mappings[f"{base}.conv2d"] = f"{keras_base}.conv2d"

    return mappings


WEIGHT_NAME_MAPPING: Dict[str, str] = {
    **base_mappings(),
    **generate_repeat_mappings(),
}


def transfer_inception_resnet_v2_weights(
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

    for variant, meta in INCEPTION_RESNETV2_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = InceptionResNetV2ImageClassify(
            **INCEPTION_RESNETV2_MODEL_CONFIG[meta["model"]],
            include_normalization=False,
        )
        transfer_inception_resnet_v2_weights(keras_model, state)

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
