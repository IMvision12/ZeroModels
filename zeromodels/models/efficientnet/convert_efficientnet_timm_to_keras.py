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
from zeromodels.models.efficientnet import EfficientNetImageClassify

# Architecture presets, moved here from efficientnet_config.py: the package config no
# longer carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer timm weights into.
EFFICIENTNET_MODEL_CONFIG = {
    "efficientnet_b0": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.0,
        "dropout_rate": 0.2,
        "default_size": 224,
        "image_size": 224,
        "num_classes": 1000,
    },
    "efficientnet_b1": {
        "width_coefficient": 1.0,
        "depth_coefficient": 1.1,
        "dropout_rate": 0.2,
        "default_size": 240,
        "image_size": 240,
        "num_classes": 1000,
    },
    "efficientnet_b2": {
        "width_coefficient": 1.1,
        "depth_coefficient": 1.2,
        "dropout_rate": 0.3,
        "default_size": 260,
        "image_size": 260,
        "num_classes": 1000,
    },
    "efficientnet_b3": {
        "width_coefficient": 1.2,
        "depth_coefficient": 1.4,
        "dropout_rate": 0.3,
        "default_size": 300,
        "image_size": 300,
        "num_classes": 1000,
    },
    "efficientnet_b4": {
        "width_coefficient": 1.4,
        "depth_coefficient": 1.8,
        "dropout_rate": 0.4,
        "default_size": 380,
        "image_size": 380,
        "num_classes": 1000,
    },
    "efficientnet_b5": {
        "width_coefficient": 1.6,
        "depth_coefficient": 2.2,
        "dropout_rate": 0.4,
        "default_size": 456,
        "image_size": 456,
        "num_classes": 1000,
    },
    "efficientnet_b6": {
        "width_coefficient": 1.8,
        "depth_coefficient": 2.6,
        "dropout_rate": 0.5,
        "default_size": 528,
        "image_size": 528,
        "num_classes": 1000,
    },
    "efficientnet_b7": {
        "width_coefficient": 2.0,
        "depth_coefficient": 3.1,
        "dropout_rate": 0.5,
        "default_size": 600,
        "image_size": 600,
        "num_classes": 1000,
    },
    "efficientnet_b8": {
        "width_coefficient": 2.2,
        "depth_coefficient": 3.6,
        "dropout_rate": 0.5,
        "default_size": 672,
        "image_size": 672,
        "num_classes": 1000,
    },
    "efficientnet_l2_800": {
        "width_coefficient": 4.3,
        "depth_coefficient": 5.3,
        "dropout_rate": 0.5,
        "default_size": 800,
        "image_size": 800,
        "num_classes": 1000,
    },
    "efficientnet_l2_475": {
        "width_coefficient": 4.3,
        "depth_coefficient": 5.3,
        "dropout_rate": 0.5,
        "default_size": 800,
        "image_size": 475,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (zm_config.json); the github release urls have been removed.
EFFICIENTNET_VARIANTS = {
    "tf_efficientnet_b0_ns_jft_in1k": {
        "model": "efficientnet_b0",
        "timm_id": "tf_efficientnet_b0.ns_jft_in1k",
    },
    "tf_efficientnet_b0_ap_in1k": {
        "model": "efficientnet_b0",
        "timm_id": "tf_efficientnet_b0.ap_in1k",
    },
    "tf_efficientnet_b0_aa_in1k": {
        "model": "efficientnet_b0",
        "timm_id": "tf_efficientnet_b0.aa_in1k",
    },
    "tf_efficientnet_b0_in1k": {
        "model": "efficientnet_b0",
        "timm_id": "tf_efficientnet_b0.in1k",
    },
    "tf_efficientnet_b1_ns_jft_in1k": {
        "model": "efficientnet_b1",
        "timm_id": "tf_efficientnet_b1.ns_jft_in1k",
    },
    "tf_efficientnet_b1_ap_in1k": {
        "model": "efficientnet_b1",
        "timm_id": "tf_efficientnet_b1.ap_in1k",
    },
    "tf_efficientnet_b1_aa_in1k": {
        "model": "efficientnet_b1",
        "timm_id": "tf_efficientnet_b1.aa_in1k",
    },
    "tf_efficientnet_b1_in1k": {
        "model": "efficientnet_b1",
        "timm_id": "tf_efficientnet_b1.in1k",
    },
    "tf_efficientnet_b2_ns_jft_in1k": {
        "model": "efficientnet_b2",
        "timm_id": "tf_efficientnet_b2.ns_jft_in1k",
    },
    "tf_efficientnet_b2_ap_in1k": {
        "model": "efficientnet_b2",
        "timm_id": "tf_efficientnet_b2.ap_in1k",
    },
    "tf_efficientnet_b2_aa_in1k": {
        "model": "efficientnet_b2",
        "timm_id": "tf_efficientnet_b2.aa_in1k",
    },
    "tf_efficientnet_b2_in1k": {
        "model": "efficientnet_b2",
        "timm_id": "tf_efficientnet_b2.in1k",
    },
    "tf_efficientnet_b3_ns_jft_in1k": {
        "model": "efficientnet_b3",
        "timm_id": "tf_efficientnet_b3.ns_jft_in1k",
    },
    "tf_efficientnet_b3_ap_in1k": {
        "model": "efficientnet_b3",
        "timm_id": "tf_efficientnet_b3.ap_in1k",
    },
    "tf_efficientnet_b3_aa_in1k": {
        "model": "efficientnet_b3",
        "timm_id": "tf_efficientnet_b3.aa_in1k",
    },
    "tf_efficientnet_b3_in1k": {
        "model": "efficientnet_b3",
        "timm_id": "tf_efficientnet_b3.in1k",
    },
    "tf_efficientnet_b4_ns_jft_in1k": {
        "model": "efficientnet_b4",
        "timm_id": "tf_efficientnet_b4.ns_jft_in1k",
    },
    "tf_efficientnet_b4_ap_in1k": {
        "model": "efficientnet_b4",
        "timm_id": "tf_efficientnet_b4.ap_in1k",
    },
    "tf_efficientnet_b4_aa_in1k": {
        "model": "efficientnet_b4",
        "timm_id": "tf_efficientnet_b4.aa_in1k",
    },
    "tf_efficientnet_b4_in1k": {
        "model": "efficientnet_b4",
        "timm_id": "tf_efficientnet_b4.in1k",
    },
    "tf_efficientnet_b5_ns_jft_in1k": {
        "model": "efficientnet_b5",
        "timm_id": "tf_efficientnet_b5.ns_jft_in1k",
    },
    "tf_efficientnet_b5_ap_in1k": {
        "model": "efficientnet_b5",
        "timm_id": "tf_efficientnet_b5.ap_in1k",
    },
    "tf_efficientnet_b5_aa_in1k": {
        "model": "efficientnet_b5",
        "timm_id": "tf_efficientnet_b5.aa_in1k",
    },
    "tf_efficientnet_b5_in1k": {
        "model": "efficientnet_b5",
        "timm_id": "tf_efficientnet_b5.in1k",
    },
    "tf_efficientnet_b6_ns_jft_in1k": {
        "model": "efficientnet_b6",
        "timm_id": "tf_efficientnet_b6.ns_jft_in1k",
    },
    "tf_efficientnet_b6_ap_in1k": {
        "model": "efficientnet_b6",
        "timm_id": "tf_efficientnet_b6.ap_in1k",
    },
    "tf_efficientnet_b6_aa_in1k": {
        "model": "efficientnet_b6",
        "timm_id": "tf_efficientnet_b6.aa_in1k",
    },
    "tf_efficientnet_b7_ns_jft_in1k": {
        "model": "efficientnet_b7",
        "timm_id": "tf_efficientnet_b7.ns_jft_in1k",
    },
    "tf_efficientnet_b7_ap_in1k": {
        "model": "efficientnet_b7",
        "timm_id": "tf_efficientnet_b7.ap_in1k",
    },
    "tf_efficientnet_b7_aa_in1k": {
        "model": "efficientnet_b7",
        "timm_id": "tf_efficientnet_b7.aa_in1k",
    },
    "tf_efficientnet_b8_ap_in1k": {
        "model": "efficientnet_b8",
        "timm_id": "tf_efficientnet_b8.ap_in1k",
    },
    "tf_efficientnet_l2_ns_jft_in1k": {
        "model": "efficientnet_l2_800",
        "timm_id": "tf_efficientnet_l2.ns_jft_in1k",
    },
    "tf_efficientnet_l2_ns_jft_in1k_475": {
        "model": "efficientnet_l2_475",
        "timm_id": "tf_efficientnet_l2.ns_jft_in1k_475",
    },
}

_BLOCK_MAPPINGS = {}
for i in range(6):
    block_prefix = f"blocks.0.{i}"
    _BLOCK_MAPPINGS[f"{block_prefix}.conv_pwl"] = f"{block_prefix}.conv_pw"
    _BLOCK_MAPPINGS[f"{block_prefix}.bn2"] = f"{block_prefix}.bn1"
    _BLOCK_MAPPINGS[f"{block_prefix}.bn3"] = f"{block_prefix}.bn2"

_BASE_MAPPINGS = {
    "_kernel": ".weight",
    "_gamma": ".weight",
    "_beta": ".bias",
    "_bias": ".bias",
    "_moving_mean": ".running_mean",
    "_moving_variance": ".running_var",
    "se_": "se.",
    "batchnorm_1": "bn1",
    "batchnorm_2": "bn2",
    "batchnorm_3": "bn3",
    "conv2d_1": "conv_pw",
    "dwconv2d": "conv_dw",
    "conv2d_2": "conv_pwl",
    "predictions": "classifier",
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {**_BASE_MAPPINGS, **_BLOCK_MAPPINGS}


def transfer_efficientnet_weights(
    keras_model, state_dict: Dict[str, np.ndarray]
) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = re.sub(
                r"blocks_(\d+)_(\d+)_",
                lambda m: f"blocks.{m.group(1)}.{m.group(2)}.",
                torch_weight_name,
            )
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

    for variant, meta in EFFICIENTNET_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = EfficientNetImageClassify(
            **EFFICIENTNET_MODEL_CONFIG[meta["model"]]
        )
        transfer_efficientnet_weights(keras_model, state)

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
