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
from zeromodels.models.mobilenetv2 import MobileNetV2ImageClassify

MOBILENETV2_MODEL_CONFIG = {
    "mobilenetv2_050": {
        "width_multiplier": 0.5,
        "depth_multiplier": 1.0,
        "fix_channels": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "mobilenetv2_100": {
        "width_multiplier": 1.0,
        "depth_multiplier": 1.0,
        "fix_channels": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "mobilenetv2_110d": {
        "width_multiplier": 1.1,
        "depth_multiplier": 1.2,
        "fix_channels": True,
        "image_size": 224,
        "num_classes": 1000,
    },
    "mobilenetv2_120d": {
        "width_multiplier": 1.2,
        "depth_multiplier": 1.4,
        "fix_channels": True,
        "image_size": 224,
        "num_classes": 1000,
    },
    "mobilenetv2_140": {
        "width_multiplier": 1.4,
        "depth_multiplier": 1.0,
        "fix_channels": False,
        "image_size": 224,
        "num_classes": 1000,
    },
}

MOBILENETV2_VARIANTS = {
    "mobilenetv2_050_lamb_in1k": {
        "model": "mobilenetv2_050",
        "timm_id": "mobilenetv2_050.lamb_in1k",
    },
    "mobilenetv2_100_ra_in1k": {
        "model": "mobilenetv2_100",
        "timm_id": "mobilenetv2_100.ra_in1k",
    },
    "mobilenetv2_110d_ra_in1k": {
        "model": "mobilenetv2_110d",
        "timm_id": "mobilenetv2_110d.ra_in1k",
    },
    "mobilenetv2_120d_ra_in1k": {
        "model": "mobilenetv2_120d",
        "timm_id": "mobilenetv2_120d.ra_in1k",
    },
    "mobilenetv2_140_ra_in1k": {
        "model": "mobilenetv2_140",
        "timm_id": "mobilenetv2_140.ra_in1k",
    },
}

_BLOCK_00 = {
    "blocks.0.0.batchnorm.2": "blocks.0.0.bn1",
    "blocks.0.0.batchnorm.3": "blocks.0.0.bn2",
    "blocks.0.0.conv.pwl": "blocks.0.0.conv_pw",
}

_BASE_MAPPINGS = {
    "stem.conv": "conv_stem",
    "stem.batchnorm": "bn1",
    "head.conv": "conv_head",
    "head.batchnorm": "bn2",
    "batchnorm.1": "bn1",
    "batchnorm.2": "bn2",
    "batchnorm.3": "bn3",
    "conv.pw": "conv_pw",
    "dwconv": "conv_dw",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "classifier",
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {**_BLOCK_00, **_BASE_MAPPINGS}


def transfer_mobilenetv2_weights(
    keras_model, state_dict: Dict[str, np.ndarray]
) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_weight_name = re.sub(
            r"blocks_(\d+)_(\d+)_",
            lambda m: f"blocks.{m.group(1)}.{m.group(2)}.",
            keras_weight_name,
        ).replace("_", ".")
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

    for variant, meta in MOBILENETV2_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = MobileNetV2ImageClassify(
            **MOBILENETV2_MODEL_CONFIG[meta["model"]], include_normalization=False
        )
        transfer_mobilenetv2_weights(keras_model, state)

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
