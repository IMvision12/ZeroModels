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
from zeromodels.models.res2net import Res2NetImageClassify

# Architecture presets, moved here from res2net_config.py: the package config no
# longer carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer timm weights into.
RES2NET_MODEL_CONFIG = {
    "res2net50_26w_4s": {
        "depth": [3, 4, 6, 3],
        "base_width": 26,
        "scale": 4,
        "cardinality": 1,
    },
    "res2net101_26w_4s": {
        "depth": [3, 4, 23, 3],
        "base_width": 26,
        "scale": 4,
        "cardinality": 1,
    },
    "res2net50_26w_6s": {
        "depth": [3, 4, 6, 3],
        "base_width": 26,
        "scale": 6,
        "cardinality": 1,
    },
    "res2net50_26w_8s": {
        "depth": [3, 4, 6, 3],
        "base_width": 26,
        "scale": 8,
        "cardinality": 1,
    },
    "res2net50_48w_2s": {
        "depth": [3, 4, 6, 3],
        "base_width": 48,
        "scale": 2,
        "cardinality": 1,
    },
    "res2net50_14w_8s": {
        "depth": [3, 4, 6, 3],
        "base_width": 14,
        "scale": 8,
        "cardinality": 1,
    },
    "res2next50": {
        "depth": [3, 4, 6, 3],
        "base_width": 4,
        "scale": 4,
        "cardinality": 8,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (zm_config.json); the github release urls have been removed.
RES2NET_VARIANTS = {
    "res2net50_26w_4s_in1k": {
        "model": "res2net50_26w_4s",
        "timm_id": "res2net50_26w_4s.in1k",
    },
    "res2net101_26w_4s_in1k": {
        "model": "res2net101_26w_4s",
        "timm_id": "res2net101_26w_4s.in1k",
    },
    "res2net50_26w_6s_in1k": {
        "model": "res2net50_26w_6s",
        "timm_id": "res2net50_26w_6s.in1k",
    },
    "res2net50_26w_8s_in1k": {
        "model": "res2net50_26w_8s",
        "timm_id": "res2net50_26w_8s.in1k",
    },
    "res2net50_48w_2s_in1k": {
        "model": "res2net50_48w_2s",
        "timm_id": "res2net50_48w_2s.in1k",
    },
    "res2net50_14w_8s_in1k": {
        "model": "res2net50_14w_8s",
        "timm_id": "res2net50_14w_8s.in1k",
    },
    "res2next50_in1k": {"model": "res2next50", "timm_id": "res2next50.in1k"},
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "batchnorm.1": "bn1",
    "batchnorm.s": "bns",
    "batchnorm.3": "bn3",
    "conv.1": "conv1",
    "conv.3": "conv3",
    "conv.s": "convs",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "fc",
}


def transfer_res2net_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
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

    for variant, meta in RES2NET_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = Res2NetImageClassify(**RES2NET_MODEL_CONFIG[meta["model"]])
        transfer_res2net_weights(keras_model, state)

        torch_model = timm.create_model(timm_id, pretrained=True).eval()
        results = verify_cls_model_equivalence(
            model_a=torch_model,
            model_b=keras_model,
            input_shape=keras_model.input_shape[1:],
            output_specs={"num_classes": keras_model.output_shape[-1]},
            comparison_type="torch_to_keras",
            run_performance=False,
            # Res2Net's deep conv/BN stack with multi-scale split blocks
            # accumulates more float error than a plain net, so the logits match
            # to ~3e-4 rather than ~1e-5; 1e-3 still catches any real mapping error.
            atol=1e-3,
            rtol=1e-3,
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
