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
    transfer_attention_weights,
    transfer_weights,
)
from zeromodels.models.nextvit import NextViTImageClassify as NextViT

# Architecture presets, moved here from nextvit_config.py: the package config no
# longer carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer timm weights into.
NEXTVIT_MODEL_CONFIG = {
    "nextvit_small": {
        "depths": [3, 4, 10, 3],
        "stem_chs": [64, 32, 64],
        "head_dim": 32,
        "mix_block_ratio": 0.75,
        "sr_ratios": [8, 4, 2, 1],
        "drop_path_rate": 0.1,
        "image_size": 224,
        "num_classes": 1000,
    },
    "nextvit_small_384": {
        "depths": [3, 4, 10, 3],
        "stem_chs": [64, 32, 64],
        "head_dim": 32,
        "mix_block_ratio": 0.75,
        "sr_ratios": [8, 4, 2, 1],
        "drop_path_rate": 0.1,
        "image_size": 384,
        "num_classes": 1000,
    },
    "nextvit_base": {
        "depths": [3, 4, 20, 3],
        "stem_chs": [64, 32, 64],
        "head_dim": 32,
        "mix_block_ratio": 0.75,
        "sr_ratios": [8, 4, 2, 1],
        "drop_path_rate": 0.1,
        "image_size": 224,
        "num_classes": 1000,
    },
    "nextvit_base_384": {
        "depths": [3, 4, 20, 3],
        "stem_chs": [64, 32, 64],
        "head_dim": 32,
        "mix_block_ratio": 0.75,
        "sr_ratios": [8, 4, 2, 1],
        "drop_path_rate": 0.1,
        "image_size": 384,
        "num_classes": 1000,
    },
    "nextvit_large": {
        "depths": [3, 4, 30, 3],
        "stem_chs": [64, 32, 64],
        "head_dim": 32,
        "mix_block_ratio": 0.75,
        "sr_ratios": [8, 4, 2, 1],
        "drop_path_rate": 0.1,
        "image_size": 224,
        "num_classes": 1000,
    },
    "nextvit_large_384": {
        "depths": [3, 4, 30, 3],
        "stem_chs": [64, 32, 64],
        "head_dim": 32,
        "mix_block_ratio": 0.75,
        "sr_ratios": [8, 4, 2, 1],
        "drop_path_rate": 0.1,
        "image_size": 384,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (zm_config.json); the github release urls have been removed.
NEXTVIT_VARIANTS = {
    "nextvit_small_bd_in1k": {
        "model": "nextvit_small",
        "timm_id": "nextvit_small.bd_in1k",
    },
    "nextvit_small_bd_in1k_384": {
        "model": "nextvit_small_384",
        "timm_id": "nextvit_small.bd_in1k_384",
    },
    "nextvit_small_bd_ssld_6m_in1k": {
        "model": "nextvit_small",
        "timm_id": "nextvit_small.bd_ssld_6m_in1k",
    },
    "nextvit_small_bd_ssld_6m_in1k_384": {
        "model": "nextvit_small_384",
        "timm_id": "nextvit_small.bd_ssld_6m_in1k_384",
    },
    "nextvit_base_bd_in1k": {
        "model": "nextvit_base",
        "timm_id": "nextvit_base.bd_in1k",
    },
    "nextvit_base_bd_in1k_384": {
        "model": "nextvit_base_384",
        "timm_id": "nextvit_base.bd_in1k_384",
    },
    "nextvit_base_bd_ssld_6m_in1k": {
        "model": "nextvit_base",
        "timm_id": "nextvit_base.bd_ssld_6m_in1k",
    },
    "nextvit_base_bd_ssld_6m_in1k_384": {
        "model": "nextvit_base_384",
        "timm_id": "nextvit_base.bd_ssld_6m_in1k_384",
    },
    "nextvit_large_bd_in1k": {
        "model": "nextvit_large",
        "timm_id": "nextvit_large.bd_in1k",
    },
    "nextvit_large_bd_in1k_384": {
        "model": "nextvit_large_384",
        "timm_id": "nextvit_large.bd_in1k_384",
    },
    "nextvit_large_bd_ssld_6m_in1k": {
        "model": "nextvit_large",
        "timm_id": "nextvit_large.bd_ssld_6m_in1k",
    },
    "nextvit_large_bd_ssld_6m_in1k_384": {
        "model": "nextvit_large_384",
        "timm_id": "nextvit_large.bd_ssld_6m_in1k_384",
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "e.mhsa": "e_mhsa",
    "group.conv3x3": "group_conv3x3",
    "patch.embed": "patch_embed",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}

_E_MHSA_NAME_REPLACEMENTS: Dict[str, str] = {"e.mhsa": "e_mhsa"}


def transfer_nextvit_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if "e_mhsa" in keras_weight_name:
            transfer_attention_weights(
                keras_weight_name,
                keras_weight,
                state_dict,
                name_replacements=_E_MHSA_NAME_REPLACEMENTS,
            )
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
        transfer_name = keras_weight_name
        if len(keras_weight.shape) == 4 and "conv" not in keras_weight_name.lower():
            transfer_name = "conv_" + keras_weight_name
        transfer_weights(transfer_name, keras_weight, torch_weight)


if __name__ == "__main__":
    import timm

    for variant, meta in NEXTVIT_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = NextViT(
            **NEXTVIT_MODEL_CONFIG[meta["model"]], include_normalization=False
        )
        transfer_nextvit_weights(keras_model, state)

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
