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
from zeromodels.models.resmlp import ResMLPImageClassify

# Architecture presets, moved here from resmlp_config.py: the package config no
# longer carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer timm weights into.
RESMLP_MODEL_CONFIG = {
    "resmlp_12": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "mlp_ratio": 4,
        "layer_scale_init": 1e-4,
        "image_size": 224,
        "num_classes": 1000,
    },
    "resmlp_24": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 24,
        "mlp_ratio": 4,
        "layer_scale_init": 1e-5,
        "image_size": 224,
        "num_classes": 1000,
    },
    "resmlp_36": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 36,
        "mlp_ratio": 4,
        "layer_scale_init": 1e-6,
        "image_size": 224,
        "num_classes": 1000,
    },
    "resmlp_big_24": {
        "patch_size": 8,
        "embed_dim": 768,
        "depth": 24,
        "mlp_ratio": 4,
        "layer_scale_init": 1e-6,
        "image_size": 224,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (zm_config.json); the github release urls have been removed.
RESMLP_VARIANTS = {
    "resmlp_12_224_fb_in1k": {
        "model": "resmlp_12",
        "timm_id": "resmlp_12_224.fb_in1k",
    },
    "resmlp_12_224_fb_distilled_in1k": {
        "model": "resmlp_12",
        "timm_id": "resmlp_12_224.fb_distilled_in1k",
    },
    "resmlp_24_224_fb_in1k": {
        "model": "resmlp_24",
        "timm_id": "resmlp_24_224.fb_in1k",
    },
    "resmlp_24_224_fb_distilled_in1k": {
        "model": "resmlp_24",
        "timm_id": "resmlp_24_224.fb_distilled_in1k",
    },
    "resmlp_36_224_fb_in1k": {
        "model": "resmlp_36",
        "timm_id": "resmlp_36_224.fb_in1k",
    },
    "resmlp_36_224_fb_distilled_in1k": {
        "model": "resmlp_36",
        "timm_id": "resmlp_36_224.fb_distilled_in1k",
    },
    "resmlp_big_24_224_fb_in1k": {
        "model": "resmlp_big_24",
        "timm_id": "resmlp_big_24_224.fb_in1k",
    },
    "resmlp_big_24_224_fb_distilled_in1k": {
        "model": "resmlp_big_24",
        "timm_id": "resmlp_big_24_224.fb_distilled_in1k",
    },
    "resmlp_big_24_224_fb_in22k_ft_in1k": {
        "model": "resmlp_big_24",
        "timm_id": "resmlp_big_24_224.fb_in22k_ft_in1k",
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "stem.conv": "stem.proj",
    "affine.1.alpha": "norm1.alpha",
    "affine.1.beta": "norm1.beta",
    "affine.2.alpha": "norm2.alpha",
    "affine.2.beta": "norm2.beta",
    "dense.1": "linear_tokens",
    "dense.2": "mlp_channels.fc1",
    "dense.3": "mlp_channels.fc2",
    "kernel": "weight",
    "gamma": "weight",
    "Final.affine": "norm",
    "predictions": "head",
}


def transfer_resmlp_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        torch_weight_name = re.sub(
            r"scale\.(\d+)\.variable(?:\.\d+)?", r"ls\1", torch_weight_name
        )

        if "affine" in keras_weight_name and (
            "alpha" in keras_weight_name or "beta" in keras_weight_name
        ):
            if torch_weight_name not in state_dict:
                raise WeightMappingError(keras_weight_name, torch_weight_name)
            torch_weight = state_dict[torch_weight_name]
            reshaped_weight = torch_weight.reshape(1, 1, -1)
            keras_weight.assign(reshaped_weight)
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

    for variant, meta in RESMLP_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = ResMLPImageClassify(
            **RESMLP_MODEL_CONFIG[meta["model"]], include_normalization=False
        )
        transfer_resmlp_weights(keras_model, state)

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
