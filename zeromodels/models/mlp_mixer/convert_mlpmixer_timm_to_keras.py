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
from zeromodels.models.mlp_mixer import MLPMixerImageClassify

# Architecture presets, moved here from mlp_mixer_config.py: the package config no
# longer carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer timm weights into.
MLP_MIXER_MODEL_CONFIG = {
    "mixer_b16_224_in21k": {
        "patch_size": 16,
        "depths": 12,
        "embed_dim": 768,
        "mlp_ratio": (0.5, 4.0),
        "image_size": 224,
        "num_classes": 21843,
    },
    "mixer_b16_224": {
        "patch_size": 16,
        "depths": 12,
        "embed_dim": 768,
        "mlp_ratio": (0.5, 4.0),
        "image_size": 224,
        "num_classes": 1000,
    },
    "mixer_l16_224_in21k": {
        "patch_size": 16,
        "depths": 24,
        "embed_dim": 1024,
        "mlp_ratio": (0.5, 4.0),
        "image_size": 224,
        "num_classes": 21843,
    },
    "mixer_l16_224": {
        "patch_size": 16,
        "depths": 24,
        "embed_dim": 1024,
        "mlp_ratio": (0.5, 4.0),
        "image_size": 224,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (zm_config.json); the github release urls have been removed.
MLP_MIXER_VARIANTS = {
    "mixer_b16_224_goog_in21k": {
        "model": "mixer_b16_224_in21k",
        "timm_id": "mixer_b16_224.goog_in21k",
    },
    "mixer_b16_224_goog_in21k_ft_in1k": {
        "model": "mixer_b16_224",
        "timm_id": "mixer_b16_224.goog_in21k_ft_in1k",
    },
    "mixer_b16_224_miil_in21k_ft_in1k": {
        "model": "mixer_b16_224",
        "timm_id": "mixer_b16_224.miil_in21k_ft_in1k",
    },
    "mixer_l16_224_goog_in21k": {
        "model": "mixer_l16_224_in21k",
        "timm_id": "mixer_l16_224.goog_in21k",
    },
    "mixer_l16_224_goog_in21k_ft_in1k": {
        "model": "mixer_l16_224",
        "timm_id": "mixer_l16_224.goog_in21k_ft_in1k",
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "layernorm.1": "norm1",
    "layernorm.2": "norm2",
    "dense.1": "mlp_tokens.fc1",
    "dense.2": "mlp_tokens.fc2",
    "dense.3": "mlp_channels.fc1",
    "dense.4": "mlp_channels.fc2",
    "stem.conv": "stem.proj",
    "final.layernorm": "norm",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
    "predictions": "head",
}


def transfer_mlp_mixer_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
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

    for variant, meta in MLP_MIXER_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = MLPMixerImageClassify(**MLP_MIXER_MODEL_CONFIG[meta["model"]])
        transfer_mlp_mixer_weights(keras_model, state)

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
