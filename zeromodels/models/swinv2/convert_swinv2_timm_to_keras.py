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
from zeromodels.models.swinv2 import SwinV2ImageClassify

# Architecture presets, moved here from swinv2_config.py: the package config no
# longer carries arch (models load by Hub repo id / kf_config). Only this converter
# builds an untrained model to transfer timm weights into.
SWINV2_MODEL_CONFIG = {
    "swinv2_tiny_window8_256": {
        "embed_dim": 96,
        "depths": (2, 2, 6, 2),
        "num_heads": (3, 6, 12, 24),
        "window_size": 8,
        "pretrain_size": 256,
        "pretrained_window_size": 0,
        "image_size": 256,
        "num_classes": 1000,
    },
    "swinv2_tiny_window16_256": {
        "embed_dim": 96,
        "depths": (2, 2, 6, 2),
        "num_heads": (3, 6, 12, 24),
        "window_size": 16,
        "pretrain_size": 256,
        "pretrained_window_size": 0,
        "image_size": 256,
        "num_classes": 1000,
    },
    "swinv2_small_window8_256": {
        "embed_dim": 96,
        "depths": (2, 2, 18, 2),
        "num_heads": (3, 6, 12, 24),
        "window_size": 8,
        "pretrain_size": 256,
        "pretrained_window_size": 0,
        "image_size": 256,
        "num_classes": 1000,
    },
    "swinv2_small_window16_256": {
        "embed_dim": 96,
        "depths": (2, 2, 18, 2),
        "num_heads": (3, 6, 12, 24),
        "window_size": 16,
        "pretrain_size": 256,
        "pretrained_window_size": 0,
        "image_size": 256,
        "num_classes": 1000,
    },
    "swinv2_base_window8_256": {
        "embed_dim": 128,
        "depths": (2, 2, 18, 2),
        "num_heads": (4, 8, 16, 32),
        "window_size": 8,
        "pretrain_size": 256,
        "pretrained_window_size": 0,
        "image_size": 256,
        "num_classes": 1000,
    },
    "swinv2_base_window12_192": {
        "embed_dim": 128,
        "depths": (2, 2, 18, 2),
        "num_heads": (4, 8, 16, 32),
        "window_size": 12,
        "pretrain_size": 192,
        "pretrained_window_size": 0,
        "image_size": 192,
        "num_classes": 21841,
    },
    "swinv2_base_window12to16_192to256": {
        "embed_dim": 128,
        "depths": (2, 2, 18, 2),
        "num_heads": (4, 8, 16, 32),
        "window_size": 16,
        "pretrain_size": 192,
        "pretrained_window_size": 12,
        "image_size": 256,
        "num_classes": 1000,
    },
    "swinv2_base_window12to24_192to384": {
        "embed_dim": 128,
        "depths": (2, 2, 18, 2),
        "num_heads": (4, 8, 16, 32),
        "window_size": 24,
        "pretrain_size": 192,
        "pretrained_window_size": 12,
        "image_size": 384,
        "num_classes": 1000,
    },
    "swinv2_base_window16_256": {
        "embed_dim": 128,
        "depths": (2, 2, 18, 2),
        "num_heads": (4, 8, 16, 32),
        "window_size": 16,
        "pretrain_size": 256,
        "pretrained_window_size": 0,
        "image_size": 256,
        "num_classes": 1000,
    },
    "swinv2_large_window12_192": {
        "embed_dim": 192,
        "depths": (2, 2, 18, 2),
        "num_heads": (6, 12, 24, 48),
        "window_size": 12,
        "pretrain_size": 192,
        "pretrained_window_size": 0,
        "image_size": 192,
        "num_classes": 21841,
    },
    "swinv2_large_window12to16_192to256": {
        "embed_dim": 192,
        "depths": (2, 2, 18, 2),
        "num_heads": (6, 12, 24, 48),
        "window_size": 16,
        "pretrain_size": 192,
        "pretrained_window_size": 12,
        "image_size": 256,
        "num_classes": 1000,
    },
    "swinv2_large_window12to24_192to384": {
        "embed_dim": 192,
        "depths": (2, 2, 18, 2),
        "num_heads": (6, 12, 24, 48),
        "window_size": 24,
        "pretrain_size": 192,
        "pretrained_window_size": 12,
        "image_size": 384,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (kf_config.json); the github release urls have been removed.
SWINV2_VARIANTS = {
    "swinv2_tiny_window8_256_ms_in1k": {
        "model": "swinv2_tiny_window8_256",
        "timm_id": "swinv2_tiny_window8_256.ms_in1k",
    },
    "swinv2_tiny_window16_256_ms_in1k": {
        "model": "swinv2_tiny_window16_256",
        "timm_id": "swinv2_tiny_window16_256.ms_in1k",
    },
    "swinv2_small_window8_256_ms_in1k": {
        "model": "swinv2_small_window8_256",
        "timm_id": "swinv2_small_window8_256.ms_in1k",
    },
    "swinv2_small_window16_256_ms_in1k": {
        "model": "swinv2_small_window16_256",
        "timm_id": "swinv2_small_window16_256.ms_in1k",
    },
    "swinv2_base_window8_256_ms_in1k": {
        "model": "swinv2_base_window8_256",
        "timm_id": "swinv2_base_window8_256.ms_in1k",
    },
    "swinv2_base_window12_192_ms_in22k": {
        "model": "swinv2_base_window12_192",
        "timm_id": "swinv2_base_window12_192.ms_in22k",
    },
    "swinv2_base_window12to16_192to256_ms_in22k_ft_in1k": {
        "model": "swinv2_base_window12to16_192to256",
        "timm_id": "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
    },
    "swinv2_base_window12to24_192to384_ms_in22k_ft_in1k": {
        "model": "swinv2_base_window12to24_192to384",
        "timm_id": "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
    },
    "swinv2_base_window16_256_ms_in1k": {
        "model": "swinv2_base_window16_256",
        "timm_id": "swinv2_base_window16_256.ms_in1k",
    },
    "swinv2_large_window12_192_ms_in22k": {
        "model": "swinv2_large_window12_192",
        "timm_id": "swinv2_large_window12_192.ms_in22k",
    },
    "swinv2_large_window12to16_192to256_ms_in22k_ft_in1k": {
        "model": "swinv2_large_window12to16_192to256",
        "timm_id": "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",
    },
    "swinv2_large_window12to24_192to384_ms_in22k_ft_in1k": {
        "model": "swinv2_large_window12to24_192to384",
        "timm_id": "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "moving_variance": "MOVVAR",
    "moving_mean": "MOVMEAN",
    "_": ".",
    "MOVVAR": "running_var",
    "MOVMEAN": "running_mean",
    "stem.conv": "patch_embed.proj",
    "stem.norm": "patch_embed.norm",
    "layernorm.1": "norm1",
    "layernorm.2": "norm2",
    "dense.1": "fc1",
    "dense.2": "fc2",
    "pm.layernorm": "norm",
    "pm.dense": "reduction",
    "final.norm": "norm",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "predictions": "head.fc",
}

_ATTN_REPLACEMENT: Dict[str, str] = {
    "cpb.mlp": "cpb_mlp",
}

_DIRECT_ATTN_WEIGHTS: Dict[str, str] = {
    "attn.logit.scale": "attn.logit_scale",
    "attn.q.bias": "attn.q_bias",
    "attn.v.bias": "attn.v_bias",
}

_SKIP_DIRECT_ATTN: tuple = (
    "attn.relative.coords.table",
    "attn.relative.position.index",
)


def transfer_swinv2_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        path_parts = keras_weight.path.split("/")

        if len(path_parts) == 2:
            flat = path_parts[-1].replace("_", ".")
            if any(skip in flat for skip in _SKIP_DIRECT_ATTN):
                continue
            matched = next((k for k in _DIRECT_ATTN_WEIGHTS if k in flat), None)
            if matched is not None:
                torch_name = flat.replace(matched, _DIRECT_ATTN_WEIGHTS[matched])
                if torch_name not in state_dict:
                    raise WeightMappingError(keras_weight_name, torch_name)
                value = np.asarray(state_dict[torch_name])
                if value.shape != tuple(keras_weight.shape):
                    value = value.reshape(tuple(keras_weight.shape))
                keras_weight.assign(value)
                continue

        if len(path_parts) >= 3 and "_attn_" in path_parts[-2]:
            transfer_attention_weights(
                keras_weight_name, keras_weight, state_dict, _ATTN_REPLACEMENT
            )
            continue

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

    for variant, meta in SWINV2_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = SwinV2ImageClassify(
            **SWINV2_MODEL_CONFIG[meta["model"]], include_normalization=False
        )
        transfer_swinv2_weights(keras_model, state)

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
