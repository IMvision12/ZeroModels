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
    transfer_attention_weights,
    transfer_weights,
)
from zeromodels.models.vit import ViTImageClassify

# Architecture presets, moved here from vit_config.py: the package config no
# longer carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer timm weights into.
VIT_MODEL_CONFIG = {
    "vit_tiny_patch16_224": {
        "patch_size": 16,
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "vit_tiny_patch16_384": {
        "patch_size": 16,
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "image_size": 384,
        "num_classes": 1000,
    },
    "vit_tiny_patch16_224_in21k": {
        "patch_size": 16,
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "image_size": 224,
        "num_classes": 21843,
    },
    "vit_small_patch16_224": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "vit_small_patch16_384": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "image_size": 384,
        "num_classes": 1000,
    },
    "vit_small_patch16_224_in21k": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "image_size": 224,
        "num_classes": 21843,
    },
    "vit_small_patch32_224": {
        "patch_size": 32,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "vit_small_patch32_384": {
        "patch_size": 32,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "image_size": 384,
        "num_classes": 1000,
    },
    "vit_small_patch32_224_in21k": {
        "patch_size": 32,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "image_size": 224,
        "num_classes": 21843,
    },
    "vit_base_patch16_224": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "vit_base_patch16_384": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "image_size": 384,
        "num_classes": 1000,
    },
    "vit_base_patch16_224_in21k": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "image_size": 224,
        "num_classes": 21843,
    },
    "vit_base_patch32_224": {
        "patch_size": 32,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "vit_base_patch32_384": {
        "patch_size": 32,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "image_size": 384,
        "num_classes": 1000,
    },
    "vit_base_patch32_224_in21k": {
        "patch_size": 32,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "image_size": 224,
        "num_classes": 21843,
    },
    "vit_large_patch16_224": {
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "vit_large_patch16_384": {
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "image_size": 384,
        "num_classes": 1000,
    },
    "vit_large_patch16_224_in21k": {
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "image_size": 224,
        "num_classes": 21843,
    },
    "vit_large_patch32_384": {
        "patch_size": 32,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_norm": False,
        "image_size": 384,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (zm_config.json); the github release urls have been removed.
VIT_VARIANTS = {
    "vit_tiny_patch16_224_augreg_in21k_ft_in1k": {
        "model": "vit_tiny_patch16_224",
        "timm_id": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    },
    "vit_tiny_patch16_384_augreg_in21k_ft_in1k": {
        "model": "vit_tiny_patch16_384",
        "timm_id": "vit_tiny_patch16_384.augreg_in21k_ft_in1k",
    },
    "vit_tiny_patch16_224_augreg_in21k": {
        "model": "vit_tiny_patch16_224_in21k",
        "timm_id": "vit_tiny_patch16_224.augreg_in21k",
    },
    "vit_small_patch16_224_augreg_in21k_ft_in1k": {
        "model": "vit_small_patch16_224",
        "timm_id": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    },
    "vit_small_patch16_384_augreg_in21k_ft_in1k": {
        "model": "vit_small_patch16_384",
        "timm_id": "vit_small_patch16_384.augreg_in21k_ft_in1k",
    },
    "vit_small_patch16_224_augreg_in1k": {
        "model": "vit_small_patch16_224",
        "timm_id": "vit_small_patch16_224.augreg_in1k",
    },
    "vit_small_patch16_384_augreg_in1k": {
        "model": "vit_small_patch16_384",
        "timm_id": "vit_small_patch16_384.augreg_in1k",
    },
    "vit_small_patch16_224_augreg_in21k": {
        "model": "vit_small_patch16_224_in21k",
        "timm_id": "vit_small_patch16_224.augreg_in21k",
    },
    "vit_small_patch32_224_augreg_in21k_ft_in1k": {
        "model": "vit_small_patch32_224",
        "timm_id": "vit_small_patch32_224.augreg_in21k_ft_in1k",
    },
    "vit_small_patch32_384_augreg_in21k_ft_in1k": {
        "model": "vit_small_patch32_384",
        "timm_id": "vit_small_patch32_384.augreg_in21k_ft_in1k",
    },
    "vit_small_patch32_224_augreg_in21k": {
        "model": "vit_small_patch32_224_in21k",
        "timm_id": "vit_small_patch32_224.augreg_in21k",
    },
    "vit_base_patch16_224_augreg_in21k_ft_in1k": {
        "model": "vit_base_patch16_224",
        "timm_id": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    },
    "vit_base_patch16_384_augreg_in21k_ft_in1k": {
        "model": "vit_base_patch16_384",
        "timm_id": "vit_base_patch16_384.augreg_in21k_ft_in1k",
    },
    "vit_base_patch16_224_orig_in21k_ft_in1k": {
        "model": "vit_base_patch16_224",
        "timm_id": "vit_base_patch16_224.orig_in21k_ft_in1k",
    },
    "vit_base_patch16_384_orig_in21k_ft_in1k": {
        "model": "vit_base_patch16_384",
        "timm_id": "vit_base_patch16_384.orig_in21k_ft_in1k",
    },
    "vit_base_patch16_224_augreg_in1k": {
        "model": "vit_base_patch16_224",
        "timm_id": "vit_base_patch16_224.augreg_in1k",
    },
    "vit_base_patch16_384_augreg_in1k": {
        "model": "vit_base_patch16_384",
        "timm_id": "vit_base_patch16_384.augreg_in1k",
    },
    "vit_base_patch16_224_augreg_in21k": {
        "model": "vit_base_patch16_224_in21k",
        "timm_id": "vit_base_patch16_224.augreg_in21k",
    },
    "vit_base_patch32_224_augreg_in21k_ft_in1k": {
        "model": "vit_base_patch32_224",
        "timm_id": "vit_base_patch32_224.augreg_in21k_ft_in1k",
    },
    "vit_base_patch32_384_augreg_in21k_ft_in1k": {
        "model": "vit_base_patch32_384",
        "timm_id": "vit_base_patch32_384.augreg_in21k_ft_in1k",
    },
    "vit_base_patch32_224_augreg_in1k": {
        "model": "vit_base_patch32_224",
        "timm_id": "vit_base_patch32_224.augreg_in1k",
    },
    "vit_base_patch32_384_augreg_in1k": {
        "model": "vit_base_patch32_384",
        "timm_id": "vit_base_patch32_384.augreg_in1k",
    },
    "vit_base_patch32_224_augreg_in21k": {
        "model": "vit_base_patch32_224_in21k",
        "timm_id": "vit_base_patch32_224.augreg_in21k",
    },
    "vit_large_patch16_224_augreg_in21k_ft_in1k": {
        "model": "vit_large_patch16_224",
        "timm_id": "vit_large_patch16_224.augreg_in21k_ft_in1k",
    },
    "vit_large_patch16_384_augreg_in21k_ft_in1k": {
        "model": "vit_large_patch16_384",
        "timm_id": "vit_large_patch16_384.augreg_in21k_ft_in1k",
    },
    "vit_large_patch16_224_augreg_in21k": {
        "model": "vit_large_patch16_224_in21k",
        "timm_id": "vit_large_patch16_224.augreg_in21k",
    },
    "vit_large_patch32_384_orig_in21k_ft_in1k": {
        "model": "vit_large_patch32_384",
        "timm_id": "vit_large_patch32_384.orig_in21k_ft_in1k",
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "conv1": "patch_embed.proj",
    "pos.embed.pos.embed": "pos_embed",
    "cls.token.cls.token": "cls_token",
    "dense.1": "mlp.fc1",
    "dense.2": "mlp.fc2",
    "layernorm.1": "norm1",
    "layernorm.2": "norm2",
    "final.layernorm": "norm",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "head",
}


def transfer_vit_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)
        torch_weight_name = re.sub(
            r"pos_embed_variable_\d+$", "pos_embed", torch_weight_name
        )
        torch_weight_name = re.sub(
            r"cls_token_variable_\d+$", "cls_token", torch_weight_name
        )

        if "attention" in torch_weight_name:
            transfer_attention_weights(keras_weight_name, keras_weight, state_dict)
            continue

        if torch_weight_name not in state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = state_dict[torch_weight_name]

        if torch_weight_name == "cls_token":
            keras_weight.assign(torch_weight)
            continue

        if torch_weight_name == "pos_embed":
            if torch_weight.shape[1] == keras_weight.shape[1] + 1:
                torch_weight = torch_weight[:, 1:, :]
            keras_weight.assign(torch_weight)
            continue

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

    for variant, meta in VIT_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = ViTImageClassify(
            **VIT_MODEL_CONFIG[meta["model"]], include_normalization=False
        )
        transfer_vit_weights(keras_model, state)

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
