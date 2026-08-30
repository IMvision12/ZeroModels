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
from zeromodels.models.deit import DeiTImageClassify

# Architecture presets, moved here from deit_config.py: the package config no
# longer carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer timm weights into.
DEIT_MODEL_CONFIG = {
    "deit_tiny_patch16_224": {
        "patch_size": 16,
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "image_size": 224,
        "num_classes": 1000,
    },
    "deit_small_patch16_224": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "image_size": 224,
        "num_classes": 1000,
    },
    "deit_base_patch16_224": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "image_size": 224,
        "num_classes": 1000,
    },
    "deit_base_patch16_384": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "image_size": 384,
        "num_classes": 1000,
    },
    "deit_tiny_distilled_patch16_224": {
        "patch_size": 16,
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "use_distillation": True,
        "image_size": 224,
        "num_classes": 1000,
    },
    "deit_small_distilled_patch16_224": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "use_distillation": True,
        "image_size": 224,
        "num_classes": 1000,
    },
    "deit_base_distilled_patch16_224": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "use_distillation": True,
        "image_size": 224,
        "num_classes": 1000,
    },
    "deit_base_distilled_patch16_384": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "use_distillation": True,
        "image_size": 384,
        "num_classes": 1000,
    },
    "deit3_small_patch16_224": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "no_embed_class": True,
        "layer_scale_init": 1e-6,
        "image_size": 224,
        "num_classes": 1000,
    },
    "deit3_small_patch16_384": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "no_embed_class": True,
        "layer_scale_init": 1e-6,
        "image_size": 384,
        "num_classes": 1000,
    },
    "deit3_medium_patch16_224": {
        "patch_size": 16,
        "embed_dim": 512,
        "depth": 12,
        "num_heads": 8,
        "no_embed_class": True,
        "layer_scale_init": 1e-6,
        "image_size": 224,
        "num_classes": 1000,
    },
    "deit3_base_patch16_224": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "no_embed_class": True,
        "layer_scale_init": 1e-6,
        "image_size": 224,
        "num_classes": 1000,
    },
    "deit3_base_patch16_384": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "no_embed_class": True,
        "layer_scale_init": 1e-6,
        "image_size": 384,
        "num_classes": 1000,
    },
    "deit3_large_patch16_224": {
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "no_embed_class": True,
        "layer_scale_init": 1e-6,
        "image_size": 224,
        "num_classes": 1000,
    },
    "deit3_large_patch16_384": {
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "no_embed_class": True,
        "layer_scale_init": 1e-6,
        "image_size": 384,
        "num_classes": 1000,
    },
    "deit3_huge_patch14_224": {
        "patch_size": 14,
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 16,
        "no_embed_class": True,
        "layer_scale_init": 1e-6,
        "image_size": 224,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (zm_config.json); the github release urls have been removed.
DEIT_VARIANTS = {
    "deit_tiny_patch16_224_fb_in1k": {
        "model": "deit_tiny_patch16_224",
        "timm_id": "deit_tiny_patch16_224.fb_in1k",
    },
    "deit_small_patch16_224_fb_in1k": {
        "model": "deit_small_patch16_224",
        "timm_id": "deit_small_patch16_224.fb_in1k",
    },
    "deit_base_patch16_224_fb_in1k": {
        "model": "deit_base_patch16_224",
        "timm_id": "deit_base_patch16_224.fb_in1k",
    },
    "deit_base_patch16_384_fb_in1k": {
        "model": "deit_base_patch16_384",
        "timm_id": "deit_base_patch16_384.fb_in1k",
    },
    "deit_tiny_distilled_patch16_224_fb_in1k": {
        "model": "deit_tiny_distilled_patch16_224",
        "timm_id": "deit_tiny_distilled_patch16_224.fb_in1k",
    },
    "deit_small_distilled_patch16_224_fb_in1k": {
        "model": "deit_small_distilled_patch16_224",
        "timm_id": "deit_small_distilled_patch16_224.fb_in1k",
    },
    "deit_base_distilled_patch16_224_fb_in1k": {
        "model": "deit_base_distilled_patch16_224",
        "timm_id": "deit_base_distilled_patch16_224.fb_in1k",
    },
    "deit_base_distilled_patch16_384_fb_in1k": {
        "model": "deit_base_distilled_patch16_384",
        "timm_id": "deit_base_distilled_patch16_384.fb_in1k",
    },
    "deit3_small_patch16_224_fb_in1k": {
        "model": "deit3_small_patch16_224",
        "timm_id": "deit3_small_patch16_224.fb_in1k",
    },
    "deit3_small_patch16_384_fb_in1k": {
        "model": "deit3_small_patch16_384",
        "timm_id": "deit3_small_patch16_384.fb_in1k",
    },
    "deit3_small_patch16_224_fb_in22k_ft_in1k": {
        "model": "deit3_small_patch16_224",
        "timm_id": "deit3_small_patch16_224.fb_in22k_ft_in1k",
    },
    "deit3_small_patch16_384_fb_in22k_ft_in1k": {
        "model": "deit3_small_patch16_384",
        "timm_id": "deit3_small_patch16_384.fb_in22k_ft_in1k",
    },
    "deit3_medium_patch16_224_fb_in1k": {
        "model": "deit3_medium_patch16_224",
        "timm_id": "deit3_medium_patch16_224.fb_in1k",
    },
    "deit3_medium_patch16_224_fb_in22k_ft_in1k": {
        "model": "deit3_medium_patch16_224",
        "timm_id": "deit3_medium_patch16_224.fb_in22k_ft_in1k",
    },
    "deit3_base_patch16_224_fb_in1k": {
        "model": "deit3_base_patch16_224",
        "timm_id": "deit3_base_patch16_224.fb_in1k",
    },
    "deit3_base_patch16_384_fb_in1k": {
        "model": "deit3_base_patch16_384",
        "timm_id": "deit3_base_patch16_384.fb_in1k",
    },
    "deit3_base_patch16_224_fb_in22k_ft_in1k": {
        "model": "deit3_base_patch16_224",
        "timm_id": "deit3_base_patch16_224.fb_in22k_ft_in1k",
    },
    "deit3_base_patch16_384_fb_in22k_ft_in1k": {
        "model": "deit3_base_patch16_384",
        "timm_id": "deit3_base_patch16_384.fb_in22k_ft_in1k",
    },
    "deit3_large_patch16_224_fb_in1k": {
        "model": "deit3_large_patch16_224",
        "timm_id": "deit3_large_patch16_224.fb_in1k",
    },
    "deit3_large_patch16_384_fb_in1k": {
        "model": "deit3_large_patch16_384",
        "timm_id": "deit3_large_patch16_384.fb_in1k",
    },
    "deit3_large_patch16_224_fb_in22k_ft_in1k": {
        "model": "deit3_large_patch16_224",
        "timm_id": "deit3_large_patch16_224.fb_in22k_ft_in1k",
    },
    "deit3_large_patch16_384_fb_in22k_ft_in1k": {
        "model": "deit3_large_patch16_384",
        "timm_id": "deit3_large_patch16_384.fb_in22k_ft_in1k",
    },
    "deit3_huge_patch14_224_fb_in1k": {
        "model": "deit3_huge_patch14_224",
        "timm_id": "deit3_huge_patch14_224.fb_in1k",
    },
    "deit3_huge_patch14_224_fb_in22k_ft_in1k": {
        "model": "deit3_huge_patch14_224",
        "timm_id": "deit3_huge_patch14_224.fb_in22k_ft_in1k",
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "conv1": "patch_embed.proj",
    "pos.embed.pos.embed": "pos_embed",
    "cls.token.cls.token": "cls_token",
    "cls.token.dist.token": "dist_token",
    "layerscale.1": "ls1",
    "layerscale.2": "ls2",
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
    "head.dist": "head_dist",
}


def transfer_deit_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
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
        torch_weight_name = re.sub(
            r"\.variable(?:[\._]\d+)?$", ".gamma", torch_weight_name
        )

        if "attention" in torch_weight_name:
            transfer_attention_weights(keras_weight_name, keras_weight, state_dict)
            continue

        if torch_weight_name not in state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = state_dict[torch_weight_name]

        if torch_weight_name in ("cls_token", "dist_token"):
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

    for variant, meta in DEIT_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = DeiTImageClassify(**DEIT_MODEL_CONFIG[meta["model"]])
        transfer_deit_weights(keras_model, state)

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
