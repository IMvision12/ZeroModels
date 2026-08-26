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
from zeromodels.models.cait import CaiTImageClassify

# Architecture presets, moved here from cait_config.py: the package config no
# longer carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer timm weights into.
CAIT_MODEL_CONFIG = {
    "cait_xxs24_224": {
        "patch_size": 16,
        "embed_dim": 192,
        "depth": 24,
        "num_heads": 4,
        "image_size": 224,
        "num_classes": 1000,
    },
    "cait_xxs24_384": {
        "patch_size": 16,
        "embed_dim": 192,
        "depth": 24,
        "num_heads": 4,
        "image_size": 384,
        "num_classes": 1000,
    },
    "cait_xxs36_224": {
        "patch_size": 16,
        "embed_dim": 192,
        "depth": 36,
        "num_heads": 4,
        "image_size": 224,
        "num_classes": 1000,
    },
    "cait_xxs36_384": {
        "patch_size": 16,
        "embed_dim": 192,
        "depth": 36,
        "num_heads": 4,
        "image_size": 384,
        "num_classes": 1000,
    },
    "cait_xs24_384": {
        "patch_size": 16,
        "embed_dim": 288,
        "depth": 24,
        "num_heads": 6,
        "image_size": 384,
        "num_classes": 1000,
    },
    "cait_s24_224": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 24,
        "num_heads": 8,
        "image_size": 224,
        "num_classes": 1000,
    },
    "cait_s24_384": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 24,
        "num_heads": 8,
        "image_size": 384,
        "num_classes": 1000,
    },
    "cait_s36_384": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 36,
        "num_heads": 8,
        "image_size": 384,
        "num_classes": 1000,
    },
    "cait_m36_384": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 36,
        "num_heads": 16,
        "image_size": 384,
        "num_classes": 1000,
    },
    "cait_m48_448": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 48,
        "num_heads": 16,
        "image_size": 448,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (zm_config.json); the github release urls have been removed.
CAIT_VARIANTS = {
    "cait_xxs24_224_fb_dist_in1k": {
        "model": "cait_xxs24_224",
        "timm_id": "cait_xxs24_224.fb_dist_in1k",
    },
    "cait_xxs24_384_fb_dist_in1k": {
        "model": "cait_xxs24_384",
        "timm_id": "cait_xxs24_384.fb_dist_in1k",
    },
    "cait_xxs36_224_fb_dist_in1k": {
        "model": "cait_xxs36_224",
        "timm_id": "cait_xxs36_224.fb_dist_in1k",
    },
    "cait_xxs36_384_fb_dist_in1k": {
        "model": "cait_xxs36_384",
        "timm_id": "cait_xxs36_384.fb_dist_in1k",
    },
    "cait_xs24_384_fb_dist_in1k": {
        "model": "cait_xs24_384",
        "timm_id": "cait_xs24_384.fb_dist_in1k",
    },
    "cait_s24_224_fb_dist_in1k": {
        "model": "cait_s24_224",
        "timm_id": "cait_s24_224.fb_dist_in1k",
    },
    "cait_s24_384_fb_dist_in1k": {
        "model": "cait_s24_384",
        "timm_id": "cait_s24_384.fb_dist_in1k",
    },
    "cait_s36_384_fb_dist_in1k": {
        "model": "cait_s36_384",
        "timm_id": "cait_s36_384.fb_dist_in1k",
    },
    "cait_m36_384_fb_dist_in1k": {
        "model": "cait_m36_384",
        "timm_id": "cait_m36_384.fb_dist_in1k",
    },
    "cait_m48_448_fb_dist_in1k": {
        "model": "cait_m48_448",
        "timm_id": "cait_m48_448.fb_dist_in1k",
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "stem.conv": "patch_embed.proj",
    "cls.token.cls.token": "cls_token",
    "pos.embed.pos.embed": "pos_embed",
    "layernorm.": "norm",
    "dense.1": "fc1",
    "dense.2": "fc2",
    "blocks.token.only": "blocks_token_only",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving_mean": "running_mean",
    "moving_variance": "running_var",
    "final.norm": "norm.",
    "predictions": "head",
}

_ATTN_REPLACEMENT: Dict[str, str] = {
    "proj.l": "proj_l",
    "proj.w": "proj_w",
    "blocks.token.only": "blocks_token_only",
}


def transfer_cait_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)
        torch_weight_name = re.sub(
            r"layerscale\.(\d+)\.variable(?:\.\d+)?", r"gamma_\1", torch_weight_name
        )

        if "attention" in torch_weight_name:
            transfer_attention_weights(
                keras_weight_name, keras_weight, state_dict, _ATTN_REPLACEMENT
            )
            continue

        if torch_weight_name not in state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = state_dict[torch_weight_name]

        if torch_weight_name in ("cls_token", "pos_embed"):
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

    for variant, meta in CAIT_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = CaiTImageClassify(
            **CAIT_MODEL_CONFIG[meta["model"]], include_normalization=False
        )
        transfer_cait_weights(keras_model, state)

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
