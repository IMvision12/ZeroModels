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
from zeromodels.models.pit import PiTImageClassify

# Architecture presets, moved here from pit_config.py: the package config no
# longer carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer timm weights into.
PIT_MODEL_CONFIG = {
    "pit_xs": {
        "patch_size": 16,
        "stride": 8,
        "embed_dim": [96, 192, 384],
        "depth": [2, 6, 4],
        "heads": [2, 4, 8],
        "mlp_ratio": 4,
        "distilled": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "pit_xs_distilled": {
        "patch_size": 16,
        "stride": 8,
        "embed_dim": [96, 192, 384],
        "depth": [2, 6, 4],
        "heads": [2, 4, 8],
        "mlp_ratio": 4,
        "distilled": True,
        "image_size": 224,
        "num_classes": 1000,
    },
    "pit_ti": {
        "patch_size": 16,
        "stride": 8,
        "embed_dim": [64, 128, 256],
        "depth": [2, 6, 4],
        "heads": [2, 4, 8],
        "mlp_ratio": 4,
        "distilled": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "pit_ti_distilled": {
        "patch_size": 16,
        "stride": 8,
        "embed_dim": [64, 128, 256],
        "depth": [2, 6, 4],
        "heads": [2, 4, 8],
        "mlp_ratio": 4,
        "distilled": True,
        "image_size": 224,
        "num_classes": 1000,
    },
    "pit_s": {
        "patch_size": 16,
        "stride": 8,
        "embed_dim": [144, 288, 576],
        "depth": [2, 6, 4],
        "heads": [3, 6, 12],
        "mlp_ratio": 4,
        "distilled": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "pit_s_distilled": {
        "patch_size": 16,
        "stride": 8,
        "embed_dim": [144, 288, 576],
        "depth": [2, 6, 4],
        "heads": [3, 6, 12],
        "mlp_ratio": 4,
        "distilled": True,
        "image_size": 224,
        "num_classes": 1000,
    },
    "pit_b": {
        "patch_size": 14,
        "stride": 7,
        "embed_dim": [256, 512, 1024],
        "depth": [3, 6, 4],
        "heads": [4, 8, 16],
        "mlp_ratio": 4,
        "distilled": False,
        "image_size": 224,
        "num_classes": 1000,
    },
    "pit_b_distilled": {
        "patch_size": 14,
        "stride": 7,
        "embed_dim": [256, 512, 1024],
        "depth": [3, 6, 4],
        "heads": [4, 8, 16],
        "mlp_ratio": 4,
        "distilled": True,
        "image_size": 224,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (zm_config.json); the github release urls have been removed.
PIT_VARIANTS = {
    "pit_xs_224_in1k": {
        "model": "pit_xs",
        "timm_id": "pit_xs_224.in1k",
    },
    "pit_xs_distilled_224_in1k": {
        "model": "pit_xs_distilled",
        "timm_id": "pit_xs_distilled_224.in1k",
    },
    "pit_ti_224_in1k": {
        "model": "pit_ti",
        "timm_id": "pit_ti_224.in1k",
    },
    "pit_ti_distilled_224_in1k": {
        "model": "pit_ti_distilled",
        "timm_id": "pit_ti_distilled_224.in1k",
    },
    "pit_s_224_in1k": {
        "model": "pit_s",
        "timm_id": "pit_s_224.in1k",
    },
    "pit_s_distilled_224_in1k": {
        "model": "pit_s_distilled",
        "timm_id": "pit_s_distilled_224.in1k",
    },
    "pit_b_224_in1k": {
        "model": "pit_b",
        "timm_id": "pit_b_224.in1k",
    },
    "pit_b_distilled_224_in1k": {
        "model": "pit_b_distilled",
        "timm_id": "pit_b_distilled_224.in1k",
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "pit": "transformers",
    "patch.embed": "patch_embed",
    "pos.embed.pos.embed": "pos_embed",
    "class.dist.token.cls.token": "cls_token",
    "dense.1": "mlp.fc1",
    "dense.2": "mlp.fc2",
    "layernorm.1": "norm1",
    "layernorm.2": "norm2",
    "layerscale.1": "ls1",
    "layerscale.2": "ls2",
    "pool.dense": "pool.fc",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "head",
    "head.dist": "head_dist",
}


def transfer_pit_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
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
            if torch_weight.ndim == 4:
                # timm stores pos_embed as (1, C, H, W); flatten to (1, H*W, C).
                _, c, h, w = torch_weight.shape
                torch_weight = torch_weight.reshape(1, c, h * w).transpose(0, 2, 1)
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

    for variant, meta in PIT_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = PiTImageClassify(**PIT_MODEL_CONFIG[meta["model"]])
        transfer_pit_weights(keras_model, state)

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
