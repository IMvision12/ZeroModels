import gc
import re
from typing import Dict, List, Tuple

import keras
import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from zeromodels.conversion.weight_split_util import split_model_weights
from zeromodels.conversion.weight_transfer_util import (
    compare_keras_torch_names,
    transfer_attention_weights,
    transfer_weights,
)
from zeromodels.models.dino import DinoResNetModel, DinoViTModel

VIT_NAME_MAPPING: Dict[str, str] = {
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
}

RESNET_NAME_MAPPING: Dict[str, str] = {
    "resnet_layer": "layer",
    "_": ".",
    "downsample.conv": "downsample.0",
    "downsample.batchnorm": "downsample.1",
    "batchnorm1": "bn1",
    "batchnorm2": "bn2",
    "batchnorm3": "bn3",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
}


def transfer_dino_weights(keras_model, torch_state_dict, name_mapping, is_vit):
    trainable, non_trainable = split_model_weights(keras_model)
    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring DINO weights"
    ):
        torch_weight_name: str = keras_weight_name
        for old, new in name_mapping.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if is_vit:
            torch_weight_name = re.sub(
                r"pos_embed_variable_\d+$", "pos_embed", torch_weight_name
            )
            torch_weight_name = re.sub(
                r"cls_token_variable_\d+$", "cls_token", torch_weight_name
            )

            if "attention" in torch_weight_name:
                transfer_attention_weights(
                    keras_weight_name, keras_weight, torch_state_dict
                )
                continue

        if torch_weight_name not in torch_state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = torch_state_dict[torch_weight_name]

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


def transfer_dino_vit_weights(
    keras_model: keras.Model, torch_state_dict: Dict[str, np.ndarray]
) -> None:
    transfer_dino_weights(keras_model, torch_state_dict, VIT_NAME_MAPPING, is_vit=True)


def transfer_dino_resnet_weights(
    keras_model: keras.Model, torch_state_dict: Dict[str, np.ndarray]
) -> None:
    transfer_dino_weights(
        keras_model, torch_state_dict, RESNET_NAME_MAPPING, is_vit=False
    )


DINO_VIT_CONVERSION_CONFIG: List[Tuple[str, str]] = [
    ("dino-vits16", "dino_vits16"),
    ("dino-vits8", "dino_vits8"),
    ("dino-vitb16", "dino_vitb16"),
    ("dino-vitb8", "dino_vitb8"),
]

DINO_RESNET_CONVERSION_CONFIG: List[Tuple[str, str]] = [
    ("dino-resnet50", "dino_resnet50"),
]

# Per-variant recipes (relocated from dino_config.py). Models load from the Hub
# by repo id; these build the arch for conversion + drive the kf_config backfill.
DINO_VIT_RECIPES = {
    "dino-vits16": {"patch_size": 16, "embed_dim": 384, "depth": 12, "num_heads": 6},
    "dino-vits8": {"patch_size": 8, "embed_dim": 384, "depth": 12, "num_heads": 6},
    "dino-vitb16": {"patch_size": 16, "embed_dim": 768, "depth": 12, "num_heads": 12},
    "dino-vitb8": {"patch_size": 8, "embed_dim": 768, "depth": 12, "num_heads": 12},
}

DINO_RESNET_RECIPES = {
    "dino-resnet50": {"depths": (3, 4, 6, 3), "filters": (64, 128, 256, 512)},
}


if __name__ == "__main__":
    import torch

    for variant, torch_hub_name in DINO_VIT_CONVERSION_CONFIG:
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  (torch.hub: {torch_hub_name})")
        print(f"{'=' * 60}")

        torch_model = torch.hub.load(
            "facebookresearch/dino:main", torch_hub_name, pretrained=True
        ).eval()
        trainable_torch, non_trainable_torch, _ = split_model_weights(torch_model)
        torch_state_dict = {
            k: v.cpu().numpy() if hasattr(v, "cpu") else v
            for k, v in {**trainable_torch, **non_trainable_torch}.items()
        }

        keras_model = DinoViTModel(
            **DINO_VIT_RECIPES[variant],
            image_size=224,
            include_normalization=False,
        )

        transfer_dino_vit_weights(keras_model, torch_state_dict)

        rng = np.random.default_rng(0)
        x = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
        with torch.no_grad():
            t_out = torch_model(torch.from_numpy(x)).cpu().numpy()
        k_in = np.transpose(x, (0, 2, 3, 1))
        last = keras_model(k_in, training=False)
        last = (
            last.detach().cpu().numpy() if hasattr(last, "detach") else np.asarray(last)
        )
        k_out = last[:, 0]
        diff = float(np.abs(k_out - t_out).max())
        if diff > 1e-3:
            raise ValueError(f"{variant}: max diff {diff:.2e}")
        print(f"  Verification OK (max diff = {diff:.2e})")

        model_filename = f"{variant}.weights.h5"
        keras_model.save_weights(model_filename)
        print(f"  Saved -> {model_filename}")

        del keras_model, torch_model, torch_state_dict
        keras.backend.clear_session()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for variant, torch_hub_name in DINO_RESNET_CONVERSION_CONFIG:
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  (torch.hub: {torch_hub_name})")
        print(f"{'=' * 60}")

        torch_model = torch.hub.load(
            "facebookresearch/dino:main", torch_hub_name, pretrained=True
        ).eval()
        trainable_torch, non_trainable_torch, _ = split_model_weights(torch_model)
        torch_state_dict = {
            k: v.cpu().numpy() if hasattr(v, "cpu") else v
            for k, v in {**trainable_torch, **non_trainable_torch}.items()
        }

        keras_model = DinoResNetModel(
            **DINO_RESNET_RECIPES[variant],
            image_size=224,
            include_normalization=False,
        )

        transfer_dino_resnet_weights(keras_model, torch_state_dict)

        rng = np.random.default_rng(0)
        x = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
        with torch.no_grad():
            t_out = torch_model(torch.from_numpy(x)).cpu().numpy()
        k_in = np.transpose(x, (0, 2, 3, 1))
        last = keras_model(k_in, training=False)
        last = (
            last.detach().cpu().numpy() if hasattr(last, "detach") else np.asarray(last)
        )
        k_out = last.mean(axis=(1, 2))
        diff = float(np.abs(k_out - t_out).max())
        if diff > 1e-3:
            raise ValueError(f"{variant}: max diff {diff:.2e}")
        print(f"  Verification OK (max diff = {diff:.2e})")

        model_filename = f"{variant}.weights.h5"
        keras_model.save_weights(model_filename)
        print(f"  Saved -> {model_filename}")

        del keras_model, torch_model, torch_state_dict
        keras.backend.clear_session()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
