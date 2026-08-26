import gc
import os
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
    transfer_weights,
)
from zeromodels.models.dino_v3 import DinoV3ConvNeXtModel, DinoV3ViTModel

DINOV3_VIT_VARIANTS: List[Tuple[str, str]] = [
    ("dinov3-vits16-pretrain-lvd1689m", "facebook/dinov3-vits16-pretrain-lvd1689m"),
    ("dinov3-vitb16-pretrain-lvd1689m", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
    ("dinov3-vitl16-pretrain-lvd1689m", "facebook/dinov3-vitl16-pretrain-lvd1689m"),
]

DINOV3_CONVNEXT_VARIANTS: List[Tuple[str, str]] = [
    (
        "dinov3-convnext-tiny-pretrain-lvd1689m",
        "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    ),
    (
        "dinov3-convnext-small-pretrain-lvd1689m",
        "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    ),
    (
        "dinov3-convnext-base-pretrain-lvd1689m",
        "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    ),
    (
        "dinov3-convnext-large-pretrain-lvd1689m",
        "facebook/dinov3-convnext-large-pretrain-lvd1689m",
    ),
]

# Per-variant recipes (relocated from dino_v3_config.py). Models load from the Hub
# by repo id; these build the arch for conversion + drive the zm_config backfill.
DINOV3_VIT_RECIPES = {
    "dinov3-vits16-pretrain-lvd1689m": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
    },
    "dinov3-vitb16-pretrain-lvd1689m": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
    },
    "dinov3-vitl16-pretrain-lvd1689m": {
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
    },
}

DINOV3_CONVNEXT_RECIPES = {
    "dinov3-convnext-tiny-pretrain-lvd1689m": {
        "depths": (3, 3, 9, 3),
        "projection_dim": (96, 192, 384, 768),
    },
    "dinov3-convnext-small-pretrain-lvd1689m": {
        "depths": (3, 3, 27, 3),
        "projection_dim": (96, 192, 384, 768),
    },
    "dinov3-convnext-base-pretrain-lvd1689m": {
        "depths": (3, 3, 27, 3),
        "projection_dim": (128, 256, 512, 1024),
    },
    "dinov3-convnext-large-pretrain-lvd1689m": {
        "depths": (3, 3, 27, 3),
        "projection_dim": (192, 384, 768, 1536),
    },
}

DINOV3_VIT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "patch.embed": "embeddings.patch_embeddings",
    "blocks.": "layer.",
    "dense.1": "mlp.up_proj",
    "dense.2": "mlp.down_proj",
    "swiglu.gate": "mlp.gate_proj",
    "swiglu.up": "mlp.up_proj",
    "swiglu.down": "mlp.down_proj",
    "layernorm.1": "norm1",
    "layernorm.2": "norm2",
    "final.layernorm": "norm",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}

VAR_MAP: Dict[str, str] = {
    "kernel": "weight",
    "gamma": "weight",
    "bias": "bias",
    "beta": "bias",
}

DINOV3_CONVNEXT_WEIGHT_MAPPING: Dict[str, str] = {
    r"stem_conv_(kernel|bias)": "stages.0.downsample_layers.0.{v}",
    r"stem_layernorm_(gamma|beta)": "stages.0.downsample_layers.1.{v}",
    r"stages_(\d+)_downsampling_layernorm_(gamma|beta)": "stages.{0}.downsample_layers.0.{v}",
    r"stages_(\d+)_downsampling_conv_(kernel|bias)": "stages.{0}.downsample_layers.1.{v}",
    r"stages_(\d+)_blocks_(\d+)_layer_scale_variable": "stages.{0}.layers.{1}.gamma",
    r"stages_(\d+)_blocks_(\d+)_depthwise_conv_(kernel|bias)": "stages.{0}.layers.{1}.depthwise_conv.{v}",
    r"stages_(\d+)_blocks_(\d+)_layernorm_(gamma|beta)": "stages.{0}.layers.{1}.layer_norm.{v}",
    r"stages_(\d+)_blocks_(\d+)_conv_1_(kernel|bias)": "stages.{0}.layers.{1}.pointwise_conv1.{v}",
    r"stages_(\d+)_blocks_(\d+)_conv_2_(kernel|bias)": "stages.{0}.layers.{1}.pointwise_conv2.{v}",
    r"final_layernorm_(gamma|beta)": "layer_norm.{v}",
}


def transfer_dinov3_vit_weights(keras_model, hf_state_dict):
    for prefix in ("dinov3_vit.", "model."):
        if any(k.startswith(prefix) for k in hf_state_dict):
            hf_state_dict = {
                (k[len(prefix) :] if k.startswith(prefix) else k): v
                for k, v in hf_state_dict.items()
            }

    trainable, non_trainable = split_model_weights(keras_model)
    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring DINOv3 ViT weights"
    ):
        path = keras_weight.path
        layer_name = path.split("/")[-2]
        var_name = path.split("/")[-1]

        if "cls_token" in path and "cls_token" in var_name:
            keras_weight.assign(hf_state_dict["embeddings.cls_token"])
            continue

        if "register_tokens" in path:
            keras_weight.assign(hf_state_dict["embeddings.register_tokens"])
            continue

        m = re.match(r"blocks_(\d+)_attn_(q|k|v)_proj$", layer_name)
        if m:
            idx = int(m.group(1))
            suffix = "weight" if "kernel" in var_name else "bias"
            hf_key = f"layer.{idx}.attention.{m.group(2)}_proj.{suffix}"
            if hf_key not in hf_state_dict:
                raise WeightMappingError(keras_weight_name, hf_key)
            transfer_weights(keras_weight_name, keras_weight, hf_state_dict[hf_key])
            continue

        m = re.match(r"blocks_(\d+)_attn_proj$", layer_name)
        if m:
            idx = int(m.group(1))
            suffix = "weight" if "kernel" in var_name else "bias"
            hf_key = f"layer.{idx}.attention.o_proj.{suffix}"
            if hf_key not in hf_state_dict:
                raise WeightMappingError(keras_weight_name, hf_key)
            transfer_weights(keras_weight_name, keras_weight, hf_state_dict[hf_key])
            continue

        m = re.match(r"blocks_(\d+)_layerscale_(1|2)$", layer_name)
        if m:
            idx = int(m.group(1))
            hf_key = f"layer.{idx}.layer_scale{m.group(2)}.lambda1"
            if hf_key not in hf_state_dict:
                raise WeightMappingError(keras_weight_name, hf_key)
            keras_weight.assign(hf_state_dict[hf_key])
            continue

        if "patch_embed" in path and len(keras_weight.shape) == 4:
            hf_key = "embeddings.patch_embeddings.weight"
            if hf_key in hf_state_dict:
                transfer_weights("conv_kernel", keras_weight, hf_state_dict[hf_key])
                continue

        torch_name = keras_weight_name
        for old, new in DINOV3_VIT_NAME_MAPPING.items():
            torch_name = torch_name.replace(old, new)

        if torch_name not in hf_state_dict:
            raise WeightMappingError(keras_weight_name, torch_name)

        torch_weight = hf_state_dict[torch_name]
        if not compare_keras_torch_names(
            keras_weight_name, keras_weight, torch_name, torch_weight
        ):
            raise WeightShapeMismatchError(
                keras_weight_name, keras_weight.shape, torch_name, torch_weight.shape
            )
        transfer_weights(keras_weight_name, keras_weight, torch_weight)


def transfer_dinov3_convnext_weights(keras_model, hf_state_dict):
    for prefix in ("dinov3_convnext.", "model."):
        if any(k.startswith(prefix) for k in hf_state_dict):
            hf_state_dict = {
                (k[len(prefix) :] if k.startswith(prefix) else k): v
                for k, v in hf_state_dict.items()
            }

    trainable, non_trainable = split_model_weights(keras_model)
    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring DINOv3 ConvNeXt weights"
    ):
        hf_key = None
        for pattern, template in DINOV3_CONVNEXT_WEIGHT_MAPPING.items():
            m = re.match(pattern, keras_weight_name)
            if m:
                groups = m.groups()
                var_group = groups[-1] if groups[-1] in VAR_MAP else None
                idx_groups = groups if var_group is None else groups[:-1]
                v = VAR_MAP[var_group] if var_group else ""
                hf_key = template.format(*idx_groups, v=v)
                break

        if hf_key is None or hf_key not in hf_state_dict:
            raise WeightMappingError(keras_weight_name, str(hf_key))

        hf_w = hf_state_dict[hf_key]

        if "layer_scale" in keras_weight_name:
            keras_weight.assign(hf_w)
        elif (
            "pointwise" in hf_key
            and len(hf_w.shape) == 2
            and len(keras_weight.shape) == 4
        ):
            keras_weight.assign(hf_w.T[np.newaxis, np.newaxis, :, :])
        else:
            transfer_weights(keras_weight_name, keras_weight, hf_w)


if __name__ == "__main__":
    import torch
    from transformers import AutoModel

    HF_TOKEN = os.environ.get("HF_TOKEN")

    for variant, hf_id in DINOV3_VIT_VARIANTS:
        print(f"\n{'=' * 60}")
        print(f"Converting ViT: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        hf_model = AutoModel.from_pretrained(hf_id, token=HF_TOKEN).eval()
        hf_sd = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}

        keras_model = DinoV3ViTModel(
            **DINOV3_VIT_RECIPES[variant],
            image_size=224,
            include_normalization=False,
        )
        transfer_dinov3_vit_weights(keras_model, hf_sd)

        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
        with torch.no_grad():
            hf_out = (
                hf_model(pixel_values=torch.from_numpy(x_np))
                .last_hidden_state.cpu()
                .numpy()
            )
        k_in = np.transpose(x_np, (0, 2, 3, 1))
        last = keras_model(k_in, training=False)
        k_out = (
            last.detach().cpu().numpy() if hasattr(last, "detach") else np.asarray(last)
        )
        diff = float(np.abs(k_out - hf_out).max())
        assert diff < 1e-2, f"{variant}: max diff {diff:.2e}"
        print(f"  Verification OK (max diff = {diff:.2e})")

        out = f"{variant}.weights.h5"
        keras_model.save_weights(out)
        print(f"  Saved -> {out}")

        del keras_model, hf_model, hf_sd
        keras.backend.clear_session()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for variant, hf_id in DINOV3_CONVNEXT_VARIANTS:
        print(f"\n{'=' * 60}")
        print(f"Converting ConvNeXt: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        hf_model = AutoModel.from_pretrained(hf_id, token=HF_TOKEN).eval()
        hf_sd = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}

        keras_model = DinoV3ConvNeXtModel(
            **DINOV3_CONVNEXT_RECIPES[variant],
            image_size=224,
            include_normalization=False,
        )
        transfer_dinov3_convnext_weights(keras_model, hf_sd)

        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
        with torch.no_grad():
            hf_out_obj = hf_model(
                pixel_values=torch.from_numpy(x_np), output_hidden_states=True
            )
            hf_feat = hf_out_obj.hidden_states[-1].permute(0, 2, 3, 1).cpu().numpy()
        k_in = np.transpose(x_np, (0, 2, 3, 1))
        last = keras_model(k_in, training=False)
        k_out = (
            last.detach().cpu().numpy() if hasattr(last, "detach") else np.asarray(last)
        )
        diff = float(np.abs(k_out - hf_feat).max())
        assert diff < 1e-2, f"{variant}: max diff {diff:.2e}"
        print(f"  Verification OK (max diff = {diff:.2e})")

        out = f"{variant}.weights.h5"
        keras_model.save_weights(out)
        print(f"  Saved -> {out}")

        del keras_model, hf_model, hf_sd
        keras.backend.clear_session()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
