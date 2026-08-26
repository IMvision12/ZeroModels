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
from zeromodels.models.dino_v2 import DinoV2Model

DINOV2_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    "conv1": "embeddings.patch_embeddings.projection",
    "pos.embed.pos.embed": "embeddings.position_embeddings",
    "cls.token.cls.token": "embeddings.cls_token",
    "blocks.": "encoder.layer.",
    "dense.1": "mlp.fc1",
    "dense.2": "mlp.fc2",
    "weights.in": "mlp.weights_in",
    "weights.out": "mlp.weights_out",
    "layernorm.1": "norm1",
    "layernorm.2": "norm2",
    "final.layernorm": "layernorm",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}


def transfer_dinov2_weights(
    keras_model: keras.Model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    import torch

    if any(k.startswith("dinov2.") for k in hf_state_dict):
        hf_state_dict = {
            k[len("dinov2.") :]: v
            for k, v in hf_state_dict.items()
            if k.startswith("dinov2.")
        }

    trainable, non_trainable = split_model_weights(keras_model)
    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring DINOv2 weights"
    ):
        path = keras_weight.path
        layer_segment = path.split("/")[-2]

        m = re.match(r"blocks_(\d+)_attn_qkv$", layer_segment)
        if m:
            idx = int(m.group(1))
            suffix = "weight" if "kernel" in path else "bias"
            base = f"encoder.layer.{idx}.attention.attention"
            q_key = f"{base}.query.{suffix}"
            k_key = f"{base}.key.{suffix}"
            v_key = f"{base}.value.{suffix}"
            for key in (q_key, k_key, v_key):
                if key not in hf_state_dict:
                    raise WeightMappingError(keras_weight_name, key)
            q, k, v = (
                hf_state_dict[q_key],
                hf_state_dict[k_key],
                hf_state_dict[v_key],
            )
            if hasattr(q, "numpy"):
                q, k, v = q.numpy(), k.numpy(), v.numpy()
            fused = np.concatenate([q, k, v], axis=0)
            transfer_weights(keras_weight_name, keras_weight, fused)
            continue

        m = re.match(r"blocks_(\d+)_attn_proj$", layer_segment)
        if m:
            idx = int(m.group(1))
            suffix = "weight" if "kernel" in path else "bias"
            hf_key = f"encoder.layer.{idx}.attention.output.dense.{suffix}"
            if hf_key not in hf_state_dict:
                raise WeightMappingError(keras_weight_name, hf_key)
            transfer_weights(keras_weight_name, keras_weight, hf_state_dict[hf_key])
            continue

        m = re.match(r"blocks_(\d+)_layerscale_(1|2)$", layer_segment)
        if m:
            idx = int(m.group(1))
            hf_key = f"encoder.layer.{idx}.layer_scale{m.group(2)}.lambda1"
            if hf_key not in hf_state_dict:
                raise WeightMappingError(keras_weight_name, hf_key)
            w = hf_state_dict[hf_key]
            keras_weight.assign(w.numpy() if hasattr(w, "numpy") else w)
            continue

        torch_weight_name = keras_weight_name
        for old, new in DINOV2_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)
        torch_weight_name = re.sub(
            r"pos_embed_variable_\d+$",
            "embeddings.position_embeddings",
            torch_weight_name,
        )
        torch_weight_name = re.sub(
            r"cls_token_variable_\d+$",
            "embeddings.cls_token",
            torch_weight_name,
        )

        if torch_weight_name not in hf_state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = hf_state_dict[torch_weight_name]

        if torch_weight_name == "embeddings.cls_token":
            keras_weight.assign(
                torch_weight.numpy() if hasattr(torch_weight, "numpy") else torch_weight
            )
            continue

        if torch_weight_name == "embeddings.position_embeddings":
            target_num_patches = keras_weight.shape[1] - 1
            pe = (
                torch_weight
                if isinstance(torch_weight, torch.Tensor)
                else torch.from_numpy(np.asarray(torch_weight))
            )
            cls_pe, spatial_pe = pe[:, :1], pe[:, 1:]
            src_size = int(round(spatial_pe.shape[1] ** 0.5))
            tgt_size = int(round(target_num_patches**0.5))
            if src_size == tgt_size:
                keras_weight.assign(pe.numpy())
            else:
                embed_dim = spatial_pe.shape[-1]
                spatial_pe = spatial_pe.reshape(
                    1, src_size, src_size, embed_dim
                ).permute(0, 3, 1, 2)
                spatial_pe = torch.nn.functional.interpolate(
                    spatial_pe.float(),
                    size=(tgt_size, tgt_size),
                    mode="bicubic",
                    align_corners=False,
                )
                spatial_pe = spatial_pe.permute(0, 2, 3, 1).reshape(
                    1, tgt_size * tgt_size, embed_dim
                )
                keras_weight.assign(torch.cat([cls_pe, spatial_pe], dim=1).numpy())
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


DINOV2_VARIANTS: List[Tuple[str, str]] = [
    ("dinov2-small", "facebook/dinov2-small"),
    ("dinov2-base", "facebook/dinov2-base"),
    ("dinov2-large", "facebook/dinov2-large"),
    ("dinov2-giant", "facebook/dinov2-giant"),
]

# Per-variant recipes (relocated from dino_v2_config.py). Models load from the Hub
# by repo id; these build the arch for conversion + drive the zm_config backfill.
DINOV2_RECIPES = {
    "dinov2-small": {
        "patch_size": 14,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "layer_scale_init": 1.0,
    },
    "dinov2-base": {
        "patch_size": 14,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "layer_scale_init": 1.0,
    },
    "dinov2-large": {
        "patch_size": 14,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "layer_scale_init": 1.0,
    },
    "dinov2-giant": {
        "patch_size": 14,
        "embed_dim": 1536,
        "depth": 40,
        "num_heads": 24,
        "layer_scale_init": 1.0,
        "use_swiglu": True,  # ViT-g/14 uses a SwiGLU FFN
    },
}


if __name__ == "__main__":
    import torch
    from transformers import Dinov2Model

    HF_TOKEN = os.environ.get("HF_TOKEN")

    for variant, hf_id in DINOV2_VARIANTS:
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        hf_model = Dinov2Model.from_pretrained(hf_id, token=HF_TOKEN).eval()
        hf_state_dict = dict(hf_model.state_dict())

        keras_model = DinoV2Model(
            **DINOV2_RECIPES[variant],
            image_size=224,
            include_normalization=False,
        )

        transfer_dinov2_weights(keras_model, hf_state_dict)

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
        if diff > 1e-3:
            raise ValueError(f"{variant}: max diff {diff:.2e}")
        print(f"  Verification OK (max diff = {diff:.2e})")

        model_filename = f"{variant}.weights.h5"
        keras_model.save_weights(model_filename)
        print(f"  Saved -> {model_filename}")

        del keras_model, hf_model, hf_state_dict
        keras.backend.clear_session()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
