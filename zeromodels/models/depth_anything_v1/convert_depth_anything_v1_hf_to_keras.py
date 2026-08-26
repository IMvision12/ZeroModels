import gc
import re
from typing import Dict, List, Tuple

import keras
import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights
from zeromodels.models.depth_anything_v1 import DepthAnythingV1DepthEstimation

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "/": ".",
    "_": ".",
    ".ln1.": ".norm1.",
    ".ln2.": ".norm2.",
    "backbone.block.": "backbone.encoder.layer.",
    "backbone.patch.embed.": "backbone.embeddings.patch_embeddings.projection.",
    "backbone.cls.token.cls.token": "backbone.embeddings.cls_token",
    "backbone.pos.embed.pos.embed": "backbone.embeddings.position_embeddings",
    "neck.reassemble.": "neck.reassemble_stage.layers.",
    "neck.fusion.": "neck.fusion_stage.layers.",
    "neck.conv.": "neck.convs.",
    ".res1.conv1.": ".residual_layer1.convolution1.",
    ".res1.conv2.": ".residual_layer1.convolution2.",
    ".res2.conv1.": ".residual_layer2.convolution1.",
    ".res2.conv2.": ".residual_layer2.convolution2.",
    ".kernel": ".weight",
    ".gamma": ".weight",
    ".beta": ".bias",
}


def transfer_depth_anything_weights(
    keras_model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    all_weights = [w for layer in keras_model.layers for w in layer.weights]
    for w in tqdm(all_weights, desc="Transferring weights"):
        path = w.path
        parts = path.split("/")

        if "_attn/" in path:
            m = re.search(r"backbone_block_(\d+)_attn", parts[0])
            if not m:
                raise WeightMappingError(path, "invalid attention path")
            layer_idx = m.group(1)
            suffix = parts[-1]
            torch_suffix = "weight" if suffix == "kernel" else "bias"
            hf_prefix = f"backbone.encoder.layer.{layer_idx}.attention"
            if "qkv" in parts[1]:
                q = hf_state_dict[f"{hf_prefix}.attention.query.{torch_suffix}"]
                k = hf_state_dict[f"{hf_prefix}.attention.key.{torch_suffix}"]
                v = hf_state_dict[f"{hf_prefix}.attention.value.{torch_suffix}"]
                torch_weight = np.concatenate([q, k, v], axis=0)
            elif "proj" in parts[1]:
                torch_weight = hf_state_dict[f"{hf_prefix}.output.dense.{torch_suffix}"]
            else:
                raise WeightMappingError(
                    path, f"unknown attention sub-layer {parts[1]}"
                )
            transfer_weights(suffix, w, torch_weight)
            continue

        m = re.match(r"backbone_block_(\d+)_ls(\d+)/variable(?:_\d+)?$", path)
        if m:
            layer_idx, ls_idx = m.group(1), m.group(2)
            torch_key = (
                f"backbone.encoder.layer.{layer_idx}.layer_scale{ls_idx}.lambda1"
            )
            w.assign(hf_state_dict[torch_key])
            continue

        torch_key = path
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_key = torch_key.replace(old, new)

        if torch_key not in hf_state_dict:
            raise WeightMappingError(path, torch_key)

        torch_weight = hf_state_dict[torch_key]
        keras_name = "conv_kernel" if len(w.shape) == 4 else path
        transfer_weights(keras_name, w, torch_weight)


DEPTH_ANYTHING_V1_VARIANTS: List[Tuple[str, str]] = [
    ("depth_anything_small", "LiheYoung/depth-anything-small-hf"),
    ("depth_anything_base", "LiheYoung/depth-anything-base-hf"),
    ("depth_anything_large", "LiheYoung/depth-anything-large-hf"),
]

# Per-variant recipes (relocated from depth_anything_v1_config.py). Models load
# from the Hub by repo id; these build the arch for conversion + drive the
# kf_config backfill.
DEPTH_ANYTHING_V1_RECIPES = {
    "depth_anything_small": {
        "backbone_dim": 384,
        "backbone_depth": 12,
        "backbone_num_heads": 6,
        "out_indices": (9, 10, 11, 12),
        "neck_hidden_sizes": (48, 96, 192, 384),
        "fusion_hidden_size": 64,
        "reassemble_factors": (4, 2, 1, 0.5),
    },
    "depth_anything_base": {
        "backbone_dim": 768,
        "backbone_depth": 12,
        "backbone_num_heads": 12,
        "out_indices": (9, 10, 11, 12),
        "neck_hidden_sizes": (96, 192, 384, 768),
        "fusion_hidden_size": 128,
        "reassemble_factors": (4, 2, 1, 0.5),
    },
    "depth_anything_large": {
        "backbone_dim": 1024,
        "backbone_depth": 24,
        "backbone_num_heads": 16,
        "out_indices": (21, 22, 23, 24),
        "neck_hidden_sizes": (256, 512, 1024, 1024),
        "fusion_hidden_size": 256,
        "reassemble_factors": (4, 2, 1, 0.5),
    },
}


if __name__ == "__main__":
    import torch
    from transformers import DepthAnythingForDepthEstimation

    for variant, hf_id in DEPTH_ANYTHING_V1_VARIANTS:
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        hf_model = DepthAnythingForDepthEstimation.from_pretrained(hf_id).eval()
        hf_sd = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}

        keras_model: keras.Model = DepthAnythingV1DepthEstimation(
            **DEPTH_ANYTHING_V1_RECIPES[variant], image_size=518
        )

        transfer_depth_anything_weights(keras_model, hf_sd)

        np.random.seed(42)
        test_image = np.random.rand(1, 518, 518, 3).astype(np.float32)

        keras_depth = keras_model.predict(test_image, verbose=0).squeeze(-1)

        with torch.no_grad():
            hf_input = torch.from_numpy(test_image.transpose(0, 3, 1, 2))
            hf_depth = hf_model(pixel_values=hf_input).predicted_depth.cpu().numpy()

        max_diff = float(np.max(np.abs(keras_depth - hf_depth)))
        mean_diff = float(np.mean(np.abs(keras_depth - hf_depth)))
        print(f"  Max depth diff:  {max_diff:.6f}")
        print(f"  Mean depth diff: {mean_diff:.6f}")
        if max_diff > 25.0:
            raise ValueError(f"{variant}: depth diff {max_diff:.2e} exceeds tolerance")
        print("  Verification OK")

        model_filename = f"{variant}.weights.h5"
        keras_model.save_weights(model_filename)
        print(f"  Saved -> {model_filename}")

        del keras_model, hf_model, hf_sd
        keras.backend.clear_session()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
