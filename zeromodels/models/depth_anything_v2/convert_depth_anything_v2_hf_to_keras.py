import gc
from typing import List, Tuple

import keras
import numpy as np

from zeromodels.models.depth_anything_v1.convert_depth_anything_v1_hf_to_keras import (
    transfer_depth_anything_weights,
)
from zeromodels.models.depth_anything_v2 import DepthAnythingV2DepthEstimation

DEPTH_ANYTHING_V2_VARIANTS: List[Tuple[str, str]] = [
    ("depth_anything_v2_small", "depth-anything/Depth-Anything-V2-Small-hf"),
    ("depth_anything_v2_base", "depth-anything/Depth-Anything-V2-Base-hf"),
    ("depth_anything_v2_large", "depth-anything/Depth-Anything-V2-Large-hf"),
    (
        "depth_anything_v2_metric_indoor_small",
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    ),
    (
        "depth_anything_v2_metric_indoor_base",
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    ),
    (
        "depth_anything_v2_metric_indoor_large",
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    ),
    (
        "depth_anything_v2_metric_outdoor_small",
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    ),
    (
        "depth_anything_v2_metric_outdoor_base",
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    ),
    (
        "depth_anything_v2_metric_outdoor_large",
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    ),
]

# Per-variant recipes (relocated from depth_anything_v2_config.py). Models load
# from the Hub by repo id; these build the arch for conversion + drive the
# zm_config backfill. The metric fine-tunes add depth_estimation_type + max_depth.
_V2_SIZES = {
    "small": {
        "backbone_dim": 384,
        "backbone_depth": 12,
        "backbone_num_heads": 6,
        "out_indices": (3, 6, 9, 12),
        "neck_hidden_sizes": (48, 96, 192, 384),
        "fusion_hidden_size": 64,
        "reassemble_factors": (4, 2, 1, 0.5),
    },
    "base": {
        "backbone_dim": 768,
        "backbone_depth": 12,
        "backbone_num_heads": 12,
        "out_indices": (3, 6, 9, 12),
        "neck_hidden_sizes": (96, 192, 384, 768),
        "fusion_hidden_size": 128,
        "reassemble_factors": (4, 2, 1, 0.5),
    },
    "large": {
        "backbone_dim": 1024,
        "backbone_depth": 24,
        "backbone_num_heads": 16,
        "out_indices": (5, 12, 18, 24),
        "neck_hidden_sizes": (256, 512, 1024, 1024),
        "fusion_hidden_size": 256,
        "reassemble_factors": (4, 2, 1, 0.5),
    },
}


def _v2_recipe(size, depth_estimation_type="relative", max_depth=None):
    recipe = dict(_V2_SIZES[size])
    if depth_estimation_type == "metric":
        recipe["depth_estimation_type"] = "metric"
        recipe["max_depth"] = max_depth
    return recipe


DEPTH_ANYTHING_V2_RECIPES = {
    "depth_anything_v2_small": _v2_recipe("small"),
    "depth_anything_v2_base": _v2_recipe("base"),
    "depth_anything_v2_large": _v2_recipe("large"),
    "depth_anything_v2_metric_indoor_small": _v2_recipe("small", "metric", 20.0),
    "depth_anything_v2_metric_indoor_base": _v2_recipe("base", "metric", 20.0),
    "depth_anything_v2_metric_indoor_large": _v2_recipe("large", "metric", 20.0),
    "depth_anything_v2_metric_outdoor_small": _v2_recipe("small", "metric", 80.0),
    "depth_anything_v2_metric_outdoor_base": _v2_recipe("base", "metric", 80.0),
    "depth_anything_v2_metric_outdoor_large": _v2_recipe("large", "metric", 80.0),
}


if __name__ == "__main__":
    import torch
    from transformers import DepthAnythingForDepthEstimation

    for variant, hf_id in DEPTH_ANYTHING_V2_VARIANTS:
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        hf_model = DepthAnythingForDepthEstimation.from_pretrained(hf_id).eval()
        hf_sd = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}

        keras_model: keras.Model = DepthAnythingV2DepthEstimation(
            **DEPTH_ANYTHING_V2_RECIPES[variant], image_size=518
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
