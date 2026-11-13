import re
from typing import Dict, List

import keras
import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from zeromodels.conversion.hf_download_utils import download_hf_state_dict
from zeromodels.conversion.weight_transfer_util import (
    compare_keras_torch_names,
    transfer_weights,
)
from zeromodels.models.deformable_detr import DeformableDETRDetect

# Keras weight path (dotted, see keras_to_torch_name) -> checkpoint key, applied in
# order. Targets the checkpoint layout stored on the Hub, which is what the ``hf:``
# loading path reads.
WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "backbone_": "model.backbone.conv_encoder.model.",
    "downsample_conv": "downsample.0",
    "downsample_bn": "downsample.1",
    "input_proj.": "model.input_proj.",
    ".conv.": ".0.",
    ".norm.": ".1.",
    "core.query_position_embeddings": "model.query_position_embeddings.weight",
    "core.encoder_layer.": "model.encoder.layers.",
    "core.decoder_layer.": "model.decoder.layers.",
    "core.class_embed": "class_embed",
    "core.bbox_embed": "bbox_embed",
    "core.": "model.",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving_mean": "running_mean",
    "moving_variance": "running_var",
}


def keras_to_torch_name(keras_weight_path):
    """Map a Keras weight path to its checkpoint key.

    ``core/decoder_layer_0/self_attn/q_proj/kernel`` turns ``_<index>`` into
    ``.<index>`` and ``/`` into ``.`` (``core.decoder_layer.0.self_attn.q_proj.kernel``),
    then :data:`WEIGHT_NAME_MAPPING` is applied.
    """
    name = re.sub(r"_(\d+)_", r".\1.", keras_weight_path)
    name = re.sub(r"_(\d+)(?=/|$)", r".\1", name).replace("/", ".")
    for keras_name_part, torch_name_part in WEIGHT_NAME_MAPPING.items():
        name = name.replace(keras_name_part, torch_name_part)
    return name


def transfer_deformable_detr_weights(keras_model, state_dict):
    for keras_weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        keras_weight_name = keras_weight.path
        torch_weight_name = keras_to_torch_name(keras_weight_name)

        # Heads live under the decoder when it uses them (box refinement /
        # two-stage) and at the top level otherwise.
        if torch_weight_name not in state_dict:
            torch_weight_name = f"model.decoder.{torch_weight_name}"
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

        if keras_weight.name in ("level_embed", "query_position_embeddings"):
            # learned tables: same layout in both frameworks (no transpose)
            keras_weight.assign(torch_weight)
        else:
            transfer_weights(keras_weight_name, keras_weight, torch_weight)


# Per-variant recipes. Models load from the Hub by repo id; these build the arch for
# conversion.
DEFORMABLE_DETR_VARIANTS = {
    "deformable-detr": {"num_feature_levels": 4},
    "deformable-detr-single-scale": {"num_feature_levels": 1},
    "deformable-detr-with-box-refine": {
        "num_feature_levels": 4,
        "with_box_refine": True,
    },
    "deformable-detr-with-box-refine-two-stage": {
        "num_feature_levels": 4,
        "with_box_refine": True,
        "two_stage": True,
    },
}


def query_matched_max_diff(ref_logits, ref_boxes, logits, boxes):
    """Max distance from each query to its closest reference query.

    Two-stage proposal scores contain exact ties (padding-like proposals share a
    score), so the top-k may order tied queries differently between frameworks;
    this comparison is invariant to that order.
    """
    ref = np.concatenate([ref_logits[0], ref_boxes[0]], axis=-1)
    out = np.concatenate([logits[0], boxes[0]], axis=-1)
    dist = np.max(np.abs(out[:, None, :] - ref[None, :, :]), axis=-1)
    return float(np.max(np.min(dist, axis=1)))


if __name__ == "__main__":
    import torch
    import torchvision.transforms as T
    from transformers import DeformableDetrForObjectDetection

    model_configs: List[Dict[str, object]] = [
        {
            "variant": variant,
            "hf_model_name": f"SenseTime/{variant}",
            "output": f"{variant.replace('-', '_')}.weights.h5",
            "num_classes": 91,
            "num_queries": 300,
            **arch,
        }
        for variant, arch in DEFORMABLE_DETR_VARIANTS.items()
    ]

    for cfg in model_configs:
        print(f"\n{'=' * 60}")
        print(f"Converting {cfg['hf_model_name']}...")
        print(f"{'=' * 60}")

        keras_model = DeformableDETRDetect(
            num_feature_levels=cfg["num_feature_levels"],
            with_box_refine=cfg.get("with_box_refine", False),
            two_stage=cfg.get("two_stage", False),
            num_classes=cfg["num_classes"],
            num_queries=cfg["num_queries"],
        )

        torch_model: torch.nn.Module = DeformableDetrForObjectDetection.from_pretrained(
            cfg["hf_model_name"]
        ).eval()
        # The checkpoint as stored on the Hub (the layout the hf: path reads).
        hf_state_dict = download_hf_state_dict(cfg["hf_model_name"])

        transfer_deformable_detr_weights(keras_model, hf_state_dict)

        print("\nVerifying model equivalence...")

        np.random.seed(42)
        test_input = np.random.rand(1, 800, 800, 3).astype(np.float32)

        hf_input = torch.tensor(test_input).permute(0, 3, 1, 2)
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        hf_input_norm = normalize(hf_input)

        with torch.no_grad():
            hf_output = torch_model(hf_input_norm)
            hf_logits = hf_output.logits.numpy()
            hf_boxes = hf_output.pred_boxes.numpy()

        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        keras_input_norm = (test_input - mean) / std

        keras_output = keras_model(keras_input_norm.astype(np.float32), training=False)
        keras_logits = keras.ops.convert_to_numpy(keras_output["logits"])
        keras_boxes = keras.ops.convert_to_numpy(keras_output["pred_boxes"])

        logits_diff = np.max(np.abs(hf_logits - keras_logits))
        boxes_diff = np.max(np.abs(hf_boxes - keras_boxes))
        matched_diff = query_matched_max_diff(
            hf_logits, hf_boxes, keras_logits, keras_boxes
        )

        print(f"Max logits diff:  {logits_diff:.6f}")
        print(f"Max boxes diff:   {boxes_diff:.6f}")
        print(f"Max query-matched diff: {matched_diff:.6f}")

        # Two-stage: tied proposal scores may permute queries, so compare by query.
        if cfg.get("two_stage", False):
            failed = matched_diff > 1e-3
        else:
            failed = logits_diff > 1e-3 or boxes_diff > 1e-3
        if failed:
            raise ValueError(
                "Model equivalence test failed - model outputs do not match "
                f"(logits: {logits_diff:.6f}, boxes: {boxes_diff:.6f}, "
                f"query-matched: {matched_diff:.6f})"
            )

        print("Model equivalence test passed!")

        model_filename = cfg["output"]
        keras_model.save_weights(model_filename)
        print(f"Model saved successfully as {model_filename}")

        del keras_model, torch_model, hf_state_dict
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
