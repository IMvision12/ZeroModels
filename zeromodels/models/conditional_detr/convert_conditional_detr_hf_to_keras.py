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
from zeromodels.models.conditional_detr import ConditionalDETRDetect

# Keras weight path (dotted, see keras_to_torch_name) -> checkpoint key, applied in
# order. Targets the checkpoint layout stored on the Hub, which is what the ``hf:``
# loading path reads.
WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "backbone_": "model.backbone.conv_encoder.model.",
    "downsample_conv": "downsample.0",
    "downsample_bn": "downsample.1",
    "input_projection": "model.input_projection",
    "query_position_embeddings.embedding.embeddings": (
        "model.query_position_embeddings.weight"
    ),
    "encoder_layers.": "model.encoder.layers.",
    "decoder.": "model.decoder.",
    "self_attn_out_proj": "self_attn.out_proj",
    "encoder_attn_out_proj": "encoder_attn.out_proj",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving_mean": "running_mean",
    "moving_variance": "running_var",
}


def keras_to_torch_name(keras_weight_path):
    """Map a Keras weight path to its checkpoint key.

    ``encoder_layers_0_self_attn/encoder_layers_0_self_attn_q_proj/kernel`` first
    drops the repeated parent prefix, then turns ``_<index>`` into ``.<index>`` and
    ``/`` into ``.`` (``encoder_layers.0.self_attn.q_proj.kernel``), and finally
    applies :data:`WEIGHT_NAME_MAPPING`.
    """
    parts = keras_weight_path.split("/")
    parts = [
        part[len(parts[i - 1]) + 1 :]
        if i and part.startswith(parts[i - 1] + "_")
        else part
        for i, part in enumerate(parts)
    ]
    name = re.sub(r"_(\d+)_", r".\1.", "/".join(parts))
    name = re.sub(r"_(\d+)(?=/|$)", r".\1", name).replace("/", ".")
    for keras_name_part, torch_name_part in WEIGHT_NAME_MAPPING.items():
        name = name.replace(keras_name_part, torch_name_part)
    return name


def transfer_conditional_detr_weights(keras_model, state_dict):
    for keras_weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        keras_weight_name = keras_weight.path
        torch_weight_name = keras_to_torch_name(keras_weight_name)

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

        transfer_weights(keras_weight_name, keras_weight, torch_weight)


# Per-variant recipes. Models load from the Hub by repo id; these build the arch for
# conversion.
CONDITIONAL_DETR_VARIANTS = {
    "conditional-detr-resnet-50": {"backbone_variant": "ResNet50"},
}


if __name__ == "__main__":
    import torch
    import torchvision.transforms as T
    from transformers import ConditionalDetrForObjectDetection

    model_configs: List[Dict[str, object]] = [
        {
            "variant": "conditional-detr-resnet-50",
            "backbone_variant": "ResNet50",
            "hf_model_name": "microsoft/conditional-detr-resnet-50",
            "output": "conditional_detr_resnet50.weights.h5",
            "num_classes": 91,
            "num_queries": 300,
        },
    ]

    for cfg in model_configs:
        print(f"\n{'=' * 60}")
        print(f"Converting {cfg['hf_model_name']}...")
        print(f"{'=' * 60}")

        keras_model = ConditionalDETRDetect(
            backbone_variant=cfg["backbone_variant"],
            num_classes=cfg["num_classes"],
            num_queries=cfg["num_queries"],
        )

        torch_model: torch.nn.Module = (
            ConditionalDetrForObjectDetection.from_pretrained(
                cfg["hf_model_name"]
            ).eval()
        )
        # The checkpoint as stored on the Hub (the layout the hf: path reads).
        hf_state_dict = download_hf_state_dict(cfg["hf_model_name"])

        transfer_conditional_detr_weights(keras_model, hf_state_dict)

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

        print(f"Max logits diff:  {logits_diff:.6f}")
        print(f"Max boxes diff:   {boxes_diff:.6f}")

        if logits_diff > 1e-3 or boxes_diff > 1e-3:
            raise ValueError(
                "Model equivalence test failed - model outputs do not match "
                f"(logits: {logits_diff:.6f}, boxes: {boxes_diff:.6f})"
            )

        print("Model equivalence test passed!")

        model_filename = cfg["output"]
        keras_model.save_weights(model_filename)
        print(f"Model saved successfully as {model_filename}")

        del keras_model, torch_model, hf_state_dict
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
