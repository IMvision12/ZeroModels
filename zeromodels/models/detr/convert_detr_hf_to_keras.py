from typing import Dict, List

import keras
import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from zeromodels.conversion.weight_transfer_util import (
    compare_keras_torch_names,
    transfer_nested_layer_weights,
    transfer_weights,
)
from zeromodels.models.detr import DETRDetect, DETRPanopticSegment

backbone_weight_name_mapping: Dict[str, str] = {
    "backbone_layer": "model.backbone.model.layer",
    "_": ".",
    "downsample.conv": "downsample.0",
    "downsample.bn": "downsample.1",
    "backbone.conv1": "model.backbone.model.conv1",
    "backbone.bn1": "model.backbone.model.bn1",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
}


def transfer_detr_weights(keras_model, state_dict):
    state_dict = {
        k.replace("model.backbone.conv_encoder.model.", "model.backbone.model."): v
        for k, v in state_dict.items()
    }
    backbone_layers = [
        layer for layer in keras_model.layers if layer.name.startswith("backbone_")
    ]

    backbone_trainable = []
    backbone_non_trainable = []
    for layer in backbone_layers:
        for weight in layer.trainable_weights:
            backbone_trainable.append((weight, f"{layer.name}_{weight.name}"))
        for weight in layer.non_trainable_weights:
            backbone_non_trainable.append((weight, f"{layer.name}_{weight.name}"))

    for keras_weight, keras_weight_name in tqdm(
        backbone_trainable + backbone_non_trainable,
        total=len(backbone_trainable + backbone_non_trainable),
        desc="Transferring backbone weights",
    ):
        torch_weight_name: str = keras_weight_name
        for keras_name_part, torch_name_part in backbone_weight_name_mapping.items():
            torch_weight_name = torch_weight_name.replace(
                keras_name_part, torch_name_part
            )

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

    input_proj = keras_model.get_layer("input_projection")
    conv_w = state_dict["model.input_projection.weight"]
    input_proj.weights[0].assign(np.transpose(conv_w, (2, 3, 1, 0)))
    input_proj.weights[1].assign(state_dict["model.input_projection.bias"])

    query_layer = keras_model.get_layer("query_position_embeddings")
    query_layer.weights[0].assign(state_dict["model.query_position_embeddings.weight"])

    ln_mapping = {"gamma": "weight", "beta": "bias"}
    dense_mapping = {"kernel": "weight"}

    for i in tqdm(
        range(keras_model.num_encoder_layers), desc="Transferring encoder weights"
    ):
        hf_prefix = f"model.encoder.layers.{i}"
        k_prefix = f"encoder_layers_{i}"

        sa_mapping = {
            f"{k_prefix}_self_attn_": "",
            "out_proj": "o_proj",
            "kernel": "weight",
        }
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_self_attn"),
            state_dict,
            f"{hf_prefix}.self_attn",
            name_mapping=sa_mapping,
        )
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_self_attn_layer_norm"),
            state_dict,
            f"{hf_prefix}.self_attn_layer_norm",
            name_mapping=ln_mapping,
        )
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_fc1"),
            state_dict,
            f"{hf_prefix}.mlp.fc1",
            name_mapping=dense_mapping,
        )
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_fc2"),
            state_dict,
            f"{hf_prefix}.mlp.fc2",
            name_mapping=dense_mapping,
        )
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_final_layer_norm"),
            state_dict,
            f"{hf_prefix}.final_layer_norm",
            name_mapping=ln_mapping,
        )

    for i in tqdm(
        range(keras_model.num_decoder_layers), desc="Transferring decoder weights"
    ):
        hf_prefix = f"model.decoder.layers.{i}"
        k_prefix = f"decoder_layers_{i}"

        sa_mapping = {
            f"{k_prefix}_self_attn_": "",
            "out_proj": "o_proj",
            "kernel": "weight",
        }
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_self_attn"),
            state_dict,
            f"{hf_prefix}.self_attn",
            name_mapping=sa_mapping,
        )
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_self_attn_layer_norm"),
            state_dict,
            f"{hf_prefix}.self_attn_layer_norm",
            name_mapping=ln_mapping,
        )
        ca_mapping = {
            f"{k_prefix}_encoder_attn_": "",
            "out_proj": "o_proj",
            "kernel": "weight",
        }
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_encoder_attn"),
            state_dict,
            f"{hf_prefix}.encoder_attn",
            name_mapping=ca_mapping,
        )
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_encoder_attn_layer_norm"),
            state_dict,
            f"{hf_prefix}.encoder_attn_layer_norm",
            name_mapping=ln_mapping,
        )
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_fc1"),
            state_dict,
            f"{hf_prefix}.mlp.fc1",
            name_mapping=dense_mapping,
        )
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_fc2"),
            state_dict,
            f"{hf_prefix}.mlp.fc2",
            name_mapping=dense_mapping,
        )
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_final_layer_norm"),
            state_dict,
            f"{hf_prefix}.final_layer_norm",
            name_mapping=ln_mapping,
        )

    transfer_nested_layer_weights(
        keras_model.get_layer("decoder_layernorm"),
        state_dict,
        "model.decoder.layernorm",
        name_mapping=ln_mapping,
    )

    transfer_nested_layer_weights(
        keras_model.get_layer("class_labels_classifier"),
        state_dict,
        "class_labels_classifier",
        name_mapping=dense_mapping,
    )

    for layer_idx in range(3):
        transfer_nested_layer_weights(
            keras_model.get_layer(f"bbox_predictor_{layer_idx}"),
            state_dict,
            f"bbox_predictor.layers.{layer_idx}",
            name_mapping=dense_mapping,
        )


def transfer_detr_segment_weights(keras_model, state_dict):
    """Transfer ``DetrForSegmentation`` weights into :class:`DETRPanopticSegment`.

    The source state dict nests the entire detection model under ``detr.*``
    and adds top-level ``bbox_attention.*`` / ``mask_head.*`` keys for
    the segmentation head. This wraps :func:`transfer_detr_weights` for
    the detection portion (after stripping the ``detr.`` prefix) and
    copies the segmentation head weights directly.
    """

    detection_state = {}
    for k, v in state_dict.items():
        if k.startswith("detr."):
            detection_state[k[len("detr.") :]] = v
        else:
            detection_state[k] = v

    transfer_detr_weights(keras_model, detection_state)

    bbox_attn = keras_model.get_layer("bbox_attention")
    bbox_attn.q_linear.kernel.assign(
        np.transpose(state_dict["bbox_attention.q_proj.weight"])
    )
    bbox_attn.q_linear.bias.assign(state_dict["bbox_attention.q_proj.bias"])
    bbox_attn.k_linear.kernel.assign(
        np.transpose(state_dict["bbox_attention.k_proj.weight"])
    )
    bbox_attn.k_linear.bias.assign(state_dict["bbox_attention.k_proj.bias"])

    mask_head = keras_model.get_layer("mask_head")

    def _assign_conv(keras_conv, hf_weight_key):
        hf_weight = state_dict[f"{hf_weight_key}.weight"]
        hf_bias = state_dict[f"{hf_weight_key}.bias"]
        keras_conv.kernel.assign(np.transpose(hf_weight, (2, 3, 1, 0)))
        keras_conv.bias.assign(hf_bias)

    def _assign_gn(keras_gn, hf_weight_key):
        keras_gn.gamma.assign(state_dict[f"{hf_weight_key}.weight"])
        keras_gn.beta.assign(state_dict[f"{hf_weight_key}.bias"])

    _assign_conv(mask_head.lay1, "mask_head.conv1.conv")
    _assign_gn(mask_head.gn1, "mask_head.conv1.norm")
    _assign_conv(mask_head.lay2, "mask_head.conv2.conv")
    _assign_gn(mask_head.gn2, "mask_head.conv2.norm")

    fpn_stage_to_layer = [
        ("0", mask_head.adapter1, mask_head.lay3, mask_head.gn3),
        ("1", mask_head.adapter2, mask_head.lay4, mask_head.gn4),
        ("2", mask_head.adapter3, mask_head.lay5, mask_head.gn5),
    ]
    for stage_idx, adapter, refine_conv, refine_gn in fpn_stage_to_layer:
        _assign_conv(adapter, f"mask_head.fpn_stages.{stage_idx}.fpn_adapter")
        _assign_conv(refine_conv, f"mask_head.fpn_stages.{stage_idx}.refine.conv")
        _assign_gn(refine_gn, f"mask_head.fpn_stages.{stage_idx}.refine.norm")

    _assign_conv(mask_head.out_lay, "mask_head.output_conv")


# Per-variant recipes (relocated from detr_config.py). Models load from the Hub
# by repo id; these build the arch for conversion + drive the backfill.
DETR_VARIANTS = {
    "detr-resnet-50": {"backbone_variant": "ResNet50"},
    "detr-resnet-101": {"backbone_variant": "ResNet101"},
}

DETR_SEGMENT_VARIANTS = {
    "detr-resnet-50-panoptic": {"backbone_variant": "ResNet50", "num_classes": 251},
    "detr-resnet-101-panoptic": {"backbone_variant": "ResNet101", "num_classes": 251},
}


if __name__ == "__main__":
    import torch
    import torchvision.transforms as T
    from transformers import DetrForObjectDetection

    model_configs: List[Dict[str, object]] = [
        {
            "variant": "detr-resnet-50",
            "backbone_variant": "ResNet50",
            "hf_model_name": "facebook/detr-resnet-50",
            "output": "detr_resnet50.weights.h5",
            "image_size": 800,
            "num_classes": 92,
            "num_queries": 100,
        },
        {
            "variant": "detr-resnet-101",
            "backbone_variant": "ResNet101",
            "hf_model_name": "facebook/detr-resnet-101",
            "output": "detr_resnet101.weights.h5",
            "image_size": 800,
            "num_classes": 92,
            "num_queries": 100,
        },
    ]

    for cfg in model_configs:
        print(f"\n{'=' * 60}")
        print(f"Converting {cfg['hf_model_name']}...")
        print(f"{'=' * 60}")

        keras_model = DETRDetect(
            backbone_variant=cfg["backbone_variant"],
            image_size=cfg["image_size"],
            num_classes=cfg["num_classes"],
            num_queries=cfg["num_queries"],
        )

        torch_model: torch.nn.Module = DetrForObjectDetection.from_pretrained(
            cfg["hf_model_name"]
        ).eval()
        pytorch_state_dict: Dict[str, np.ndarray] = {
            k: v.cpu().numpy() for k, v in torch_model.state_dict().items()
        }

        transfer_detr_weights(keras_model, pytorch_state_dict)

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

        del keras_model, torch_model, pytorch_state_dict
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    segment_configs: List[Dict[str, object]] = [
        {
            "variant": "detr-resnet-50-panoptic",
            "backbone_variant": "ResNet50",
            "hf_model_name": "facebook/detr-resnet-50-panoptic",
            "output": "detr_resnet50_panoptic.weights.h5",
            "image_size": 800,
            "num_classes": 251,
            "num_queries": 100,
        },
        {
            "variant": "detr-resnet-101-panoptic",
            "backbone_variant": "ResNet101",
            "hf_model_name": "facebook/detr-resnet-101-panoptic",
            "output": "detr_resnet101_panoptic.weights.h5",
            "image_size": 800,
            "num_classes": 251,
            "num_queries": 100,
        },
    ]

    from transformers import DetrForSegmentation

    for cfg in segment_configs:
        print(f"\n{'=' * 60}")
        print(f"Converting {cfg['hf_model_name']}...")
        print(f"{'=' * 60}")

        keras_model = DETRPanopticSegment(
            backbone_variant=cfg["backbone_variant"],
            image_size=cfg["image_size"],
            num_classes=cfg["num_classes"],
            num_queries=cfg["num_queries"],
        )

        torch_model: torch.nn.Module = DetrForSegmentation.from_pretrained(
            cfg["hf_model_name"]
        ).eval()
        pytorch_state_dict: Dict[str, np.ndarray] = {
            k: v.cpu().numpy() for k, v in torch_model.state_dict().items()
        }

        transfer_detr_segment_weights(keras_model, pytorch_state_dict)

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
            hf_masks = hf_output.pred_masks.numpy()

        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        keras_input_norm = (test_input - mean) / std

        keras_output = keras_model(keras_input_norm.astype(np.float32), training=False)
        keras_logits = keras.ops.convert_to_numpy(keras_output["logits"])
        keras_boxes = keras.ops.convert_to_numpy(keras_output["pred_boxes"])
        keras_masks = keras.ops.convert_to_numpy(keras_output["pred_masks"])

        logits_diff = np.max(np.abs(hf_logits - keras_logits))
        boxes_diff = np.max(np.abs(hf_boxes - keras_boxes))
        masks_diff = np.max(np.abs(hf_masks - keras_masks))

        print(f"Max logits diff:  {logits_diff:.6f}")
        print(f"Max boxes diff:   {boxes_diff:.6f}")
        print(f"Max masks diff:   {masks_diff:.6f}")

        if logits_diff > 1e-3 or boxes_diff > 1e-3 or masks_diff > 1e-2:
            raise ValueError(
                "Model equivalence test failed - model outputs do not match "
                f"(logits: {logits_diff:.6f}, boxes: {boxes_diff:.6f}, "
                f"masks: {masks_diff:.6f})"
            )

        print("Model equivalence test passed!")

        model_filename = cfg["output"]
        keras_model.save_weights(model_filename)
        print(f"Model saved successfully as {model_filename}")

        del keras_model, torch_model, pytorch_state_dict
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
