import gc
import os
from typing import Any, Dict, List

import keras
import numpy as np
from PIL import Image
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
from zeromodels.models.owlvit import OwlViTDetect

weight_name_mapping: Dict[str, str] = {
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "embeddings": "weight",
}


def has_layer(keras_model, name):
    try:
        keras_model.get_layer(name)
        return True
    except ValueError:
        return False


def transfer_owlvit_encoder_weights(keras_model, state_dict, prefix=None):
    if prefix is None:
        prefix = (
            "owlvit."
            if "owlvit.vision_model.embeddings.class_embedding" in state_dict
            else ""
        )

    has_vision = has_layer(keras_model, "vision_model_embeddings")
    has_text = has_layer(keras_model, "text_model_embeddings")

    if has_vision:
        vision_num_layers = getattr(keras_model, "vision_num_layers", None)
        if vision_num_layers is None:
            vision_num_layers = getattr(
                keras_model, "vision_model", keras_model
            ).vision_num_layers

        embed = keras_model.get_layer("vision_model_embeddings")
        cls_torch_name = f"{prefix}vision_model.embeddings.class_embedding"
        if cls_torch_name not in state_dict:
            raise WeightMappingError(
                "vision_model_embeddings/class_embedding", cls_torch_name
            )
        cls_torch = state_dict[cls_torch_name]
        if not compare_keras_torch_names(
            "class_embedding", embed.class_embedding, cls_torch_name, cls_torch
        ):
            raise WeightShapeMismatchError(
                "vision_model_embeddings/class_embedding",
                tuple(embed.class_embedding.shape),
                cls_torch_name,
                cls_torch.shape,
            )
        embed.class_embedding.assign(cls_torch)

        patch_torch_name = f"{prefix}vision_model.embeddings.patch_embedding.weight"
        if patch_torch_name not in state_dict:
            raise WeightMappingError(
                "vision_model_embeddings/patch_embedding/kernel", patch_torch_name
            )
        transfer_weights(
            "conv_kernel",
            embed.patch_embedding.kernel,
            state_dict[patch_torch_name],
        )

        pos_torch_name = f"{prefix}vision_model.embeddings.position_embedding.weight"
        if pos_torch_name not in state_dict:
            raise WeightMappingError(
                "vision_model_embeddings/position_embedding/embeddings", pos_torch_name
            )
        transfer_weights(
            "position_embedding/embeddings",
            embed.position_embedding.weights[0],
            state_dict[pos_torch_name],
        )

        transfer_nested_layer_weights(
            keras_layer=keras_model.get_layer("vision_model_pre_layernorm"),
            torch_weights_dict=state_dict,
            torch_prefix=f"{prefix}vision_model.pre_layernorm",
            name_mapping=weight_name_mapping,
        )
        transfer_nested_layer_weights(
            keras_layer=keras_model.get_layer("vision_model_post_layernorm"),
            torch_weights_dict=state_dict,
            torch_prefix=f"{prefix}vision_model.post_layernorm",
            name_mapping=weight_name_mapping,
        )

        for i in tqdm(
            range(vision_num_layers), desc="Transferring vision encoder weights"
        ):
            kp = f"vision_model_layers_{i}"
            tp = f"{prefix}vision_model.encoder.layers.{i}"
            for sublayer in ("self_attn", "layer_norm1", "layer_norm2"):
                transfer_nested_layer_weights(
                    keras_layer=keras_model.get_layer(f"{kp}_{sublayer}"),
                    torch_weights_dict=state_dict,
                    torch_prefix=f"{tp}.{sublayer}",
                    name_mapping=weight_name_mapping,
                )
            for fc in ("fc1", "fc2"):
                transfer_nested_layer_weights(
                    keras_layer=keras_model.get_layer(f"{kp}_mlp_{fc}"),
                    torch_weights_dict=state_dict,
                    torch_prefix=f"{tp}.mlp.{fc}",
                    name_mapping=weight_name_mapping,
                )

    if has_text:
        text_layers = getattr(keras_model, "text_num_layers", None)
        if text_layers is None:
            text_layers = getattr(
                keras_model, "text_model", keras_model
            ).text_num_layers

        transfer_nested_layer_weights(
            keras_layer=keras_model.get_layer("text_model_embeddings"),
            torch_weights_dict=state_dict,
            torch_prefix=f"{prefix}text_model.embeddings",
            name_mapping=weight_name_mapping,
        )
        transfer_nested_layer_weights(
            keras_layer=keras_model.get_layer("text_model_final_layer_norm"),
            torch_weights_dict=state_dict,
            torch_prefix=f"{prefix}text_model.final_layer_norm",
            name_mapping=weight_name_mapping,
        )

        for i in tqdm(range(text_layers), desc="Transferring text encoder weights"):
            kp = f"text_model_layers_{i}"
            tp = f"{prefix}text_model.encoder.layers.{i}"
            for sublayer in ("self_attn", "layer_norm1", "layer_norm2"):
                transfer_nested_layer_weights(
                    keras_layer=keras_model.get_layer(f"{kp}_{sublayer}"),
                    torch_weights_dict=state_dict,
                    torch_prefix=f"{tp}.{sublayer}",
                    name_mapping=weight_name_mapping,
                )
            for fc in ("fc1", "fc2"):
                transfer_nested_layer_weights(
                    keras_layer=keras_model.get_layer(f"{kp}_mlp_{fc}"),
                    torch_weights_dict=state_dict,
                    torch_prefix=f"{tp}.mlp.{fc}",
                    name_mapping=weight_name_mapping,
                )

    if has_layer(keras_model, "text_projection"):
        transfer_nested_layer_weights(
            keras_layer=keras_model.get_layer("text_projection"),
            torch_weights_dict=state_dict,
            torch_prefix=f"{prefix}text_projection",
            name_mapping=weight_name_mapping,
        )


def transfer_owlvit_detection_weights(keras_model, state_dict):
    transfer_owlvit_encoder_weights(keras_model, state_dict, prefix="owlvit.")

    for d in ("dense0", "dense1", "dense2"):
        transfer_nested_layer_weights(
            keras_layer=keras_model.get_layer(f"box_head_{d}"),
            torch_weights_dict=state_dict,
            torch_prefix=f"box_head.{d}",
            name_mapping=weight_name_mapping,
        )
    for d in ("dense0", "logit_shift", "logit_scale"):
        transfer_nested_layer_weights(
            keras_layer=keras_model.get_layer(f"class_head_{d}"),
            torch_weights_dict=state_dict,
            torch_prefix=f"class_head.{d}",
            name_mapping=weight_name_mapping,
        )
    transfer_nested_layer_weights(
        keras_layer=keras_model.get_layer("layer_norm"),
        torch_weights_dict=state_dict,
        torch_prefix="layer_norm",
        name_mapping=weight_name_mapping,
    )


# Per-variant recipes (relocated from owlvit_config.py). Models load from the
# Hub by repo id; these build the arch for conversion + drive the backfill.
OWLVIT_VARIANTS = {
    "owlvit-base-patch32": {
        "vision_image_size": 768,
        "vision_patch_size": 32,
        "vision_hidden_dim": 768,
        "vision_intermediate_size": 3072,
        "vision_num_layers": 12,
        "vision_num_heads": 12,
        "text_hidden_dim": 512,
        "text_intermediate_size": 2048,
        "text_num_heads": 8,
        "projection_dim": 512,
    },
    "owlvit-base-patch16": {
        "vision_image_size": 768,
        "vision_patch_size": 16,
        "vision_hidden_dim": 768,
        "vision_intermediate_size": 3072,
        "vision_num_layers": 12,
        "vision_num_heads": 12,
        "text_hidden_dim": 512,
        "text_intermediate_size": 2048,
        "text_num_heads": 8,
        "projection_dim": 512,
    },
    "owlvit-large-patch14": {
        "vision_image_size": 840,
        "vision_patch_size": 14,
        "vision_hidden_dim": 1024,
        "vision_intermediate_size": 4096,
        "vision_num_layers": 24,
        "vision_num_heads": 16,
        "text_hidden_dim": 768,
        "text_intermediate_size": 3072,
        "text_num_heads": 16,
        "projection_dim": 768,
    },
}


if __name__ == "__main__":
    import torch
    from transformers import OwlViTForObjectDetection, OwlViTProcessor

    model_configs: List[Dict[str, Any]] = [
        {
            "variant": "owlvit-base-patch32",
            "hf_model_name": "google/owlvit-base-patch32",
            "image_size": 768,
        },
        {
            "variant": "owlvit-base-patch16",
            "hf_model_name": "google/owlvit-base-patch16",
            "image_size": 768,
        },
        {
            "variant": "owlvit-large-patch14",
            "hf_model_name": "google/owlvit-large-patch14",
            "image_size": 840,
        },
    ]

    for cfg in model_configs:
        print(f"\n{'=' * 60}")
        print(f"Converting {cfg['hf_model_name']}...")
        print(f"{'=' * 60}")

        image_size: int = cfg["image_size"]

        keras_model = OwlViTDetect(**OWLVIT_VARIANTS[cfg["variant"]])

        torch_model: torch.nn.Module = OwlViTForObjectDetection.from_pretrained(
            cfg["hf_model_name"],
            token=os.environ.get("HF_TOKEN"),
        ).eval()

        pytorch_state_dict: Dict[str, np.ndarray] = {
            k: v.cpu().numpy() for k, v in torch_model.state_dict().items()
        }

        transfer_owlvit_detection_weights(keras_model, pytorch_state_dict)

        print("\nVerifying model equivalence...")

        rng = np.random.default_rng(42)
        img_np = rng.integers(0, 255, size=(image_size, image_size, 3), dtype=np.uint8)
        image = Image.fromarray(img_np)
        text_queries = [["a photo of a cat", "a photo of a dog"]]

        hf_processor = OwlViTProcessor.from_pretrained(
            cfg["hf_model_name"],
            token=os.environ.get("HF_TOKEN"),
        )
        hf_inputs = hf_processor(text=text_queries, images=image, return_tensors="pt")

        with torch.no_grad():
            hf_output = torch_model(
                input_ids=hf_inputs["input_ids"],
                pixel_values=hf_inputs["pixel_values"],
                attention_mask=hf_inputs.get("attention_mask"),
            )
            hf_logits = hf_output.logits.cpu().numpy()
            hf_boxes = hf_output.pred_boxes.cpu().numpy()

        pix_chw = hf_inputs["pixel_values"].cpu().numpy()
        input_ids_np = hf_inputs["input_ids"].cpu().numpy()
        del torch_model, hf_output, hf_inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pix_hwc = np.transpose(pix_chw, (0, 2, 3, 1))
        keras_inputs = {
            "pixel_values": keras.ops.convert_to_tensor(pix_hwc, dtype="float32"),
            "input_ids": keras.ops.convert_to_tensor(input_ids_np, dtype="int32"),
        }
        keras_output = keras_model(keras_inputs)
        keras_logits = keras.ops.convert_to_numpy(keras_output["logits"])
        keras_boxes = keras.ops.convert_to_numpy(keras_output["pred_boxes"])

        logits_diff = float(np.max(np.abs(hf_logits - keras_logits)))
        boxes_diff = float(np.max(np.abs(hf_boxes - keras_boxes)))

        hf_flat = hf_logits.flatten()
        k_flat = keras_logits.flatten()
        logits_cos = float(
            np.dot(hf_flat, k_flat)
            / (np.linalg.norm(hf_flat) * np.linalg.norm(k_flat) + 1e-8)
        )

        print(f"Max logits diff:   {logits_diff:.6f}")
        print(f"Max boxes diff:    {boxes_diff:.6f}")
        print(f"Logits cosine sim: {logits_cos:.6f}")

        if logits_cos < 0.95:
            raise ValueError(
                f"Equivalence test failed: logits cosine similarity "
                f"{logits_cos:.4f} < 0.95"
            )

        print("Model equivalence test passed!")

        model_filename = (
            f"{cfg['hf_model_name'].split('/')[-1].replace('-', '_')}.weights.h5"
        )
        keras_model.save_weights(model_filename)
        print(f"Model saved successfully as {model_filename}")

        del keras_model, pytorch_state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
