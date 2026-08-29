import re
from typing import Dict

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
from zeromodels.models.table_transformer import TableTransformerDetect

WEIGHT_NAME_MAPPING: Dict[str, str] = {
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


def transfer_table_transformer_weights(keras_model, state_dict):
    state_dict = {
        k.replace("model.backbone.conv_encoder.model.", "model.backbone.model."): v
        for k, v in state_dict.items()
    }

    is_timm = "model.backbone.model.conv1.weight" in state_dict

    backbone_layers = [
        layer for layer in keras_model.layers if layer.name.startswith("backbone_")
    ]
    backbone_weights = []
    for layer in backbone_layers:
        for weight in layer.trainable_weights + layer.non_trainable_weights:
            backbone_weights.append((weight, layer.name, weight.name))

    for keras_weight, layer_name, weight_name in tqdm(
        backbone_weights, desc="Transferring backbone weights"
    ):
        keras_weight_name = f"{layer_name}_{weight_name}"
        # timm layout rewrites the flattened name; HF-native builds the path per module.
        if is_timm:
            torch_weight_name = keras_weight_name
            for old, new in WEIGHT_NAME_MAPPING.items():
                torch_weight_name = torch_weight_name.replace(old, new)
        else:
            sub = "convolution" if "conv" in layer_name else "normalization"
            suffix = WEIGHT_NAME_MAPPING[weight_name.replace("_", ".")]
            if layer_name in ("backbone_conv1", "backbone_bn1"):
                torch_weight_name = (
                    f"model.backbone.model.embedder.embedder.{sub}.{suffix}"
                )
            else:
                match = re.match(r"backbone_layer(\d+)_(\d+)_(.+)", layer_name)
                stage = int(match.group(1)) - 1
                block = int(match.group(2))
                tail = match.group(3)
                if tail in ("conv1", "bn1"):
                    path = f"encoder.stages.{stage}.layers.{block}.layer.0.{sub}"
                elif tail in ("conv2", "bn2"):
                    path = f"encoder.stages.{stage}.layers.{block}.layer.1.{sub}"
                else:  # downsample_conv / downsample_bn
                    path = f"encoder.stages.{stage}.layers.{block}.shortcut.{sub}"
                torch_weight_name = f"model.backbone.model.{path}.{suffix}"

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

        sa_mapping = {f"{k_prefix}_self_attn_": "", "kernel": "weight"}
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
            f"{hf_prefix}.fc1",
            name_mapping=dense_mapping,
        )
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_fc2"),
            state_dict,
            f"{hf_prefix}.fc2",
            name_mapping=dense_mapping,
        )
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_final_layer_norm"),
            state_dict,
            f"{hf_prefix}.final_layer_norm",
            name_mapping=ln_mapping,
        )

    transfer_nested_layer_weights(
        keras_model.get_layer("encoder_layernorm"),
        state_dict,
        "model.encoder.layernorm",
        name_mapping=ln_mapping,
    )

    for i in tqdm(
        range(keras_model.num_decoder_layers), desc="Transferring decoder weights"
    ):
        hf_prefix = f"model.decoder.layers.{i}"
        k_prefix = f"decoder_layers_{i}"

        sa_mapping = {f"{k_prefix}_self_attn_": "", "kernel": "weight"}
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
        ca_mapping = {f"{k_prefix}_encoder_attn_": "", "kernel": "weight"}
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
            f"{hf_prefix}.fc1",
            name_mapping=dense_mapping,
        )
        transfer_nested_layer_weights(
            keras_model.get_layer(f"{k_prefix}_fc2"),
            state_dict,
            f"{hf_prefix}.fc2",
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


TABLE_TRANSFORMER_VARIANTS = {
    "table-transformer-detection": "microsoft/table-transformer-detection",
    "table-transformer-structure-recognition": (
        "microsoft/table-transformer-structure-recognition"
    ),
    "table-transformer-structure-recognition-v1.1-all": (
        "microsoft/table-transformer-structure-recognition-v1.1-all"
    ),
    "table-transformer-structure-recognition-v1.1-fin": (
        "microsoft/table-transformer-structure-recognition-v1.1-fin"
    ),
    "table-transformer-structure-recognition-v1.1-pub": (
        "microsoft/table-transformer-structure-recognition-v1.1-pub"
    ),
}


def load_hf_model(hf_id, raw_config):
    import transformers

    if raw_config.get("dilation") is None:
        cfg = transformers.TableTransformerConfig.from_dict(
            {**raw_config, "dilation": False}
        )
        return transformers.TableTransformerForObjectDetection.from_pretrained(
            hf_id, config=cfg
        ).eval()
    return transformers.TableTransformerForObjectDetection.from_pretrained(hf_id).eval()


if __name__ == "__main__":
    import gc
    import json

    import torch
    from huggingface_hub import hf_hub_download

    from zeromodels.conversion.hf_download_utils import download_hf_state_dict

    for variant, hf_id in TABLE_TRANSFORMER_VARIANTS.items():
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        with open(hf_hub_download(hf_id, "config.json"), encoding="utf-8") as f:
            hf_config = json.load(f)

        keras_model = TableTransformerDetect(
            **TableTransformerDetect.config_from_hf(hf_config),
            image_size=800,
        )
        state = download_hf_state_dict(hf_id)
        transfer_table_transformer_weights(keras_model, state)

        hf_model = load_hf_model(hf_id, hf_config)

        np.random.seed(0)
        test_input = np.random.rand(1, 800, 800, 3).astype(np.float32)
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        keras_input = ((test_input - mean) / std).astype(np.float32)

        hf_input = torch.tensor(keras_input).permute(0, 3, 1, 2)
        with torch.no_grad():
            hf_out = hf_model(pixel_values=hf_input)
            hf_logits = hf_out.logits.numpy()
            hf_boxes = hf_out.pred_boxes.numpy()

        keras_out = keras_model(keras_input, training=False)
        keras_logits = keras.ops.convert_to_numpy(keras_out["logits"])
        keras_boxes = keras.ops.convert_to_numpy(keras_out["pred_boxes"])

        logits_diff = float(np.max(np.abs(hf_logits - keras_logits)))
        boxes_diff = float(np.max(np.abs(hf_boxes - keras_boxes)))
        cos = float(
            np.dot(hf_logits.ravel(), keras_logits.ravel())
            / (np.linalg.norm(hf_logits.ravel()) * np.linalg.norm(keras_logits.ravel()))
        )
        print(f"  logits max|diff|={logits_diff:.3e}  cosine={cos:.8f}")
        print(f"  boxes  max|diff|={boxes_diff:.3e}")

        if logits_diff > 1e-3 or boxes_diff > 1e-3:
            raise ValueError(
                f"Parity failed for {variant} "
                f"(logits {logits_diff:.3e}, boxes {boxes_diff:.3e})"
            )

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, hf_model, state
        keras.backend.clear_session()
        gc.collect()
