from typing import Dict

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

HF_SOURCES: Dict[str, str] = {
    "rfdetr-nano": "Roboflow/rf-detr-nano",
    "rfdetr-small": "Roboflow/rf-detr-small",
    "rfdetr-medium": "Roboflow/rf-detr-medium",
    "rfdetr-base": "Roboflow/rf-detr-base",
    "rfdetr-large": "Roboflow/rf-detr-large",
}

weight_name_mapping: Dict[str, str] = {
    "backbone_encoder_layer_": "backbone.0.encoder.encoder.encoder.layer.",
    "backbone_encoder_layernorm_": "backbone.0.encoder.encoder.layernorm.",
    "_attention_query_": ".attention.attention.query.",
    "_attention_key_": ".attention.attention.key.",
    "_attention_value_": ".attention.attention.value.",
    "_attention_out_proj_": ".attention.output.dense.",
    "_norm1_": ".norm1.",
    "_norm2_": ".norm2.",
    "_layer_scale1_": ".layer_scale1.",
    "_layer_scale2_": ".layer_scale2.",
    "_mlp_fc1_": ".mlp.fc1.",
    "_mlp_fc2_": ".mlp.fc2.",
    "_mlp_weights_in_": ".mlp.fc1.",
    "_mlp_weights_out_": ".mlp.fc2.",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}

decoder_name_mapping: Dict[str, str] = {
    "self_attn_out_proj": "self_attn.out_proj",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}


def transfer_rf_detr_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    sd = state_dict

    layer_names = {layer.name for layer in keras_model.layers}
    dec_layers = 0
    while f"decoder_layer_{dec_layers}" in layer_names:
        dec_layers += 1

    backbone_encoder_weights = []
    for layer in keras_model.layers:
        if layer.name.startswith("backbone_encoder_layer_") or (
            layer.name == "backbone_encoder_layernorm"
        ):
            for weight in layer.trainable_weights:
                backbone_encoder_weights.append((weight, f"{layer.name}_{weight.name}"))
            for weight in layer.non_trainable_weights:
                backbone_encoder_weights.append((weight, f"{layer.name}_{weight.name}"))

    for keras_weight, keras_weight_name in tqdm(
        backbone_encoder_weights,
        total=len(backbone_encoder_weights),
        desc="Transferring backbone encoder weights",
    ):
        torch_weight_name = keras_weight_name
        for keras_name_part, torch_name_part in weight_name_mapping.items():
            torch_weight_name = torch_weight_name.replace(
                keras_name_part, torch_name_part
            )

        if torch_weight_name not in sd:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = sd[torch_weight_name]
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

    transfer_nested_layer_weights(
        keras_model.get_layer("backbone_embeddings"),
        sd,
        "backbone.0.encoder.encoder.embeddings",
        name_mapping={
            "conv_projection": "patch_embeddings.projection",
            "kernel": "weight",
        },
    )

    pt_proj = "backbone.0.projector.stages.0"
    projector_conv_ln_pairs = [
        ("projector_c2f_cv1_conv", f"{pt_proj}.0.cv1.conv"),
        ("projector_c2f_cv1_ln", f"{pt_proj}.0.cv1.bn"),
        ("projector_c2f_cv2_conv", f"{pt_proj}.0.cv2.conv"),
        ("projector_c2f_cv2_ln", f"{pt_proj}.0.cv2.bn"),
    ]
    for b_idx in range(3):
        for cv in ("cv1", "cv2"):
            projector_conv_ln_pairs.append(
                (
                    f"projector_c2f_bottleneck_{b_idx}_{cv}_conv",
                    f"{pt_proj}.0.m.{b_idx}.{cv}.conv",
                )
            )
            projector_conv_ln_pairs.append(
                (
                    f"projector_c2f_bottleneck_{b_idx}_{cv}_ln",
                    f"{pt_proj}.0.m.{b_idx}.{cv}.bn",
                )
            )

    for keras_name, pt_name in tqdm(
        projector_conv_ln_pairs, desc="Transferring projector weights"
    ):
        layer = keras_model.get_layer(keras_name)
        if keras_name.endswith("_conv"):
            layer.weights[0].assign(np.transpose(sd[f"{pt_name}.weight"], (2, 3, 1, 0)))
        else:
            layer.weights[0].assign(sd[f"{pt_name}.weight"])
            layer.weights[1].assign(sd[f"{pt_name}.bias"])

    proj_ln = keras_model.get_layer("projector_ln")
    proj_ln.weights[0].assign(sd[f"{pt_proj}.1.weight"])
    proj_ln.weights[1].assign(sd[f"{pt_proj}.1.bias"])

    enc_output = keras_model.get_layer("enc_output_0")
    enc_output.weights[0].assign(sd["transformer.enc_output.0.weight"].T)
    enc_output.weights[1].assign(sd["transformer.enc_output.0.bias"])

    enc_output_norm = keras_model.get_layer("enc_output_norm_0")
    enc_output_norm.weights[0].assign(sd["transformer.enc_output_norm.0.weight"])
    enc_output_norm.weights[1].assign(sd["transformer.enc_output_norm.0.bias"])

    enc_cls = keras_model.get_layer("enc_out_class_embed_0")
    enc_cls.weights[0].assign(sd["transformer.enc_out_class_embed.0.weight"].T)
    enc_cls.weights[1].assign(sd["transformer.enc_out_class_embed.0.bias"])

    for i in range(3):
        bbox_layer = keras_model.get_layer(f"enc_bbox_{i}")
        bbox_layer.weights[0].assign(
            sd[f"transformer.enc_out_bbox_embed.0.layers.{i}.weight"].T
        )
        bbox_layer.weights[1].assign(
            sd[f"transformer.enc_out_bbox_embed.0.layers.{i}.bias"]
        )

    for i in range(2):
        rph = keras_model.get_layer(f"ref_point_head_{i}")
        rph.weights[0].assign(
            sd[f"transformer.decoder.ref_point_head.layers.{i}.weight"].T
        )
        rph.weights[1].assign(sd[f"transformer.decoder.ref_point_head.layers.{i}.bias"])

    for i in tqdm(range(dec_layers), desc="Transferring decoder weights"):
        pt_dl = f"transformer.decoder.layers.{i}"
        k_dl = f"decoder_layer_{i}"
        dec_layer = keras_model.get_layer(k_dl)

        q_w, k_w, v_w = np.split(sd[f"{pt_dl}.self_attn.in_proj_weight"], 3, axis=0)
        q_b, k_b, v_b = np.split(sd[f"{pt_dl}.self_attn.in_proj_bias"], 3, axis=0)
        weight_dict = {w.path: w for w in dec_layer.weights}
        weight_dict[f"{k_dl}/self_attn_q_proj/kernel"].assign(q_w.T)
        weight_dict[f"{k_dl}/self_attn_q_proj/bias"].assign(q_b)
        weight_dict[f"{k_dl}/self_attn_k_proj/kernel"].assign(k_w.T)
        weight_dict[f"{k_dl}/self_attn_k_proj/bias"].assign(k_b)
        weight_dict[f"{k_dl}/self_attn_v_proj/kernel"].assign(v_w.T)
        weight_dict[f"{k_dl}/self_attn_v_proj/bias"].assign(v_b)

        transfer_nested_layer_weights(
            dec_layer,
            sd,
            pt_dl,
            name_mapping=decoder_name_mapping,
            skip_paths=["self_attn_q_proj", "self_attn_k_proj", "self_attn_v_proj"],
        )

    dec_norm = keras_model.get_layer("decoder_norm")
    dec_norm.weights[0].assign(sd["transformer.decoder.norm.weight"])
    dec_norm.weights[1].assign(sd["transformer.decoder.norm.bias"])

    num_queries = int(keras_model.get_layer("query_feat_embed").weights[0].shape[0])

    keras_model.get_layer("refpoint_embed_layer").weights[0].assign(
        sd["refpoint_embed.weight"][:num_queries]
    )
    keras_model.get_layer("query_feat_embed").weights[0].assign(
        sd["query_feat.weight"][:num_queries]
    )

    cls_embed = keras_model.get_layer("class_embed")
    cls_embed.weights[0].assign(sd["class_embed.weight"].T)
    cls_embed.weights[1].assign(sd["class_embed.bias"])

    for i in range(3):
        bbox_layer = keras_model.get_layer(f"bbox_embed_{i}")
        bbox_layer.weights[0].assign(sd[f"bbox_embed.layers.{i}.weight"].T)
        bbox_layer.weights[1].assign(sd[f"bbox_embed.layers.{i}.bias"])


HF_SEG_SOURCES: Dict[str, str] = {
    "rfdetr-seg-preview": "Roboflow/rf-detr-seg-preview",
    "rfdetr-seg-nano": "Roboflow/rf-detr-seg-nano",
    "rfdetr-seg-small": "Roboflow/rf-detr-seg-small",
    "rfdetr-seg-medium": "Roboflow/rf-detr-seg-medium",
    "rfdetr-seg-large": "Roboflow/rf-detr-seg-large",
    "rfdetr-seg-xlarge": "Roboflow/rf-detr-seg-xlarge",
    "rfdetr-seg-xxlarge": "Roboflow/rf-detr-seg-xxlarge",
}


def transfer_rf_detr_seg_weights(
    keras_model, state_dict: Dict[str, np.ndarray]
) -> None:
    sd = state_dict
    transfer_rf_detr_weights(keras_model, sd)

    layer_names = {layer.name for layer in keras_model.layers}
    num_blocks = 0
    while f"seg_block_{num_blocks}_dwconv" in layer_names:
        num_blocks += 1

    for i in range(num_blocks):
        prefix = f"segmentation_head.blocks.{i}"
        dwconv = keras_model.get_layer(f"seg_block_{i}_dwconv")
        dwconv.weights[0].assign(
            np.transpose(sd[f"{prefix}.dwconv.weight"], (2, 3, 0, 1))
        )
        dwconv.weights[1].assign(sd[f"{prefix}.dwconv.bias"])
        norm = keras_model.get_layer(f"seg_block_{i}_norm")
        norm.weights[0].assign(sd[f"{prefix}.norm.weight"])
        norm.weights[1].assign(sd[f"{prefix}.norm.bias"])
        pwconv = keras_model.get_layer(f"seg_block_{i}_pwconv")
        pwconv.weights[0].assign(sd[f"{prefix}.pwconv1.weight"].T)
        pwconv.weights[1].assign(sd[f"{prefix}.pwconv1.bias"])

    sp = keras_model.get_layer("seg_spatial_features_proj")
    sp.weights[0].assign(
        np.transpose(sd["segmentation_head.spatial_features_proj.weight"], (2, 3, 1, 0))
    )
    sp.weights[1].assign(sd["segmentation_head.spatial_features_proj.bias"])

    qn = keras_model.get_layer("seg_query_features_block_norm")
    qn.weights[0].assign(sd["segmentation_head.query_features_block.norm_in.weight"])
    qn.weights[1].assign(sd["segmentation_head.query_features_block.norm_in.bias"])
    fc1 = keras_model.get_layer("seg_query_features_block_fc1")
    fc1.weights[0].assign(
        sd["segmentation_head.query_features_block.layers.0.weight"].T
    )
    fc1.weights[1].assign(sd["segmentation_head.query_features_block.layers.0.bias"])
    fc2 = keras_model.get_layer("seg_query_features_block_fc2")
    fc2.weights[0].assign(
        sd["segmentation_head.query_features_block.layers.2.weight"].T
    )
    fc2.weights[1].assign(sd["segmentation_head.query_features_block.layers.2.bias"])
    qp = keras_model.get_layer("seg_query_features_proj")
    qp.weights[0].assign(sd["segmentation_head.query_features_proj.weight"].T)
    qp.weights[1].assign(sd["segmentation_head.query_features_proj.bias"])

    keras_model.get_layer("seg_bias").weights[0].assign(sd["segmentation_head.bias"])


# Per-variant recipes (relocated from rf_detr_config.py). Models load from the
# Hub by repo id; these build the arch for conversion + drive the backfill.
RF_DETR_DETECT_VARIANTS = {
    "rfdetr-nano": {
        "out_feature_indexes": [3, 6, 9, 12],
        "patch_size": 16,
        "num_windows": 2,
        "positional_encoding_size": 24,
        "resolution": 384,
        "dec_layers": 2,
    },
    "rfdetr-small": {
        "out_feature_indexes": [3, 6, 9, 12],
        "patch_size": 16,
        "num_windows": 2,
        "positional_encoding_size": 32,
        "resolution": 512,
        "dec_layers": 3,
    },
    "rfdetr-medium": {
        "out_feature_indexes": [3, 6, 9, 12],
        "patch_size": 16,
        "num_windows": 2,
        "positional_encoding_size": 36,
        "resolution": 576,
        "dec_layers": 4,
    },
    "rfdetr-base": {
        "out_feature_indexes": [2, 5, 8, 11],
        "patch_size": 14,
        "num_windows": 4,
        "positional_encoding_size": 37,
        "resolution": 560,
        "dec_layers": 3,
    },
    "rfdetr-large": {
        "out_feature_indexes": [3, 6, 9, 12],
        "patch_size": 16,
        "num_windows": 2,
        "positional_encoding_size": 44,
        "resolution": 704,
        "dec_layers": 4,
    },
}

RF_DETR_SEGMENT_VARIANTS = {
    "rfdetr-seg-preview": {
        "out_feature_indexes": [3, 6, 9, 12],
        "patch_size": 12,
        "num_windows": 2,
        "positional_encoding_size": 36,
        "resolution": 432,
        "dec_layers": 4,
        "num_queries": 200,
    },
    "rfdetr-seg-nano": {
        "out_feature_indexes": [3, 6, 9, 12],
        "patch_size": 12,
        "num_windows": 1,
        "positional_encoding_size": 26,
        "resolution": 312,
        "dec_layers": 4,
        "num_queries": 100,
    },
    "rfdetr-seg-small": {
        "out_feature_indexes": [3, 6, 9, 12],
        "patch_size": 12,
        "num_windows": 2,
        "positional_encoding_size": 32,
        "resolution": 384,
        "dec_layers": 4,
        "num_queries": 100,
    },
    "rfdetr-seg-medium": {
        "out_feature_indexes": [3, 6, 9, 12],
        "patch_size": 12,
        "num_windows": 2,
        "positional_encoding_size": 36,
        "resolution": 432,
        "dec_layers": 5,
        "num_queries": 200,
    },
    "rfdetr-seg-large": {
        "out_feature_indexes": [3, 6, 9, 12],
        "patch_size": 12,
        "num_windows": 2,
        "positional_encoding_size": 42,
        "resolution": 504,
        "dec_layers": 5,
        "num_queries": 300,
    },
    "rfdetr-seg-xlarge": {
        "out_feature_indexes": [3, 6, 9, 12],
        "patch_size": 12,
        "num_windows": 2,
        "positional_encoding_size": 52,
        "resolution": 624,
        "dec_layers": 6,
        "num_queries": 300,
    },
    "rfdetr-seg-xxlarge": {
        "out_feature_indexes": [3, 6, 9, 12],
        "patch_size": 12,
        "num_windows": 2,
        "positional_encoding_size": 64,
        "resolution": 768,
        "dec_layers": 6,
        "num_queries": 300,
    },
}


if __name__ == "__main__":
    import keras
    import torch

    from zeromodels.conversion.hf_download_utils import (
        download_hf_state_dict,
    )
    from zeromodels.models.rf_detr.rf_detr_model import RFDETRDetect

    def cosine(a, b):
        a, b = a.ravel(), b.ravel()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    for variant, hf_id in HF_SOURCES.items():
        config = RF_DETR_DETECT_VARIANTS[variant]
        res = config["resolution"]
        print(f"\n{'=' * 60}\nConverting {variant}  <-  {hf_id}\n{'=' * 60}")

        keras_model = RFDETRDetect(
            **config,
            backbone_use_swiglu=False,
            num_register_tokens=0,
            image_size=res,
        )
        _ = keras_model(keras.random.uniform((1, res, res, 3), dtype="float32"))
        print(f"  Parameters: {keras_model.count_params():,}")

        state_dict = download_hf_state_dict(hf_id)
        print(f"  HF keys: {len(state_dict)}")
        transfer_rf_detr_weights(keras_model, state_dict)

        print("\nVerifying parity against transformers RfDetrForObjectDetection...")
        from transformers import RfDetrForObjectDetection

        hf_model = RfDetrForObjectDetection.from_pretrained(hf_id).eval()

        np.random.seed(42)
        test_input = np.random.rand(1, res, res, 3).astype(np.float32)
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        norm = ((test_input - mean) / std).astype(np.float32)

        with torch.no_grad():
            pt_out = hf_model(pixel_values=torch.from_numpy(norm).permute(0, 3, 1, 2))
        pt_logits = pt_out.logits.cpu().numpy()
        pt_boxes = pt_out.pred_boxes.cpu().numpy()

        keras_out = keras_model(norm, training=False)
        keras_logits = keras.ops.convert_to_numpy(keras_out["logits"])
        keras_boxes = keras.ops.convert_to_numpy(keras_out["pred_boxes"])

        logits_diff = float(np.max(np.abs(pt_logits - keras_logits)))
        boxes_diff = float(np.max(np.abs(pt_boxes - keras_boxes)))
        logits_cos = cosine(pt_logits, keras_logits)
        boxes_cos = cosine(pt_boxes, keras_boxes)
        class_agree = float(
            (pt_logits[0].argmax(-1) == keras_logits[0].argmax(-1)).mean()
        )

        print(f"  Max logits diff: {logits_diff:.6f}   cosine: {logits_cos:.6f}")
        print(f"  Max boxes diff:  {boxes_diff:.6f}   cosine: {boxes_cos:.6f}")
        print(f"  Class agreement: {class_agree * 100:.1f}%")

        if logits_cos < 0.99 or boxes_cos < 0.99 or class_agree < 0.95:
            raise ValueError(
                f"{variant}: HF parity failed (logits cos {logits_cos:.4f}, "
                f"boxes cos {boxes_cos:.4f}, class agreement {class_agree:.3f})"
            )

        print("Model equivalence test passed!")

        keras_model.save_weights(f"rf_detr_{variant.split('-')[1]}.weights.h5")
        print(f"  Saved -> rf_detr_{variant.split('-')[1]}.weights.h5")

        del keras_model, hf_model, state_dict
        keras.backend.clear_session()

    from zeromodels.models.rf_detr.rf_detr_model import RFDETRInstanceSegment

    print(f"\n\n{'=' * 60}\nSegmentation variants\n{'=' * 60}")
    for variant, hf_id in HF_SEG_SOURCES.items():
        seg_config = RF_DETR_SEGMENT_VARIANTS[variant]
        res = seg_config["resolution"]
        print(f"\n{'=' * 60}\nConverting {variant}  <-  {hf_id}\n{'=' * 60}")

        keras_model = RFDETRInstanceSegment(
            **seg_config,
            backbone_use_swiglu=False,
            num_register_tokens=0,
            image_size=res,
        )
        _ = keras_model(keras.random.uniform((1, res, res, 3), dtype="float32"))
        print(f"  Parameters: {keras_model.count_params():,}")

        state_dict = download_hf_state_dict(hf_id)
        print(f"  HF keys: {len(state_dict)}")
        transfer_rf_detr_seg_weights(keras_model, state_dict)

        print(
            "\nVerifying parity against transformers RfDetrForInstanceSegmentation..."
        )
        from transformers import RfDetrForInstanceSegmentation

        hf_model = RfDetrForInstanceSegmentation.from_pretrained(hf_id).eval()

        np.random.seed(42)
        test_input = np.random.rand(1, res, res, 3).astype(np.float32)
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        norm = ((test_input - mean) / std).astype(np.float32)

        with torch.no_grad():
            pt_out = hf_model(pixel_values=torch.from_numpy(norm).permute(0, 3, 1, 2))
        pt_logits = pt_out.logits.cpu().numpy()
        pt_boxes = pt_out.pred_boxes.cpu().numpy()
        pt_masks = pt_out.pred_masks.cpu().numpy()

        keras_out = keras_model(norm, training=False)
        keras_logits = keras.ops.convert_to_numpy(keras_out["logits"])
        keras_boxes = keras.ops.convert_to_numpy(keras_out["pred_boxes"])
        keras_masks = keras.ops.convert_to_numpy(keras_out["pred_masks"])

        logits_diff = float(np.max(np.abs(pt_logits - keras_logits)))
        boxes_diff = float(np.max(np.abs(pt_boxes - keras_boxes)))
        masks_diff = float(np.max(np.abs(pt_masks - keras_masks)))
        logits_cos = cosine(pt_logits, keras_logits)
        boxes_cos = cosine(pt_boxes, keras_boxes)
        masks_cos = cosine(pt_masks, keras_masks)
        class_agree = float(
            (pt_logits[0].argmax(-1) == keras_logits[0].argmax(-1)).mean()
        )

        print(f"  Max logits diff: {logits_diff:.6f}   cosine: {logits_cos:.6f}")
        print(f"  Max boxes diff:  {boxes_diff:.6f}   cosine: {boxes_cos:.6f}")
        print(f"  Max masks diff:  {masks_diff:.6f}   cosine: {masks_cos:.6f}")
        print(f"  Class agreement: {class_agree * 100:.1f}%")

        if (
            logits_cos < 0.99
            or boxes_cos < 0.99
            or masks_cos < 0.99
            or class_agree < 0.95
        ):
            raise ValueError(
                f"{variant}: HF parity failed (logits cos {logits_cos:.4f}, "
                f"boxes cos {boxes_cos:.4f}, masks cos {masks_cos:.4f}, "
                f"class agreement {class_agree:.3f})"
            )

        print("Model equivalence test passed!")

        out_name = f"rf_detr_seg_{variant.split('-')[-1]}.weights.h5"
        keras_model.save_weights(out_name)
        print(f"  Saved -> {out_name}")

        del keras_model, hf_model, state_dict
        keras.backend.clear_session()
