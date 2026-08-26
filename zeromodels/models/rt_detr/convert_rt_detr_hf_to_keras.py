from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.weight_transfer_util import (
    compare_keras_torch_names,
    transfer_nested_layer_weights,
    transfer_weights,
)
from zeromodels.models.rt_detr import RTDETRDetect

backbone_name_mapping: Dict[str, str] = {
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving_mean": "running_mean",
    "moving_variance": "running_var",
}


def transfer_rt_detr_weights(keras_model, state_dict):
    sd = state_dict
    depths = keras_model._backbone_block_repeats
    layer_type = keras_model._backbone_layer_type
    num_convs = 2 if layer_type == "basic" else 3
    num_dec = keras_model._decoder_layers

    for i in tqdm(range(3), desc="Transferring backbone stem"):
        hf_pre = f"model.backbone.model.embedder.embedder.{i}"
        conv = keras_model.get_layer(f"backbone_embedder_{i}_conv")
        transfer_weights("conv_kernel", conv.kernel, sd[f"{hf_pre}.convolution.weight"])
        bn = keras_model.get_layer(f"backbone_embedder_{i}_bn")
        transfer_nested_layer_weights(
            bn,
            sd,
            f"{hf_pre}.normalization",
            name_mapping=backbone_name_mapping,
        )

    stage_pairs: List[Tuple[str, str]] = []
    for si, nb in enumerate(depths):
        for bi in range(nb):
            hf_pre = f"model.backbone.model.encoder.stages.{si}.layers.{bi}"
            k_pre = f"backbone_stage{si}_{bi}"
            for ci in range(num_convs):
                stage_pairs.append(
                    (f"{k_pre}_conv{ci + 1}", f"{hf_pre}.layer.{ci}.convolution")
                )
                stage_pairs.append(
                    (f"{k_pre}_bn{ci + 1}", f"{hf_pre}.layer.{ci}.normalization")
                )
            suf = "shortcut" if si == 0 and bi == 0 else "shortcut.1"
            stage_pairs.append(
                (f"{k_pre}_shortcut_conv", f"{hf_pre}.{suf}.convolution")
            )
            stage_pairs.append(
                (f"{k_pre}_shortcut_bn", f"{hf_pre}.{suf}.normalization")
            )

    for keras_name, hf_prefix in tqdm(
        stage_pairs, desc=f"Transferring backbone stages ({layer_type})"
    ):
        try:
            layer = keras_model.get_layer(keras_name)
        except ValueError:
            continue
        if keras_name.endswith("_conv"):
            hf_key = f"{hf_prefix}.weight"
            if not compare_keras_torch_names(
                keras_name, layer.kernel, hf_key, sd[hf_key]
            ):
                continue
            transfer_weights("conv_kernel", layer.kernel, sd[hf_key])
        else:
            transfer_nested_layer_weights(
                layer,
                sd,
                hf_prefix,
                name_mapping=backbone_name_mapping,
            )

    for i in tqdm(range(3), desc="Transferring encoder input projections"):
        conv = keras_model.get_layer(f"encoder_input_proj_{i}_conv")
        transfer_weights(
            "conv_kernel", conv.kernel, sd[f"model.encoder_input_proj.{i}.0.weight"]
        )
        bn = keras_model.get_layer(f"encoder_input_proj_{i}_bn")
        transfer_nested_layer_weights(
            bn,
            sd,
            f"model.encoder_input_proj.{i}.1",
            name_mapping=backbone_name_mapping,
        )

    print("Transferring AIFI encoder...")
    hf_aifi = "model.encoder.aifi.0.layers.0"
    sa = keras_model.get_layer("aifi_0_layers_0_self_attn")

    aifi_sa_mapping: Dict[str, str] = {
        "aifi_0_layers_0_self_attn_": "",
        "out_proj": "o_proj",
        "kernel": "weight",
        "gamma": "weight",
        "beta": "bias",
    }
    transfer_nested_layer_weights(
        sa,
        sd,
        f"{hf_aifi}.self_attn",
        name_mapping=aifi_sa_mapping,
    )

    for layer_name, hf_suffix in [
        ("aifi_0_layers_0_self_attn_layer_norm", "self_attn_layer_norm"),
        ("aifi_0_layers_0_final_layer_norm", "final_layer_norm"),
    ]:
        ln = keras_model.get_layer(layer_name)
        transfer_nested_layer_weights(
            ln,
            sd,
            f"{hf_aifi}.{hf_suffix}",
            name_mapping=backbone_name_mapping,
        )

    aifi_fc_mapping: Dict[str, str] = {
        "aifi_0_layers_0_": "",
        "kernel": "weight",
        "beta": "bias",
    }
    for layer_name, hf_suffix in [
        ("aifi_0_layers_0_fc1", "mlp.fc1"),
        ("aifi_0_layers_0_fc2", "mlp.fc2"),
    ]:
        fc = keras_model.get_layer(layer_name)
        transfer_nested_layer_weights(
            fc,
            sd,
            f"{hf_aifi}.{hf_suffix}",
            name_mapping=aifi_fc_mapping,
        )

    conv_norm_pairs: List[Tuple[str, str]] = []
    for i in range(2):
        conv_norm_pairs.append(
            (f"lateral_convs_{i}", f"model.encoder.lateral_convs.{i}")
        )
        conv_norm_pairs.append(
            (f"downsample_convs_{i}", f"model.encoder.downsample_convs.{i}")
        )
    for block_type in ["fpn_blocks", "pan_blocks"]:
        for i in range(2):
            hf_blk = f"model.encoder.{block_type}.{i}"
            k_blk = f"{block_type}_{i}"
            conv_norm_pairs.append((f"{k_blk}_conv1", f"{hf_blk}.conv1"))
            conv_norm_pairs.append((f"{k_blk}_conv2", f"{hf_blk}.conv2"))
            if f"{hf_blk}.conv3.conv.weight" in sd:
                conv_norm_pairs.append((f"{k_blk}_conv3", f"{hf_blk}.conv3"))
            for bi in range(3):
                conv_norm_pairs.append(
                    (
                        f"{k_blk}_bottlenecks_{bi}_conv1",
                        f"{hf_blk}.bottlenecks.{bi}.conv1",
                    )
                )
                conv_norm_pairs.append(
                    (
                        f"{k_blk}_bottlenecks_{bi}_conv2",
                        f"{hf_blk}.bottlenecks.{bi}.conv2",
                    )
                )

    for keras_name, hf_prefix in tqdm(
        conv_norm_pairs, desc="Transferring CCFM (FPN + PAN)"
    ):
        conv = keras_model.get_layer(f"{keras_name}_conv")
        transfer_weights("conv_kernel", conv.kernel, sd[f"{hf_prefix}.conv.weight"])
        bn = keras_model.get_layer(f"{keras_name}_norm")
        transfer_nested_layer_weights(
            bn,
            sd,
            f"{hf_prefix}.norm",
            name_mapping=backbone_name_mapping,
        )

    for i in tqdm(range(3), desc="Transferring decoder input projections"):
        conv = keras_model.get_layer(f"decoder_input_proj_{i}_conv")
        transfer_weights(
            "conv_kernel", conv.kernel, sd[f"model.decoder_input_proj.{i}.0.weight"]
        )
        bn = keras_model.get_layer(f"decoder_input_proj_{i}_bn")
        transfer_nested_layer_weights(
            bn,
            sd,
            f"model.decoder_input_proj.{i}.1",
            name_mapping=backbone_name_mapping,
        )

    print("Transferring encoder output heads...")
    for keras_name, hf_key in [
        ("enc_output_linear", "model.enc_output.0"),
        ("enc_score_head", "model.enc_score_head"),
    ]:
        layer = keras_model.get_layer(keras_name)
        transfer_weights("kernel", layer.weights[0], sd[f"{hf_key}.weight"])
        layer.weights[1].assign(sd[f"{hf_key}.bias"])

    enc_ln = keras_model.get_layer("enc_output_layernorm")
    transfer_nested_layer_weights(
        enc_ln,
        sd,
        "model.enc_output.1",
        name_mapping=backbone_name_mapping,
    )

    for j in range(3):
        layer = keras_model.get_layer(f"enc_bbox_head_{j}")
        transfer_weights(
            "kernel", layer.weights[0], sd[f"model.enc_bbox_head.layers.{j}.weight"]
        )
        layer.weights[1].assign(sd[f"model.enc_bbox_head.layers.{j}.bias"])

    for j in range(2):
        layer = keras_model.get_layer(f"query_pos_head_{j}")
        transfer_weights(
            "kernel",
            layer.weights[0],
            sd[f"model.decoder.query_pos_head.layers.{j}.weight"],
        )
        layer.weights[1].assign(sd[f"model.decoder.query_pos_head.layers.{j}.bias"])

    for i in tqdm(range(num_dec), desc="Transferring decoder layers"):
        hf_dl = f"model.decoder.layers.{i}"
        k_dl = f"decoder_layers_{i}"
        dec_layer = keras_model.get_layer(k_dl)

        decoder_name_mapping: Dict[str, str] = {
            f"{k_dl}_self_attn.{k_dl}_self_attn_": "self_attn.",
            f"{k_dl}_encoder_attn.": "encoder_attn.",
            f"{k_dl}_self_attn_layer_norm.": "self_attn_layer_norm.",
            f"{k_dl}_encoder_attn_layer_norm.": "encoder_attn_layer_norm.",
            f"{k_dl}_final_layer_norm.": "final_layer_norm.",
            f"{k_dl}_fc1.": "mlp.fc1.",
            f"{k_dl}_fc2.": "mlp.fc2.",
            "out_proj": "o_proj",
            "kernel": "weight",
            "gamma": "weight",
            "beta": "bias",
        }

        transfer_nested_layer_weights(
            dec_layer,
            sd,
            hf_dl,
            name_mapping=decoder_name_mapping,
        )

    for i in tqdm(range(num_dec), desc="Transferring detection heads"):
        try:
            cls_layer = keras_model.get_layer(f"class_embed_{i}")
            transfer_weights(
                "kernel",
                cls_layer.weights[0],
                sd[f"model.decoder.class_embed.{i}.weight"],
            )
            cls_layer.weights[1].assign(sd[f"model.decoder.class_embed.{i}.bias"])
        except ValueError:
            pass

        for j in range(3):
            bbox = keras_model.get_layer(f"bbox_embed_{i}_{j}")
            transfer_weights(
                "kernel",
                bbox.weights[0],
                sd[f"model.decoder.bbox_embed.{i}.layers.{j}.weight"],
            )
            bbox.weights[1].assign(sd[f"model.decoder.bbox_embed.{i}.layers.{j}.bias"])


# Per-variant recipes (relocated from rt_detr_config.py). Models load from the
# Hub by repo id; these build the arch for conversion + drive the backfill.
RT_DETR_VARIANTS = {
    "rtdetr-r18vd": {
        "backbone_hidden_sizes": (64, 128, 256, 512),
        "backbone_block_repeats": (2, 2, 2, 2),
        "backbone_layer_type": "basic",
        "encoder_in_channels": (128, 256, 512),
        "hidden_expansion": 0.5,
        "decoder_num_layers": 3,
    },
    "rtdetr-r18vd-coco-o365": {
        "backbone_hidden_sizes": (64, 128, 256, 512),
        "backbone_block_repeats": (2, 2, 2, 2),
        "backbone_layer_type": "basic",
        "encoder_in_channels": (128, 256, 512),
        "hidden_expansion": 0.5,
        "decoder_num_layers": 3,
    },
    "rtdetr-r34vd": {
        "backbone_hidden_sizes": (64, 128, 256, 512),
        "backbone_block_repeats": (3, 4, 6, 3),
        "backbone_layer_type": "basic",
        "encoder_in_channels": (128, 256, 512),
        "hidden_expansion": 0.5,
        "decoder_num_layers": 4,
    },
    "rtdetr-r50vd": {
        "backbone_hidden_sizes": (256, 512, 1024, 2048),
        "backbone_block_repeats": (3, 4, 6, 3),
        "backbone_layer_type": "bottleneck",
        "encoder_in_channels": (512, 1024, 2048),
        "decoder_num_layers": 6,
    },
    "rtdetr-r50vd-coco-o365": {
        "backbone_hidden_sizes": (256, 512, 1024, 2048),
        "backbone_block_repeats": (3, 4, 6, 3),
        "backbone_layer_type": "bottleneck",
        "encoder_in_channels": (512, 1024, 2048),
        "decoder_num_layers": 6,
    },
    "rtdetr-r101vd": {
        "backbone_hidden_sizes": (256, 512, 1024, 2048),
        "backbone_block_repeats": (3, 4, 23, 3),
        "backbone_layer_type": "bottleneck",
        "encoder_in_channels": (512, 1024, 2048),
        "encoder_hidden_dim": 384,
        "encoder_ffn_dim": 2048,
        "decoder_num_layers": 6,
    },
    "rtdetr-r101vd-coco-o365": {
        "backbone_hidden_sizes": (256, 512, 1024, 2048),
        "backbone_block_repeats": (3, 4, 23, 3),
        "backbone_layer_type": "bottleneck",
        "encoder_in_channels": (512, 1024, 2048),
        "encoder_hidden_dim": 384,
        "encoder_ffn_dim": 2048,
        "decoder_num_layers": 6,
    },
}


if __name__ == "__main__":
    import torch
    from transformers import RTDetrForObjectDetection

    model_configs: List[Dict[str, Any]] = [
        {
            "variant": "rtdetr-r50vd",
            "hf_name": "PekingU/rtdetr_r50vd",
            "output": "rt_detr_r50vd.weights.h5",
        },
        {
            "variant": "rtdetr-r18vd",
            "hf_name": "PekingU/rtdetr_r18vd",
            "output": "rt_detr_r18vd.weights.h5",
        },
        {
            "variant": "rtdetr-r34vd",
            "hf_name": "PekingU/rtdetr_r34vd",
            "output": "rt_detr_r34vd.weights.h5",
        },
        {
            "variant": "rtdetr-r101vd",
            "hf_name": "PekingU/rtdetr_r101vd",
            "output": "rt_detr_r101vd.weights.h5",
        },
        {
            "variant": "rtdetr-r18vd-coco-o365",
            "hf_name": "PekingU/rtdetr_r18vd_coco_o365",
            "output": "rt_detr_r18vd_coco_o365.weights.h5",
        },
        {
            "variant": "rtdetr-r50vd-coco-o365",
            "hf_name": "PekingU/rtdetr_r50vd_coco_o365",
            "output": "rt_detr_r50vd_coco_o365.weights.h5",
        },
        {
            "variant": "rtdetr-r101vd-coco-o365",
            "hf_name": "PekingU/rtdetr_r101vd_coco_o365",
            "output": "rt_detr_r101vd_coco_o365.weights.h5",
        },
    ]

    for cfg in model_configs:
        hf_name = cfg["hf_name"]
        output = cfg["output"]

        print(f"\n{'=' * 60}")
        print(f"Converting {hf_name}...")
        print(f"{'=' * 60}")

        torch_model = RTDetrForObjectDetection.from_pretrained(
            hf_name,
            attn_implementation="eager",
        ).eval()
        sd: Dict[str, np.ndarray] = {
            k: v.cpu().numpy() for k, v in torch_model.state_dict().items()
        }

        keras_model = RTDETRDetect(
            **RT_DETR_VARIANTS[cfg["variant"]],
            image_size=640,
            num_queries=300,
            num_classes=80,
        )
        print(f"  Parameters: {keras_model.count_params():,}")

        transfer_rt_detr_weights(keras_model, sd)

        print("\nVerifying model equivalence...")

        np.random.seed(42)
        test_input = np.random.rand(1, 640, 640, 3).astype(np.float32)

        hf_input = torch.tensor(test_input).permute(0, 3, 1, 2)
        with torch.no_grad():
            hf_output = torch_model(hf_input)
            hf_logits = hf_output.logits.numpy()
            hf_boxes = hf_output.pred_boxes.numpy()

        keras_output = keras_model.predict(test_input, verbose=0)
        keras_logits = np.asarray(keras_output["logits"])
        keras_boxes = np.asarray(keras_output["pred_boxes"])

        raw_logits_diff = float(np.max(np.abs(hf_logits - keras_logits)))
        raw_boxes_diff = float(np.max(np.abs(hf_boxes - keras_boxes)))

        # RT-DETR's two-stage decoder selects queries via top-k on encoder
        # scores. fp32-level encoder differences reorder near-tied scores,
        # so the *same* set of queries can land in different positions: a
        # benign permutation (detection is set prediction, order-invariant).
        # Match each reference query to its nearest keras query by logits before
        # diffing so the metric reflects the true error.
        matched_logits_diff = 0.0
        matched_boxes_diff = 0.0
        for b in range(hf_logits.shape[0]):
            cost = np.linalg.norm(
                hf_logits[b][:, None, :] - keras_logits[b][None, :, :], axis=-1
            )
            match = cost.argmin(axis=1)
            matched_logits_diff = max(
                matched_logits_diff,
                float(np.abs(hf_logits[b] - keras_logits[b][match]).max()),
            )
            matched_boxes_diff = max(
                matched_boxes_diff,
                float(np.abs(hf_boxes[b] - keras_boxes[b][match]).max()),
            )

        print(f"Max logits diff (raw):       {raw_logits_diff:.6f}")
        print(f"Max boxes diff  (raw):       {raw_boxes_diff:.6f}")
        print(f"Max logits diff (matched):   {matched_logits_diff:.6f}")
        print(f"Max boxes diff  (matched):   {matched_boxes_diff:.6f}")

        if matched_logits_diff > 1e-2 or matched_boxes_diff > 1e-2:
            raise ValueError(
                "Equivalence test failed (query-matched): "
                f"logits {matched_logits_diff:.4f}, boxes {matched_boxes_diff:.4f}"
            )
        print("Equivalence test passed!")

        keras_model.save_weights(output)
        print(f"Model saved as {output}")

        del keras_model, torch_model, sd
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
