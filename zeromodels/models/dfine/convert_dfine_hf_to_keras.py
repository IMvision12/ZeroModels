import os
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightShapeMismatchError
from zeromodels.conversion.weight_transfer_util import (
    compare_keras_torch_names,
    transfer_nested_layer_weights,
    transfer_weights,
)

backbone_name_mapping: Dict[str, str] = {
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving_mean": "running_mean",
    "moving_variance": "running_var",
}


def transfer_dfine_weights(keras_model, state_dict):
    sd = state_dict
    stage_num_blocks = keras_model._stage_num_blocks
    stage_numb_of_layers = keras_model._stage_numb_of_layers
    stage_light_block = (False, False, True, True)
    stage_downsample = (False, True, True, True)
    use_lab = keras_model._use_lab
    num_feature_levels = keras_model._num_feature_levels
    num_dec = keras_model._decoder_layers
    n_enc_proj = len(keras_model._encoder_in_channels)

    stem_conv_bn_pairs: List[Tuple[str, str]] = []
    for stem_name in ["stem1", "stem2a", "stem2b", "stem3", "stem4"]:
        stem_conv_bn_pairs.append(
            (f"backbone_{stem_name}", f"model.backbone.model.embedder.{stem_name}")
        )

    for k_prefix, hf_prefix in tqdm(
        stem_conv_bn_pairs, desc="Transferring backbone stem"
    ):
        conv = keras_model.get_layer(f"{k_prefix}_conv")
        transfer_weights(
            "conv_kernel", conv.kernel, sd[f"{hf_prefix}.convolution.weight"]
        )
        bn = keras_model.get_layer(f"{k_prefix}_bn")
        transfer_nested_layer_weights(
            bn, sd, f"{hf_prefix}.normalization", name_mapping=backbone_name_mapping
        )
        lab_key = f"{hf_prefix}.lab.scale"
        if lab_key in sd and use_lab:
            try:
                lab = keras_model.get_layer(f"{k_prefix}_lab")
                lab.scale.assign(sd[lab_key])
                lab.bias.assign(sd[f"{hf_prefix}.lab.bias"])
            except ValueError:
                pass

    stage_conv_bn_pairs: List[Tuple[str, str]] = []
    for si in range(4):
        if stage_downsample[si]:
            hf_ds = f"model.backbone.model.encoder.stages.{si}.downsample"
            ds_key = f"{hf_ds}.convolution.weight"
            if ds_key in sd:
                stage_conv_bn_pairs.append((f"backbone_stage{si}_downsample", hf_ds))

        for bi in range(stage_num_blocks[si]):
            hf_blk = f"model.backbone.model.encoder.stages.{si}.blocks.{bi}"
            k_blk = f"backbone_stage{si}_block{bi}"

            for li in range(stage_numb_of_layers[si]):
                if stage_light_block[si]:
                    for sub in ["conv1", "conv2"]:
                        stage_conv_bn_pairs.append(
                            (
                                f"{k_blk}_layers_{li}_{sub}",
                                f"{hf_blk}.layers.{li}.{sub}",
                            )
                        )
                else:
                    stage_conv_bn_pairs.append(
                        (f"{k_blk}_layers_{li}", f"{hf_blk}.layers.{li}")
                    )

            for ai in range(2):
                stage_conv_bn_pairs.append(
                    (f"{k_blk}_agg_{ai}", f"{hf_blk}.aggregation.{ai}")
                )

    for k_prefix, hf_prefix in tqdm(
        stage_conv_bn_pairs, desc="Transferring backbone stages"
    ):
        try:
            conv = keras_model.get_layer(f"{k_prefix}_conv")
        except ValueError:
            continue
        hf_conv_key = f"{hf_prefix}.convolution.weight"
        if hf_conv_key not in sd:
            continue
        if not compare_keras_torch_names(
            k_prefix, conv.kernel, hf_conv_key, sd[hf_conv_key]
        ):
            raise WeightShapeMismatchError(
                k_prefix, conv.kernel.shape, hf_conv_key, sd[hf_conv_key].shape
            )
        transfer_weights("conv_kernel", conv.kernel, sd[hf_conv_key])
        bn = keras_model.get_layer(f"{k_prefix}_bn")
        transfer_nested_layer_weights(
            bn, sd, f"{hf_prefix}.normalization", name_mapping=backbone_name_mapping
        )
        lab_key = f"{hf_prefix}.lab.scale"
        if lab_key in sd and use_lab:
            try:
                lab = keras_model.get_layer(f"{k_prefix}_lab")
                lab.scale.assign(sd[lab_key])
                lab.bias.assign(sd[f"{hf_prefix}.lab.bias"])
            except ValueError:
                pass

    for i in tqdm(range(n_enc_proj), desc="Transferring encoder input projections"):
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
    if f"{hf_aifi}.self_attn.q_proj.weight" not in sd:
        hf_aifi = "model.encoder.encoder.0.layers.0"

    sa = keras_model.get_layer("aifi_0_layers_0_self_attn")
    hf_out_proj = "o_proj" if f"{hf_aifi}.self_attn.o_proj.weight" in sd else "out_proj"
    for proj_name, hf_proj in [
        ("q_proj", "q_proj"),
        ("k_proj", "k_proj"),
        ("v_proj", "v_proj"),
        ("out_proj", hf_out_proj),
    ]:
        dense = getattr(sa, proj_name)
        transfer_weights(
            "kernel", dense.kernel, sd[f"{hf_aifi}.self_attn.{hf_proj}.weight"]
        )
        dense.bias.assign(sd[f"{hf_aifi}.self_attn.{hf_proj}.bias"])
    transfer_nested_layer_weights(
        keras_model.get_layer("aifi_0_layers_0_self_attn_layer_norm"),
        sd,
        f"{hf_aifi}.self_attn_layer_norm",
        name_mapping=backbone_name_mapping,
    )
    transfer_nested_layer_weights(
        keras_model.get_layer("aifi_0_layers_0_final_layer_norm"),
        sd,
        f"{hf_aifi}.final_layer_norm",
        name_mapping=backbone_name_mapping,
    )
    if f"{hf_aifi}.mlp.layers.0.weight" in sd:
        aifi_ffn_map = [
            ("aifi_0_layers_0_fc1", "mlp.layers.0"),
            ("aifi_0_layers_0_fc2", "mlp.layers.1"),
        ]
    else:
        aifi_ffn_map = [
            ("aifi_0_layers_0_fc1", "fc1"),
            ("aifi_0_layers_0_fc2", "fc2"),
        ]
    for layer_name, hf_suffix in aifi_ffn_map:
        fc = keras_model.get_layer(layer_name)
        transfer_weights("kernel", fc.kernel, sd[f"{hf_aifi}.{hf_suffix}.weight"])
        fc.bias.assign(sd[f"{hf_aifi}.{hf_suffix}.bias"])

    print("Transferring CCFM...")
    num_fpn = num_feature_levels - 1

    ccfm_conv_norm_pairs: List[Tuple[str, str]] = []
    for i in range(num_fpn):
        ccfm_conv_norm_pairs.append(
            (f"lateral_convs_{i}", f"model.encoder.lateral_convs.{i}")
        )
    for i in range(num_fpn):
        for sub in ["conv1", "conv2"]:
            ccfm_conv_norm_pairs.append(
                (
                    f"downsample_convs_{i}_{sub}",
                    f"model.encoder.downsample_convs.{i}.{sub}",
                )
            )

    for keras_name, hf_prefix in tqdm(
        ccfm_conv_norm_pairs, desc="Transferring CCFM lateral/downsample"
    ):
        conv = keras_model.get_layer(f"{keras_name}_conv")
        transfer_weights("conv_kernel", conv.kernel, sd[f"{hf_prefix}.conv.weight"])
        bn = keras_model.get_layer(f"{keras_name}_norm")
        transfer_nested_layer_weights(
            bn, sd, f"{hf_prefix}.norm", name_mapping=backbone_name_mapping
        )

    for block_type in ["fpn_blocks", "pan_blocks"]:
        for i in range(num_fpn):
            hf_blk = f"model.encoder.{block_type}.{i}"
            k_blk = f"{block_type}_{i}"
            rep_pairs: List[Tuple[str, str]] = []
            for conv_name in ["conv1", "conv2", "conv3", "conv4"]:
                if f"{hf_blk}.{conv_name}.conv.weight" in sd:
                    rep_pairs.append((f"{k_blk}_{conv_name}", f"{hf_blk}.{conv_name}"))
            for csp in ["csp_rep1", "csp_rep2"]:
                for sub in ["conv1", "conv2", "conv3"]:
                    if f"{hf_blk}.{csp}.{sub}.conv.weight" in sd:
                        rep_pairs.append(
                            (f"{k_blk}_{csp}_{sub}", f"{hf_blk}.{csp}.{sub}")
                        )
                for bi in range(10):
                    for sub in ["conv1", "conv2"]:
                        key = f"{hf_blk}.{csp}.bottlenecks.{bi}.{sub}.conv.weight"
                        if key in sd:
                            rep_pairs.append(
                                (
                                    f"{k_blk}_{csp}_bottlenecks_{bi}_{sub}",
                                    f"{hf_blk}.{csp}.bottlenecks.{bi}.{sub}",
                                )
                            )

            for keras_name, hf_prefix in rep_pairs:
                try:
                    conv = keras_model.get_layer(f"{keras_name}_conv")
                    transfer_weights(
                        "conv_kernel", conv.kernel, sd[f"{hf_prefix}.conv.weight"]
                    )
                    bn = keras_model.get_layer(f"{keras_name}_norm")
                    transfer_nested_layer_weights(
                        bn, sd, f"{hf_prefix}.norm", name_mapping=backbone_name_mapping
                    )
                except (ValueError, KeyError):
                    pass

    for i in tqdm(
        range(num_feature_levels), desc="Transferring decoder input projections"
    ):
        try:
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
        except (ValueError, KeyError):
            pass

    print("Transferring encoder output heads...")
    for layer_name, hf_key in [
        ("enc_output_linear", "model.enc_output.0"),
        ("enc_score_head", "model.enc_score_head"),
    ]:
        layer = keras_model.get_layer(layer_name)
        transfer_weights("kernel", layer.weights[0], sd[f"{hf_key}.weight"])
        layer.weights[1].assign(sd[f"{hf_key}.bias"])

    transfer_nested_layer_weights(
        keras_model.get_layer("enc_output_layernorm"),
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

    for j in range(3):
        layer = keras_model.get_layer(f"pre_bbox_head_{j}")
        transfer_weights(
            "kernel",
            layer.weights[0],
            sd[f"model.decoder.pre_bbox_head.layers.{j}.weight"],
        )
        layer.weights[1].assign(sd[f"model.decoder.pre_bbox_head.layers.{j}.bias"])

    dp = keras_model.get_layer("decoder_params")
    dp.up.assign(sd["model.decoder.up"])
    dp.reg_scale.assign(sd["model.decoder.reg_scale"])

    for i in tqdm(range(num_dec), desc="Transferring decoder layers"):
        hf_dl = f"model.decoder.layers.{i}"
        dec_layer = keras_model.get_layer(f"decoder_layers_{i}")

        sa = dec_layer.self_attn
        dec_out_proj = (
            "o_proj" if f"{hf_dl}.self_attn.o_proj.weight" in sd else "out_proj"
        )
        for proj_name, hf_proj in [
            ("q_proj", "q_proj"),
            ("k_proj", "k_proj"),
            ("v_proj", "v_proj"),
            ("out_proj", dec_out_proj),
        ]:
            dense = getattr(sa, proj_name)
            transfer_weights(
                "kernel", dense.kernel, sd[f"{hf_dl}.self_attn.{hf_proj}.weight"]
            )
            dense.bias.assign(sd[f"{hf_dl}.self_attn.{hf_proj}.bias"])

        dec_layer.self_attn_layer_norm.gamma.assign(
            sd[f"{hf_dl}.self_attn_layer_norm.weight"]
        )
        dec_layer.self_attn_layer_norm.beta.assign(
            sd[f"{hf_dl}.self_attn_layer_norm.bias"]
        )

        ea = dec_layer.encoder_attn
        transfer_weights(
            "kernel",
            ea.sampling_offsets.kernel,
            sd[f"{hf_dl}.encoder_attn.sampling_offsets.weight"],
        )
        ea.sampling_offsets.bias.assign(
            sd[f"{hf_dl}.encoder_attn.sampling_offsets.bias"]
        )
        transfer_weights(
            "kernel",
            ea.attention_weights_proj.kernel,
            sd[f"{hf_dl}.encoder_attn.attention_weights.weight"],
        )
        ea.attention_weights_proj.bias.assign(
            sd[f"{hf_dl}.encoder_attn.attention_weights.bias"]
        )
        ea.num_points_scale.assign(sd[f"{hf_dl}.encoder_attn.num_points_scale"])

        transfer_weights(
            "kernel", dec_layer.gateway_gate.kernel, sd[f"{hf_dl}.gateway.gate.weight"]
        )
        dec_layer.gateway_gate.bias.assign(sd[f"{hf_dl}.gateway.gate.bias"])
        dec_layer.gateway_norm.gamma.assign(sd[f"{hf_dl}.gateway.norm.weight"])
        dec_layer.gateway_norm.beta.assign(sd[f"{hf_dl}.gateway.norm.bias"])

        if f"{hf_dl}.mlp.layers.0.weight" in sd:
            dec_fc1_key = f"{hf_dl}.mlp.layers.0"
            dec_fc2_key = f"{hf_dl}.mlp.layers.1"
        else:
            dec_fc1_key = f"{hf_dl}.fc1"
            dec_fc2_key = f"{hf_dl}.fc2"
        transfer_weights("kernel", dec_layer.fc1.kernel, sd[f"{dec_fc1_key}.weight"])
        dec_layer.fc1.bias.assign(sd[f"{dec_fc1_key}.bias"])
        transfer_weights("kernel", dec_layer.fc2.kernel, sd[f"{dec_fc2_key}.weight"])
        dec_layer.fc2.bias.assign(sd[f"{dec_fc2_key}.bias"])

        dec_layer.final_layer_norm.gamma.assign(sd[f"{hf_dl}.final_layer_norm.weight"])
        dec_layer.final_layer_norm.beta.assign(sd[f"{hf_dl}.final_layer_norm.bias"])

    for i in tqdm(range(num_dec), desc="Transferring detection heads"):
        for j in range(3):
            bb_layer = keras_model.get_layer(f"bbox_embed_{i}_{j}")
            transfer_weights(
                "kernel",
                bb_layer.weights[0],
                sd[f"model.decoder.bbox_embed.{i}.layers.{j}.weight"],
            )
            bb_layer.weights[1].assign(
                sd[f"model.decoder.bbox_embed.{i}.layers.{j}.bias"]
            )

    last = num_dec - 1
    cls_layer = keras_model.get_layer(f"class_embed_{last}")
    transfer_weights(
        "kernel", cls_layer.weights[0], sd[f"model.decoder.class_embed.{last}.weight"]
    )
    cls_layer.weights[1].assign(sd[f"model.decoder.class_embed.{last}.bias"])

    for j, suffix in enumerate(["layers.0", "layers.1"]):
        lqe = keras_model.get_layer(f"lqe_{last}_{j}")
        transfer_weights(
            "kernel",
            lqe.weights[0],
            sd[f"model.decoder.lqe_layers.{last}.reg_conf.{suffix}.weight"],
        )
        lqe.weights[1].assign(
            sd[f"model.decoder.lqe_layers.{last}.reg_conf.{suffix}.bias"]
        )


# Per-variant recipes (relocated from dfine_config.py). Models load from the
# Hub by repo id; these build the arch for conversion + drive the backfill.
DFINE_VARIANTS = {
    "dfine-nano": {
        "stem_channels": (3, 16, 16),
        "stage_in_channels": (16, 64, 256, 512),
        "stage_mid_channels": (16, 32, 64, 128),
        "stage_out_channels": (64, 256, 512, 1024),
        "stage_num_blocks": (1, 1, 2, 1),
        "stage_numb_of_layers": (3, 3, 3, 3),
        "use_lab": True,
        "encoder_in_channels": (512, 1024),
        "encoder_hidden_dim": 128,
        "hidden_dim": 128,
        "decoder_num_layers": 3,
        "decoder_n_points": [6, 6],
        "hidden_expansion": 0.34,
        "ccfm_num_blocks": 2,
        "num_feature_levels": 2,
        "feat_strides": (16, 32),
        "encode_proj_layers": (1,),
        "encoder_ffn_dim": 512,
        "decoder_ffn_dim": 512,
    },
    "dfine-small": {
        "stem_channels": (3, 16, 16),
        "stage_in_channels": (16, 64, 256, 512),
        "stage_mid_channels": (16, 32, 64, 128),
        "stage_out_channels": (64, 256, 512, 1024),
        "stage_num_blocks": (1, 1, 2, 1),
        "stage_numb_of_layers": (3, 3, 3, 3),
        "use_lab": True,
        "encoder_in_channels": (256, 512, 1024),
        "decoder_num_layers": 3,
        "decoder_n_points": [3, 6, 3],
        "hidden_expansion": 0.5,
    },
    "dfine-medium": {
        "stem_channels": (3, 24, 32),
        "stage_in_channels": (32, 96, 384, 768),
        "stage_mid_channels": (32, 64, 128, 256),
        "stage_out_channels": (96, 384, 768, 1536),
        "stage_num_blocks": (1, 1, 3, 1),
        "stage_numb_of_layers": (4, 4, 4, 4),
        "use_lab": True,
        "encoder_in_channels": (384, 768, 1536),
        "ccfm_num_blocks": 2,
        "decoder_num_layers": 4,
        "decoder_n_points": [3, 6, 3],
    },
    "dfine-large": {
        "stem_channels": (3, 32, 48),
        "stage_in_channels": (48, 128, 512, 1024),
        "stage_mid_channels": (48, 96, 192, 384),
        "stage_out_channels": (128, 512, 1024, 2048),
        "stage_num_blocks": (1, 1, 3, 1),
        "stage_numb_of_layers": (6, 6, 6, 6),
        "use_lab": False,
        "encoder_in_channels": (512, 1024, 2048),
        "ccfm_num_blocks": 3,
        "decoder_num_layers": 6,
        "decoder_n_points": [3, 6, 3],
    },
    "dfine-xlarge": {
        "stem_channels": (3, 32, 64),
        "stage_in_channels": (64, 128, 512, 1024),
        "stage_mid_channels": (64, 128, 256, 512),
        "stage_out_channels": (128, 512, 1024, 2048),
        "stage_num_blocks": (1, 2, 5, 2),
        "stage_numb_of_layers": (6, 6, 6, 6),
        "use_lab": False,
        "encoder_in_channels": (512, 1024, 2048),
        "encoder_hidden_dim": 384,
        "ccfm_num_blocks": 3,
        "decoder_num_layers": 6,
        "decoder_n_points": [3, 6, 3],
        "encoder_ffn_dim": 2048,
    },
}


if __name__ == "__main__":
    os.environ.setdefault("KERAS_BACKEND", "torch")

    import torch
    from transformers import AutoModelForObjectDetection

    from zeromodels.models.dfine import DFineDetect

    model_configs: List[Dict[str, Any]] = [
        {
            "variant": "dfine-nano",
            "hf_name": "ustc-community/dfine-nano-coco",
            "output": "dfine_nano.weights.h5",
        },
        {
            "variant": "dfine-small",
            "hf_name": "ustc-community/dfine-small-coco",
            "output": "dfine_small.weights.h5",
        },
        {
            "variant": "dfine-medium",
            "hf_name": "ustc-community/dfine-medium-coco",
            "output": "dfine_medium.weights.h5",
        },
        {
            "variant": "dfine-large",
            "hf_name": "ustc-community/dfine-large-coco",
            "output": "dfine_large.weights.h5",
        },
        {
            "variant": "dfine-xlarge",
            "hf_name": "ustc-community/dfine-xlarge-coco",
            "output": "dfine_xlarge.weights.h5",
        },
    ]

    for cfg in model_configs:
        hf_name = cfg["hf_name"]
        output = cfg["output"]

        print(f"\n{'=' * 60}")
        print(f"Converting {hf_name}...")
        print(f"{'=' * 60}")

        torch_model = AutoModelForObjectDetection.from_pretrained(
            hf_name,
            trust_remote_code=True,
        ).eval()
        sd: Dict[str, np.ndarray] = {
            k: v.cpu().numpy() for k, v in torch_model.state_dict().items()
        }

        keras_model = DFineDetect(
            **DFINE_VARIANTS[cfg["variant"]],
            image_size=640,
            num_queries=300,
            num_classes=80,
        )
        print(f"  Parameters: {keras_model.count_params():,}")

        transfer_dfine_weights(keras_model, sd)

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
