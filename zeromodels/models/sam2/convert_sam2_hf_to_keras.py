import gc
import os
from typing import Dict

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.weight_transfer_util import (
    transfer_nested_layer_weights,
    transfer_weights,
)

BACKBONE_NAME_MAPPING = {
    "mlp_proj_in": "mlp.proj_in",
    "mlp_proj_out": "mlp.proj_out",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}


def transfer_sam2_encoder_weights(
    keras_model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    """Transfer Hiera backbone + FPN neck + no_memory_embedding weights.

    Works for both ``SAM2Model`` (encoder-only) and
    ``SAM2PromptableSegment`` (full pipeline) since both build the
    encoder with identical layer names.
    """
    patch_conv = keras_model.get_layer("backbone_patch_embed_projection")
    transfer_weights(
        "conv_kernel",
        patch_conv.kernel,
        hf_state_dict["vision_encoder.backbone.patch_embed.projection.weight"],
    )
    patch_conv.bias.assign(
        hf_state_dict["vision_encoder.backbone.patch_embed.projection.bias"]
    )

    pos_layer = keras_model.get_layer("backbone_pos_embed")
    pos_embed_hf = hf_state_dict["vision_encoder.backbone.pos_embed"]
    pos_layer.pos_embed.assign(np.transpose(pos_embed_hf, (0, 2, 3, 1)))
    pos_embed_window_hf = hf_state_dict["vision_encoder.backbone.pos_embed_window"]
    pos_layer.pos_embed_window.assign(np.transpose(pos_embed_window_hf, (0, 2, 3, 1)))
    pos_layer._recompute_full_pos()

    total_blocks = sum(keras_model.blocks_per_stage)
    for i in tqdm(range(total_blocks), desc="Transferring vision encoder blocks"):
        layer = keras_model.get_layer(f"backbone_blocks_{i}")
        transfer_nested_layer_weights(
            layer,
            hf_state_dict,
            f"vision_encoder.backbone.blocks.{i}",
            name_mapping=BACKBONE_NAME_MAPPING,
        )

    n_fpn = len(keras_model.backbone_channel_list)
    for i in range(n_fpn):
        neck_conv = keras_model.get_layer(f"neck_convs_{i}")
        transfer_weights(
            "conv_kernel",
            neck_conv.kernel,
            hf_state_dict[f"vision_encoder.neck.convs.{i}.weight"],
        )
        neck_conv.bias.assign(hf_state_dict[f"vision_encoder.neck.convs.{i}.bias"])

    no_mem_layer = keras_model.get_layer("no_memory_embedding")
    no_mem_layer.embedding.assign(hf_state_dict["no_memory_embedding"].reshape(-1))


def transfer_sam2_weights(keras_model, hf_state_dict: Dict[str, np.ndarray]) -> None:
    """Transfer full SAM2 weights (encoder + prompt encoder + mask decoder)."""
    transfer_sam2_encoder_weights(keras_model, hf_state_dict)

    prompt_enc = keras_model.get_layer("prompt_encoder")
    prompt_enc.shared_embedding.positional_embedding.assign(
        hf_state_dict["shared_image_embedding.positional_embedding"]
    )
    hf_point_embed = hf_state_dict["prompt_encoder.point_embed.weight"]
    for i in range(prompt_enc.num_point_embeddings):
        prompt_enc.point_embeddings[i].assign(hf_point_embed[i : i + 1])
    prompt_enc.not_a_point_embed.assign(
        hf_state_dict["prompt_encoder.not_a_point_embed.weight"]
    )
    prompt_enc.no_mask_embed.assign(
        hf_state_dict["prompt_encoder.no_mask_embed.weight"]
    )

    mask_dec = keras_model.get_layer("mask_decoder")
    mask_dec.obj_score_token.assign(
        hf_state_dict["mask_decoder.obj_score_token.weight"]
    )
    mask_dec.iou_token.assign(hf_state_dict["mask_decoder.iou_token.weight"])
    mask_dec.mask_tokens.assign(hf_state_dict["mask_decoder.mask_tokens.weight"])

    for i in tqdm(
        range(mask_dec.num_hidden_layers), desc="Transferring mask decoder layers"
    ):
        hf_prefix = f"mask_decoder.transformer.layers.{i}"

        for attn_layer, attn_suffix in [
            (mask_dec.transformer_self_attns[i], "self_attn"),
            (
                mask_dec.transformer_cross_attn_token_to_images[i],
                "cross_attn_token_to_image",
            ),
            (
                mask_dec.transformer_cross_attn_image_to_tokens[i],
                "cross_attn_image_to_token",
            ),
        ]:
            for proj_name, hf_proj_name in [
                ("q_proj", "q_proj"),
                ("k_proj", "k_proj"),
                ("v_proj", "v_proj"),
                ("out_proj", "o_proj"),
            ]:
                proj = getattr(attn_layer, proj_name)
                transfer_weights(
                    "kernel",
                    proj.kernel,
                    hf_state_dict[f"{hf_prefix}.{attn_suffix}.{hf_proj_name}.weight"],
                )
                proj.bias.assign(
                    hf_state_dict[f"{hf_prefix}.{attn_suffix}.{hf_proj_name}.bias"]
                )

        for ln_attr, hf_name in [
            ("transformer_layer_norm1s", "layer_norm1"),
            ("transformer_layer_norm2s", "layer_norm2"),
            ("transformer_layer_norm3s", "layer_norm3"),
            ("transformer_layer_norm4s", "layer_norm4"),
        ]:
            ln = getattr(mask_dec, ln_attr)[i]
            ln.gamma.assign(hf_state_dict[f"{hf_prefix}.{hf_name}.weight"])
            ln.beta.assign(hf_state_dict[f"{hf_prefix}.{hf_name}.bias"])

        mlp_lin1 = mask_dec.transformer_mlp_lin1s[i]
        transfer_weights(
            "kernel", mlp_lin1.kernel, hf_state_dict[f"{hf_prefix}.mlp.proj_in.weight"]
        )
        mlp_lin1.bias.assign(hf_state_dict[f"{hf_prefix}.mlp.proj_in.bias"])

        mlp_lin2 = mask_dec.transformer_mlp_lin2s[i]
        transfer_weights(
            "kernel", mlp_lin2.kernel, hf_state_dict[f"{hf_prefix}.mlp.proj_out.weight"]
        )
        mlp_lin2.bias.assign(hf_state_dict[f"{hf_prefix}.mlp.proj_out.bias"])

    hf_final_attn = "mask_decoder.transformer.final_attn_token_to_image"
    for proj_name, hf_proj_name in [
        ("q_proj", "q_proj"),
        ("k_proj", "k_proj"),
        ("v_proj", "v_proj"),
        ("out_proj", "o_proj"),
    ]:
        proj = getattr(mask_dec.final_attn_token_to_image, proj_name)
        transfer_weights(
            "kernel",
            proj.kernel,
            hf_state_dict[f"{hf_final_attn}.{hf_proj_name}.weight"],
        )
        proj.bias.assign(hf_state_dict[f"{hf_final_attn}.{hf_proj_name}.bias"])

    final_ln = mask_dec.layer_norm_final_attn
    final_ln.gamma.assign(
        hf_state_dict["mask_decoder.transformer.layer_norm_final_attn.weight"]
    )
    final_ln.beta.assign(
        hf_state_dict["mask_decoder.transformer.layer_norm_final_attn.bias"]
    )

    for conv_attr, hf_name in [
        ("upscale_conv1", "upscale_conv1"),
        ("upscale_conv2", "upscale_conv2"),
        ("conv_s0", "conv_s0"),
        ("conv_s1", "conv_s1"),
    ]:
        conv = getattr(mask_dec, conv_attr)
        transfer_weights(
            "conv_kernel", conv.kernel, hf_state_dict[f"mask_decoder.{hf_name}.weight"]
        )
        conv.bias.assign(hf_state_dict[f"mask_decoder.{hf_name}.bias"])

    mask_dec.upscale_layer_norm.gamma.assign(
        hf_state_dict["mask_decoder.upscale_layer_norm.weight"]
    )
    mask_dec.upscale_layer_norm.beta.assign(
        hf_state_dict["mask_decoder.upscale_layer_norm.bias"]
    )

    for i in range(mask_dec.num_mask_tokens):
        hf_prefix = f"mask_decoder.output_hypernetworks_mlps.{i}"

        proj_in = mask_dec.output_hypernetworks_mlps_proj_ins[i]
        transfer_weights(
            "kernel", proj_in.kernel, hf_state_dict[f"{hf_prefix}.proj_in.weight"]
        )
        proj_in.bias.assign(hf_state_dict[f"{hf_prefix}.proj_in.bias"])

        hidden = mask_dec.output_hypernetworks_mlps_hidden_layers[i]
        transfer_weights(
            "kernel", hidden.kernel, hf_state_dict[f"{hf_prefix}.layers.0.weight"]
        )
        hidden.bias.assign(hf_state_dict[f"{hf_prefix}.layers.0.bias"])

        proj_out = mask_dec.output_hypernetworks_mlps_proj_outs[i]
        transfer_weights(
            "kernel", proj_out.kernel, hf_state_dict[f"{hf_prefix}.proj_out.weight"]
        )
        proj_out.bias.assign(hf_state_dict[f"{hf_prefix}.proj_out.bias"])

    for head_prefix, proj_in_attr, hidden_attr, proj_out_attr in [
        (
            "mask_decoder.iou_prediction_head",
            "iou_head_proj_in",
            "iou_head_hidden_layers",
            "iou_head_proj_out",
        ),
        (
            "mask_decoder.pred_obj_score_head",
            "obj_score_proj_in",
            "obj_score_hidden_layers",
            "obj_score_proj_out",
        ),
    ]:
        proj_in = getattr(mask_dec, proj_in_attr)
        transfer_weights(
            "kernel", proj_in.kernel, hf_state_dict[f"{head_prefix}.proj_in.weight"]
        )
        proj_in.bias.assign(hf_state_dict[f"{head_prefix}.proj_in.bias"])

        for j, hidden in enumerate(getattr(mask_dec, hidden_attr)):
            transfer_weights(
                "kernel",
                hidden.kernel,
                hf_state_dict[f"{head_prefix}.layers.{j}.weight"],
            )
            hidden.bias.assign(hf_state_dict[f"{head_prefix}.layers.{j}.bias"])

        proj_out = getattr(mask_dec, proj_out_attr)
        transfer_weights(
            "kernel", proj_out.kernel, hf_state_dict[f"{head_prefix}.proj_out.weight"]
        )
        proj_out.bias.assign(hf_state_dict[f"{head_prefix}.proj_out.bias"])


# Per-variant recipes (relocated from sam2_config.py). Models load from the Hub
# by repo id; these build the arch for conversion + drive the backfill.
SAM2_VARIANTS = {
    "sam2_hiera_tiny": {
        "hidden_dim": 96,
        "blocks_per_stage": [1, 2, 7, 2],
        "embed_dim_per_stage": [96, 192, 384, 768],
        "num_attention_heads_per_stage": [1, 2, 4, 8],
        "window_size_per_stage": [8, 4, 14, 7],
        "global_attention_blocks": [5, 7, 9],
        "backbone_channel_list": [768, 384, 192, 96],
    },
    "sam2_hiera_small": {
        "hidden_dim": 96,
        "blocks_per_stage": [1, 2, 11, 2],
        "embed_dim_per_stage": [96, 192, 384, 768],
        "num_attention_heads_per_stage": [1, 2, 4, 8],
        "window_size_per_stage": [8, 4, 14, 7],
        "global_attention_blocks": [7, 10, 13],
        "backbone_channel_list": [768, 384, 192, 96],
    },
    "sam2_hiera_base_plus": {
        "hidden_dim": 112,
        "blocks_per_stage": [2, 3, 16, 3],
        "embed_dim_per_stage": [112, 224, 448, 896],
        "num_attention_heads_per_stage": [2, 4, 8, 16],
        "window_size_per_stage": [8, 4, 14, 7],
        "global_attention_blocks": [12, 16, 20],
        "backbone_channel_list": [896, 448, 224, 112],
        "window_pos_embed_bg_size": [14, 14],
    },
    "sam2_hiera_large": {
        "hidden_dim": 144,
        "blocks_per_stage": [2, 6, 36, 4],
        "embed_dim_per_stage": [144, 288, 576, 1152],
        "num_attention_heads_per_stage": [2, 4, 8, 16],
        "window_size_per_stage": [8, 4, 16, 8],
        "global_attention_blocks": [23, 33, 43],
        "backbone_channel_list": [1152, 576, 288, 144],
    },
}


if __name__ == "__main__":
    os.environ.setdefault("KERAS_BACKEND", "torch")

    import keras
    import torch
    from transformers import Sam2Model as HFSam2Model

    from zeromodels.models.sam2 import SAM2PromptableSegment

    SAM2_CONVERSION_CONFIG = [
        ("sam2_hiera_tiny", "facebook/sam2-hiera-tiny"),
        ("sam2_hiera_small", "facebook/sam2-hiera-small"),
        ("sam2_hiera_base_plus", "facebook/sam2-hiera-base-plus"),
        ("sam2_hiera_large", "facebook/sam2-hiera-large"),
    ]

    for variant, hf_id in SAM2_CONVERSION_CONFIG:
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        hf_model = HFSam2Model.from_pretrained(
            hf_id, attn_implementation="eager"
        ).eval()
        state = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}

        keras_model = SAM2PromptableSegment(**SAM2_VARIANTS[variant])
        transfer_sam2_weights(keras_model, state)

        np.random.seed(42)
        test_image = np.random.rand(1, 1024, 1024, 3).astype(np.float32)
        test_points = np.array([[[[500.0, 500.0]]]], dtype=np.float32)
        test_labels = np.array([[[1]]], dtype=np.int32)

        keras_output = keras_model.predict(
            {
                "pixel_values": test_image,
                "input_points": test_points,
                "input_labels": test_labels,
            },
            verbose=0,
        )

        with torch.no_grad():
            hf_output = hf_model(
                pixel_values=torch.from_numpy(test_image.transpose(0, 3, 1, 2)),
                input_points=torch.from_numpy(test_points),
                input_labels=torch.from_numpy(test_labels),
                multimask_output=True,
            )
        mask_diff = np.max(
            np.abs(keras_output["pred_masks"] - hf_output.pred_masks.cpu().numpy())
        )
        iou_diff = np.max(
            np.abs(keras_output["iou_scores"] - hf_output.iou_scores.cpu().numpy())
        )
        obj_diff = np.max(
            np.abs(
                keras_output["object_score_logits"]
                - hf_output.object_score_logits.cpu().numpy()
            )
        )
        print(
            f"  mask diff = {mask_diff:.4e}, iou diff = {iou_diff:.4e}, "
            f"obj diff = {obj_diff:.4e}"
        )
        assert mask_diff < 0.5 and iou_diff < 0.05 and obj_diff < 0.05

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, hf_model, state
        keras.backend.clear_session()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
