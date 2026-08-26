import gc
from typing import Dict, List, Tuple

import keras
import numpy as np
from tqdm import tqdm

from zeromodels.conversion.weight_transfer_util import (
    transfer_nested_layer_weights,
    transfer_weights,
)
from zeromodels.models.sam import SAMPromptableSegment

vision_encoder_name_mapping = {
    "mlp_lin1": "mlp.lin1",
    "mlp_lin2": "mlp.lin2",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}


def transfer_sam_encoder_weights(
    keras_model: keras.Model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    """Transfer SAM ViT encoder + neck weights.

    Works for both ``SAMModel`` (encoder-only) and
    ``SAMPromptableSegment`` (full pipeline) since both build the
    encoder with identical layer names.
    """
    patch_conv = keras_model.get_layer("vision_encoder_patch_embed_projection")
    transfer_weights(
        "conv_kernel",
        patch_conv.kernel,
        hf_state_dict["vision_encoder.patch_embed.projection.weight"],
    )
    patch_conv.bias.assign(hf_state_dict["vision_encoder.patch_embed.projection.bias"])

    pos_layer = keras_model.get_layer("vision_encoder_pos_embed")
    pos_layer.pos_embed.assign(hf_state_dict["vision_encoder.pos_embed"])

    num_layers = keras_model.vision_num_hidden_layers
    for i in tqdm(range(num_layers), desc="Transferring SAM vision encoder layers"):
        layer = keras_model.get_layer(f"vision_encoder_layers_{i}")
        skipped = transfer_nested_layer_weights(
            layer,
            hf_state_dict,
            f"vision_encoder.layers.{i}",
            name_mapping=vision_encoder_name_mapping,
            skip_paths=["rel_pos"],
        )
        for keras_weight, _ in skipped:
            w_name = keras_weight.path.split("/")[-1]
            hf_key = f"vision_encoder.layers.{i}.attn.{w_name}"
            keras_weight.assign(hf_state_dict[hf_key])

    neck_conv1 = keras_model.get_layer("vision_encoder_neck_conv1")
    transfer_weights(
        "conv_kernel",
        neck_conv1.kernel,
        hf_state_dict["vision_encoder.neck.conv1.weight"],
    )
    neck_ln1 = keras_model.get_layer("vision_encoder_neck_layer_norm1")
    neck_ln1.gamma.assign(hf_state_dict["vision_encoder.neck.layer_norm1.weight"])
    neck_ln1.beta.assign(hf_state_dict["vision_encoder.neck.layer_norm1.bias"])

    neck_conv2 = keras_model.get_layer("vision_encoder_neck_conv2")
    transfer_weights(
        "conv_kernel",
        neck_conv2.kernel,
        hf_state_dict["vision_encoder.neck.conv2.weight"],
    )
    neck_ln2 = keras_model.get_layer("vision_encoder_neck_layer_norm2")
    neck_ln2.gamma.assign(hf_state_dict["vision_encoder.neck.layer_norm2.weight"])
    neck_ln2.beta.assign(hf_state_dict["vision_encoder.neck.layer_norm2.bias"])


def transfer_sam_weights(
    keras_model: keras.Model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    """Transfer full SAM weights (encoder + prompt encoder + mask decoder)."""
    transfer_sam_encoder_weights(keras_model, hf_state_dict)

    image_pe_layer = keras_model.get_layer("image_positional_embeddings")
    image_pe_layer.shared_embedding.positional_embedding.assign(
        hf_state_dict["shared_image_embedding.positional_embedding"]
    )

    prompt_enc = keras_model.get_layer("prompt_encoder")
    for i in range(prompt_enc.num_point_embeddings):
        prompt_enc.point_embeddings[i].assign(
            hf_state_dict[f"prompt_encoder.point_embed.{i}.weight"]
        )

    prompt_enc.not_a_point_embed.assign(
        hf_state_dict["prompt_encoder.not_a_point_embed.weight"]
    )
    prompt_enc.no_mask_embed.assign(
        hf_state_dict["prompt_encoder.no_mask_embed.weight"]
    )

    if getattr(keras_model, "enable_masks", False):
        for idx in (1, 2):
            conv = keras_model.get_layer(f"prompt_encoder_mask_embed_conv{idx}")
            transfer_weights(
                "conv_kernel",
                conv.kernel,
                hf_state_dict[f"prompt_encoder.mask_embed.conv{idx}.weight"],
            )
            conv.bias.assign(hf_state_dict[f"prompt_encoder.mask_embed.conv{idx}.bias"])
            ln = keras_model.get_layer(f"prompt_encoder_mask_embed_layer_norm{idx}")
            ln.gamma.assign(
                hf_state_dict[f"prompt_encoder.mask_embed.layer_norm{idx}.weight"]
            )
            ln.beta.assign(
                hf_state_dict[f"prompt_encoder.mask_embed.layer_norm{idx}.bias"]
            )

        conv3 = keras_model.get_layer("prompt_encoder_mask_embed_conv3")
        transfer_weights(
            "conv_kernel",
            conv3.kernel,
            hf_state_dict["prompt_encoder.mask_embed.conv3.weight"],
        )
        conv3.bias.assign(hf_state_dict["prompt_encoder.mask_embed.conv3.bias"])

    mask_dec = keras_model.get_layer("mask_decoder")

    mask_dec.iou_token.assign(hf_state_dict["mask_decoder.iou_token.weight"])
    mask_dec.mask_tokens.assign(hf_state_dict["mask_decoder.mask_tokens.weight"])

    for i in range(mask_dec.num_hidden_layers):
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
            for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                proj = getattr(attn_layer, proj_name)
                transfer_weights(
                    "kernel",
                    proj.kernel,
                    hf_state_dict[f"{hf_prefix}.{attn_suffix}.{proj_name}.weight"],
                )
                proj.bias.assign(
                    hf_state_dict[f"{hf_prefix}.{attn_suffix}.{proj_name}.bias"]
                )

        for ln_attr, hf_ln in [
            (mask_dec.transformer_layer_norm1s[i], "layer_norm1"),
            (mask_dec.transformer_layer_norm2s[i], "layer_norm2"),
            (mask_dec.transformer_layer_norm3s[i], "layer_norm3"),
            (mask_dec.transformer_layer_norm4s[i], "layer_norm4"),
        ]:
            ln_attr.gamma.assign(hf_state_dict[f"{hf_prefix}.{hf_ln}.weight"])
            ln_attr.beta.assign(hf_state_dict[f"{hf_prefix}.{hf_ln}.bias"])

        mlp_lin1 = mask_dec.transformer_mlp_lin1s[i]
        transfer_weights(
            "kernel", mlp_lin1.kernel, hf_state_dict[f"{hf_prefix}.mlp.lin1.weight"]
        )
        mlp_lin1.bias.assign(hf_state_dict[f"{hf_prefix}.mlp.lin1.bias"])

        mlp_lin2 = mask_dec.transformer_mlp_lin2s[i]
        transfer_weights(
            "kernel", mlp_lin2.kernel, hf_state_dict[f"{hf_prefix}.mlp.lin2.weight"]
        )
        mlp_lin2.bias.assign(hf_state_dict[f"{hf_prefix}.mlp.lin2.bias"])

    hf_final_attn = "mask_decoder.transformer.final_attn_token_to_image"
    for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
        proj = getattr(mask_dec.final_attn_token_to_image, proj_name)
        transfer_weights(
            "kernel",
            proj.kernel,
            hf_state_dict[f"{hf_final_attn}.{proj_name}.weight"],
        )
        proj.bias.assign(hf_state_dict[f"{hf_final_attn}.{proj_name}.bias"])

    final_ln = mask_dec.layer_norm_final_attn
    final_ln.gamma.assign(
        hf_state_dict["mask_decoder.transformer.layer_norm_final_attn.weight"]
    )
    final_ln.beta.assign(
        hf_state_dict["mask_decoder.transformer.layer_norm_final_attn.bias"]
    )

    transfer_weights(
        "conv_kernel",
        mask_dec.upscale_conv1.kernel,
        hf_state_dict["mask_decoder.upscale_conv1.weight"],
    )
    mask_dec.upscale_conv1.bias.assign(hf_state_dict["mask_decoder.upscale_conv1.bias"])

    mask_dec.upscale_layer_norm.gamma.assign(
        hf_state_dict["mask_decoder.upscale_layer_norm.weight"]
    )
    mask_dec.upscale_layer_norm.beta.assign(
        hf_state_dict["mask_decoder.upscale_layer_norm.bias"]
    )

    transfer_weights(
        "conv_kernel",
        mask_dec.upscale_conv2.kernel,
        hf_state_dict["mask_decoder.upscale_conv2.weight"],
    )
    mask_dec.upscale_conv2.bias.assign(hf_state_dict["mask_decoder.upscale_conv2.bias"])

    num_mask_tokens = mask_dec.num_mask_tokens
    for i in range(num_mask_tokens):
        hf_prefix = f"mask_decoder.output_hypernetworks_mlps.{i}"

        proj_in = mask_dec.output_hypernetworks_mlps_proj_ins[i]
        transfer_weights(
            "kernel", proj_in.kernel, hf_state_dict[f"{hf_prefix}.proj_in.weight"]
        )
        proj_in.bias.assign(hf_state_dict[f"{hf_prefix}.proj_in.bias"])

        for j in range(mask_dec._hyper_num_hidden):
            idx = i * mask_dec._hyper_num_hidden + j
            hidden = mask_dec.output_hypernetworks_mlps_hidden_layers[idx]
            transfer_weights(
                "kernel", hidden.kernel, hf_state_dict[f"{hf_prefix}.layers.{j}.weight"]
            )
            hidden.bias.assign(hf_state_dict[f"{hf_prefix}.layers.{j}.bias"])

        proj_out = mask_dec.output_hypernetworks_mlps_proj_outs[i]
        transfer_weights(
            "kernel", proj_out.kernel, hf_state_dict[f"{hf_prefix}.proj_out.weight"]
        )
        proj_out.bias.assign(hf_state_dict[f"{hf_prefix}.proj_out.bias"])

    hf_prefix = "mask_decoder.iou_prediction_head"
    transfer_weights(
        "kernel",
        mask_dec.iou_head_proj_in.kernel,
        hf_state_dict[f"{hf_prefix}.proj_in.weight"],
    )
    mask_dec.iou_head_proj_in.bias.assign(hf_state_dict[f"{hf_prefix}.proj_in.bias"])
    for j, hidden_layer in enumerate(mask_dec.iou_head_hidden_layers):
        transfer_weights(
            "kernel",
            hidden_layer.kernel,
            hf_state_dict[f"{hf_prefix}.layers.{j}.weight"],
        )
        hidden_layer.bias.assign(hf_state_dict[f"{hf_prefix}.layers.{j}.bias"])
    transfer_weights(
        "kernel",
        mask_dec.iou_head_proj_out.kernel,
        hf_state_dict[f"{hf_prefix}.proj_out.weight"],
    )
    mask_dec.iou_head_proj_out.bias.assign(hf_state_dict[f"{hf_prefix}.proj_out.bias"])


SAM_CONVERSION_CONFIG: List[Tuple[str, str]] = [
    ("sam_vit_base", "facebook/sam-vit-base"),
    ("sam_vit_large", "facebook/sam-vit-large"),
    ("sam_vit_huge", "facebook/sam-vit-huge"),
]


# Per-variant recipes (relocated from sam_config.py). Models load from the Hub
# by repo id; these build the arch for conversion + drive the backfill.
SAM_VARIANTS = {
    "sam_vit_base": {
        "vision_hidden_size": 768,
        "vision_num_hidden_layers": 12,
        "vision_num_attention_heads": 12,
        "vision_mlp_dim": 3072,
        "vision_global_attn_indexes": [2, 5, 8, 11],
        "image_size": 1024,
    },
    "sam_vit_large": {
        "vision_hidden_size": 1024,
        "vision_num_hidden_layers": 24,
        "vision_num_attention_heads": 16,
        "vision_mlp_dim": 4096,
        "vision_global_attn_indexes": [5, 11, 17, 23],
        "image_size": 1024,
    },
    "sam_vit_huge": {
        "vision_hidden_size": 1280,
        "vision_num_hidden_layers": 32,
        "vision_num_attention_heads": 16,
        "vision_mlp_dim": 5120,
        "vision_global_attn_indexes": [7, 15, 23, 31],
        "image_size": 1024,
    },
}


if __name__ == "__main__":
    import torch
    from transformers import SamModel

    for variant, hf_id in SAM_CONVERSION_CONFIG:
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        hf_model = SamModel.from_pretrained(hf_id).eval()
        hf_state_dict = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}

        keras_model: keras.Model = SAMPromptableSegment(**SAM_VARIANTS[variant])

        transfer_sam_weights(keras_model, hf_state_dict)

        print("Verifying model equivalence...")
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
        keras_masks = keras_output["pred_masks"]
        keras_iou = keras_output["iou_scores"]

        with torch.no_grad():
            hf_input = {
                "pixel_values": torch.from_numpy(test_image.transpose(0, 3, 1, 2)),
                "input_points": torch.from_numpy(test_points),
                "input_labels": torch.from_numpy(test_labels),
                "multimask_output": True,
            }
            hf_output = hf_model(**hf_input)
            hf_masks = hf_output.pred_masks.cpu().numpy()
            hf_iou = hf_output.iou_scores.cpu().numpy()

        mask_diff = float(np.max(np.abs(keras_masks - hf_masks)))
        iou_diff = float(np.max(np.abs(keras_iou - hf_iou)))
        print(f"  Max mask diff: {mask_diff:.6f}")
        print(f"  Max IoU diff:  {iou_diff:.6f}")

        if mask_diff > 0.8:
            raise ValueError(f"{variant}: mask diff {mask_diff:.6f} > 0.8")
        if iou_diff > 1e-2:
            raise ValueError(f"{variant}: iou diff {iou_diff:.6f} > 1e-2")
        print("  Verification OK")

        total_params = sum(int(np.prod(w.shape)) for w in keras_model.weights)
        total_gb = (total_params * 4) / (1024**3)
        if total_gb > 1.7:
            model_filename = f"{variant}.weights.json"
            keras_model.save_weights(model_filename, max_shard_size=1.7)
            print(f"  Saved -> {model_filename} (sharded, ~{total_gb:.2f} GB)")
        else:
            model_filename = f"{variant}.weights.h5"
            keras_model.save_weights(model_filename)
            print(f"  Saved -> {model_filename} (~{total_gb:.2f} GB)")

        del keras_model, hf_model, hf_state_dict
        keras.backend.clear_session()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
