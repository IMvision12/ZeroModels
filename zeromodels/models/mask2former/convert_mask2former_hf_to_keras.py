import gc
import os
from typing import Any, Dict, List

import keras
import numpy as np
from tqdm import tqdm

from zeromodels.conversion.weight_transfer_util import transfer_weights
from zeromodels.models.mask2former import Mask2FormerUniversalSegment

# Per-variant recipes (relocated from mask2former_config.py). Models load from
# the Hub by repo id; these build the arch for conversion + drive the backfill.
MASK2FORMER_VARIANTS = {
    "mask2former-swin-tiny-coco-instance": {
        "backbone_embed_dim": 96,
        "backbone_depths": (2, 2, 6, 2),
        "backbone_num_heads": (3, 6, 12, 24),
        "backbone_window_size": 7,
        "num_queries": 100,
        "num_classes": 80,
        "image_size": 384,
    },
    "mask2former-swin-small-coco-instance": {
        "backbone_embed_dim": 96,
        "backbone_depths": (2, 2, 18, 2),
        "backbone_num_heads": (3, 6, 12, 24),
        "backbone_window_size": 7,
        "num_queries": 100,
        "num_classes": 80,
        "image_size": 384,
    },
    "mask2former-swin-base-coco-instance": {
        "backbone_embed_dim": 128,
        "backbone_depths": (2, 2, 18, 2),
        "backbone_num_heads": (4, 8, 16, 32),
        "backbone_window_size": 12,
        "num_queries": 100,
        "num_classes": 80,
        "image_size": 384,
    },
    "mask2former-swin-large-coco-instance": {
        "backbone_embed_dim": 192,
        "backbone_depths": (2, 2, 18, 2),
        "backbone_num_heads": (6, 12, 24, 48),
        "backbone_window_size": 12,
        "num_queries": 200,
        "num_classes": 80,
        "image_size": 384,
    },
    "mask2former-swin-tiny-ade-semantic": {
        "backbone_embed_dim": 96,
        "backbone_depths": (2, 2, 6, 2),
        "backbone_num_heads": (3, 6, 12, 24),
        "backbone_window_size": 7,
        "num_queries": 100,
        "num_classes": 150,
        "image_size": 512,
    },
    "mask2former-swin-tiny-coco-panoptic": {
        "backbone_embed_dim": 96,
        "backbone_depths": (2, 2, 6, 2),
        "backbone_num_heads": (3, 6, 12, 24),
        "backbone_window_size": 7,
        "num_queries": 100,
        "num_classes": 133,
        "image_size": 384,
    },
}


def transfer_mask2former_weights(keras_model, hf_state_dict):
    """Transfer a Mask2Former state dict into the Keras Mask2Former model.

    Walks the Swin backbone, the MSDeformAttn pixel decoder, the masked-attention
    transformer decoder, and the class / mask-embedding heads, copying each source
    tensor into the matching Keras weight (transposing kernels and attention
    projections as needed).

    Args:
        keras_model: A freshly-built Mask2Former Keras model (unloaded weights).
        hf_state_dict: The source model ``state_dict`` mapping names to
            numpy arrays.
    """
    sd = hf_state_dict
    backbone = keras_model.get_layer("backbone")

    backbone_prefix = "model.pixel_level_module.encoder"
    pixel_decoder_prefix = "model.pixel_level_module.decoder"
    transformer_prefix = "model.transformer_module"

    print("Transferring Swin backbone...", flush=True)
    proj = backbone.patch_embeddings.projection
    transfer_weights(
        "conv_kernel",
        proj.weights[0],
        sd[f"{backbone_prefix}.embeddings.patch_embeddings.projection.weight"],
    )
    transfer_weights(
        "bias",
        proj.weights[1],
        sd[f"{backbone_prefix}.embeddings.patch_embeddings.projection.bias"],
    )
    transfer_weights(
        "gamma",
        backbone.embeddings_norm.weights[0],
        sd[f"{backbone_prefix}.embeddings.norm.weight"],
    )
    transfer_weights(
        "beta",
        backbone.embeddings_norm.weights[1],
        sd[f"{backbone_prefix}.embeddings.norm.bias"],
    )

    for stage_idx, stage in enumerate(backbone.stages):
        for block_idx, block in enumerate(stage.blocks):
            p = f"{backbone_prefix}.encoder.layers.{stage_idx}.blocks.{block_idx}"
            transfer_weights(
                "gamma",
                block.layernorm_before.weights[0],
                sd[f"{p}.layernorm_before.weight"],
            )
            transfer_weights(
                "beta",
                block.layernorm_before.weights[1],
                sd[f"{p}.layernorm_before.bias"],
            )
            attn = block.attention.self_attn
            for name in ("query", "key", "value"):
                dense = getattr(attn, name)
                transfer_weights(
                    "kernel", dense.weights[0], sd[f"{p}.attention.self.{name}.weight"]
                )
                transfer_weights(
                    "bias", dense.weights[1], sd[f"{p}.attention.self.{name}.bias"]
                )
            attn.relative_position_bias_table.assign(
                sd[f"{p}.attention.self.relative_position_bias_table"]
            )
            transfer_weights(
                "kernel",
                block.attention.output_dense.weights[0],
                sd[f"{p}.attention.output.dense.weight"],
            )
            transfer_weights(
                "bias",
                block.attention.output_dense.weights[1],
                sd[f"{p}.attention.output.dense.bias"],
            )
            transfer_weights(
                "gamma",
                block.layernorm_after.weights[0],
                sd[f"{p}.layernorm_after.weight"],
            )
            transfer_weights(
                "beta",
                block.layernorm_after.weights[1],
                sd[f"{p}.layernorm_after.bias"],
            )
            transfer_weights(
                "kernel",
                block.intermediate_dense.weights[0],
                sd[f"{p}.intermediate.dense.weight"],
            )
            transfer_weights(
                "bias",
                block.intermediate_dense.weights[1],
                sd[f"{p}.intermediate.dense.bias"],
            )
            transfer_weights(
                "kernel", block.output_dense.weights[0], sd[f"{p}.output.dense.weight"]
            )
            transfer_weights(
                "bias", block.output_dense.weights[1], sd[f"{p}.output.dense.bias"]
            )

        if stage.downsample is not None:
            ds_prefix = f"{backbone_prefix}.encoder.layers.{stage_idx}.downsample"
            transfer_weights(
                "kernel",
                stage.downsample.reduction.weights[0],
                sd[f"{ds_prefix}.reduction.weight"],
            )
            transfer_weights(
                "gamma",
                stage.downsample.norm.weights[0],
                sd[f"{ds_prefix}.norm.weight"],
            )
            transfer_weights(
                "beta", stage.downsample.norm.weights[1], sd[f"{ds_prefix}.norm.bias"]
            )

    for i in range(len(backbone.hidden_states_norms)):
        nrm = backbone.hidden_states_norms[i]
        transfer_weights(
            "gamma",
            nrm.weights[0],
            sd[f"{backbone_prefix}.hidden_states_norms.stage{i + 1}.weight"],
        )
        transfer_weights(
            "beta",
            nrm.weights[1],
            sd[f"{backbone_prefix}.hidden_states_norms.stage{i + 1}.bias"],
        )

    print("Transferring pixel decoder...", flush=True)
    for i in range(3):
        p_proj = f"{pixel_decoder_prefix}.input_projections.{i}"
        conv = keras_model.get_layer(f"pixel_decoder_input_projections_{i}_conv")
        transfer_weights("conv_kernel", conv.weights[0], sd[f"{p_proj}.0.weight"])
        transfer_weights("bias", conv.weights[1], sd[f"{p_proj}.0.bias"])
        nrm = keras_model.get_layer(f"pixel_decoder_input_projections_{i}_norm")
        transfer_weights("gamma", nrm.weights[0], sd[f"{p_proj}.1.weight"])
        transfer_weights("beta", nrm.weights[1], sd[f"{p_proj}.1.bias"])

    for i in tqdm(
        range(keras_model.encoder_num_layers), desc="Transferring encoder layers"
    ):
        p = f"{pixel_decoder_prefix}.encoder.layers.{i}"
        prefix_k = f"pixel_decoder_encoder_layers_{i}"
        attn = keras_model.get_layer(f"{prefix_k}_self_attn")
        for name in (
            "sampling_offsets",
            "attention_weights",
            "value_proj",
            "output_proj",
        ):
            dense = getattr(attn, name)
            transfer_weights(
                "kernel", dense.weights[0], sd[f"{p}.self_attn.{name}.weight"]
            )
            transfer_weights("bias", dense.weights[1], sd[f"{p}.self_attn.{name}.bias"])
        sa_ln = keras_model.get_layer(f"{prefix_k}_self_attn_layer_norm")
        transfer_weights(
            "gamma", sa_ln.weights[0], sd[f"{p}.self_attn_layer_norm.weight"]
        )
        transfer_weights("beta", sa_ln.weights[1], sd[f"{p}.self_attn_layer_norm.bias"])
        fc1 = keras_model.get_layer(f"{prefix_k}_fc1")
        transfer_weights("kernel", fc1.weights[0], sd[f"{p}.fc1.weight"])
        transfer_weights("bias", fc1.weights[1], sd[f"{p}.fc1.bias"])
        fc2 = keras_model.get_layer(f"{prefix_k}_fc2")
        transfer_weights("kernel", fc2.weights[0], sd[f"{p}.fc2.weight"])
        transfer_weights("bias", fc2.weights[1], sd[f"{p}.fc2.bias"])
        fln = keras_model.get_layer(f"{prefix_k}_final_layer_norm")
        transfer_weights("gamma", fln.weights[0], sd[f"{p}.final_layer_norm.weight"])
        transfer_weights("beta", fln.weights[1], sd[f"{p}.final_layer_norm.bias"])

    keras_model.get_layer("pixel_decoder_level_embed").weight.assign(
        sd[f"{pixel_decoder_prefix}.level_embed"]
    )

    adapter_conv = keras_model.get_layer("pixel_decoder_adapter_1_conv")
    transfer_weights(
        "conv_kernel",
        adapter_conv.weights[0],
        sd[f"{pixel_decoder_prefix}.adapter_1.0.weight"],
    )
    adapter_norm = keras_model.get_layer("pixel_decoder_adapter_1_norm")
    transfer_weights(
        "gamma",
        adapter_norm.weights[0],
        sd[f"{pixel_decoder_prefix}.adapter_1.1.weight"],
    )
    transfer_weights(
        "beta", adapter_norm.weights[1], sd[f"{pixel_decoder_prefix}.adapter_1.1.bias"]
    )
    layer1_conv = keras_model.get_layer("pixel_decoder_layer_1_conv")
    transfer_weights(
        "conv_kernel",
        layer1_conv.weights[0],
        sd[f"{pixel_decoder_prefix}.layer_1.0.weight"],
    )
    layer1_norm = keras_model.get_layer("pixel_decoder_layer_1_norm")
    transfer_weights(
        "gamma", layer1_norm.weights[0], sd[f"{pixel_decoder_prefix}.layer_1.1.weight"]
    )
    transfer_weights(
        "beta", layer1_norm.weights[1], sd[f"{pixel_decoder_prefix}.layer_1.1.bias"]
    )
    mask_proj = keras_model.get_layer("pixel_decoder_mask_projection")
    transfer_weights(
        "conv_kernel",
        mask_proj.weights[0],
        sd[f"{pixel_decoder_prefix}.mask_projection.weight"],
    )
    transfer_weights(
        "bias", mask_proj.weights[1], sd[f"{pixel_decoder_prefix}.mask_projection.bias"]
    )

    print("Transferring transformer decoder...", flush=True)
    keras_model.get_layer("transformer_decoder_queries_features").weight.assign(
        sd[f"{transformer_prefix}.queries_features.weight"]
    )
    keras_model.get_layer("transformer_decoder_queries_embedder").weight.assign(
        sd[f"{transformer_prefix}.queries_embedder.weight"]
    )
    keras_model.get_layer("transformer_decoder_level_embed").weight.assign(
        sd[f"{transformer_prefix}.level_embed.weight"]
    )

    for i in tqdm(
        range(keras_model.decoder_num_layers), desc="Transferring decoder layers"
    ):
        p = f"{transformer_prefix}.decoder.layers.{i}"
        prefix_k = f"transformer_decoder_layers_{i}"

        sa = keras_model.get_layer(f"{prefix_k}_self_attn")
        hf_out_proj = "o_proj" if f"{p}.self_attn.o_proj.weight" in sd else "out_proj"
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            hf_proj = hf_out_proj if proj == "out_proj" else proj
            layer = getattr(sa, proj)
            transfer_weights(
                "kernel", layer.weights[0], sd[f"{p}.self_attn.{hf_proj}.weight"]
            )
            transfer_weights(
                "bias", layer.weights[1], sd[f"{p}.self_attn.{hf_proj}.bias"]
            )

        sa_ln = keras_model.get_layer(f"{prefix_k}_self_attn_layer_norm")
        transfer_weights(
            "gamma", sa_ln.weights[0], sd[f"{p}.self_attn_layer_norm.weight"]
        )
        transfer_weights("beta", sa_ln.weights[1], sd[f"{p}.self_attn_layer_norm.bias"])

        ca = keras_model.get_layer(f"{prefix_k}_cross_attn")
        ca.in_proj_weight.assign(sd[f"{p}.cross_attn.in_proj_weight"])
        ca.in_proj_bias.assign(sd[f"{p}.cross_attn.in_proj_bias"])
        transfer_weights(
            "kernel", ca.out_proj.weights[0], sd[f"{p}.cross_attn.out_proj.weight"]
        )
        transfer_weights(
            "bias", ca.out_proj.weights[1], sd[f"{p}.cross_attn.out_proj.bias"]
        )
        ca_ln = keras_model.get_layer(f"{prefix_k}_cross_attn_layer_norm")
        transfer_weights(
            "gamma", ca_ln.weights[0], sd[f"{p}.cross_attn_layer_norm.weight"]
        )
        transfer_weights(
            "beta", ca_ln.weights[1], sd[f"{p}.cross_attn_layer_norm.bias"]
        )

        fc1 = keras_model.get_layer(f"{prefix_k}_fc1")
        transfer_weights("kernel", fc1.weights[0], sd[f"{p}.fc1.weight"])
        transfer_weights("bias", fc1.weights[1], sd[f"{p}.fc1.bias"])
        fc2 = keras_model.get_layer(f"{prefix_k}_fc2")
        transfer_weights("kernel", fc2.weights[0], sd[f"{p}.fc2.weight"])
        transfer_weights("bias", fc2.weights[1], sd[f"{p}.fc2.bias"])
        fln = keras_model.get_layer(f"{prefix_k}_final_layer_norm")
        transfer_weights("gamma", fln.weights[0], sd[f"{p}.final_layer_norm.weight"])
        transfer_weights("beta", fln.weights[1], sd[f"{p}.final_layer_norm.bias"])

    dec_ln = keras_model.get_layer("transformer_decoder_layernorm")
    transfer_weights(
        "gamma", dec_ln.weights[0], sd[f"{transformer_prefix}.decoder.layernorm.weight"]
    )
    transfer_weights(
        "beta", dec_ln.weights[1], sd[f"{transformer_prefix}.decoder.layernorm.bias"]
    )

    for i in range(3):
        emb = keras_model.get_layer(f"transformer_decoder_mask_embedder_{i}")
        base = f"{transformer_prefix}.decoder.mask_predictor.mask_embedder.{i}.0"
        transfer_weights("kernel", emb.weights[0], sd[f"{base}.weight"])
        transfer_weights("bias", emb.weights[1], sd[f"{base}.bias"])

    print("Transferring class_predictor...", flush=True)
    cp = keras_model.get_layer("class_predictor")
    transfer_weights("kernel", cp.weights[0], sd["class_predictor.weight"])
    transfer_weights("bias", cp.weights[1], sd["class_predictor.bias"])


MASK2FORMER_CONVERSION_CONFIG: List[Dict[str, Any]] = [
    {"variant": variant, "hf_id": f"facebook/{variant}"}
    for variant in MASK2FORMER_VARIANTS
]


if __name__ == "__main__":
    import torch
    from transformers import Mask2FormerForUniversalSegmentation

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for cfg in MASK2FORMER_CONVERSION_CONFIG:
        variant = cfg["variant"]
        hf_id = cfg["hf_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        keras_model = Mask2FormerUniversalSegment.from_weights(
            variant, load_weights=False
        )

        hf_model = Mask2FormerForUniversalSegmentation.from_pretrained(
            hf_id, token=os.environ.get("HF_TOKEN")
        ).eval()
        sd = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}
        hf_model = hf_model.to(device)

        transfer_mask2former_weights(keras_model, sd)

        input_size = keras_model.image_size[0]
        rng = np.random.default_rng(42)
        pix_hwc = rng.standard_normal((1, input_size, input_size, 3)).astype(np.float32)

        with torch.no_grad():
            hf_pixel_values = torch.from_numpy(np.transpose(pix_hwc, (0, 3, 1, 2))).to(
                device
            )
            hf_out = hf_model(pixel_values=hf_pixel_values)
            hf_class = hf_out.class_queries_logits.cpu().numpy()
            hf_mask = hf_out.masks_queries_logits.cpu().numpy()

        del hf_model, hf_out, sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        keras_input = keras.ops.convert_to_tensor(pix_hwc, dtype="float32")
        keras_out = keras_model(keras_input)
        keras_class = keras.ops.convert_to_numpy(keras_out["class_queries_logits"])
        keras_mask = keras.ops.convert_to_numpy(keras_out["masks_queries_logits"])

        class_diff = float(np.max(np.abs(hf_class - keras_class)))
        mask_diff = float(np.max(np.abs(hf_mask - keras_mask)))
        print(f"  Max class logits diff: {class_diff:.6f}", flush=True)
        print(f"  Max mask logits diff:  {mask_diff:.6f}", flush=True)

        out_filename = f"{variant}.weights.h5"
        keras_model.save_weights(out_filename)
        print(f"  Saved -> {out_filename}", flush=True)

        del keras_model
        keras.backend.clear_session()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
