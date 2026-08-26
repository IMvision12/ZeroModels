import gc
import os
from typing import Any, Dict, List

import keras
import numpy as np
from tqdm import tqdm

from zeromodels.conversion.weight_transfer_util import transfer_weights
from zeromodels.models.maskformer import MaskFormerUniversalSegment


def transfer_maskformer_weights(keras_model, hf_state_dict):
    sd = hf_state_dict
    backbone = keras_model.get_layer("backbone")

    sample_key = next(iter(sd))
    model_prefix = "model." if sample_key.startswith("model.") else ""

    backbone_prefix = f"{model_prefix}pixel_level_module.encoder.model"
    pixel_decoder_prefix = f"{model_prefix}pixel_level_module.decoder"
    transformer_prefix = f"{model_prefix}transformer_module"

    print("Transferring Swin backbone...")
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

    encoder_prefix = backbone_prefix.rsplit(".model", 1)[0]
    for i in range(len(backbone.hidden_states_norms)):
        nrm = backbone.hidden_states_norms[i]
        transfer_weights(
            "gamma",
            nrm.weights[0],
            sd[f"{encoder_prefix}.hidden_states_norms.{i}.weight"],
        )
        transfer_weights(
            "beta", nrm.weights[1], sd[f"{encoder_prefix}.hidden_states_norms.{i}.bias"]
        )

    print("Transferring pixel decoder...")
    stem_conv = keras_model.get_layer("pixel_decoder_fpn_stem_conv")
    transfer_weights(
        "conv_kernel",
        stem_conv.weights[0],
        sd[f"{pixel_decoder_prefix}.fpn.stem.0.weight"],
    )
    stem_norm = keras_model.get_layer("pixel_decoder_fpn_stem_norm")
    transfer_weights(
        "gamma", stem_norm.weights[0], sd[f"{pixel_decoder_prefix}.fpn.stem.1.weight"]
    )
    transfer_weights(
        "beta", stem_norm.weights[1], sd[f"{pixel_decoder_prefix}.fpn.stem.1.bias"]
    )

    for i in range(3):
        p = f"{pixel_decoder_prefix}.fpn.layers.{i}"
        proj_conv = keras_model.get_layer(f"pixel_decoder_fpn_layer_{i}_proj_conv")
        transfer_weights("conv_kernel", proj_conv.weights[0], sd[f"{p}.proj.0.weight"])
        proj_norm = keras_model.get_layer(f"pixel_decoder_fpn_layer_{i}_proj_norm")
        transfer_weights("gamma", proj_norm.weights[0], sd[f"{p}.proj.1.weight"])
        transfer_weights("beta", proj_norm.weights[1], sd[f"{p}.proj.1.bias"])
        block_conv = keras_model.get_layer(f"pixel_decoder_fpn_layer_{i}_block_conv")
        transfer_weights(
            "conv_kernel", block_conv.weights[0], sd[f"{p}.block.0.weight"]
        )
        block_norm = keras_model.get_layer(f"pixel_decoder_fpn_layer_{i}_block_norm")
        transfer_weights("gamma", block_norm.weights[0], sd[f"{p}.block.1.weight"])
        transfer_weights("beta", block_norm.weights[1], sd[f"{p}.block.1.bias"])

    mask_proj = keras_model.get_layer("pixel_decoder_mask_projection")
    transfer_weights(
        "conv_kernel",
        mask_proj.weights[0],
        sd[f"{pixel_decoder_prefix}.mask_projection.weight"],
    )
    transfer_weights(
        "bias", mask_proj.weights[1], sd[f"{pixel_decoder_prefix}.mask_projection.bias"]
    )

    print("Transferring transformer decoder...")
    keras_model.get_layer("transformer_decoder_queries_embedder").weight.assign(
        sd[f"{transformer_prefix}.queries_embedder.weight"]
    )
    input_proj = keras_model.get_layer("transformer_decoder_input_projection")
    transfer_weights(
        "conv_kernel",
        input_proj.weights[0],
        sd[f"{transformer_prefix}.input_projection.weight"],
    )
    transfer_weights(
        "bias", input_proj.weights[1], sd[f"{transformer_prefix}.input_projection.bias"]
    )

    for i in tqdm(
        range(keras_model.decoder_num_layers), desc="Transferring decoder layers"
    ):
        p = f"{transformer_prefix}.decoder.layers.{i}"
        prefix_k = f"transformer_decoder_layers_{i}"

        for attn_name in ("self_attn", "encoder_attn"):
            attn = keras_model.get_layer(f"{prefix_k}_{attn_name}")
            hf_out_proj = (
                "o_proj" if f"{p}.{attn_name}.o_proj.weight" in sd else "out_proj"
            )
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                hf_proj = hf_out_proj if proj == "o_proj" else proj
                layer = getattr(attn, proj)
                transfer_weights(
                    "kernel", layer.weights[0], sd[f"{p}.{attn_name}.{hf_proj}.weight"]
                )
                transfer_weights(
                    "bias", layer.weights[1], sd[f"{p}.{attn_name}.{hf_proj}.bias"]
                )
            ln = keras_model.get_layer(f"{prefix_k}_{attn_name}_layer_norm")
            transfer_weights(
                "gamma", ln.weights[0], sd[f"{p}.{attn_name}_layer_norm.weight"]
            )
            transfer_weights(
                "beta", ln.weights[1], sd[f"{p}.{attn_name}_layer_norm.bias"]
            )

        ffn = "mlp." if f"{p}.mlp.fc1.weight" in sd else ""
        fc1 = keras_model.get_layer(f"{prefix_k}_fc1")
        transfer_weights("kernel", fc1.weights[0], sd[f"{p}.{ffn}fc1.weight"])
        transfer_weights("bias", fc1.weights[1], sd[f"{p}.{ffn}fc1.bias"])
        fc2 = keras_model.get_layer(f"{prefix_k}_fc2")
        transfer_weights("kernel", fc2.weights[0], sd[f"{p}.{ffn}fc2.weight"])
        transfer_weights("bias", fc2.weights[1], sd[f"{p}.{ffn}fc2.bias"])
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

    print("Transferring heads...")
    cp = keras_model.get_layer("class_predictor")
    transfer_weights("kernel", cp.weights[0], sd["class_predictor.weight"])
    transfer_weights("bias", cp.weights[1], sd["class_predictor.bias"])
    for i in range(3):
        emb = keras_model.get_layer(f"mask_embedder_{i}")
        transfer_weights("kernel", emb.weights[0], sd[f"mask_embedder.{i}.0.weight"])
        transfer_weights("bias", emb.weights[1], sd[f"mask_embedder.{i}.0.bias"])


MASKFORMER_CONVERSION_CONFIG: List[Dict[str, Any]] = [
    {
        "variant": "maskformer-swin-tiny-ade",
        "hf_id": "facebook/maskformer-swin-tiny-ade",
    },
    {
        "variant": "maskformer-swin-tiny-coco",
        "hf_id": "facebook/maskformer-swin-tiny-coco",
    },
    {
        "variant": "maskformer-swin-small-coco",
        "hf_id": "facebook/maskformer-swin-small-coco",
    },
    {
        "variant": "maskformer-swin-base-ade",
        "hf_id": "facebook/maskformer-swin-base-ade",
    },
    {
        "variant": "maskformer-swin-base-coco",
        "hf_id": "facebook/maskformer-swin-base-coco",
    },
]


if __name__ == "__main__":
    import torch
    from transformers import MaskFormerForInstanceSegmentation

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for cfg in MASKFORMER_CONVERSION_CONFIG:
        variant = cfg["variant"]
        hf_id = cfg["hf_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        keras_model = MaskFormerUniversalSegment.from_weights(
            variant, load_weights=False
        )

        hf_model = MaskFormerForInstanceSegmentation.from_pretrained(
            hf_id, token=os.environ.get("HF_TOKEN")
        ).eval()
        sd = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}
        hf_model = hf_model.to(device)

        transfer_maskformer_weights(keras_model, sd)

        image_size = keras_model.image_size
        rng = np.random.default_rng(42)
        pix_hwc = rng.standard_normal((1, image_size[0], image_size[1], 3)).astype(
            np.float32
        )

        with torch.no_grad():
            hf_pixel_values = torch.from_numpy(np.transpose(pix_hwc, (0, 3, 1, 2))).to(
                device
            )
            hf_out = hf_model(pixel_values=hf_pixel_values)
            hf_class_logits = hf_out.class_queries_logits.cpu().numpy()
            hf_mask_logits = hf_out.masks_queries_logits.cpu().numpy()

        del hf_model, hf_out, sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        keras_input = keras.ops.convert_to_tensor(pix_hwc, dtype="float32")
        keras_out = keras_model(keras_input)
        keras_class = keras.ops.convert_to_numpy(keras_out["class_queries_logits"])
        keras_mask = keras.ops.convert_to_numpy(keras_out["masks_queries_logits"])

        class_diff = float(np.max(np.abs(hf_class_logits - keras_class)))
        mask_diff = float(np.max(np.abs(hf_mask_logits - keras_mask)))
        print(f"  Max class logits diff: {class_diff:.6f}")
        print(f"  Max mask logits diff:  {mask_diff:.6f}")

        if class_diff > 5e-3 or mask_diff > 1e-2:
            raise ValueError(
                f"{variant}: class {class_diff:.6f}, mask {mask_diff:.6f} exceed tolerance"
            )

        out_filename = f"{variant}.weights.h5"
        keras_model.save_weights(out_filename)
        print(f"  Saved -> {out_filename}")

        del keras_model
        keras.backend.clear_session()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
