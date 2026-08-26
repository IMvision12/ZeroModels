import os

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.weight_transfer_util import (
    transfer_nested_layer_weights,
    transfer_weights,
)
from zeromodels.models.sam3 import SAM3Model

vit_name_mapping = {
    "mlp_fc1": "mlp.fc1",
    "mlp_fc2": "mlp.fc2",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}


DETECTOR_PREFIX = "detector_model"
PROBE_KEY = "vision_encoder.backbone.embeddings.patch_embeddings.projection.weight"


def transfer_sam3_weights(sam3_model, hf, prefix=""):
    if not prefix and f"{DETECTOR_PREFIX}.{PROBE_KEY}" in hf:
        prefix = DETECTOR_PREFIX
    p = f"{prefix}." if prefix else ""
    det = sam3_model

    num_clip_layers = 24
    num_geom_layers = len(sam3_model.geometry_encoder.transformer_layers)
    total_layers = (
        det.vit_num_hidden_layers
        + det.detr_encoder_num_layers
        + det.detr_decoder_num_layers
        + num_clip_layers
        + num_geom_layers
    )
    pbar = tqdm(total=total_layers, desc="Converting SAM3", unit="layer")

    patch_conv = det.get_layer("backbone_patch_embed")
    transfer_weights(
        "conv_kernel",
        patch_conv.kernel,
        hf[f"{p}vision_encoder.backbone.embeddings.patch_embeddings.projection.weight"],
    )
    pos_embed = det.get_layer("backbone_position_embedding")
    pos_embed.embeddings.assign(
        hf[f"{p}vision_encoder.backbone.embeddings.position_embeddings"].squeeze(0)
    )

    for i in range(det.vit_num_hidden_layers):
        layer = det.get_layer(f"backbone_layers_{i}")
        skipped = transfer_nested_layer_weights(
            layer,
            hf,
            f"{p}vision_encoder.backbone.layers.{i}",
            name_mapping=vit_name_mapping,
        )
        if skipped:
            for w, path in skipped:
                pbar.write(f"  WARNING: Skipped {path}")
        pbar.update(1)

    bb_ln = det.get_layer("backbone_layer_norm")
    bb_ln.gamma.assign(hf[f"{p}vision_encoder.backbone.layer_norm.weight"])
    bb_ln.beta.assign(hf[f"{p}vision_encoder.backbone.layer_norm.bias"])

    for idx, sf in enumerate(det.fpn_scale_factors):
        fp = f"{p}vision_encoder.neck.fpn_layers.{idx}"
        if sf == 4.0:
            conv = det.get_layer(f"fpn_level_{idx}_deconv1")
            transfer_weights(
                "conv_kernel", conv.kernel, hf[f"{fp}.scale_layers.0.weight"]
            )
            conv.bias.assign(hf[f"{fp}.scale_layers.0.bias"])
            conv = det.get_layer(f"fpn_level_{idx}_deconv2")
            transfer_weights(
                "conv_kernel", conv.kernel, hf[f"{fp}.scale_layers.2.weight"]
            )
            conv.bias.assign(hf[f"{fp}.scale_layers.2.bias"])
        elif sf == 2.0:
            conv = det.get_layer(f"fpn_level_{idx}_deconv1")
            transfer_weights(
                "conv_kernel", conv.kernel, hf[f"{fp}.scale_layers.0.weight"]
            )
            conv.bias.assign(hf[f"{fp}.scale_layers.0.bias"])
        for pn in ["proj1", "proj2"]:
            if f"{fp}.{pn}.weight" in hf:
                conv = det.get_layer(f"fpn_level_{idx}_{pn}")
                transfer_weights("conv_kernel", conv.kernel, hf[f"{fp}.{pn}.weight"])
                conv.bias.assign(hf[f"{fp}.{pn}.bias"])

    tp = det.get_layer("text_projection")
    transfer_weights("kernel", tp.kernel, hf[f"{p}text_projection.weight"])
    tp.bias.assign(hf[f"{p}text_projection.bias"])

    for i in range(det.detr_encoder_num_layers):
        ep = f"{p}detr_encoder.layers.{i}"
        kp = f"detr_encoder_layers_{i}"
        for an in ["self_attn", "cross_attn"]:
            attn = det.get_layer(f"{kp}_{an}")
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                dense = getattr(attn, proj)
                transfer_weights("kernel", dense.kernel, hf[f"{ep}.{an}.{proj}.weight"])
                dense.bias.assign(hf[f"{ep}.{an}.{proj}.bias"])
        for ln in ["layer_norm1", "layer_norm2", "layer_norm3"]:
            layer = det.get_layer(f"{kp}_{ln}")
            layer.gamma.assign(hf[f"{ep}.{ln}.weight"])
            layer.beta.assign(hf[f"{ep}.{ln}.bias"])
        fc1 = det.get_layer(f"{kp}_fc1")
        transfer_weights("kernel", fc1.kernel, hf[f"{ep}.mlp.fc1.weight"])
        fc1.bias.assign(hf[f"{ep}.mlp.fc1.bias"])
        fc2 = det.get_layer(f"{kp}_fc2")
        transfer_weights("kernel", fc2.kernel, hf[f"{ep}.mlp.fc2.weight"])
        fc2.bias.assign(hf[f"{ep}.mlp.fc2.bias"])
        pbar.update(1)

    det.get_layer("detr_decoder_query_embed").embeddings.assign(
        hf[f"{p}detr_decoder.query_embed.weight"]
    )
    det.get_layer("detr_decoder_reference_points").embeddings.assign(
        hf[f"{p}detr_decoder.reference_points.weight"]
    )
    det.get_layer("detr_decoder_presence_token").embeddings.assign(
        hf[f"{p}detr_decoder.presence_token.weight"]
    )
    for head in ["box_head", "presence_head", "ref_point_head"]:
        h = det.get_layer(f"detr_decoder_{head}")
        for j, d in enumerate(h.dense_layers):
            transfer_weights(
                "kernel", d.kernel, hf[f"{p}detr_decoder.{head}.layer{j + 1}.weight"]
            )
            d.bias.assign(hf[f"{p}detr_decoder.{head}.layer{j + 1}.bias"])

    dec_ln = {
        "layer_norm1": "self_attn_layer_norm",
        "layer_norm2": "text_cross_attn_layer_norm",
        "layer_norm3": "vision_cross_attn_layer_norm",
        "layer_norm4": "mlp_layer_norm",
    }
    for i in range(det.detr_decoder_num_layers):
        dp = f"{p}detr_decoder.layers.{i}"
        kp = f"detr_decoder_layers_{i}"
        for an in ["self_attn", "text_cross_attn", "vision_cross_attn"]:
            attn = det.get_layer(f"{kp}_{an}")
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                dense = getattr(attn, proj)
                transfer_weights("kernel", dense.kernel, hf[f"{dp}.{an}.{proj}.weight"])
                dense.bias.assign(hf[f"{dp}.{an}.{proj}.bias"])
        for kln, hln in dec_ln.items():
            layer = det.get_layer(f"{kp}_{kln}")
            layer.gamma.assign(hf[f"{dp}.{hln}.weight"])
            layer.beta.assign(hf[f"{dp}.{hln}.bias"])
        fc1 = det.get_layer(f"{kp}_fc1")
        transfer_weights("kernel", fc1.kernel, hf[f"{dp}.mlp.fc1.weight"])
        fc1.bias.assign(hf[f"{dp}.mlp.fc1.bias"])
        fc2 = det.get_layer(f"{kp}_fc2")
        transfer_weights("kernel", fc2.kernel, hf[f"{dp}.mlp.fc2.weight"])
        fc2.bias.assign(hf[f"{dp}.mlp.fc2.bias"])
        pbar.update(1)

    out_ln = det.get_layer("detr_decoder_output_layer_norm")
    out_ln.gamma.assign(hf[f"{p}detr_decoder.output_layer_norm.weight"])
    out_ln.beta.assign(hf[f"{p}detr_decoder.output_layer_norm.bias"])
    pres_ln = det.get_layer("detr_decoder_presence_layer_norm")
    pres_ln.gamma.assign(hf[f"{p}detr_decoder.presence_layer_norm.weight"])
    pres_ln.beta.assign(hf[f"{p}detr_decoder.presence_layer_norm.bias"])

    rpb = det.get_layer("detr_decoder_box_rpb")
    for ax in ["x", "y"]:
        mlp = getattr(rpb, f"box_rpb_embed_{ax}")
        for j, d in enumerate(mlp.dense_layers):
            transfer_weights(
                "kernel",
                d.kernel,
                hf[f"{p}detr_decoder.box_rpb_embed_{ax}.layer{j + 1}.weight"],
            )
            d.bias.assign(hf[f"{p}detr_decoder.box_rpb_embed_{ax}.layer{j + 1}.bias"])

    sp = f"{p}dot_product_scoring"
    for n, hn in [
        ("text_mlp_fc1", "text_mlp.layer1"),
        ("text_mlp_fc2", "text_mlp.layer2"),
    ]:
        dense = det.get_layer(f"dot_product_scoring_{n}")
        transfer_weights("kernel", dense.kernel, hf[f"{sp}.{hn}.weight"])
        dense.bias.assign(hf[f"{sp}.{hn}.bias"])

    norm = det.get_layer("dot_product_scoring_text_mlp_out_norm")
    norm.gamma.assign(hf[f"{sp}.text_mlp_out_norm.weight"])
    norm.beta.assign(hf[f"{sp}.text_mlp_out_norm.bias"])

    for n in ["text_proj", "query_proj"]:
        dense = det.get_layer(f"dot_product_scoring_{n}")
        transfer_weights("kernel", dense.kernel, hf[f"{sp}.{n}.weight"])
        dense.bias.assign(hf[f"{sp}.{n}.bias"])

    for s in range(len(det.fpn_scale_factors) - 2):
        conv = det.get_layer(f"pixel_decoder_stage_{s}_conv")
        transfer_weights(
            "conv_kernel",
            conv.kernel,
            hf[f"{p}mask_decoder.pixel_decoder.conv_layers.{s}.weight"],
        )
        conv.bias.assign(hf[f"{p}mask_decoder.pixel_decoder.conv_layers.{s}.bias"])
        gn = det.get_layer(f"pixel_decoder_stage_{s}_gn")
        gn.gamma.assign(hf[f"{p}mask_decoder.pixel_decoder.norms.{s}.weight"])
        gn.beta.assign(hf[f"{p}mask_decoder.pixel_decoder.norms.{s}.bias"])

    conv = det.get_layer("mask_decoder_instance_proj")
    transfer_weights(
        "conv_kernel", conv.kernel, hf[f"{p}mask_decoder.instance_projection.weight"]
    )
    conv.bias.assign(hf[f"{p}mask_decoder.instance_projection.bias"])
    conv = det.get_layer("mask_decoder_semantic_proj")
    transfer_weights(
        "conv_kernel", conv.kernel, hf[f"{p}mask_decoder.semantic_projection.weight"]
    )
    conv.bias.assign(hf[f"{p}mask_decoder.semantic_projection.bias"])

    for j in range(3):
        dense = det.get_layer(f"mask_embedder_linear{j + 1}")
        transfer_weights(
            "kernel",
            dense.kernel,
            hf[f"{p}mask_decoder.mask_embedder.layers.{j}.weight"],
        )
        dense.bias.assign(hf[f"{p}mask_decoder.mask_embedder.layers.{j}.bias"])

    pca = det.get_layer("mask_decoder_prompt_cross_attn")
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        dense = getattr(pca, proj)
        transfer_weights(
            "kernel",
            dense.kernel,
            hf[f"{p}mask_decoder.prompt_cross_attn.{proj}.weight"],
        )
        dense.bias.assign(hf[f"{p}mask_decoder.prompt_cross_attn.{proj}.bias"])

    pca_norm = det.get_layer("mask_decoder_prompt_cross_attn_norm")
    pca_norm.gamma.assign(hf[f"{p}mask_decoder.prompt_cross_attn_norm.weight"])
    pca_norm.beta.assign(hf[f"{p}mask_decoder.prompt_cross_attn_norm.bias"])

    te = sam3_model.text_encoder
    te.get_layer("token_embedding").weights[0].assign(
        hf[f"{p}text_encoder.text_model.embeddings.token_embedding.weight"]
    )
    te.get_layer("add_position").position_embedding.weights[0].assign(
        hf[f"{p}text_encoder.text_model.embeddings.position_embedding.weight"]
    )
    for i in range(num_clip_layers):
        hp = f"{p}text_encoder.text_model.encoder.layers.{i}"
        kp = f"layers_{i}"
        ln1 = te.get_layer(f"{kp}_layer_norm1")
        ln1.gamma.assign(hf[f"{hp}.layer_norm1.weight"])
        ln1.beta.assign(hf[f"{hp}.layer_norm1.bias"])
        ln2 = te.get_layer(f"{kp}_layer_norm2")
        ln2.gamma.assign(hf[f"{hp}.layer_norm2.weight"])
        ln2.beta.assign(hf[f"{hp}.layer_norm2.bias"])
        attn = te.get_layer(f"{kp}_self_attn")
        for proj in ["q_proj", "k_proj", "v_proj"]:
            dense = getattr(attn, proj)
            transfer_weights(
                "kernel", dense.kernel, hf[f"{hp}.self_attn.{proj}.weight"]
            )
            dense.bias.assign(hf[f"{hp}.self_attn.{proj}.bias"])
        transfer_weights(
            "kernel", attn.o_proj.kernel, hf[f"{hp}.self_attn.out_proj.weight"]
        )
        attn.o_proj.bias.assign(hf[f"{hp}.self_attn.out_proj.bias"])
        fc1 = te.get_layer(f"{kp}_fc1")
        transfer_weights("kernel", fc1.kernel, hf[f"{hp}.mlp.fc1.weight"])
        fc1.bias.assign(hf[f"{hp}.mlp.fc1.bias"])
        fc2 = te.get_layer(f"{kp}_fc2")
        transfer_weights("kernel", fc2.kernel, hf[f"{hp}.mlp.fc2.weight"])
        fc2.bias.assign(hf[f"{hp}.mlp.fc2.bias"])
        pbar.update(1)

    fln = te.get_layer("final_layer_norm")
    fln.gamma.assign(hf[f"{p}text_encoder.text_model.final_layer_norm.weight"])
    fln.beta.assign(hf[f"{p}text_encoder.text_model.final_layer_norm.bias"])

    geo = sam3_model.geometry_encoder
    transfer_weights(
        "kernel",
        geo.boxes_direct_project.kernel,
        hf[f"{p}geometry_encoder.boxes_direct_project.weight"],
    )
    geo.boxes_direct_project.bias.assign(
        hf[f"{p}geometry_encoder.boxes_direct_project.bias"]
    )
    transfer_weights(
        "kernel",
        geo.boxes_pos_enc_project.kernel,
        hf[f"{p}geometry_encoder.boxes_pos_enc_project.weight"],
    )
    geo.boxes_pos_enc_project.bias.assign(
        hf[f"{p}geometry_encoder.boxes_pos_enc_project.bias"]
    )
    transfer_weights(
        "conv_kernel",
        geo.boxes_pool_project.kernel,
        hf[f"{p}geometry_encoder.boxes_pool_project.weight"],
    )
    geo.boxes_pool_project.bias.assign(
        hf[f"{p}geometry_encoder.boxes_pool_project.bias"]
    )
    geo.label_embed.weights[0].assign(hf[f"{p}geometry_encoder.label_embed.weight"])
    geo.cls_embed.weights[0].assign(hf[f"{p}geometry_encoder.cls_embed.weight"])
    geo.vision_layer_norm.gamma.assign(
        hf[f"{p}geometry_encoder.vision_layer_norm.weight"]
    )
    geo.vision_layer_norm.beta.assign(hf[f"{p}geometry_encoder.vision_layer_norm.bias"])
    geo.prompt_layer_norm.gamma.assign(
        hf[f"{p}geometry_encoder.prompt_layer_norm.weight"]
    )
    geo.prompt_layer_norm.beta.assign(hf[f"{p}geometry_encoder.prompt_layer_norm.bias"])
    geo.output_layer_norm.gamma.assign(
        hf[f"{p}geometry_encoder.output_layer_norm.weight"]
    )
    geo.output_layer_norm.beta.assign(hf[f"{p}geometry_encoder.output_layer_norm.bias"])
    transfer_weights(
        "kernel",
        geo.final_proj.kernel,
        hf[f"{p}geometry_encoder.final_proj.weight"],
    )
    geo.final_proj.bias.assign(hf[f"{p}geometry_encoder.final_proj.bias"])

    for i in range(len(geo.transformer_layers)):
        layer = geo.transformer_layers[i]
        gp = f"{p}geometry_encoder.layers.{i}"
        for an in ["self_attn", "cross_attn"]:
            attn = getattr(layer, an)
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                dense = getattr(attn, proj)
                transfer_weights("kernel", dense.kernel, hf[f"{gp}.{an}.{proj}.weight"])
                dense.bias.assign(hf[f"{gp}.{an}.{proj}.bias"])
        layer.layer_norm1.gamma.assign(hf[f"{gp}.layer_norm1.weight"])
        layer.layer_norm1.beta.assign(hf[f"{gp}.layer_norm1.bias"])
        layer.layer_norm2.gamma.assign(hf[f"{gp}.layer_norm2.weight"])
        layer.layer_norm2.beta.assign(hf[f"{gp}.layer_norm2.bias"])
        layer.layer_norm3.gamma.assign(hf[f"{gp}.layer_norm3.weight"])
        layer.layer_norm3.beta.assign(hf[f"{gp}.layer_norm3.bias"])
        transfer_weights("kernel", layer.fc1.kernel, hf[f"{gp}.mlp.fc1.weight"])
        layer.fc1.bias.assign(hf[f"{gp}.mlp.fc1.bias"])
        transfer_weights("kernel", layer.fc2.kernel, hf[f"{gp}.mlp.fc2.weight"])
        layer.fc2.bias.assign(hf[f"{gp}.mlp.fc2.bias"])
        pbar.update(1)

    pbar.close()


# Per-variant recipe (relocated from sam3_config.py). SAM3Model ignores the
# geometry_* keys (they are legacy); it builds the arch for conversion + the backfill.
SAM3_VARIANTS = {
    "sam3_saco": {
        "vit_hidden_size": 1024,
        "vit_intermediate_size": 4736,
        "vit_num_hidden_layers": 32,
        "vit_num_attention_heads": 16,
        "vit_image_size": 1008,
        "vit_patch_size": 14,
        "vit_window_size": 24,
        "vit_global_attn_indexes": [7, 15, 23, 31],
        "vit_rope_theta": 10000.0,
        "vit_pretrain_image_size": 336,
        "fpn_hidden_size": 256,
        "fpn_scale_factors": [4.0, 2.0, 1.0, 0.5],
        "geometry_hidden_size": 256,
        "geometry_num_layers": 3,
        "geometry_num_attention_heads": 8,
        "geometry_intermediate_size": 2048,
        "geometry_dropout": 0.1,
        "geometry_roi_size": 7,
        "detr_encoder_hidden_size": 256,
        "detr_encoder_num_layers": 6,
        "detr_encoder_num_attention_heads": 8,
        "detr_encoder_intermediate_size": 2048,
        "detr_encoder_dropout": 0.1,
        "detr_decoder_hidden_size": 256,
        "detr_decoder_num_layers": 6,
        "detr_decoder_num_queries": 200,
        "detr_decoder_num_attention_heads": 8,
        "detr_decoder_intermediate_size": 2048,
        "detr_decoder_dropout": 0.1,
        "mask_decoder_hidden_size": 256,
        "mask_decoder_num_upsampling_stages": 3,
        "mask_decoder_num_attention_heads": 8,
        "text_hidden_size": 1024,
        "text_projection_dim": 512,
    },
}


if __name__ == "__main__":
    from transformers import Sam3Model as HFSam3Model

    HF_TOKEN = os.environ.get("HF_TOKEN")
    variant = "sam3_saco"

    print("Loading HF Sam3Model...")
    hf_model = HFSam3Model.from_pretrained(
        "facebook/sam3", attn_implementation="eager", token=HF_TOKEN
    ).eval()
    hf = {k: v.cpu().numpy() for k, v in hf_model.state_dict().items()}
    print(f"HF: {len(hf)} keys")
    del hf_model

    print("\nBuilding Keras SAM3Model...")
    sam3 = SAM3Model(**SAM3_VARIANTS[variant])

    print("\nTransferring detector weights...")
    transfer_sam3_weights(sam3, hf)

    all_weights = (
        sam3.weights + sam3.text_encoder.weights + sam3.geometry_encoder.weights
    )
    total_params = sum(int(np.prod(w.shape)) for w in all_weights)
    total_gb = (total_params * 4) / (1024**3)
    print(f"\n  Total params: {total_params:,}  (~{total_gb:.2f} GB)")
    if total_gb > 1.7:
        out_path = f"{variant}.weights.json"
        sam3.save_weights(out_path, max_shard_size=1.7)
        print(f"  Saved -> {out_path} (sharded, ~{total_gb:.2f} GB)")
    else:
        out_path = f"{variant}.weights.h5"
        sam3.save_weights(out_path)
        print(f"  Saved -> {out_path} (~{total_gb:.2f} GB)")

    print("\nDone!")
