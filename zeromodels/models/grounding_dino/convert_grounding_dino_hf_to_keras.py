import re
from typing import Dict

import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights

SWIN_PREFIX = "model.backbone.conv_encoder.model"
BB_PREFIX = "model.backbone.conv_encoder.model"

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "attention.relative_position_bias_table": (
        "attention.self.relative_position_bias_table"
    ),
    "attention.q_proj": "attention.self.query",
    "attention.k_proj": "attention.self.key",
    "attention.v_proj": "attention.self.value",
    "attention.o_proj": "attention.output.dense",
    "mlp_fc1": "intermediate.dense",
    "mlp_fc2": "output.dense",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}


def keras_to_hf(name):
    """Map a Keras weight path (dots) to its HF Grounding DINO key."""
    if name.startswith("backbone."):
        n = name[len("backbone.") :]
        if n.startswith("patch_embeddings_projection"):
            return f"{SWIN_PREFIX}.embeddings.patch_embeddings.projection." + (
                "weight" if n.endswith("kernel") else "bias"
            )
        if n.startswith("embed_norm"):
            return f"{SWIN_PREFIX}.embeddings.norm." + (
                "weight" if n.endswith("gamma") else "bias"
            )
        m = re.match(r"hidden_states_norms_stage(\d+)\.(gamma|beta)", n)
        if m:
            return f"{BB_PREFIX}.hidden_states_norms.stage{m.group(1)}." + (
                "weight" if m.group(2) == "gamma" else "bias"
            )
        m = re.match(r"stage_(\d+)\.(.*)", n)
        if m:
            stage, rest = m.group(1), m.group(2)
            rest = rest.replace("blocks_", "blocks.")
            for old, new in WEIGHT_NAME_MAPPING.items():
                rest = rest.replace(old, new)
            return f"{SWIN_PREFIX}.encoder.layers.{stage}.{rest}"
    if name.startswith("text_backbone."):
        n = name[len("text_backbone.") :]
        if n.startswith("embeddings."):
            tail = n[len("embeddings.") :]
            if tail.endswith(".embeddings"):
                return f"model.text_backbone.embeddings.{tail[: -len('.embeddings')]}.weight"
            tail = tail.replace("gamma", "weight").replace("beta", "bias")
            return f"model.text_backbone.embeddings.{tail}"
        m = re.match(r"layer_(\d+)\.(.*)", n)
        if m:
            i, rest = m.group(1), m.group(2)
            mapping = {
                "query": "attention.self.query",
                "key": "attention.self.key",
                "value": "attention.self.value",
                "attn_output": "attention.output.dense",
                "attn_norm": "attention.output.LayerNorm",
                "intermediate": "intermediate.dense",
                "output_dense": "output.dense",
                "output_norm": "output.LayerNorm",
            }
            for k, v in mapping.items():
                if rest.startswith(k + "."):
                    rest = v + rest[len(k) :]
                    break
            rest = (
                rest.replace("kernel", "weight")
                .replace("gamma", "weight")
                .replace("beta", "bias")
            )
            return f"model.text_backbone.encoder.layer.{i}.{rest}"
    if name.startswith("text_projection."):
        return "model.text_projection." + (
            "weight" if name.endswith("kernel") else "bias"
        )
    m = re.match(r"input_proj_(\d+)_conv\.(kernel|bias)", name)
    if m:
        return f"model.input_proj_vision.{m.group(1)}.0." + (
            "weight" if m.group(2) == "kernel" else "bias"
        )
    m = re.match(r"input_proj_(\d+)_norm\.(gamma|beta)", name)
    if m:
        return f"model.input_proj_vision.{m.group(1)}.1." + (
            "weight" if m.group(2) == "gamma" else "bias"
        )
    if name == "level_embed":
        return "model.level_embed"
    m = re.match(r"encoder_layer_(\d+)\.(.*)", name)
    if m:
        rest = _common(m.group(2))
        return f"model.encoder.layers.{m.group(1)}.{rest}"
    if name.startswith("enc_output_norm."):
        return "model.enc_output_norm." + (
            "weight" if name.endswith("gamma") else "bias"
        )
    if name.startswith("enc_output."):
        return "model.enc_output." + ("weight" if name.endswith("kernel") else "bias")
    m = re.match(r"encoder_output_bbox_embed\.layers_(\d+)\.(kernel|bias)", name)
    if m:
        return f"model.encoder_output_bbox_embed.layers.{m.group(1)}." + (
            "weight" if m.group(2) == "kernel" else "bias"
        )
    if name.startswith("query_position_embeddings.embeddings"):
        return "model.query_position_embeddings.weight"
    if name.startswith("decoder_norm."):
        return "model.decoder.layer_norm." + (
            "weight" if name.endswith("gamma") else "bias"
        )
    m = re.match(r"reference_points_head\.layers_(\d+)\.(kernel|bias)", name)
    if m:
        return f"model.decoder.reference_points_head.layers.{m.group(1)}." + (
            "weight" if m.group(2) == "kernel" else "bias"
        )
    m = re.match(r"decoder_layer_(\d+)\.(.*)", name)
    if m:
        rest = _common(m.group(2))
        return f"model.decoder.layers.{m.group(1)}.{rest}"
    m = re.match(r"bbox_embed_(\d+)\.layers_(\d+)\.(kernel|bias)", name)
    if m:
        return f"bbox_embed.{m.group(1)}.layers.{m.group(2)}." + (
            "weight" if m.group(3) == "kernel" else "bias"
        )
    raise WeightMappingError(name, name)


def _common(rest):
    """Shared encoder/decoder sublayer name fixups (Dense/LayerNorm/params)."""
    rest = rest.replace("self_attn_layer_norm", "self_attn_layer_norm")
    rest = (
        rest.replace("kernel", "weight")
        .replace("gamma", "weight")
        .replace("beta", "bias")
    )
    return rest


def resolve_key(name, state):
    """First candidate key present in ``state``, or None.

    All six decoder layers share one bbox head (``decoder_bbox_embed_share``),
    and the two released checkpoints disagree on where it lives: tiny stores it
    top-level as ``bbox_embed.0.*``, base under ``model.decoder.bbox_embed.0.*``.
    """
    candidates = [name]
    shared = re.sub(r"bbox_embed\.\d+\.", "bbox_embed.0.", name)
    if shared != name:
        candidates.append(shared)
    for cand in list(candidates):
        if cand.startswith("bbox_embed."):
            candidates.append("model.decoder." + cand)
        elif cand.startswith("model."):
            candidates.append(cand[len("model.") :])
    return next((c for c in candidates if c in state), None)


def transfer_grounding_dino_weights(keras_model, hf_state_dict):
    if not keras_model.built or not keras_model.weights:
        keras_model(
            {
                "input_ids": np.array(
                    [[101, 102, 1012, 1029, 102, 102]], dtype="int64"
                ),
                "attention_mask": np.ones((1, 6), dtype="int64"),
                "pixel_values": np.zeros((1, 384, 384, 3), dtype="float32"),
            }
        )
    state = hf_state_dict
    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        kname = weight.path.split("/", 1)[1].replace("/", ".")
        mapped = keras_to_hf(kname)
        hf = resolve_key(mapped, state)
        if hf is None:
            raise WeightMappingError(weight.path, mapped)
        value = state[hf]
        if kname.endswith("patch_embeddings_projection.kernel") or re.search(
            r"input_proj_\d+_conv\.kernel", kname
        ):
            weight.assign(np.transpose(np.asarray(value), (2, 3, 1, 0)))
        elif (
            kname == "level_embed"
            or kname.endswith("relative_position_bias_table")
            or kname.endswith("fusion_layer.vision_param")
            or kname.endswith("fusion_layer.text_param")
        ):
            weight.assign(np.asarray(value))
        else:
            transfer_weights(weight.path, weight, value)


# Per-variant recipes (relocated from grounding_dino_config.py). Conversion goes
# through the hf: path; these drive the backfill's zm_config.json.
GROUNDING_DINO_VARIANTS = {
    "grounding_dino_tiny": {
        "d_model": 256,
        "encoder_layers": 6,
        "encoder_ffn_dim": 2048,
        "encoder_attention_heads": 8,
        "decoder_layers": 6,
        "decoder_ffn_dim": 2048,
        "decoder_attention_heads": 8,
        "num_queries": 900,
        "num_feature_levels": 4,
        "encoder_n_points": 4,
        "decoder_n_points": 4,
        "max_text_len": 256,
        "query_dim": 4,
        "two_stage": True,
        "positional_embedding_temperature": 20.0,
        "layer_norm_eps": 1e-5,
        "activation_function": "relu",
        "backbone_embed_dim": 96,
        "backbone_depths": (2, 2, 6, 2),
        "backbone_num_heads": (3, 6, 12, 24),
        "backbone_window_size": 7,
        "backbone_out_indices": (2, 3, 4),
        "text_vocab_size": 30522,
        "text_hidden_size": 768,
        "text_num_layers": 12,
        "text_num_heads": 12,
        "text_intermediate_size": 3072,
        "text_max_position_embeddings": 512,
        "text_layer_norm_eps": 1e-12,
    },
    "grounding_dino_base": {
        "d_model": 256,
        "encoder_layers": 6,
        "encoder_ffn_dim": 2048,
        "encoder_attention_heads": 8,
        "decoder_layers": 6,
        "decoder_ffn_dim": 2048,
        "decoder_attention_heads": 8,
        "num_queries": 900,
        "num_feature_levels": 4,
        "encoder_n_points": 4,
        "decoder_n_points": 4,
        "max_text_len": 256,
        "query_dim": 4,
        "two_stage": True,
        "positional_embedding_temperature": 20.0,
        "layer_norm_eps": 1e-5,
        "activation_function": "relu",
        "backbone_embed_dim": 128,
        "backbone_depths": (2, 2, 18, 2),
        "backbone_num_heads": (4, 8, 16, 32),
        "backbone_window_size": 12,
        "backbone_out_indices": (2, 3, 4),
        "text_vocab_size": 30522,
        "text_hidden_size": 768,
        "text_num_layers": 12,
        "text_num_heads": 12,
        "text_intermediate_size": 3072,
        "text_max_position_embeddings": 512,
        "text_layer_norm_eps": 1e-12,
    },
}


if __name__ == "__main__":
    import gc

    import keras
    import torch
    import transformers
    from PIL import Image

    from zeromodels.models.grounding_dino import (
        GroundingDinoDetect,
        GroundingDinoProcessor,
    )

    HF_SOURCES = {
        "grounding_dino_tiny": "IDEA-Research/grounding-dino-tiny",
        "grounding_dino_base": "IDEA-Research/grounding-dino-base",
    }
    MAX_SHARD_GB = 1.7
    rng = np.random.default_rng(0)

    def cosine(a, b):
        a = np.nan_to_num(a.astype("float64"), neginf=-50.0, posinf=50.0).ravel()
        b = np.nan_to_num(b.astype("float64"), neginf=-50.0, posinf=50.0).ravel()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def detections(logits, boxes, threshold=0.3):
        """Confident queries only, sorted by score. The full 900-query tensor is
        top-k ordered, so tiny numeric drift permutes the low-scoring tail and
        makes an elementwise comparison meaningless."""
        lg = np.nan_to_num(logits[0], neginf=-50.0, posinf=50.0)
        scores = (1.0 / (1.0 + np.exp(-np.clip(lg, -50, 50)))).max(axis=-1)
        keep = np.where(scores > threshold)[0]
        order = keep[np.argsort(-scores[keep])]
        return scores[order], boxes[0][order]

    for variant, hf_id in HF_SOURCES.items():
        out_path = variant
        print(f"\n{'=' * 60}\nConverting: {variant}  <-  {hf_id}\n{'=' * 60}")

        # The detection class, not the backbone: the six decoder bbox_embed
        # heads live on it, and a backbone-only export leaves them unsaved.
        model = GroundingDinoDetect.from_weights("hf:" + hf_id)

        img = Image.fromarray(rng.integers(0, 255, (480, 480, 3), dtype="uint8"))
        proc = GroundingDinoProcessor.from_weights("hf:" + hf_id)
        kin = proc(images=img, text="a cat. a remote control.")
        pv = np.transpose(
            np.asarray(keras.ops.convert_to_numpy(kin["pixel_values"])), (0, 3, 1, 2)
        )
        ids = np.asarray(keras.ops.convert_to_numpy(kin["input_ids"])).astype("int64")
        am = np.asarray(keras.ops.convert_to_numpy(kin["attention_mask"])).astype(
            "int64"
        )
        with torch.no_grad():
            k_out = model(kin)
            k_enc = model.encode(kin)
        k_logits = np.asarray(keras.ops.convert_to_numpy(k_out["logits"]))
        k_boxes = np.asarray(keras.ops.convert_to_numpy(k_out["pred_boxes"]))
        k_vis = np.asarray(keras.ops.convert_to_numpy(k_enc["vision"]))
        k_txt = np.asarray(keras.ops.convert_to_numpy(k_enc["text"]))
        del k_out, k_enc
        n_bytes = sum(int(np.prod(w.shape)) * 4 for w in model.weights)
        if out_path.endswith(".json"):
            model.save_weights(out_path, max_shard_size=MAX_SHARD_GB)
        elif n_bytes > 2 * 1024**3:
            raise ValueError(
                f"{variant} is {n_bytes / 1024**3:.2f} GB (> 2 GB); set its config "
                f"URL extension to .weights.json so it shards."
            )
        else:
            model.save_weights(out_path)
        print(f"  Saved -> {out_path}  ({n_bytes / 1024**3:.2f} GB)")
        del model
        keras.backend.clear_session()
        gc.collect()

        hf_model = transformers.GroundingDinoForObjectDetection.from_pretrained(
            hf_id
        ).eval()
        with torch.no_grad():
            hf_out = hf_model(
                pixel_values=torch.from_numpy(pv),
                input_ids=torch.from_numpy(ids),
                attention_mask=torch.from_numpy(am),
                output_hidden_states=True,
            )
        cos_vis = cosine(k_vis, hf_out.encoder_last_hidden_state_vision.numpy())
        cos_txt = cosine(k_txt, hf_out.encoder_last_hidden_state_text.numpy())
        cos_logits = cosine(k_logits, hf_out.logits.numpy())
        print(f"  encoder vision cosine: {cos_vis:.6f}")
        print(f"  encoder text cosine:   {cos_txt:.6f}")
        print(f"  logits cosine:         {cos_logits:.6f}")

        k_scores, k_dets = detections(k_logits, k_boxes)
        h_scores, h_dets = detections(hf_out.logits.numpy(), hf_out.pred_boxes.numpy())
        print(f"  detections: keras {len(k_scores)}, hf {len(h_scores)}")
        if len(k_scores) != len(h_scores):
            raise ValueError(
                f"{variant}: detection count differs "
                f"(keras={len(k_scores)}, hf={len(h_scores)})"
            )
        box_diff = float(np.abs(k_dets - h_dets).max()) if len(k_dets) else 0.0
        score_diff = float(np.abs(k_scores - h_scores).max()) if len(k_scores) else 0.0
        print(f"  detected box max|diff|:   {box_diff:.3e}")
        print(f"  detected score max|diff|: {score_diff:.3e}")
        if min(cos_vis, cos_txt, cos_logits) < 0.99 or box_diff > 5e-3:
            raise ValueError(
                f"{variant}: Grounding DINO parity failed (vision={cos_vis:.4f}, "
                f"text={cos_txt:.4f}, logits={cos_logits:.4f}, boxes={box_diff:.2e})"
            )
        del hf_model
        gc.collect()
