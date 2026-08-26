import gc
from typing import Dict

import keras
import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from zeromodels.conversion.weight_split_util import split_model_weights
from zeromodels.conversion.weight_transfer_util import (
    compare_keras_torch_names,
    transfer_attention_weights,
    transfer_weights,
)
from zeromodels.models.clip import CLIPZeroShotClassify

WEIGHT_NAME_MAPPING = {
    "_": ".",
    "vision.model": "vision_model",
    "text.model": "text_model",
    "conv": "embeddings.patch_embedding",
    "class.embedding": "class_embedding",
    "pos.embed": "position_embedding.weight",
    "vision_model.layernorm.1": "vision_model.pre_layrnorm",
    "text_model.encoder": "text_model.encoder.layers",
    "vision_model.encoder": "vision_model.encoder.layers",
    "text_model.layernorm": "text_model.final_layer_norm",
    "layernorm.1": "layer_norm1",
    "layernorm.2": "layer_norm2",
    "vision_model.layer_norm2": "vision_model.post_layernorm",
    "text.projection": "text_projection",
    "visual.projection": "visual_projection",
    "logit_scale_logit_scale": "logit_scale",
    "dense.1": "mlp.fc1",
    "dense.2": "mlp.fc2",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
}

ATTN_NAME_REPLACE = {
    "text.model": "text_model",
    "vision.model": "vision_model",
    "encoder": "encoder.layers",
    "attn": "self_attn",
    "q.proj": "q_proj",
    "k.proj": "k_proj",
    "v.proj": "v_proj",
    "out.proj": "out_proj",
}


def strip_model_prefix(state_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if not any(k.startswith("model.") for k in state_dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            out[k[len("model.") :]] = v
        else:
            out[k] = v
    return out


def transfer_clip_weights(keras_model, hf_state_dict: Dict[str, np.ndarray]) -> None:
    state = strip_model_prefix(hf_state_dict)
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_weight_name: str = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if "attention" in torch_weight_name:
            transfer_attention_weights(
                keras_weight_name, keras_weight, state, ATTN_NAME_REPLACE
            )
            continue

        if keras_weight_name == "text_model_embedding_embeddings":
            if "token_embedding" in keras_weight.path:
                keras_weight.assign(
                    state["text_model.embeddings.token_embedding.weight"]
                )
                continue
            if "positional_embedding" in keras_weight.path:
                keras_weight.assign(
                    state["text_model.embeddings.position_embedding.weight"]
                )
                continue

        if keras_weight_name == "logit_scale_logit_scale":
            keras_weight.assign(state["logit_scale"])
            continue

        if keras_weight_name == "vision_model_embeddings_pos_embed":
            pos = state["vision_model.embeddings.position_embedding.weight"]
            keras_weight.assign(np.expand_dims(pos, 0))
            continue

        if torch_weight_name not in state:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = state[torch_weight_name]
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


def transfer_clip_image_classify_weights(
    keras_model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    state = strip_model_prefix(hf_state_dict)
    has_classifier = "classifier.weight" in state and "classifier.bias" in state
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        if keras_weight_name in ("classifier_kernel", "classifier_bias"):
            if not has_classifier:
                continue
            if "kernel" in keras_weight.path:
                keras_weight.assign(np.transpose(state["classifier.weight"]))
            else:
                keras_weight.assign(state["classifier.bias"])
            continue

        torch_weight_name: str = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if "attention" in torch_weight_name:
            transfer_attention_weights(
                keras_weight_name, keras_weight, state, ATTN_NAME_REPLACE
            )
            continue

        if keras_weight_name == "vision_model_embeddings_pos_embed":
            pos = state["vision_model.embeddings.position_embedding.weight"]
            keras_weight.assign(np.expand_dims(pos, 0))
            continue

        if torch_weight_name not in state:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = state[torch_weight_name]
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


# Per-variant recipes (relocated from clip_config.py). The zeromodels repos hold
# a CLIPZeroShotClassify checkpoint (the superset with logit_scale); its
# zm_config.json declares CLIPZeroShotClassify and every CLIP head loads from it.
CLIP_RECIPES = {
    "clip_vit_base_16": {
        "embed_dim": 512,
        "image_size": 224,
        "vision_num_layers": 12,
        "vision_hidden_dim": 768,
        "vision_patch_size": 16,
        "max_seq_len": 77,
        "vocab_size": 49408,
        "text_hidden_dim": 512,
        "text_num_heads": 8,
        "text_num_layers": 12,
        "vision_mlp_ratio": 4.0,
        "text_mlp_ratio": 4.0,
        "hidden_act": "quick_gelu",
    },
    "clip_vit_base_32": {
        "embed_dim": 512,
        "image_size": 224,
        "vision_num_layers": 12,
        "vision_hidden_dim": 768,
        "vision_patch_size": 32,
        "max_seq_len": 77,
        "vocab_size": 49408,
        "text_hidden_dim": 512,
        "text_num_heads": 8,
        "text_num_layers": 12,
        "vision_mlp_ratio": 4.0,
        "text_mlp_ratio": 4.0,
        "hidden_act": "quick_gelu",
    },
    "clip_vit_large_14": {
        "embed_dim": 768,
        "image_size": 224,
        "vision_num_layers": 24,
        "vision_hidden_dim": 1024,
        "vision_patch_size": 14,
        "max_seq_len": 77,
        "vocab_size": 49408,
        "text_hidden_dim": 768,
        "text_num_heads": 12,
        "text_num_layers": 12,
        "vision_mlp_ratio": 4.0,
        "text_mlp_ratio": 4.0,
        "hidden_act": "quick_gelu",
    },
    "clip_vit_large_14_336": {
        "embed_dim": 768,
        "image_size": 336,
        "vision_num_layers": 24,
        "vision_hidden_dim": 1024,
        "vision_patch_size": 14,
        "max_seq_len": 77,
        "vocab_size": 49408,
        "text_hidden_dim": 768,
        "text_num_heads": 12,
        "text_num_layers": 12,
        "vision_mlp_ratio": 4.0,
        "text_mlp_ratio": 4.0,
        "hidden_act": "quick_gelu",
    },
    "clip_vit_g_14": {
        "embed_dim": 1024,
        "image_size": 224,
        "vision_num_layers": 40,
        "vision_hidden_dim": 1408,
        "vision_patch_size": 14,
        "max_seq_len": 77,
        "vocab_size": 49408,
        "text_hidden_dim": 1024,
        "text_num_heads": 16,
        "text_num_layers": 24,
        "vision_mlp_ratio": 6144 / 1408,
        "text_mlp_ratio": 4096 / 1024,
        "hidden_act": "gelu",
    },
    "clip_vit_bigg_14": {
        "embed_dim": 1280,
        "image_size": 224,
        "vision_num_layers": 48,
        "vision_hidden_dim": 1664,
        "vision_patch_size": 14,
        "max_seq_len": 77,
        "vocab_size": 49408,
        "text_hidden_dim": 1280,
        "text_num_heads": 20,
        "text_num_layers": 32,
        "vision_mlp_ratio": 8192 / 1664,
        "text_mlp_ratio": 5120 / 1280,
        "hidden_act": "gelu",
    },
}


if __name__ == "__main__":
    from transformers import AutoModel

    CLIP_CONVERSION_CONFIG = [
        ("clip_vit_base_16", "openai/clip-vit-base-patch16"),
        ("clip_vit_base_32", "openai/clip-vit-base-patch32"),
        ("clip_vit_large_14", "openai/clip-vit-large-patch14"),
        ("clip_vit_large_14_336", "openai/clip-vit-large-patch14-336"),
        ("clip_vit_g_14", "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"),
        ("clip_vit_bigg_14", "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"),
    ]

    for variant, hf_id in CLIP_CONVERSION_CONFIG:
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        hf_model = AutoModel.from_pretrained(hf_id).eval()
        state = {k: v.detach().cpu().numpy() for k, v in hf_model.state_dict().items()}

        keras_model = CLIPZeroShotClassify(**CLIP_RECIPES[variant])
        transfer_clip_weights(keras_model, state)

        total_params = sum(int(np.prod(w.shape)) for w in keras_model.weights)
        total_gb = (total_params * 4) / (1024**3)

        del state
        gc.collect()

        if total_gb <= 2.0:
            import torch

            ctx = keras_model.max_seq_len
            vocab = keras_model.vocab_size
            ishape = keras_model.image_size
            if keras.config.image_data_format() == "channels_first":
                img_h, img_w = ishape[1], ishape[2]
            else:
                img_h, img_w = ishape[0], ishape[1]

            rng = np.random.default_rng(0)
            pixel = rng.standard_normal((2, img_h, img_w, 3)).astype(np.float32)
            token_ids = rng.integers(0, vocab - 1, size=(2, ctx)).astype(np.int32)
            token_ids[:, -1] = vocab - 1  # EOT id (max) -> the pooled position
            attn = np.ones((2, ctx), dtype=np.int32)

            with torch.no_grad():
                hf_out = hf_model(
                    pixel_values=torch.from_numpy(pixel.transpose(0, 3, 1, 2)),
                    input_ids=torch.from_numpy(token_ids.astype(np.int64)),
                    attention_mask=torch.from_numpy(attn.astype(np.int64)),
                )
                hf_logits = hf_out.logits_per_image.cpu().numpy()
                scale = float(hf_model.logit_scale.exp().cpu().numpy())

            k_out = keras_model(
                {"images": pixel, "token_ids": token_ids, "padding_mask": attn},
                training=False,
            )
            k_logits = keras.ops.convert_to_numpy(k_out["image_logits"])

            logits_diff = float(np.abs(hf_logits - k_logits).max())
            cosine_diff = logits_diff / scale
            print(
                f"  Max logits diff: {logits_diff:.6f}  "
                f"(cosine-level: {cosine_diff:.2e})"
            )
            if cosine_diff > 1e-2:
                raise ValueError(
                    f"{variant}: equivalence check failed "
                    f"(logits diff {logits_diff:.4f}, cosine {cosine_diff:.2e})"
                )
        else:
            print(
                f"  Equivalence check skipped (~{total_gb:.1f} GB model exceeds "
                f"RAM budget; weights validated by name-based mapping)"
            )

        if total_gb > 1.7:
            out_path = f"{variant}.weights.json"
            keras_model.save_weights(out_path, max_shard_size=1.7)
            print(f"  Saved -> {out_path} (sharded, ~{total_gb:.2f} GB)")
        else:
            out_path = f"{variant}.weights.h5"
            keras_model.save_weights(out_path)
            print(f"  Saved -> {out_path} (~{total_gb:.2f} GB)")

        del keras_model, hf_model
        keras.backend.clear_session()
        gc.collect()
