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
from zeromodels.models.siglip import SigLIPZeroShotClassify

WEIGHT_NAME_MAPPING = {
    "_": ".",
    "vision.model": "vision_model",
    "text.model": "text_model",
    "patch.embedding.conv": "patch_embedding",
    "position.embedding.embeddings": "position_embedding.weight",
    "token.embedding.embeddings": "token_embedding.weight",
    "text_model.post_layernorm": "text_model.final_layer_norm",
    "layernorm.1": "layer_norm1",
    "layernorm.2": "layer_norm2",
    "dense.1": "mlp.fc1",
    "dense.2": "mlp.fc2",
    "vision_model.final.layernorm": "vision_model.post_layernorm",
    "text_model.final.layernorm": "text_model.final_layer_norm",
    "probe.probe": "probe",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
}

ATTN_NAME_REPLACE = {
    "_": ".",
    "self.attn": "self_attn",
    "vision.model": "vision_model",
    "text.model": "text_model",
    "in.proj": "in_proj",
    "out.proj": "out_proj",
    "q.proj": "q_proj",
    "k.proj": "k_proj",
    "v.proj": "v_proj",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "bias": "bias",
}


def transfer_siglip_weights(keras_model, hf_state_dict: Dict[str, np.ndarray]) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if "attention" in torch_weight_name:
            if "in_proj" in keras_weight.path:
                if "kernel" in keras_weight.path:
                    keras_weight.assign(
                        np.transpose(
                            hf_state_dict["vision_model.head.attention.in_proj_weight"]
                        )
                    )
                else:
                    keras_weight.assign(
                        hf_state_dict["vision_model.head.attention.in_proj_bias"]
                    )
                continue
            transfer_attention_weights(
                keras_weight_name, keras_weight, hf_state_dict, ATTN_NAME_REPLACE
            )
            continue

        if "probe" in torch_weight_name:
            keras_weight.assign(hf_state_dict["vision_model.head.probe"])
            continue

        if "logit" in torch_weight_name:
            if torch_weight_name.split(".")[-1] == "scale":
                keras_weight.assign(hf_state_dict["logit_scale"].reshape(()))
            else:
                keras_weight.assign(hf_state_dict["logit_bias"].reshape(()))
            continue

        if "position.ids" in torch_weight_name:
            if "vision_model" in torch_weight_name:
                key = "vision_model.embeddings.position_ids"
            else:
                key = "text_model.embeddings.position_ids"
            if key in hf_state_dict:
                keras_weight.assign(hf_state_dict[key])
            continue

        if "head.attention" in torch_weight_name:
            continue

        if torch_weight_name not in hf_state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = hf_state_dict[torch_weight_name]
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


def transfer_siglip_image_classify_weights(
    keras_model, hf_state_dict: Dict[str, np.ndarray]
) -> None:
    has_classifier = (
        "classifier.weight" in hf_state_dict and "classifier.bias" in hf_state_dict
    )
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        if keras_weight_name in ("classifier_kernel", "classifier_bias"):
            if not has_classifier:
                continue
            if "kernel" in keras_weight.path:
                keras_weight.assign(np.transpose(hf_state_dict["classifier.weight"]))
            else:
                keras_weight.assign(hf_state_dict["classifier.bias"])
            continue

        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if "attention" in torch_weight_name:
            transfer_attention_weights(
                keras_weight_name, keras_weight, hf_state_dict, ATTN_NAME_REPLACE
            )
            continue

        if "position.ids" in torch_weight_name:
            key = "vision_model.embeddings.position_ids"
            if key in hf_state_dict:
                keras_weight.assign(hf_state_dict[key])
            continue

        if torch_weight_name not in hf_state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = hf_state_dict[torch_weight_name]
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


# Per-variant recipes (relocated from siglip_config.py). The zeromodels repos
# hold a SigLIPZeroShotClassify checkpoint; its kf_config.json declares
# SigLIPZeroShotClassify and every SigLIP head loads from it.
SIGLIP_RECIPES = {
    "siglip_base_p16_224": {
        "image_size": 224,
        "patch_size": 16,
        "vision_hidden_dim": 768,
        "vision_num_layers": 12,
        "vision_num_heads": 12,
        "vision_mlp_dim": 3072,
        "vocab_size": 32000,
        "embed_dim": 768,
        "text_hidden_dim": 768,
        "text_num_layers": 12,
        "text_num_heads": 12,
        "text_mlp_dim": 3072,
        "max_seq_len": 64,
    },
    "siglip_base_p16_256": {
        "image_size": 256,
        "patch_size": 16,
        "vision_hidden_dim": 768,
        "vision_num_layers": 12,
        "vision_num_heads": 12,
        "vision_mlp_dim": 3072,
        "vocab_size": 32000,
        "embed_dim": 768,
        "text_hidden_dim": 768,
        "text_num_layers": 12,
        "text_num_heads": 12,
        "text_mlp_dim": 3072,
        "max_seq_len": 64,
    },
    "siglip_base_p16_multilingual_256": {
        "image_size": 256,
        "patch_size": 16,
        "vision_hidden_dim": 768,
        "vision_num_layers": 12,
        "vision_num_heads": 12,
        "vision_mlp_dim": 3072,
        "vocab_size": 250000,
        "embed_dim": 768,
        "text_hidden_dim": 768,
        "text_num_layers": 12,
        "text_num_heads": 12,
        "text_mlp_dim": 3072,
        "max_seq_len": 64,
    },
    "siglip_base_p16_384": {
        "image_size": 384,
        "patch_size": 16,
        "vision_hidden_dim": 768,
        "vision_num_layers": 12,
        "vision_num_heads": 12,
        "vision_mlp_dim": 3072,
        "vocab_size": 32000,
        "embed_dim": 768,
        "text_hidden_dim": 768,
        "text_num_layers": 12,
        "text_num_heads": 12,
        "text_mlp_dim": 3072,
        "max_seq_len": 64,
    },
    "siglip_base_p16_512": {
        "image_size": 512,
        "patch_size": 16,
        "vision_hidden_dim": 768,
        "vision_num_layers": 12,
        "vision_num_heads": 12,
        "vision_mlp_dim": 3072,
        "vocab_size": 32000,
        "embed_dim": 768,
        "text_hidden_dim": 768,
        "text_num_layers": 12,
        "text_num_heads": 12,
        "text_mlp_dim": 3072,
        "max_seq_len": 64,
    },
    "siglip_large_p16_256": {
        "image_size": 256,
        "patch_size": 16,
        "vision_hidden_dim": 1024,
        "vision_num_layers": 24,
        "vision_num_heads": 16,
        "vision_mlp_dim": 4096,
        "vocab_size": 32000,
        "embed_dim": 1024,
        "text_hidden_dim": 1024,
        "text_num_layers": 24,
        "text_num_heads": 16,
        "text_mlp_dim": 4096,
        "max_seq_len": 64,
    },
    "siglip_large_p16_384": {
        "image_size": 384,
        "patch_size": 16,
        "vision_hidden_dim": 1024,
        "vision_num_layers": 24,
        "vision_num_heads": 16,
        "vision_mlp_dim": 4096,
        "vocab_size": 32000,
        "embed_dim": 1024,
        "text_hidden_dim": 1024,
        "text_num_layers": 24,
        "text_num_heads": 16,
        "text_mlp_dim": 4096,
        "max_seq_len": 64,
    },
    "siglip_so400m_p14_224": {
        "image_size": 224,
        "patch_size": 14,
        "vision_hidden_dim": 1152,
        "vision_num_layers": 27,
        "vision_num_heads": 16,
        "vision_mlp_dim": 4304,
        "vocab_size": 32000,
        "embed_dim": 1152,
        "text_hidden_dim": 1152,
        "text_num_layers": 27,
        "text_num_heads": 16,
        "text_mlp_dim": 4304,
        "max_seq_len": 16,
    },
    "siglip_so400m_p14_384": {
        "image_size": 384,
        "patch_size": 14,
        "vision_hidden_dim": 1152,
        "vision_num_layers": 27,
        "vision_num_heads": 16,
        "vision_mlp_dim": 4304,
        "vocab_size": 32000,
        "embed_dim": 1152,
        "text_hidden_dim": 1152,
        "text_num_layers": 27,
        "text_num_heads": 16,
        "text_mlp_dim": 4304,
        "max_seq_len": 64,
    },
}


if __name__ == "__main__":
    from transformers import SiglipModel

    SIGLIP_CONVERSION_CONFIG = [
        ("siglip_base_p16_224", "google/siglip-base-patch16-224"),
        ("siglip_base_p16_256", "google/siglip-base-patch16-256"),
        (
            "siglip_base_p16_multilingual_256",
            "google/siglip-base-patch16-256-multilingual",
        ),
        ("siglip_base_p16_384", "google/siglip-base-patch16-384"),
        ("siglip_base_p16_512", "google/siglip-base-patch16-512"),
        ("siglip_large_p16_256", "google/siglip-large-patch16-256"),
        ("siglip_large_p16_384", "google/siglip-large-patch16-384"),
        ("siglip_so400m_p14_224", "google/siglip-so400m-patch14-224"),
        ("siglip_so400m_p14_384", "google/siglip-so400m-patch14-384"),
    ]

    for variant, hf_id in SIGLIP_CONVERSION_CONFIG:
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        hf_model = SiglipModel.from_pretrained(hf_id).eval()
        state = {k: v.detach().cpu().numpy() for k, v in hf_model.state_dict().items()}

        keras_model = SigLIPZeroShotClassify(**SIGLIP_RECIPES[variant])
        transfer_siglip_weights(keras_model, state)

        total_params = sum(int(np.prod(w.shape)) for w in keras_model.weights)
        total_gb = (total_params * 4) / (1024**3)

        del state
        gc.collect()

        if total_gb <= 5.0:
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

            with torch.no_grad():
                hf_out = hf_model(
                    pixel_values=torch.from_numpy(pixel.transpose(0, 3, 1, 2)),
                    input_ids=torch.from_numpy(token_ids.astype(np.int64)),
                )
                hf_logits = hf_out.logits_per_image.cpu().numpy()
                scale = hf_model.logit_scale.exp().item()

            k_out = keras_model(
                {"images": pixel, "token_ids": token_ids}, training=False
            )
            k_logits = keras.ops.convert_to_numpy(k_out["image_logits"])

            logits_diff = float(np.abs(hf_logits - k_logits).max())
            cosine_diff = logits_diff / (scale + 1e-8)
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
