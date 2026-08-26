import gc
from typing import Dict

import keras
import numpy as np
from tqdm import tqdm

from zeromodels.conversion import verify_cls_model_equivalence
from zeromodels.conversion.exceptions import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from zeromodels.conversion.hf_download_utils import download_hf_state_dict
from zeromodels.conversion.weight_transfer_util import (
    compare_keras_torch_names,
    transfer_weights,
)
from zeromodels.models.maxvit import MaxViTImageClassify as MaxViT

# Architecture presets, moved here from maxvit_config.py: the package config no
# longer carries arch (models load by Hub repo id / kf_config). Only this converter
# builds an untrained model to transfer timm weights into.
MAXVIT_MODEL_CONFIG = {
    "maxvit_tiny_224": {
        "stem_width": 64,
        "depths": [2, 2, 5, 2],
        "embed_dim": [64, 128, 256, 512],
        "num_heads": [2, 4, 8, 16],
        "window_size": 7,
        "image_size": 224,
        "num_classes": 1000,
    },
    "maxvit_tiny_384": {
        "stem_width": 64,
        "depths": [2, 2, 5, 2],
        "embed_dim": [64, 128, 256, 512],
        "num_heads": [2, 4, 8, 16],
        "window_size": 12,
        "image_size": 384,
        "num_classes": 1000,
    },
    "maxvit_tiny_512": {
        "stem_width": 64,
        "depths": [2, 2, 5, 2],
        "embed_dim": [64, 128, 256, 512],
        "num_heads": [2, 4, 8, 16],
        "window_size": 16,
        "image_size": 512,
        "num_classes": 1000,
    },
    "maxvit_small_224": {
        "stem_width": 64,
        "depths": [2, 2, 5, 2],
        "embed_dim": [96, 192, 384, 768],
        "num_heads": [3, 6, 12, 24],
        "window_size": 7,
        "image_size": 224,
        "num_classes": 1000,
    },
    "maxvit_small_384": {
        "stem_width": 64,
        "depths": [2, 2, 5, 2],
        "embed_dim": [96, 192, 384, 768],
        "num_heads": [3, 6, 12, 24],
        "window_size": 12,
        "image_size": 384,
        "num_classes": 1000,
    },
    "maxvit_small_512": {
        "stem_width": 64,
        "depths": [2, 2, 5, 2],
        "embed_dim": [96, 192, 384, 768],
        "num_heads": [3, 6, 12, 24],
        "window_size": 16,
        "image_size": 512,
        "num_classes": 1000,
    },
    "maxvit_base_224": {
        "stem_width": 64,
        "depths": [2, 6, 14, 2],
        "embed_dim": [96, 192, 384, 768],
        "num_heads": [3, 6, 12, 24],
        "window_size": 7,
        "image_size": 224,
        "num_classes": 1000,
    },
    "maxvit_base_384": {
        "stem_width": 64,
        "depths": [2, 6, 14, 2],
        "embed_dim": [96, 192, 384, 768],
        "num_heads": [3, 6, 12, 24],
        "window_size": 12,
        "image_size": 384,
        "num_classes": 1000,
    },
    "maxvit_base_512": {
        "stem_width": 64,
        "depths": [2, 6, 14, 2],
        "embed_dim": [96, 192, 384, 768],
        "num_heads": [3, 6, 12, 24],
        "window_size": 16,
        "image_size": 512,
        "num_classes": 1000,
    },
    "maxvit_base_224_in21k": {
        "stem_width": 64,
        "depths": [2, 6, 14, 2],
        "embed_dim": [96, 192, 384, 768],
        "num_heads": [3, 6, 12, 24],
        "window_size": 7,
        "image_size": 224,
        "num_classes": 21843,
    },
    "maxvit_large_224": {
        "stem_width": 128,
        "depths": [2, 6, 14, 2],
        "embed_dim": [128, 256, 512, 1024],
        "num_heads": [4, 8, 16, 32],
        "window_size": 7,
        "image_size": 224,
        "num_classes": 1000,
    },
    "maxvit_large_384": {
        "stem_width": 128,
        "depths": [2, 6, 14, 2],
        "embed_dim": [128, 256, 512, 1024],
        "num_heads": [4, 8, 16, 32],
        "window_size": 12,
        "image_size": 384,
        "num_classes": 1000,
    },
    "maxvit_large_512": {
        "stem_width": 128,
        "depths": [2, 6, 14, 2],
        "embed_dim": [128, 256, 512, 1024],
        "num_heads": [4, 8, 16, 32],
        "window_size": 16,
        "image_size": 512,
        "num_classes": 1000,
    },
    "maxvit_large_224_in21k": {
        "stem_width": 128,
        "depths": [2, 6, 14, 2],
        "embed_dim": [128, 256, 512, 1024],
        "num_heads": [4, 8, 16, 32],
        "window_size": 7,
        "image_size": 224,
        "num_classes": 21843,
    },
    "maxvit_xlarge_224_in21k": {
        "stem_width": 192,
        "depths": [2, 6, 14, 2],
        "embed_dim": [192, 384, 768, 1536],
        "num_heads": [6, 12, 24, 48],
        "window_size": 7,
        "image_size": 224,
        "num_classes": 21843,
    },
    "maxvit_xlarge_384": {
        "stem_width": 192,
        "depths": [2, 6, 14, 2],
        "embed_dim": [192, 384, 768, 1536],
        "num_heads": [6, 12, 24, 48],
        "window_size": 12,
        "image_size": 384,
        "num_classes": 1000,
    },
    "maxvit_xlarge_512": {
        "stem_width": 192,
        "depths": [2, 6, 14, 2],
        "embed_dim": [192, 384, 768, 1536],
        "num_heads": [6, 12, 24, 48],
        "window_size": 16,
        "image_size": 512,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (kf_config.json); the github release urls have been removed.
MAXVIT_VARIANTS = {
    "maxvit_tiny_tf_224_in1k": {
        "model": "maxvit_tiny_224",
        "timm_id": "maxvit_tiny_tf_224.in1k",
    },
    "maxvit_tiny_tf_384_in1k": {
        "model": "maxvit_tiny_384",
        "timm_id": "maxvit_tiny_tf_384.in1k",
    },
    "maxvit_tiny_tf_512_in1k": {
        "model": "maxvit_tiny_512",
        "timm_id": "maxvit_tiny_tf_512.in1k",
    },
    "maxvit_small_tf_224_in1k": {
        "model": "maxvit_small_224",
        "timm_id": "maxvit_small_tf_224.in1k",
    },
    "maxvit_small_tf_384_in1k": {
        "model": "maxvit_small_384",
        "timm_id": "maxvit_small_tf_384.in1k",
    },
    "maxvit_small_tf_512_in1k": {
        "model": "maxvit_small_512",
        "timm_id": "maxvit_small_tf_512.in1k",
    },
    "maxvit_base_tf_224_in1k": {
        "model": "maxvit_base_224",
        "timm_id": "maxvit_base_tf_224.in1k",
    },
    "maxvit_base_tf_384_in1k": {
        "model": "maxvit_base_384",
        "timm_id": "maxvit_base_tf_384.in1k",
    },
    "maxvit_base_tf_512_in1k": {
        "model": "maxvit_base_512",
        "timm_id": "maxvit_base_tf_512.in1k",
    },
    "maxvit_base_tf_224_in21k": {
        "model": "maxvit_base_224_in21k",
        "timm_id": "maxvit_base_tf_224.in21k",
    },
    "maxvit_base_tf_384_in21k_ft_in1k": {
        "model": "maxvit_base_384",
        "timm_id": "maxvit_base_tf_384.in21k_ft_in1k",
    },
    "maxvit_base_tf_512_in21k_ft_in1k": {
        "model": "maxvit_base_512",
        "timm_id": "maxvit_base_tf_512.in21k_ft_in1k",
    },
    "maxvit_large_tf_224_in1k": {
        "model": "maxvit_large_224",
        "timm_id": "maxvit_large_tf_224.in1k",
    },
    "maxvit_large_tf_384_in1k": {
        "model": "maxvit_large_384",
        "timm_id": "maxvit_large_tf_384.in1k",
    },
    "maxvit_large_tf_512_in1k": {
        "model": "maxvit_large_512",
        "timm_id": "maxvit_large_tf_512.in1k",
    },
    "maxvit_large_tf_224_in21k": {
        "model": "maxvit_large_224_in21k",
        "timm_id": "maxvit_large_tf_224.in21k",
    },
    "maxvit_large_tf_384_in21k_ft_in1k": {
        "model": "maxvit_large_384",
        "timm_id": "maxvit_large_tf_384.in21k_ft_in1k",
    },
    "maxvit_large_tf_512_in21k_ft_in1k": {
        "model": "maxvit_large_512",
        "timm_id": "maxvit_large_tf_512.in21k_ft_in1k",
    },
    "maxvit_xlarge_tf_224_in21k": {
        "model": "maxvit_xlarge_224_in21k",
        "timm_id": "maxvit_xlarge_tf_224.in21k",
    },
    "maxvit_xlarge_tf_384_in21k_ft_in1k": {
        "model": "maxvit_xlarge_384",
        "timm_id": "maxvit_xlarge_tf_384.in21k_ft_in1k",
    },
    "maxvit_xlarge_tf_512_in21k_ft_in1k": {
        "model": "maxvit_xlarge_512",
        "timm_id": "maxvit_xlarge_tf_512.in21k_ft_in1k",
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "relative_position_bias_table": "RPBT",
    "moving_variance": "MOVVAR",
    "moving_mean": "MOVMEAN",
    "attn_block": "ATTNBLOCK",
    "attn_grid": "ATTNGRID",
    "shortcut_expand": "SHORTCUTEXPAND",
    "pre_logits": "PRELOGITS",
    "pre_norm": "PRENORM",
    "conv1_1x1": "CONV11X1",
    "conv2_kxk": "CONV2KXK",
    "conv3_1x1": "CONV31X1",
    "rel_pos": "RELPOS",
    "se_fc": "SEFC",
    "attn_qkv": "ATTNQKV",
    "attn_proj": "ATTNPROJ",
    "mlp_fc": "MLPFC",
    "_": ".",
    "RPBT": "relative_position_bias_table",
    "MOVVAR": "running_var",
    "MOVMEAN": "running_mean",
    "ATTNBLOCK": "attn_block",
    "ATTNGRID": "attn_grid",
    "SHORTCUTEXPAND": "shortcut.expand",
    "PRELOGITS": "pre_logits",
    "PRENORM": "pre_norm",
    "CONV11X1": "conv1_1x1",
    "CONV2KXK": "conv2_kxk",
    "CONV31X1": "conv3_1x1",
    "RELPOS": "rel_pos",
    "SEFC": "conv.se.fc",
    "ATTNQKV": "attn.qkv",
    "ATTNPROJ": "attn.proj",
    "MLPFC": "mlp.fc",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
}


def transfer_maxvit_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    all_keras_weights = []
    for layer in keras_model.layers:
        for w in layer.weights:
            path = w.path
            parts = path.split("/")
            layer_name = parts[-2] if len(parts) >= 2 else parts[0]
            weight_suffix = parts[-1]
            keras_weight_name = f"{layer_name}_{weight_suffix}"
            all_keras_weights.append((w, keras_weight_name))

    for keras_weight, keras_weight_name in tqdm(
        all_keras_weights, desc="Transferring weights to Keras"
    ):
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        if torch_weight_name not in state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = state_dict[torch_weight_name]
        if not compare_keras_torch_names(
            keras_weight_name, keras_weight, torch_weight_name, torch_weight
        ):
            raise WeightShapeMismatchError(
                keras_weight_name,
                keras_weight.shape,
                torch_weight_name,
                torch_weight.shape,
            )

        transfer_name = keras_weight_name
        if "conv2_kxk" in keras_weight_name:
            transfer_name = "dwconv_" + keras_weight_name
        elif "se_fc" in keras_weight_name:
            transfer_name = "conv_" + keras_weight_name
        transfer_weights(transfer_name, keras_weight, torch_weight)


if __name__ == "__main__":
    import timm

    for variant, meta in MAXVIT_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = MaxViT(
            **MAXVIT_MODEL_CONFIG[meta["model"]], include_normalization=False
        )
        transfer_maxvit_weights(keras_model, state)

        torch_model = timm.create_model(timm_id, pretrained=True).eval()
        results = verify_cls_model_equivalence(
            model_a=torch_model,
            model_b=keras_model,
            input_shape=keras_model.input_shape[1:],
            output_specs={"num_classes": keras_model.output_shape[-1]},
            comparison_type="torch_to_keras",
            run_performance=False,
            atol=1e-4,
            rtol=1e-4,
        )
        if not results["standard_input"]:
            raise ValueError(
                "Model equivalence test failed - model outputs do not match for standard input"
            )

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state, torch_model
        keras.backend.clear_session()
        gc.collect()
