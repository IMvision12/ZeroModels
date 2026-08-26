import gc
import re
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
from zeromodels.conversion.weight_split_util import split_model_weights
from zeromodels.conversion.weight_transfer_util import (
    compare_keras_torch_names,
    transfer_weights,
)
from zeromodels.models.convnext import ConvNeXtImageClassify

# Architecture presets, moved here from convnext_config.py: the package config no
# longer carries arch (models load by Hub repo id / kf_config). Only this converter
# builds an untrained model to transfer timm weights into.
CONVNEXT_MODEL_CONFIG = {
    "convnext_atto": {
        "depths": [2, 2, 6, 2],
        "projection_dim": [40, 80, 160, 320],
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_femto": {
        "depths": [2, 2, 6, 2],
        "projection_dim": [48, 96, 192, 384],
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_pico": {
        "depths": [2, 2, 6, 2],
        "projection_dim": [64, 128, 256, 512],
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_nano": {
        "depths": [2, 2, 8, 2],
        "projection_dim": [80, 160, 320, 640],
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_tiny": {
        "depths": [3, 3, 9, 3],
        "projection_dim": [96, 192, 384, 768],
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_tiny_in22k": {
        "depths": [3, 3, 9, 3],
        "projection_dim": [96, 192, 384, 768],
        "image_size": 224,
        "num_classes": 21841,
    },
    "convnext_tiny_384": {
        "depths": [3, 3, 9, 3],
        "projection_dim": [96, 192, 384, 768],
        "image_size": 384,
        "num_classes": 1000,
    },
    "convnext_small": {
        "depths": [3, 3, 27, 3],
        "projection_dim": [96, 192, 384, 768],
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_small_in22k": {
        "depths": [3, 3, 27, 3],
        "projection_dim": [96, 192, 384, 768],
        "image_size": 224,
        "num_classes": 21841,
    },
    "convnext_small_384": {
        "depths": [3, 3, 27, 3],
        "projection_dim": [96, 192, 384, 768],
        "image_size": 384,
        "num_classes": 1000,
    },
    "convnext_base": {
        "depths": [3, 3, 27, 3],
        "projection_dim": [128, 256, 512, 1024],
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_base_in22k": {
        "depths": [3, 3, 27, 3],
        "projection_dim": [128, 256, 512, 1024],
        "image_size": 224,
        "num_classes": 21841,
    },
    "convnext_base_384": {
        "depths": [3, 3, 27, 3],
        "projection_dim": [128, 256, 512, 1024],
        "image_size": 384,
        "num_classes": 1000,
    },
    "convnext_large": {
        "depths": [3, 3, 27, 3],
        "projection_dim": [192, 384, 768, 1536],
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_large_in22k": {
        "depths": [3, 3, 27, 3],
        "projection_dim": [192, 384, 768, 1536],
        "image_size": 224,
        "num_classes": 21841,
    },
    "convnext_large_384": {
        "depths": [3, 3, 27, 3],
        "projection_dim": [192, 384, 768, 1536],
        "image_size": 384,
        "num_classes": 1000,
    },
    "convnext_xlarge": {
        "depths": [3, 3, 27, 3],
        "projection_dim": [256, 512, 1024, 2048],
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnext_xlarge_in22k": {
        "depths": [3, 3, 27, 3],
        "projection_dim": [256, 512, 1024, 2048],
        "image_size": 224,
        "num_classes": 21841,
    },
    "convnext_xlarge_384": {
        "depths": [3, 3, 27, 3],
        "projection_dim": [256, 512, 1024, 2048],
        "image_size": 384,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (kf_config.json); the github release urls have been removed.
CONVNEXT_VARIANTS = {
    "convnext_atto_d2_in1k": {
        "model": "convnext_atto",
        "timm_id": "convnext_atto.d2_in1k",
    },
    "convnext_femto_d1_in1k": {
        "model": "convnext_femto",
        "timm_id": "convnext_femto.d1_in1k",
    },
    "convnext_pico_d1_in1k": {
        "model": "convnext_pico",
        "timm_id": "convnext_pico.d1_in1k",
    },
    "convnext_nano_d1h_in1k": {
        "model": "convnext_nano",
        "timm_id": "convnext_nano.d1h_in1k",
    },
    "convnext_nano_in12k_ft_in1k": {
        "model": "convnext_nano",
        "timm_id": "convnext_nano.in12k_ft_in1k",
    },
    "convnext_tiny_fb_in1k": {
        "model": "convnext_tiny",
        "timm_id": "convnext_tiny.fb_in1k",
    },
    "convnext_tiny_fb_in22k": {
        "model": "convnext_tiny_in22k",
        "timm_id": "convnext_tiny.fb_in22k",
    },
    "convnext_tiny_fb_in22k_ft_in1k": {
        "model": "convnext_tiny",
        "timm_id": "convnext_tiny.fb_in22k_ft_in1k",
    },
    "convnext_tiny_fb_in22k_ft_in1k_384": {
        "model": "convnext_tiny_384",
        "timm_id": "convnext_tiny.fb_in22k_ft_in1k_384",
    },
    "convnext_small_fb_in1k": {
        "model": "convnext_small",
        "timm_id": "convnext_small.fb_in1k",
    },
    "convnext_small_fb_in22k": {
        "model": "convnext_small_in22k",
        "timm_id": "convnext_small.fb_in22k",
    },
    "convnext_small_fb_in22k_ft_in1k": {
        "model": "convnext_small",
        "timm_id": "convnext_small.fb_in22k_ft_in1k",
    },
    "convnext_small_fb_in22k_ft_in1k_384": {
        "model": "convnext_small_384",
        "timm_id": "convnext_small.fb_in22k_ft_in1k_384",
    },
    "convnext_base_fb_in1k": {
        "model": "convnext_base",
        "timm_id": "convnext_base.fb_in1k",
    },
    "convnext_base_fb_in22k": {
        "model": "convnext_base_in22k",
        "timm_id": "convnext_base.fb_in22k",
    },
    "convnext_base_fb_in22k_ft_in1k": {
        "model": "convnext_base",
        "timm_id": "convnext_base.fb_in22k_ft_in1k",
    },
    "convnext_base_fb_in22k_ft_in1k_384": {
        "model": "convnext_base_384",
        "timm_id": "convnext_base.fb_in22k_ft_in1k_384",
    },
    "convnext_large_fb_in1k": {
        "model": "convnext_large",
        "timm_id": "convnext_large.fb_in1k",
    },
    "convnext_large_fb_in22k": {
        "model": "convnext_large_in22k",
        "timm_id": "convnext_large.fb_in22k",
    },
    "convnext_large_fb_in22k_ft_in1k": {
        "model": "convnext_large",
        "timm_id": "convnext_large.fb_in22k_ft_in1k",
    },
    "convnext_large_fb_in22k_ft_in1k_384": {
        "model": "convnext_large_384",
        "timm_id": "convnext_large.fb_in22k_ft_in1k_384",
    },
    "convnext_xlarge_fb_in22k": {
        "model": "convnext_xlarge_in22k",
        "timm_id": "convnext_xlarge.fb_in22k",
    },
    "convnext_xlarge_fb_in22k_ft_in1k": {
        "model": "convnext_xlarge",
        "timm_id": "convnext_xlarge.fb_in22k_ft_in1k",
    },
    "convnext_xlarge_fb_in22k_ft_in1k_384": {
        "model": "convnext_xlarge_384",
        "timm_id": "convnext_xlarge.fb_in22k_ft_in1k_384",
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "stem_conv_": "stem.0.",
    "stem_layernorm_": "stem.1.",
    "_": ".",
    "layernorm": "norm",
    "depthwise.conv": "conv_dw",
    "grn": "mlp.grn",
    "dense.1": "mlp.fc1",
    "dense.2": "mlp.fc2",
    "conv.1": "mlp.fc1",
    "conv.2": "mlp.fc2",
    "downsampling.norm": "downsample.0",
    "downsampling.conv": "downsample.1",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "final.norm": "head.norm",
    "predictions": "head.fc",
}


def transfer_convnext_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)
        torch_weight_name = re.sub(
            r"\.layer\.scale\.variable(?:\.\d+)?$", ".gamma", torch_weight_name
        )

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
        transfer_weights(keras_weight_name, keras_weight, torch_weight)


if __name__ == "__main__":
    import timm

    for variant, meta in CONVNEXT_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = ConvNeXtImageClassify(
            **CONVNEXT_MODEL_CONFIG[meta["model"]], include_normalization=False
        )
        transfer_convnext_weights(keras_model, state)

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

        total_params = sum(int(np.prod(w.shape)) for w in keras_model.weights)
        total_gb = (total_params * 4) / (1024**3)
        if total_gb > 1.7:
            out_path = f"{variant}.weights.json"
            keras_model.save_weights(out_path, max_shard_size=1.7)
            print(f"  Saved -> {out_path} (sharded, ~{total_gb:.2f} GB)")
        else:
            out_path = f"{variant}.weights.h5"
            keras_model.save_weights(out_path)
            print(f"  Saved -> {out_path} (~{total_gb:.2f} GB)")

        del keras_model, state, torch_model
        keras.backend.clear_session()
        gc.collect()
