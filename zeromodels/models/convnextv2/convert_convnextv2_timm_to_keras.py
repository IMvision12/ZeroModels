import gc

import keras
import numpy as np

from zeromodels.conversion import verify_cls_model_equivalence
from zeromodels.conversion.hf_download_utils import download_hf_state_dict
from zeromodels.models.convnext.convert_convnext_timm_to_keras import (
    transfer_convnext_weights as transfer_convnextv2_weights,
)
from zeromodels.models.convnextv2 import ConvNeXtV2ImageClassify

# Architecture presets, moved here from convnextv2_config.py: the package config no
# longer carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer timm weights into.
CONVNEXTV2_MODEL_CONFIG = {
    "convnextv2_atto": {
        "depths": [2, 2, 6, 2],
        "projection_dim": [40, 80, 160, 320],
        "use_conv": True,
        "use_grn": True,
        "layer_scale_init": None,
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnextv2_femto": {
        "depths": [2, 2, 6, 2],
        "projection_dim": [48, 96, 192, 384],
        "use_conv": True,
        "use_grn": True,
        "layer_scale_init": None,
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnextv2_pico": {
        "depths": [2, 2, 6, 2],
        "projection_dim": [64, 128, 256, 512],
        "use_conv": True,
        "use_grn": True,
        "layer_scale_init": None,
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnextv2_nano": {
        "depths": [2, 2, 8, 2],
        "projection_dim": [80, 160, 320, 640],
        "use_conv": True,
        "use_grn": True,
        "layer_scale_init": None,
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnextv2_nano_384": {
        "depths": [2, 2, 8, 2],
        "projection_dim": [80, 160, 320, 640],
        "use_conv": True,
        "use_grn": True,
        "layer_scale_init": None,
        "image_size": 384,
        "num_classes": 1000,
    },
    "convnextv2_tiny": {
        "depths": [3, 3, 9, 3],
        "projection_dim": [96, 192, 384, 768],
        "use_grn": True,
        "layer_scale_init": None,
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnextv2_tiny_384": {
        "depths": [3, 3, 9, 3],
        "projection_dim": [96, 192, 384, 768],
        "use_grn": True,
        "layer_scale_init": None,
        "image_size": 384,
        "num_classes": 1000,
    },
    "convnextv2_base": {
        "depths": [3, 3, 27, 3],
        "projection_dim": [128, 256, 512, 1024],
        "use_grn": True,
        "layer_scale_init": None,
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnextv2_base_384": {
        "depths": [3, 3, 27, 3],
        "projection_dim": [128, 256, 512, 1024],
        "use_grn": True,
        "layer_scale_init": None,
        "image_size": 384,
        "num_classes": 1000,
    },
    "convnextv2_large": {
        "depths": [3, 3, 27, 3],
        "projection_dim": [192, 384, 768, 1536],
        "use_grn": True,
        "layer_scale_init": None,
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnextv2_large_384": {
        "depths": [3, 3, 27, 3],
        "projection_dim": [192, 384, 768, 1536],
        "use_grn": True,
        "layer_scale_init": None,
        "image_size": 384,
        "num_classes": 1000,
    },
    "convnextv2_huge": {
        "depths": [3, 3, 27, 3],
        "projection_dim": [352, 704, 1408, 2816],
        "use_grn": True,
        "layer_scale_init": None,
        "image_size": 224,
        "num_classes": 1000,
    },
    "convnextv2_huge_384": {
        "depths": [3, 3, 27, 3],
        "projection_dim": [352, 704, 1408, 2816],
        "use_grn": True,
        "layer_scale_init": None,
        "image_size": 384,
        "num_classes": 1000,
    },
    "convnextv2_huge_512": {
        "depths": [3, 3, 27, 3],
        "projection_dim": [352, 704, 1408, 2816],
        "use_grn": True,
        "layer_scale_init": None,
        "image_size": 512,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (zm_config.json); the github release urls have been removed.
CONVNEXTV2_VARIANTS = {
    "convnextv2_atto_fcmae_ft_in1k": {
        "model": "convnextv2_atto",
        "timm_id": "convnextv2_atto.fcmae_ft_in1k",
    },
    "convnextv2_femto_fcmae_ft_in1k": {
        "model": "convnextv2_femto",
        "timm_id": "convnextv2_femto.fcmae_ft_in1k",
    },
    "convnextv2_pico_fcmae_ft_in1k": {
        "model": "convnextv2_pico",
        "timm_id": "convnextv2_pico.fcmae_ft_in1k",
    },
    "convnextv2_nano_fcmae_ft_in1k": {
        "model": "convnextv2_nano",
        "timm_id": "convnextv2_nano.fcmae_ft_in1k",
    },
    "convnextv2_nano_fcmae_ft_in22k_in1k": {
        "model": "convnextv2_nano",
        "timm_id": "convnextv2_nano.fcmae_ft_in22k_in1k",
    },
    "convnextv2_nano_fcmae_ft_in22k_in1k_384": {
        "model": "convnextv2_nano_384",
        "timm_id": "convnextv2_nano.fcmae_ft_in22k_in1k_384",
    },
    "convnextv2_tiny_fcmae_ft_in1k": {
        "model": "convnextv2_tiny",
        "timm_id": "convnextv2_tiny.fcmae_ft_in1k",
    },
    "convnextv2_tiny_fcmae_ft_in22k_in1k": {
        "model": "convnextv2_tiny",
        "timm_id": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    },
    "convnextv2_tiny_fcmae_ft_in22k_in1k_384": {
        "model": "convnextv2_tiny_384",
        "timm_id": "convnextv2_tiny.fcmae_ft_in22k_in1k_384",
    },
    "convnextv2_base_fcmae_ft_in1k": {
        "model": "convnextv2_base",
        "timm_id": "convnextv2_base.fcmae_ft_in1k",
    },
    "convnextv2_base_fcmae_ft_in22k_in1k": {
        "model": "convnextv2_base",
        "timm_id": "convnextv2_base.fcmae_ft_in22k_in1k",
    },
    "convnextv2_base_fcmae_ft_in22k_in1k_384": {
        "model": "convnextv2_base_384",
        "timm_id": "convnextv2_base.fcmae_ft_in22k_in1k_384",
    },
    "convnextv2_large_fcmae_ft_in1k": {
        "model": "convnextv2_large",
        "timm_id": "convnextv2_large.fcmae_ft_in1k",
    },
    "convnextv2_large_fcmae_ft_in22k_in1k": {
        "model": "convnextv2_large",
        "timm_id": "convnextv2_large.fcmae_ft_in22k_in1k",
    },
    "convnextv2_large_fcmae_ft_in22k_in1k_384": {
        "model": "convnextv2_large_384",
        "timm_id": "convnextv2_large.fcmae_ft_in22k_in1k_384",
    },
    "convnextv2_huge_fcmae_ft_in1k": {
        "model": "convnextv2_huge",
        "timm_id": "convnextv2_huge.fcmae_ft_in1k",
    },
    "convnextv2_huge_fcmae_ft_in22k_in1k_384": {
        "model": "convnextv2_huge_384",
        "timm_id": "convnextv2_huge.fcmae_ft_in22k_in1k_384",
    },
    "convnextv2_huge_fcmae_ft_in22k_in1k_512": {
        "model": "convnextv2_huge_512",
        "timm_id": "convnextv2_huge.fcmae_ft_in22k_in1k_512",
    },
}

__all__ = ["transfer_convnextv2_weights"]


if __name__ == "__main__":
    import timm

    for variant, meta in CONVNEXTV2_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = ConvNeXtV2ImageClassify(**CONVNEXTV2_MODEL_CONFIG[meta["model"]])
        transfer_convnextv2_weights(keras_model, state)

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
