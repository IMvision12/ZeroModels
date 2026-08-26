import gc

import keras

from zeromodels.conversion import verify_cls_model_equivalence
from zeromodels.conversion.hf_download_utils import download_hf_state_dict
from zeromodels.models.flexivit import FlexiViTImageClassify
from zeromodels.models.vit.convert_vit_timm_to_keras import (
    transfer_vit_weights as transfer_flexivit_weights,
)

__all__ = ["transfer_flexivit_weights"]

# Architecture presets, moved here from flexivit_config.py: the package config no
# longer carries arch (models load by Hub repo id / kf_config). Only this converter
# builds an untrained model to transfer timm weights into.
FLEXIVIT_MODEL_CONFIG = {
    "flexivit_small": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "no_embed_class": True,
        "image_size": 240,
        "num_classes": 1000,
    },
    "flexivit_base": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "no_embed_class": True,
        "image_size": 240,
        "num_classes": 1000,
    },
    "flexivit_base_in21k": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "no_embed_class": True,
        "image_size": 240,
        "num_classes": 21843,
    },
    "flexivit_large": {
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "no_embed_class": True,
        "image_size": 240,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (kf_config.json); the github release urls have been removed.
FLEXIVIT_VARIANTS = {
    "flexivit_small_1200ep_in1k": {
        "model": "flexivit_small",
        "timm_id": "flexivit_small.1200ep_in1k",
    },
    "flexivit_small_600ep_in1k": {
        "model": "flexivit_small",
        "timm_id": "flexivit_small.600ep_in1k",
    },
    "flexivit_small_300ep_in1k": {
        "model": "flexivit_small",
        "timm_id": "flexivit_small.300ep_in1k",
    },
    "flexivit_base_1200ep_in1k": {
        "model": "flexivit_base",
        "timm_id": "flexivit_base.1200ep_in1k",
    },
    "flexivit_base_300ep_in1k": {
        "model": "flexivit_base",
        "timm_id": "flexivit_base.300ep_in1k",
    },
    "flexivit_base_1000ep_in21k": {
        "model": "flexivit_base_in21k",
        "timm_id": "flexivit_base.1000ep_in21k",
    },
    "flexivit_base_300ep_in21k": {
        "model": "flexivit_base_in21k",
        "timm_id": "flexivit_base.300ep_in21k",
    },
    "flexivit_large_1200ep_in1k": {
        "model": "flexivit_large",
        "timm_id": "flexivit_large.1200ep_in1k",
    },
    "flexivit_large_600ep_in1k": {
        "model": "flexivit_large",
        "timm_id": "flexivit_large.600ep_in1k",
    },
    "flexivit_large_300ep_in1k": {
        "model": "flexivit_large",
        "timm_id": "flexivit_large.300ep_in1k",
    },
}


if __name__ == "__main__":
    import timm

    for variant, meta in FLEXIVIT_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = FlexiViTImageClassify(
            **FLEXIVIT_MODEL_CONFIG[meta["model"]], include_normalization=False
        )
        transfer_flexivit_weights(keras_model, state)

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
