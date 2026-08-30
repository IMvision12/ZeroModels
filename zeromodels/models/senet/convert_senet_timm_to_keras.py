import gc

import keras

from zeromodels.conversion import verify_cls_model_equivalence
from zeromodels.conversion.hf_download_utils import download_hf_state_dict
from zeromodels.models.resnet.convert_resnet_timm_to_keras import (
    transfer_resnet_weights as transfer_senet_weights,
)
from zeromodels.models.senet import SENetImageClassify

# Architecture presets, moved here from senet_config.py: the package config no
# longer carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer timm weights into.
SENET_MODEL_CONFIG = {
    "seresnet50": {
        "depths": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
        "senet": True,
    },
    "seresnext50_32x4d": {
        "depths": [3, 4, 6, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 2,
        "senet": True,
        "block_fn_name": "resnext_block",
    },
    "seresnext101_32x4d": {
        "depths": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 2,
        "senet": True,
        "block_fn_name": "resnext_block",
    },
    "seresnext101_32x8d": {
        "depths": [3, 4, 23, 3],
        "filters": [64, 128, 256, 512],
        "groups": 32,
        "width_factor": 4,
        "senet": True,
        "block_fn_name": "resnext_block",
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (zm_config.json); the github release urls have been removed.
SENET_VARIANTS = {
    "seresnet50_a1_in1k": {"model": "seresnet50", "timm_id": "seresnet50.a1_in1k"},
    "seresnext50_32x4d_racm_in1k": {
        "model": "seresnext50_32x4d",
        "timm_id": "seresnext50_32x4d.racm_in1k",
    },
    "seresnext50_32x4d_gluon_in1k": {
        "model": "seresnext50_32x4d",
        "timm_id": "seresnext50_32x4d.gluon_in1k",
    },
    "seresnext101_32x4d_gluon_in1k": {
        "model": "seresnext101_32x4d",
        "timm_id": "seresnext101_32x4d.gluon_in1k",
    },
    "seresnext101_32x8d_ah_in1k": {
        "model": "seresnext101_32x8d",
        "timm_id": "seresnext101_32x8d.ah_in1k",
    },
}

if __name__ == "__main__":
    import timm

    for variant, meta in SENET_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = SENetImageClassify(**SENET_MODEL_CONFIG[meta["model"]])
        transfer_senet_weights(keras_model, state)

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
