import gc
import re
import sys
from typing import Dict

import keras
import numpy as np
from tqdm import tqdm

from zeromodels.conversion import verify_cls_model_equivalence
from zeromodels.conversion.exceptions import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from zeromodels.conversion.weight_split_util import split_model_weights
from zeromodels.conversion.weight_transfer_util import (
    compare_keras_torch_names,
    transfer_weights,
)
from zeromodels.models.xception import XceptionImageClassify

# Architecture presets, moved here from xception_config.py: the package config no
# longer carries arch (models load by Hub repo id / zm_config). Only this converter
# builds an untrained model to transfer timm weights into.
XCEPTION_MODEL_CONFIG = {
    "xception41": {
        "config": "41",
        "preact": False,
        "bn_epsilon": 1e-3,
        "image_size": 299,
        "num_classes": 1000,
    },
    "xception41p": {
        "config": "41p",
        "preact": True,
        "bn_epsilon": 1e-5,
        "image_size": 299,
        "num_classes": 1000,
    },
    "xception65": {
        "config": "65",
        "preact": False,
        "bn_epsilon": 1e-3,
        "image_size": 299,
        "num_classes": 1000,
    },
    "xception65p": {
        "config": "65p",
        "preact": True,
        "bn_epsilon": 1e-3,
        "image_size": 299,
        "num_classes": 1000,
    },
    "xception71": {
        "config": "71",
        "preact": False,
        "bn_epsilon": 1e-3,
        "image_size": 299,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (zm_config.json); the github release urls have been removed.
XCEPTION_VARIANTS = {
    "xception41_tf_in1k": {
        "model": "xception41",
        "timm_id": "xception41.tf_in1k",
    },
    "xception41p_ra3_in1k": {
        "model": "xception41p",
        "timm_id": "xception41p.ra3_in1k",
    },
    "xception65_ra3_in1k": {
        "model": "xception65",
        "timm_id": "xception65.ra3_in1k",
    },
    "xception65_tf_in1k": {
        "model": "xception65",
        "timm_id": "xception65.tf_in1k",
    },
    "xception65p_ra3_in1k": {
        "model": "xception65p",
        "timm_id": "xception65p.ra3_in1k",
    },
    "xception71_tf_in1k": {
        "model": "xception71",
        "timm_id": "xception71.tf_in1k",
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "block.": "blocks.",
    "dwconv": "conv_dw",
    "conv.pw": "conv_pw",
    "bn.dw": "bn_dw",
    "bn.pw": "bn_pw",
    "norm.bn": "norm",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "predictions": "head.fc",
}


def transfer_xception_weights(
    keras_model,
    state_dict: Dict[str, np.ndarray],
    preact: bool,
) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_weight_name = keras_weight_name.replace("_", ".")
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)
        if preact:
            torch_weight_name = torch_weight_name.replace(
                "stem.1.conv.weight", "stem.1.weight"
            )
            torch_weight_name = re.sub(
                r"blocks\.(\d+)\.shortcut\.conv\.weight",
                r"blocks.\1.shortcut.weight",
                torch_weight_name,
            )

        if torch_weight_name not in state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = state_dict[torch_weight_name]
        if (
            keras_weight.ndim == 2
            and torch_weight.ndim == 4
            and torch_weight.shape[-2:] == (1, 1)
        ):
            torch_weight = torch_weight.squeeze(axis=(-1, -2))
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

    sys.setrecursionlimit(10000)

    for variant, meta in XCEPTION_VARIANTS.items():
        model_cfg = dict(XCEPTION_MODEL_CONFIG[meta["model"]])
        model_cfg.pop("num_classes", None)
        preact = model_cfg["preact"]
        timm_id = meta["timm_id"]

        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        torch_model = timm.create_model(timm_id, pretrained=True).eval()
        state = {
            k: v.detach().cpu().numpy() for k, v in torch_model.state_dict().items()
        }
        num_classes = int(state["head.fc.weight"].shape[0])

        keras_model = XceptionImageClassify(
            **model_cfg,
            num_classes=num_classes,
            include_normalization=False,
        )

        transfer_xception_weights(keras_model, state, preact)

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
            raise ValueError(f"{variant}: model equivalence test failed")

        out_path = f"{variant}.weights.h5"
        keras_model.save_weights(out_path)
        print(f"  Saved -> {out_path}")

        del keras_model, state, torch_model
        keras.backend.clear_session()
        gc.collect()
