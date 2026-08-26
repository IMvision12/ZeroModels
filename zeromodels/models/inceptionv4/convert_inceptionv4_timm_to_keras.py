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
from zeromodels.models.inceptionv4 import InceptionV4ImageClassify

# Architecture presets, moved here from inceptionv4_config.py: the package config no
# longer carries arch (models load by Hub repo id / kf_config). Only this converter
# builds an untrained model to transfer timm weights into.
INCEPTIONV4_MODEL_CONFIG = {
    "inception_v4": {
        "image_size": 299,
        "num_classes": 1000,
    },
}

# Hosted variants -> (model arch key, timm id). Weights load by Hub repo id
# (kf_config.json); the github release urls have been removed.
INCEPTIONV4_VARIANTS = {
    "inception_v4_tf_in1k": {
        "model": "inception_v4",
        "timm_id": "inception_v4.tf_in1k",
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "features_": "features.",
    "_conv": ".conv",
    "_kernel": ".weight",
    "_gamma": ".weight",
    "_beta": ".bias",
    "_bias": ".bias",
    "_bn": ".bn",
    "_moving_mean": ".running_mean",
    "_moving_variance": ".running_var",
    "predictions": "last_linear",
}


def transfer_inceptionv4_weights(
    keras_model, state_dict: Dict[str, np.ndarray]
) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        torch_weight_name = keras_weight_name
        for old, new in WEIGHT_NAME_MAPPING.items():
            torch_weight_name = torch_weight_name.replace(old, new)

        torch_weight_name = re.sub(
            r"features\.(\d+)_", r"features.\1.", torch_weight_name
        )

        if re.match(r"features\.(19|20|21)\.", torch_weight_name):
            if "branch3_1" in torch_weight_name:
                torch_weight_name = torch_weight_name.replace("branch3_1", "branch3.1")
            else:
                torch_weight_name = re.sub(
                    r"\.branch([12])\.([0-9]+[ab]?)",
                    r".branch\1_\2",
                    torch_weight_name,
                )
        else:
            torch_weight_name = re.sub(
                r"\.branch([0-9])_([0-9][ab]?)",
                r".branch\1.\2",
                torch_weight_name,
            )
            torch_weight_name = torch_weight_name.replace("branch3_1", "branch3.1")

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

    for variant, meta in INCEPTIONV4_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = InceptionV4ImageClassify(
            **INCEPTIONV4_MODEL_CONFIG[meta["model"]], include_normalization=False
        )
        transfer_inceptionv4_weights(keras_model, state)

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
