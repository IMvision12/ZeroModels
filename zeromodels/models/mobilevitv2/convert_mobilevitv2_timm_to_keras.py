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
from zeromodels.conversion.weight_split_util import split_model_weights
from zeromodels.conversion.weight_transfer_util import (
    compare_keras_torch_names,
    transfer_weights,
)
from zeromodels.models.mobilevitv2 import MobileViTV2ImageClassify

# Architecture presets, moved here from mobilevitv2_config.py: the package config no
# longer carries classification arch (models load by Hub repo id / kf_config). Only
# this converter builds an untrained model to transfer timm weights into.
MOBILEVITV2_MODEL_CONFIG = {
    "mobilevitv2_050": {"multiplier": 0.5, "image_size": 256, "num_classes": 1000},
    "mobilevitv2_075": {"multiplier": 0.75, "image_size": 256, "num_classes": 1000},
    "mobilevitv2_100": {"multiplier": 1.0, "image_size": 256, "num_classes": 1000},
    "mobilevitv2_125": {"multiplier": 1.25, "image_size": 256, "num_classes": 1000},
    "mobilevitv2_150": {"multiplier": 1.5, "image_size": 256, "num_classes": 1000},
    "mobilevitv2_150_384": {"multiplier": 1.5, "image_size": 384, "num_classes": 1000},
    "mobilevitv2_175": {"multiplier": 1.75, "image_size": 256, "num_classes": 1000},
    "mobilevitv2_175_384": {"multiplier": 1.75, "image_size": 384, "num_classes": 1000},
    "mobilevitv2_200": {"multiplier": 2.0, "image_size": 256, "num_classes": 1000},
    "mobilevitv2_200_384": {"multiplier": 2.0, "image_size": 384, "num_classes": 1000},
}

# Hosted classification variants -> (model arch key, timm id). Weights load by Hub
# repo id (kf_config.json); this converter builds an untrained model from the arch
# key above and transfers the matching timm weights.
MOBILEVITV2_VARIANTS = {
    "mobilevitv2_050_cvnets_in1k": {
        "model": "mobilevitv2_050",
        "timm_id": "mobilevitv2_050.cvnets_in1k",
    },
    "mobilevitv2_075_cvnets_in1k": {
        "model": "mobilevitv2_075",
        "timm_id": "mobilevitv2_075.cvnets_in1k",
    },
    "mobilevitv2_100_cvnets_in1k": {
        "model": "mobilevitv2_100",
        "timm_id": "mobilevitv2_100.cvnets_in1k",
    },
    "mobilevitv2_125_cvnets_in1k": {
        "model": "mobilevitv2_125",
        "timm_id": "mobilevitv2_125.cvnets_in1k",
    },
    "mobilevitv2_150_cvnets_in1k": {
        "model": "mobilevitv2_150",
        "timm_id": "mobilevitv2_150.cvnets_in1k",
    },
    "mobilevitv2_150_cvnets_in22k_ft_in1k": {
        "model": "mobilevitv2_150",
        "timm_id": "mobilevitv2_150.cvnets_in22k_ft_in1k",
    },
    "mobilevitv2_150_cvnets_in22k_ft_in1k_384": {
        "model": "mobilevitv2_150_384",
        "timm_id": "mobilevitv2_150.cvnets_in22k_ft_in1k_384",
    },
    "mobilevitv2_175_cvnets_in1k": {
        "model": "mobilevitv2_175",
        "timm_id": "mobilevitv2_175.cvnets_in1k",
    },
    "mobilevitv2_175_cvnets_in22k_ft_in1k": {
        "model": "mobilevitv2_175",
        "timm_id": "mobilevitv2_175.cvnets_in22k_ft_in1k",
    },
    "mobilevitv2_175_cvnets_in22k_ft_in1k_384": {
        "model": "mobilevitv2_175_384",
        "timm_id": "mobilevitv2_175.cvnets_in22k_ft_in1k_384",
    },
    "mobilevitv2_200_cvnets_in1k": {
        "model": "mobilevitv2_200",
        "timm_id": "mobilevitv2_200.cvnets_in1k",
    },
    "mobilevitv2_200_cvnets_in22k_ft_in1k": {
        "model": "mobilevitv2_200",
        "timm_id": "mobilevitv2_200.cvnets_in22k_ft_in1k",
    },
    "mobilevitv2_200_cvnets_in22k_ft_in1k_384": {
        "model": "mobilevitv2_200_384",
        "timm_id": "mobilevitv2_200.cvnets_in22k_ft_in1k_384",
    },
}

WEIGHT_NAME_MAPPING: Dict[str, str] = {
    "_": ".",
    ".ir.conv.1.": ".conv1_1x1.conv.",
    ".ir.batchnorm.1.": ".conv1_1x1.bn.",
    ".ir.dwconv.": ".conv2_kxk.conv.",
    ".ir.batchnorm.2.": ".conv2_kxk.bn.",
    ".ir.conv.2.": ".conv3_1x1.conv.",
    ".ir.batchnorm.3.": ".conv3_1x1.bn.",
    ".mv2.dwconv.": ".conv_kxk.conv.",
    ".mv2.batchnorm.1.": ".conv_kxk.bn.",
    ".mv2.conv.1.": ".conv_1x1.",
    ".mv2.proj.conv.": ".conv_proj.conv.",
    ".mv2.proj.batchnorm.": ".conv_proj.bn.",
    ".attn.conv.1.": ".attn.qkv_proj.",
    ".attn.conv.2.": ".attn.out_proj.",
    ".mlp.conv.1.": ".mlp.fc1.",
    ".mlp.conv.2.": ".mlp.fc2.",
    ".groupnorm.1.": ".norm1.",
    ".groupnorm.2.": ".norm2.",
    ".groupnorm.": ".norm.",
    "stem.batchnorm.": "stem.bn.",
    "predictions.": "head.fc.",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
}


def transfer_mobilevitv2_weights(
    keras_model, state_dict: Dict[str, np.ndarray]
) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
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
        transfer_weights(keras_weight_name, keras_weight, torch_weight)


if __name__ == "__main__":
    import timm

    for variant, meta in MOBILEVITV2_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = MobileViTV2ImageClassify(
            **MOBILEVITV2_MODEL_CONFIG[meta["model"]], include_normalization=False
        )
        transfer_mobilevitv2_weights(keras_model, state)

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
