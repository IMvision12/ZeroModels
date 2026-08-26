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
from zeromodels.models.mobilevit import MobileViTImageClassify

# Architecture presets, moved here from mobilevit_config.py: the package config no
# longer carries classification arch (models load by Hub repo id / kf_config). Only
# this converter builds an untrained model to transfer timm weights into.
MOBILEVIT_MODEL_CONFIG = {
    "mobilevit_xxs": {
        "initial_dims": 16,
        "head_dims": 320,
        "block_dims": [16, 24, 48, 64, 80],
        "expansion_ratio": [2.0, 2.0, 2.0, 2.0, 2.0],
        "attention_dims": [None, None, 64, 80, 96],
        "image_size": 256,
        "num_classes": 1000,
    },
    "mobilevit_xs": {
        "initial_dims": 16,
        "head_dims": 384,
        "block_dims": [32, 48, 64, 80, 96],
        "expansion_ratio": [4.0, 4.0, 4.0, 4.0, 4.0],
        "attention_dims": [None, None, 96, 120, 144],
        "image_size": 256,
        "num_classes": 1000,
    },
    "mobilevit_s": {
        "initial_dims": 16,
        "head_dims": 640,
        "block_dims": [32, 64, 96, 128, 160],
        "expansion_ratio": [4.0, 4.0, 4.0, 4.0, 4.0],
        "attention_dims": [None, None, 144, 192, 240],
        "image_size": 256,
        "num_classes": 1000,
    },
}

# Hosted classification variants -> (model arch key, timm id). Weights load by Hub
# repo id (kf_config.json); this converter builds an untrained model from the arch
# key above and transfers the matching timm weights.
MOBILEVIT_VARIANTS = {
    "mobilevit_xxs_cvnets_in1k": {
        "model": "mobilevit_xxs",
        "timm_id": "mobilevit_xxs.cvnets_in1k",
    },
    "mobilevit_xs_cvnets_in1k": {
        "model": "mobilevit_xs",
        "timm_id": "mobilevit_xs.cvnets_in1k",
    },
    "mobilevit_s_cvnets_in1k": {
        "model": "mobilevit_s",
        "timm_id": "mobilevit_s.cvnets_in1k",
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
    ".mv.conv.1.": ".conv_kxk.conv.",
    ".mv.batchnorm.1.": ".conv_kxk.bn.",
    ".mv.conv.2.": ".conv_1x1.",
    ".mv.conv.3.": ".conv_proj.conv.",
    ".mv.batchnorm.2.": ".conv_proj.bn.",
    ".mv.conv.4.": ".conv_fusion.conv.",
    ".mv.batchnorm.3.": ".conv_fusion.bn.",
    ".layernorm.1.": ".norm1.",
    ".layernorm.2.": ".norm2.",
    ".layernorm.": ".norm.",
    "final.conv.": "final_conv.conv.",
    "final.batchnorm.": "final_conv.bn.",
    "stem.batchnorm.": "stem.bn.",
    "predictions.": "head.fc.",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
}

# Attention qkv/proj live in a nested layer whose split name does not encode the
# stage/block, so they are resolved from the weight path instead of the mapping.
ATTENTION_PATH = re.compile(
    r".+/stages_(\d+)_1_transformer_(\d+)_attn_(qkv|proj)/(kernel|bias)$"
)


def transfer_mobilevit_weights(keras_model, state_dict: Dict[str, np.ndarray]) -> None:
    trainable, non_trainable = split_model_weights(keras_model)

    for keras_weight, keras_weight_name in tqdm(
        trainable + non_trainable, desc="Transferring weights to Keras"
    ):
        attn = ATTENTION_PATH.match(keras_weight.path)
        if attn:
            stage, t_idx, kind, var = attn.groups()
            torch_var = "weight" if var == "kernel" else "bias"
            torch_weight_name = (
                f"stages.{stage}.1.transformer.{t_idx}.attn.{kind}.{torch_var}"
            )
        else:
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

    for variant, meta in MOBILEVIT_VARIANTS.items():
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        state = download_hf_state_dict(f"timm/{timm_id}")
        keras_model = MobileViTImageClassify(
            **MOBILEVIT_MODEL_CONFIG[meta["model"]], include_normalization=False
        )
        transfer_mobilevit_weights(keras_model, state)

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
