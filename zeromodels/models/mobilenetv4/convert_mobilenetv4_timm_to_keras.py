import gc
import re
import sys
from typing import Dict

import keras
import numpy as np
from keras import layers
from tqdm import tqdm

from zeromodels.conversion import verify_cls_model_equivalence
from zeromodels.conversion.exceptions import WeightMappingError
from zeromodels.conversion.weight_transfer_util import transfer_weights
from zeromodels.models.mobilenetv4 import MobileNetV4ImageClassify

MOBILENETV4_MODEL_CONFIG = {
    "mobilenetv4_conv_small": {"config": "conv_small", "image_size": 224},
    "mobilenetv4_conv_medium": {"config": "conv_medium", "image_size": 256},
    "mobilenetv4_conv_large": {"config": "conv_large", "image_size": 384},
    "mobilenetv4_hybrid_medium": {"config": "hybrid_medium", "image_size": 224},
    "mobilenetv4_hybrid_large": {"config": "hybrid_large", "image_size": 384},
}

MOBILENETV4_VARIANTS = {
    "mobilenetv4_conv_small_e2400_r224_in1k": {
        "model": "mobilenetv4_conv_small",
        "timm_id": "mobilenetv4_conv_small.e2400_r224_in1k",
    },
    "mobilenetv4_conv_medium_e500_r256_in1k": {
        "model": "mobilenetv4_conv_medium",
        "timm_id": "mobilenetv4_conv_medium.e500_r256_in1k",
    },
    "mobilenetv4_conv_large_e600_r384_in1k": {
        "model": "mobilenetv4_conv_large",
        "timm_id": "mobilenetv4_conv_large.e600_r384_in1k",
    },
    "mobilenetv4_hybrid_medium_e500_r224_in1k": {
        "model": "mobilenetv4_hybrid_medium",
        "timm_id": "mobilenetv4_hybrid_medium.e500_r224_in1k",
    },
    "mobilenetv4_hybrid_large_e600_r384_in1k": {
        "model": "mobilenetv4_hybrid_large",
        "timm_id": "mobilenetv4_hybrid_large.e600_r384_in1k",
    },
}

WEIGHT_NAME_MAPPING = {
    "conv.stem": "conv_stem",
    "conv.head": "conv_head",
    "norm.head": "norm_head",
    "conv.exp": "conv_exp",
    "conv.pwl": "conv_pwl",
    "dw.start.conv": "dw_start.conv",
    "dw.start.bn": "dw_start.bn",
    "dw.mid.conv": "dw_mid.conv",
    "dw.mid.bn": "dw_mid.bn",
    "pw.exp.conv": "pw_exp.conv",
    "pw.exp.bn": "pw_exp.bn",
    "pw.proj.conv": "pw_proj.conv",
    "pw.proj.bn": "pw_proj.bn",
    "down.conv": "down_conv",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
    "layer.scale.weight": "layer_scale.gamma",
}

_BLOCK_RE = re.compile(r"blocks_(\d+)_(\d+)_")


def keras_name_to_timm(keras_name: str) -> str:
    name = _BLOCK_RE.sub(lambda m: f"blocks.{m.group(1)}.{m.group(2)}.", keras_name)
    name = name.replace("_", ".")
    for old, new in WEIGHT_NAME_MAPPING.items():
        name = name.replace(old, new)
    return name


def transfer_mobilenetv4_weights(
    keras_model, state_dict: Dict[str, np.ndarray]
) -> None:
    for layer in tqdm(keras_model.layers, desc="Transferring weights to Keras"):
        if not layer.weights:
            continue
        if isinstance(layer, layers.DepthwiseConv2D):
            hint = "depthwise_conv2d"
        elif isinstance(layer, layers.Conv2D):
            hint = "conv2d"
        else:
            hint = None
        for weight in layer.weights:
            keras_name = f"{layer.name}_{weight.name}"
            torch_name = keras_name_to_timm(keras_name)
            if torch_name not in state_dict:
                raise WeightMappingError(keras_name, torch_name)
            transpose_name = f"{hint}/{weight.name}" if hint else keras_name
            transfer_weights(transpose_name, weight, state_dict[torch_name])


if __name__ == "__main__":
    import timm

    sys.setrecursionlimit(10000)

    for variant, meta in MOBILENETV4_VARIANTS.items():
        model_cfg = dict(MOBILENETV4_MODEL_CONFIG[meta["model"]])
        timm_id = meta["timm_id"]
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  timm/{timm_id}")
        print(f"{'=' * 60}")

        torch_model = timm.create_model(timm_id, pretrained=True).eval()
        state = {
            k: v.detach().cpu().numpy() for k, v in torch_model.state_dict().items()
        }
        num_classes = int(state["classifier.weight"].shape[0])

        keras_model = MobileNetV4ImageClassify(
            config=model_cfg["config"],
            image_size=model_cfg["image_size"],
            num_classes=num_classes,
            include_normalization=False,
        )

        transfer_mobilenetv4_weights(keras_model, state)

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
