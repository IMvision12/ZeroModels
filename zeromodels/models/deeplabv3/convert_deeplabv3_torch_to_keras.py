import gc
from typing import Callable, Dict, List, Tuple

import keras
import numpy as np
from tqdm import tqdm

from zeromodels.conversion.exceptions import (
    WeightMappingError,
    WeightShapeMismatchError,
)
from zeromodels.conversion.weight_split_util import split_model_weights
from zeromodels.conversion.weight_transfer_util import (
    compare_keras_torch_names,
    transfer_weights,
)
from zeromodels.models.deeplabv3 import DeepLabV3SemanticSegment

weight_name_mapping: Dict[str, str] = {
    "backbone_layer": "backbone.layer",
    "_": ".",
    "downsample.conv": "downsample.0",
    "downsample.bn": "downsample.1",
    "backbone.conv1": "backbone.conv1",
    "backbone.bn1": "backbone.bn1",
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving.mean": "running_mean",
    "moving.variance": "running_var",
}

_classifier_conv_bn_pairs: List[Tuple[str, str, str, str]] = [
    (
        "classifier_0_convs_0_0",
        "classifier_0_convs_0_1",
        "classifier.0.convs.0.0",
        "classifier.0.convs.0.1",
    ),
    (
        "classifier_0_convs_1_0",
        "classifier_0_convs_1_1",
        "classifier.0.convs.1.0",
        "classifier.0.convs.1.1",
    ),
    (
        "classifier_0_convs_2_0",
        "classifier_0_convs_2_1",
        "classifier.0.convs.2.0",
        "classifier.0.convs.2.1",
    ),
    (
        "classifier_0_convs_3_0",
        "classifier_0_convs_3_1",
        "classifier.0.convs.3.0",
        "classifier.0.convs.3.1",
    ),
    (
        "classifier_0_convs_4_1",
        "classifier_0_convs_4_2",
        "classifier.0.convs.4.1",
        "classifier.0.convs.4.2",
    ),
    (
        "classifier_0_project_0",
        "classifier_0_project_1",
        "classifier.0.project.0",
        "classifier.0.project.1",
    ),
    ("classifier_1", "classifier_2", "classifier.1", "classifier.2"),
]


def transfer_deeplabv3_weights(
    keras_model: keras.Model, torch_state_dict: Dict[str, np.ndarray]
) -> None:
    trainable, non_trainable = split_model_weights(keras_model)
    all_keras_weights = trainable + non_trainable
    backbone_weights = [
        (w, name) for w, name in all_keras_weights if name.startswith("backbone_")
    ]

    for keras_weight, keras_weight_name in tqdm(
        backbone_weights, desc="Transferring DeepLabV3 backbone weights"
    ):
        torch_weight_name = keras_weight_name
        for keras_part, torch_part in weight_name_mapping.items():
            torch_weight_name = torch_weight_name.replace(keras_part, torch_part)

        if torch_weight_name not in torch_state_dict:
            raise WeightMappingError(keras_weight_name, torch_weight_name)

        torch_weight = torch_state_dict[torch_weight_name]

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

    for k_conv, k_bn, pt_conv, pt_bn in tqdm(
        _classifier_conv_bn_pairs, desc="Transferring DeepLabV3 classifier weights"
    ):
        conv_layer = keras_model.get_layer(k_conv)
        conv_w = torch_state_dict[f"{pt_conv}.weight"]
        conv_layer.weights[0].assign(np.transpose(conv_w, (2, 3, 1, 0)))

        bn_layer = keras_model.get_layer(k_bn)
        bn_layer.weights[0].assign(torch_state_dict[f"{pt_bn}.weight"])
        bn_layer.weights[1].assign(torch_state_dict[f"{pt_bn}.bias"])
        bn_layer.weights[2].assign(torch_state_dict[f"{pt_bn}.running_mean"])
        bn_layer.weights[3].assign(torch_state_dict[f"{pt_bn}.running_var"])

    cls_layer = keras_model.get_layer("classifier_4")
    conv_w = torch_state_dict["classifier.4.weight"]
    cls_layer.weights[0].assign(np.transpose(conv_w, (2, 3, 1, 0)))
    cls_layer.weights[1].assign(torch_state_dict["classifier.4.bias"])


if __name__ == "__main__":
    import torch
    from torchvision.models.segmentation import (
        DeepLabV3_ResNet50_Weights,
        DeepLabV3_ResNet101_Weights,
        deeplabv3_resnet50,
        deeplabv3_resnet101,
    )

    DEEPLABV3_CONVERSION_CONFIG: List[Tuple[str, Callable, object]] = [
        (
            "deeplabv3_resnet50_coco_voc",
            deeplabv3_resnet50,
            DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
        ),
        (
            "deeplabv3_resnet101_coco_voc",
            deeplabv3_resnet101,
            DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1,
        ),
    ]

    for variant, torch_model_fn, torch_weights in DEEPLABV3_CONVERSION_CONFIG:
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}")
        print(f"{'=' * 60}")

        keras_model: keras.Model = DeepLabV3SemanticSegment.from_weights(
            variant, load_weights=False
        )
        torch_model = torch_model_fn(weights=torch_weights).eval()

        trainable_torch, non_trainable_torch, _ = split_model_weights(torch_model)
        torch_state_dict = {
            k: v.numpy() for k, v in {**trainable_torch, **non_trainable_torch}.items()
        }

        transfer_deeplabv3_weights(keras_model, torch_state_dict)

        print("Verifying model equivalence...")
        np.random.seed(42)
        image_size = keras_model.image_size
        test_input = np.random.rand(1, *image_size).astype(np.float32)
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        normalized_input = (test_input - mean) / std

        torch_input = torch.tensor(normalized_input).permute(0, 3, 1, 2).float()
        with torch.no_grad():
            torch_output = torch_model(torch_input)["out"]
            torch_output_np = torch_output.permute(0, 2, 3, 1).numpy()

        keras_output = keras_model(normalized_input.astype(np.float32), training=False)
        keras_output_np = keras.ops.convert_to_numpy(keras_output)

        torch_h, torch_w = torch_output_np.shape[1], torch_output_np.shape[2]
        keras_h, keras_w = keras_output_np.shape[1], keras_output_np.shape[2]

        if torch_h != keras_h or torch_w != keras_w:
            keras_no_upsample = keras.Model(
                inputs=keras_model.input, outputs=keras_model.layers[-2].output
            )
            keras_raw = keras_no_upsample(
                normalized_input.astype(np.float32), training=False
            )
            keras_raw_np = keras.ops.convert_to_numpy(keras_raw)
            max_diff = float(np.max(np.abs(torch_output_np - keras_raw_np)))
        else:
            max_diff = float(np.max(np.abs(torch_output_np - keras_output_np)))

        print(f"  Max output diff: {max_diff:.6f}")
        if max_diff > 1e-3:
            raise ValueError(
                f"{variant}: max diff {max_diff:.6f} exceeds 1e-3 tolerance"
            )
        print("  Verification OK")

        model_filename = f"{variant}.weights.h5"
        keras_model.save_weights(model_filename)
        print(f"  Saved -> {model_filename}")

        del keras_model, torch_model, torch_state_dict
        keras.backend.clear_session()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
