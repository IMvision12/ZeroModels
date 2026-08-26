from typing import Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from PIL import Image

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.image_util import get_data_format
from zeromodels.utils.labels_util import PASCAL_VOC_CLASSES


@keras.saving.register_keras_serializable(package="zeromodels")
class DeepLabV3ImageProcessor(BaseImageProcessor):
    """Preprocess images for DeepLabV3 inference.

    Handles loading, resizing, rescaling, and ImageNet normalization to
    match the preprocessing used during DeepLabV3 training (torchvision
    convention).

    Args:
        size: Target size as ``{"height": H, "width": W}``.
            Default: ``{"height": 520, "width": 520}``.
        resample: Interpolation method (``"nearest"``, ``"bilinear"``,
            or ``"bicubic"``).
        do_rescale: Whether to divide pixel values by 255.
        rescale_factor: Rescale factor (default ``1/255``).
        do_normalize: Whether to apply ImageNet normalization.
        image_mean: Per-channel mean for normalization.
            Default: ``(0.485, 0.456, 0.406)``.
        image_std: Per-channel std for normalization.
            Default: ``(0.229, 0.224, 0.225)``.
        return_tensor: If True return a Keras tensor, otherwise numpy
            array.
        data_format: ``"channels_first"`` / ``"channels_last"``;
            ``None`` resolves to ``keras.backend.image_data_format()``.

    Example:
        ```python
        from zeromodels.models.deeplabv3 import (
            DeepLabV3ImageProcessor, DeepLabV3SemanticSegment,
        )

        model = DeepLabV3SemanticSegment.from_weights("deeplabv3_resnet50_coco_voc")
        processor = DeepLabV3ImageProcessor()
        pixels = processor("photo.jpg")["pixel_values"]
        output = model(pixels, training=False)
        ```
    """

    def __init__(
        self,
        size: Optional[Dict[str, int]] = None,
        resample: str = "bilinear",
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        return_tensor: bool = True,
        data_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = size if size is not None else {"height": 520, "width": 520}
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = (
            image_mean if image_mean is not None else (0.485, 0.456, 0.406)
        )
        self.image_std = image_std if image_std is not None else (0.229, 0.224, 0.225)
        self.return_tensor = return_tensor
        self.data_format = data_format

    def __call__(
        self, image: Union[str, np.ndarray, "Image.Image"]
    ) -> Union["keras.KerasTensor", np.ndarray]:
        return self.call(image)

    def call(
        self, image: Union[str, np.ndarray, "Image.Image"]
    ) -> Union["keras.KerasTensor", np.ndarray]:
        image, _, _, _ = self.preprocess_image(
            image,
            target_size=(self.size["height"], self.size["width"]),
            image_mean=self.image_mean if self.do_normalize else None,
            image_std=self.image_std if self.do_normalize else None,
            rescale=self.do_rescale,
            interpolation=self.resample,
            antialias=False,
            data_format=self.data_format,
        )
        if self.do_rescale and self.rescale_factor != 1 / 255:
            image = image * (self.rescale_factor * 255)

        if not self.return_tensor:
            image = keras.ops.convert_to_numpy(image)

        return {"pixel_values": image}

    def post_process_semantic_segmentation(
        self, outputs, target_size=None, label_names=None, data_format=None
    ):
        return deeplabv3_post_process_semantic_segmentation(
            outputs,
            target_size=target_size,
            label_names=label_names,
            data_format=data_format,
        )


def deeplabv3_post_process_semantic_segmentation(
    outputs: "keras.KerasTensor",
    target_size: Optional[Tuple[int, int]] = None,
    label_names: Optional[List[str]] = None,
    data_format: Optional[str] = None,
) -> Dict:
    """Post-process raw DeepLabV3 outputs into semantic segmentation results.

    Takes the raw logits from DeepLabV3, computes the argmax class map,
    optionally resizes to the original image size, and maps class indices
    to human-readable names.

    Args:
        outputs: Raw model output tensor of shape ``(1, H, W, num_classes)``
            when ``data_format="channels_last"`` or
            ``(1, num_classes, H, W)`` when ``data_format="channels_first"``.
        target_size: Original image ``(height, width)`` for resizing the
            prediction mask. If ``None``, the mask is returned at model
            output resolution.
        label_names: Custom class name list for mapping label indices to
            names. If ``None``, defaults to Pascal VOC class names (21
            classes). Provide this when using a model fine-tuned on a
            custom dataset.
        data_format: Layout of the channel axis in ``outputs``. ``None``
            resolves to the global setting from
            ``keras.config.image_data_format()``.

    Returns:
        Dict with:
            - ``"segmentation"``: Integer array of shape ``(H, W)`` with
              class indices.
            - ``"class_names"``: List of unique class names detected in the
              image.
            - ``"unique_classes"``: Array of unique class indices.

    Example:
        ```python
        from zeromodels.models.deeplabv3 import (
            DeepLabV3ImageProcessor,
            DeepLabV3SemanticSegment,
        )
        from zeromodels.models.deeplabv3.deeplabv3_image_processor import (
            deeplabv3_post_process_semantic_segmentation,
        )

        model = DeepLabV3SemanticSegment.from_weights("deeplabv3_resnet50_coco_voc")
        pixels = DeepLabV3ImageProcessor()("photo.jpg")["pixel_values"]
        output = model(pixels, training=False)
        result = deeplabv3_post_process_semantic_segmentation(
            output, target_size=(orig_h, orig_w)
        )
        print(result["class_names"])
        ```
    """
    _names = label_names if label_names is not None else PASCAL_VOC_CLASSES

    logits = keras.ops.convert_to_numpy(outputs)
    channel_axis = 0 if get_data_format(data_format) == "channels_first" else -1
    # Upsample the logits, then take the argmax. Doing it the other way round
    # quantises the decision to the model's output stride and gives visibly
    # blocky boundaries.
    if target_size is not None:
        if channel_axis == 0:
            logits = np.transpose(logits, (0, 2, 3, 1))
        resized = keras.ops.image.resize(
            logits, (target_size[0], target_size[1]), interpolation="bilinear"
        )
        pred_mask = np.argmax(keras.ops.convert_to_numpy(resized)[0], axis=-1)
    else:
        pred_mask = np.argmax(logits[0], axis=channel_axis)  # (H, W)

    unique_classes = np.unique(pred_mask)
    class_names = [
        _names[c] if c < len(_names) else f"class_{c}" for c in unique_classes
    ]

    return {
        "segmentation": pred_mask,
        "class_names": class_names,
        "unique_classes": unique_classes,
    }
