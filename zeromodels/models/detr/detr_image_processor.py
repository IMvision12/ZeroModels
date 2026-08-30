from typing import Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from PIL import Image

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.labels_util import COCO_91_CLASSES


@keras.saving.register_keras_serializable(package="zeromodels")
class DETRImageProcessor(BaseImageProcessor):
    """Preprocess images for DETR inference.

    The model takes already-normalized input, so run pixels through this processor
    first.

    Args:
        size: Target size as ``{"height": H, "width": W}``.
            Default: ``{"height": 800, "width": 800}``.
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
        self.size = size if size is not None else {"height": 800, "width": 800}
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
        self, image: Union[str, np.ndarray, Image.Image]
    ) -> Dict[str, Union[keras.KerasTensor, np.ndarray]]:
        return self.call(image)

    def call(
        self, image: Union[str, np.ndarray, Image.Image, List]
    ) -> Dict[str, Union[keras.KerasTensor, np.ndarray]]:
        if isinstance(image, (list, tuple)):
            return self.stack_images(image)
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

    def post_process_object_detection(
        self, outputs, threshold=0.7, target_sizes=None, label_names=None
    ):
        return detr_post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
            label_names=label_names,
        )


def detr_post_process_object_detection(
    outputs: Dict[str, keras.KerasTensor],
    threshold: float = 0.7,
    target_sizes: Optional[List[Tuple[int, int]]] = None,
    label_names: Optional[List[str]] = None,
) -> List[Dict[str, np.ndarray]]:
    logits = keras.ops.convert_to_numpy(outputs["logits"])
    boxes = keras.ops.convert_to_numpy(outputs["pred_boxes"])

    batch_size = logits.shape[0]

    probs = softmax(logits)

    results = []
    for i in range(batch_size):
        obj_probs = probs[i, :, :-1]
        scores = np.max(obj_probs, axis=-1)
        labels = np.argmax(obj_probs, axis=-1)

        keep = scores > threshold
        scores = scores[keep]
        labels = labels[keep]
        kept_boxes = boxes[i][keep]

        cx, cy, w, h = (
            kept_boxes[:, 0],
            kept_boxes[:, 1],
            kept_boxes[:, 2],
            kept_boxes[:, 3],
        )
        x_min = cx - w / 2
        y_min = cy - h / 2
        x_max = cx + w / 2
        y_max = cy + h / 2
        xyxy_boxes = np.stack([x_min, y_min, x_max, y_max], axis=-1)

        if target_sizes is not None:
            img_h, img_w = target_sizes[i]
            scale = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
            xyxy_boxes = xyxy_boxes * scale

        _names = label_names if label_names is not None else COCO_91_CLASSES
        mapped_names = [_names[l] if l < len(_names) else f"class_{l}" for l in labels]

        results.append(
            {
                "scores": scores,
                "labels": labels,
                "label_names": mapped_names,
                "boxes": xyxy_boxes,
            }
        )

    return results


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
