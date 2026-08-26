from typing import Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from keras import ops
from PIL import Image

from zeromodels.base import BaseImageProcessor


@keras.saving.register_keras_serializable(package="zeromodels")
class OwlViTImageProcessor(BaseImageProcessor):
    """Preprocess images for OWL-ViT inference.

    Use this when the model is created with
    ``include_normalization=False``.

    Args:
        size: Target size as ``{"height": H, "width": W}``.
            Default: ``{"height": 768, "width": 768}``.
        resample: Interpolation method (``"nearest"``, ``"bilinear"``,
            or ``"bicubic"``). Defaults to ``"bicubic"``.
        do_rescale: Whether to divide pixel values by 255.
        rescale_factor: Rescale factor (default ``1/255``).
        do_normalize: Whether to apply CLIP normalization.
        image_mean: Per-channel mean for normalization.
            Default: ``(0.48145466, 0.4578275, 0.40821073)``.
        image_std: Per-channel std for normalization.
            Default: ``(0.26862954, 0.26130258, 0.27577711)``.
        return_tensor: If True return a Keras tensor, otherwise numpy
            array.
        data_format: ``"channels_first"`` / ``"channels_last"``;
            ``None`` resolves to ``keras.backend.image_data_format()``.
    """

    def __init__(
        self,
        size: Optional[Dict[str, int]] = None,
        resample: str = "bicubic",
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
        self.size = size if size is not None else {"height": 768, "width": 768}
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = (
            image_mean
            if image_mean is not None
            else (0.48145466, 0.4578275, 0.40821073)
        )
        self.image_std = (
            image_std if image_std is not None else (0.26862954, 0.26130258, 0.27577711)
        )
        self.return_tensor = return_tensor
        self.data_format = data_format

    def __call__(
        self, images: Union[str, np.ndarray, Image.Image, List]
    ) -> Dict[str, Union[keras.KerasTensor, np.ndarray]]:
        return self.call(images)

    def call(
        self, images: Union[str, np.ndarray, Image.Image, List]
    ) -> Dict[str, Union[keras.KerasTensor, np.ndarray]]:
        x, _, _, _ = self.preprocess_image(
            images,
            target_size=(self.size["height"], self.size["width"]),
            image_mean=self.image_mean if self.do_normalize else None,
            image_std=self.image_std if self.do_normalize else None,
            rescale=self.do_rescale,
            interpolation=self.resample,
            antialias=True,
            data_format=self.data_format,
        )
        if self.do_rescale and self.rescale_factor != 1 / 255:
            x = x * (self.rescale_factor * 255)

        if not self.return_tensor:
            x = ops.convert_to_numpy(x)

        return {"pixel_values": x}

    def post_process_object_detection(
        self,
        outputs,
        threshold: float = 0.1,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
        text_labels: Optional[List[List[str]]] = None,
    ):
        return owlvit_post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
            text_labels=text_labels,
        )


def owlvit_post_process_object_detection(
    outputs: Dict,
    threshold: float = 0.1,
    target_sizes: Optional[List[Tuple[int, int]]] = None,
    text_labels: Optional[List[List[str]]] = None,
) -> List[Dict[str, np.ndarray]]:
    """Post-process raw OWL-ViT outputs into per-image detection dicts.

    Converts raw model outputs (logits + normalized boxes) into filtered
    detections with the best query per box, confidence scores, and
    bounding boxes in ``[x_min, y_min, x_max, y_max]`` pixel coordinates.

    Args:
        outputs: Raw model output dict with keys ``"logits"`` of shape
            ``(B, num_patches, num_queries)`` and ``"pred_boxes"`` of
            shape ``(B, num_patches, 4)`` in normalized
            ``(cx, cy, w, h)`` format.
        threshold: Minimum sigmoid score to keep a detection.
        target_sizes: List of ``(height, width)`` for each image. If
            ``None``, boxes stay in normalized ``[0, 1]`` coordinates.
        text_labels: ``[[...] per image]`` of text query strings used
            to populate the ``"text_labels"`` field.

    Returns:
        List of dicts (one per image) with keys ``"scores"``,
        ``"labels"`` (winning query index per detection),
        ``"text_labels"`` (only if ``text_labels`` is given), and
        ``"boxes"``.
    """
    logits = ops.convert_to_numpy(outputs["logits"])
    boxes = ops.convert_to_numpy(outputs["pred_boxes"])

    probs = 1.0 / (1.0 + np.exp(-logits))
    scores_max = probs.max(axis=-1)
    labels_max = probs.argmax(axis=-1)

    results = []
    for i in range(logits.shape[0]):
        keep = scores_max[i] > threshold
        scores = scores_max[i][keep]
        labels = labels_max[i][keep]
        kept_boxes = boxes[i][keep]

        cx, cy, w, h = (
            kept_boxes[:, 0],
            kept_boxes[:, 1],
            kept_boxes[:, 2],
            kept_boxes[:, 3],
        )
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        xyxy = np.stack([x1, y1, x2, y2], axis=-1)

        if target_sizes is not None:
            img_h, img_w = target_sizes[i]
            scale = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
            xyxy = xyxy * scale

        result = {"scores": scores, "labels": labels, "boxes": xyxy}
        if text_labels is not None:
            queries = text_labels[i]
            result["text_labels"] = [
                queries[int(l)] if int(l) < len(queries) else f"query_{int(l)}"
                for l in labels
            ]
        results.append(result)

    return results
