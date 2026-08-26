from typing import Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from PIL import Image

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.labels_util import COCO_80_CLASSES


@keras.saving.register_keras_serializable(package="zeromodels")
class RTDETRImageProcessor(BaseImageProcessor):
    """Preprocess images for RT-DETR inference.

    RT-DETR's published checkpoints do **not** apply ImageNet
    normalization (only rescale to ``[0, 1]``), but
    ``do_normalize=True`` is exposed for fine-tuning on custom datasets.

    Args:
        size: Target size as ``{"height": H, "width": W}``.
            Default: ``{"height": 640, "width": 640}``.
        resample: Interpolation method (``"nearest"``, ``"bilinear"``,
            or ``"bicubic"``).
        do_rescale: Whether to divide pixel values by 255.
        rescale_factor: Rescale factor (default ``1/255``).
        do_normalize: Whether to apply ImageNet normalization.
            Defaults to ``False``.
        image_mean: Per-channel mean. Used only when
            ``do_normalize=True``. Defaults to ImageNet mean.
        image_std: Per-channel std. Used only when
            ``do_normalize=True``. Defaults to ImageNet std.
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
        do_normalize: bool = False,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        return_tensor: bool = True,
        data_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = size if size is not None else {"height": 640, "width": 640}
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
        self,
        outputs,
        threshold=0.5,
        num_top_queries=300,
        target_sizes=None,
        label_names=None,
    ):
        return rt_detr_post_process_object_detection(
            outputs,
            threshold=threshold,
            num_top_queries=num_top_queries,
            target_sizes=target_sizes,
            label_names=label_names,
        )


def rt_detr_post_process_object_detection(
    outputs: Dict[str, keras.KerasTensor],
    threshold: float = 0.5,
    num_top_queries: int = 300,
    target_sizes: Optional[List[Tuple[int, int]]] = None,
    label_names: Optional[List[str]] = None,
) -> List[Dict[str, np.ndarray]]:
    """Post-process raw RT-DETR outputs into usable detections.

    RT-DETR uses sigmoid activation (not softmax) and has no dedicated
    background class. This post-processor applies sigmoid to raw logits,
    selects top-K scoring ``(query, class)`` pairs, converts boxes from
    normalised ``(cx, cy, w, h)`` to ``(x_min, y_min, x_max, y_max)``
    pixel coordinates, and filters by score threshold.

    Args:
        outputs: Raw model output dict with keys ``"logits"`` of shape
            ``(B, num_queries, num_classes)`` and ``"pred_boxes"`` of
            shape ``(B, num_queries, 4)`` in normalised
            ``(cx, cy, w, h)`` format.
        threshold: Minimum confidence score to keep a detection.
        num_top_queries: Number of top ``(query, class)`` pairs to
            consider before threshold filtering. Default ``300``.
        target_sizes: List of ``(height, width)`` tuples for each image
            in the batch. Used to convert normalised boxes to pixel
            coordinates. If ``None``, boxes stay in ``[0, 1]``
            coordinates.
        label_names: Custom class name list for mapping label indices to
            names. If ``None``, defaults to COCO class names. Provide this
            when using a model fine-tuned on a custom dataset.

    Returns:
        List of dicts (one per image in the batch), each containing:
            - ``"scores"``: Confidence scores ``(num_detections,)``.
            - ``"labels"``: Integer class IDs ``(num_detections,)``.
            - ``"label_names"``: Human-readable COCO class names.
            - ``"boxes"``: ``[x_min, y_min, x_max, y_max]``
              ``(num_detections, 4)``.

    Example:
        ```python
        from zeromodels.models.rt_detr import (
            RTDETRDetect, RTDETRImageProcessor, rt_detr_post_process_object_detection,
        )

        model = RTDETRDetect.from_weights("rtdetr-r50vd")
        img = RTDETRImageProcessor("photo.jpg")
        output = model(img, training=False)
        results = rt_detr_post_process_object_detection(
            output, threshold=0.5, target_sizes=[(orig_h, orig_w)],
        )
        for r in results:
            for name, score in zip(r["label_names"], r["scores"]):
                print(f"{name}: {score:.2f}")
        ```
    """
    ops = keras.ops
    logits = ops.convert_to_tensor(outputs["logits"])
    boxes = ops.convert_to_tensor(outputs["pred_boxes"])

    batch_size = int(logits.shape[0])
    num_queries = int(logits.shape[1])
    num_classes = int(logits.shape[2])

    probs = ops.sigmoid(logits)

    flat_probs = ops.reshape(probs, (batch_size, num_queries * num_classes))
    num_select = min(num_top_queries, num_queries * num_classes)
    topk_scores, topk_indices = ops.top_k(flat_probs, k=num_select)

    topk_box_indices = topk_indices // num_classes
    topk_labels = topk_indices % num_classes

    gather_idx = ops.expand_dims(topk_box_indices, axis=-1)
    gather_idx = ops.repeat(gather_idx, 4, axis=-1)
    topk_boxes = ops.take_along_axis(boxes, gather_idx, axis=1)

    cx = topk_boxes[..., 0]
    cy = topk_boxes[..., 1]
    w = topk_boxes[..., 2]
    h = topk_boxes[..., 3]
    xyxy_boxes = ops.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)

    if target_sizes is not None:
        scales = ops.convert_to_tensor(
            [[w_i, h_i, w_i, h_i] for (h_i, w_i) in target_sizes],
            dtype="float32",
        )
        xyxy_boxes = xyxy_boxes * ops.expand_dims(scales, axis=1)

    topk_scores_np = ops.convert_to_numpy(topk_scores)
    topk_labels_np = ops.convert_to_numpy(topk_labels)
    xyxy_boxes_np = ops.convert_to_numpy(xyxy_boxes)

    _names = label_names if label_names is not None else COCO_80_CLASSES

    results = []
    for i in range(batch_size):
        keep = topk_scores_np[i] > threshold
        scores = topk_scores_np[i][keep]
        labels = topk_labels_np[i][keep]
        kept_boxes = xyxy_boxes_np[i][keep]

        mapped_names = [_names[l] if l < len(_names) else f"class_{l}" for l in labels]

        results.append(
            {
                "scores": scores,
                "labels": labels,
                "label_names": mapped_names,
                "boxes": kept_boxes,
            }
        )

    return results
