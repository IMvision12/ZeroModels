from typing import Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from PIL import Image

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.labels_util import COCO_91_CLASSES


@keras.saving.register_keras_serializable(package="zeromodels")
class RFDETRImageProcessor(BaseImageProcessor):
    """Preprocess images for RF-DETR inference.

    Every variant trains at its own resolution, so prefer
    ``RFDETRImageProcessor.from_weights("zeromodels/<variant>")``, which reads
    the right size from the repo's ``zm_preprocessor.json``. Constructing the
    class bare (or with just ``variant=``) gives rfdetr-base's 560, so pass
    ``size`` explicitly if you build it directly for a non-base variant.

    Args:
        variant: Release variant label, kept for reference. Size is NOT derived
            from it (per-variant resolution lives in the repo's
            ``zm_preprocessor.json``); pass ``size`` for a specific resolution.
        size: Target size as ``{"height": H, "width": W}``. Defaults to
            ``{"height": 560, "width": 560}`` (rfdetr-base) when not given. Each
            variant's own resolution:

            * Detection (``RFDETRDetect``): 384 (nano), 512 (small),
              576 (medium), 560 (base), 704 (large).
            * Instance segmentation (``RFDETRInstanceSegment``): 312 (seg-nano),
              384 (seg-small), 432 (seg-preview / seg-medium), 504 (seg-large),
              624 (seg-xlarge), 768 (seg-xxlarge).

            The same processor serves both: preprocessing is identical
            (rescale + ImageNet normalize + resize); only the target size
            and the post-processor differ.
        resample: Interpolation method (``"nearest"``, ``"bilinear"``,
            or ``"bicubic"``).
        do_rescale: Whether to divide pixel values by 255.
        rescale_factor: Rescale factor (default ``1/255``).
        do_normalize: Whether to apply ImageNet normalization.
        image_mean: Per-channel mean. Default: ``(0.485, 0.456, 0.406)``.
        image_std: Per-channel std. Default: ``(0.229, 0.224, 0.225)``.
        return_tensor: If True return a Keras tensor, otherwise numpy.
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
        variant: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.variant = variant
        self.size = size if size is not None else self.variant_size(variant)
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

    @staticmethod
    def variant_size(variant: Optional[str]) -> Dict[str, int]:
        """Default square target size (rfdetr-base's 560).

        The per-variant resolution is no longer kept in the package: it travels
        with the weights in the repo's ``zm_preprocessor.json`` and is applied
        when loading via ``from_weights("zeromodels/<variant>")``. Direct
        construction without an explicit ``size`` falls back to 560.
        """
        return {"height": 560, "width": 560}

    def __call__(
        self, image: Union[str, np.ndarray, Image.Image, List]
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
            antialias=True,
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
        return rf_detr_post_process_object_detection(
            outputs,
            threshold=threshold,
            num_top_queries=num_top_queries,
            target_sizes=target_sizes,
            label_names=label_names,
        )

    def post_process_instance_segmentation(
        self,
        outputs,
        threshold=0.5,
        num_top_queries=300,
        target_sizes=None,
        label_names=None,
        mask_threshold=0.5,
    ):
        return rf_detr_post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            num_top_queries=num_top_queries,
            target_sizes=target_sizes,
            label_names=label_names,
            mask_threshold=mask_threshold,
        )


def rf_detr_post_process_object_detection(
    outputs: Dict[str, keras.KerasTensor],
    threshold: float = 0.5,
    num_top_queries: int = 300,
    target_sizes: Optional[List[Tuple[int, int]]] = None,
    label_names: Optional[List[str]] = None,
) -> List[Dict[str, np.ndarray]]:
    """Post-process raw RF-DETR outputs into usable detections.

    RF-DETR uses sigmoid activation (not softmax) and does not have a
    dedicated background class. This post-processor applies sigmoid to raw
    logits, selects top-K scoring (query, class) pairs, converts boxes from
    normalized cxcywh to xyxy pixel coordinates, and filters by score threshold.

    Args:
        outputs: Raw model output dict with keys ``"logits"`` of shape
            ``(B, num_queries, num_classes)`` and ``"pred_boxes"`` of shape
            ``(B, num_queries, 4)`` in normalized ``(cx, cy, w, h)`` format.
        threshold: Minimum confidence score to keep a detection.
        num_top_queries: Number of top (query, class) pairs to consider
            before threshold filtering. Default 300.
        target_sizes: List of ``(height, width)`` tuples for each image in
            the batch. Used to convert normalized boxes to pixel coordinates.
            If None, boxes are returned in normalized ``[0, 1]`` coordinates.
        label_names: Custom class name list for mapping label indices to
            names. If ``None``, defaults to COCO class names. Provide this
            when using a model fine-tuned on a custom dataset.

    Returns:
        List of dicts (one per image in the batch), each containing:
            - ``"scores"``: Confidence scores, shape ``(num_detections,)``.
            - ``"labels"``: Integer class IDs, shape ``(num_detections,)``.
            - ``"label_names"``: Human-readable COCO class names.
            - ``"boxes"``: Bounding boxes as ``[x_min, y_min, x_max, y_max]``,
              shape ``(num_detections, 4)``.

    Example:
        ```python
        from zeromodels.models.rf_detr import RFDETRDetect, RFDETRImageProcessor, rf_detr_post_process_object_detection

        model = RFDETRDetect.from_weights("rfdetr-base")
        img = RFDETRImageProcessor("photo.jpg", size={"height": 560, "width": 560})
        output = model(img, training=False)
        results = rf_detr_post_process_object_detection(output, threshold=0.5,
                                      target_sizes=[(orig_h, orig_w)])
        for r in results:
            for name, score in zip(r["label_names"], r["scores"]):
                print(f"{name}: {score:.2f}")
        ```
    """
    logits = keras.ops.convert_to_numpy(outputs["logits"])
    boxes = keras.ops.convert_to_numpy(outputs["pred_boxes"])

    batch_size = logits.shape[0]
    num_classes = logits.shape[2]

    probs = sigmoid(logits)

    results = []
    for i in range(batch_size):
        prob_i = probs[i]
        boxes_i = boxes[i]

        flat_scores = prob_i.reshape(-1)
        num_select = min(num_top_queries, flat_scores.shape[0])
        topk_indices = np.argpartition(flat_scores, -num_select)[-num_select:]
        topk_indices = topk_indices[np.argsort(-flat_scores[topk_indices])]

        topk_scores = flat_scores[topk_indices]
        topk_box_indices = topk_indices // num_classes
        topk_labels = topk_indices % num_classes

        topk_boxes = boxes_i[topk_box_indices]

        keep = topk_scores > threshold
        scores = topk_scores[keep]
        labels = topk_labels[keep]
        kept_boxes = topk_boxes[keep]

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


def rf_detr_post_process_instance_segmentation(
    outputs,
    threshold: float = 0.5,
    num_top_queries: int = 300,
    target_sizes: Optional[List[Tuple[int, int]]] = None,
    label_names: Optional[List[str]] = None,
    mask_threshold: float = 0.5,
) -> List[Dict[str, np.ndarray]]:
    """Post-process ``RFDETRInstanceSegment`` outputs into instance masks + scores/labels/boxes.

    Mirrors :func:`rf_detr_post_process_object_detection`'s sigmoid + flat top-k
    scoring; a query is emitted at most once (its highest-scoring class). For each
    kept detection, its mask logits are bilinearly upsampled to ``target_sizes``
    (or kept at the model's mask resolution if ``target_sizes`` is None),
    sigmoid-activated, and thresholded at ``mask_threshold`` to produce a binary
    mask.

    Args:
        outputs: Dict with ``logits`` ``(B, Q, num_classes)``, ``pred_boxes``
            ``(B, Q, 4)`` in normalized (cx, cy, w, h), and ``pred_masks``
            ``(B, Q, mh, mw)`` of mask logits.
        threshold: Minimum class score to keep a detection.
        num_top_queries: Top-k queries (× classes) to consider before threshold.
        target_sizes: Per-image ``(height, width)`` to scale boxes and upsample
            masks to. If None, boxes stay normalized and masks stay at the model's
            mask resolution.
        label_names: Custom class names (defaults to COCO_91_CLASSES).
        mask_threshold: Probability threshold for the binary mask (post-sigmoid).

    Returns:
        List of per-image dicts with keys ``"scores"``, ``"labels"``,
        ``"label_names"``, ``"boxes"`` (xyxy), and ``"masks"``: a boolean array of
        shape ``(K, H, W)`` for each image.
    """
    logits = keras.ops.convert_to_numpy(outputs["logits"])
    boxes = keras.ops.convert_to_numpy(outputs["pred_boxes"])
    mask_logits = keras.ops.convert_to_numpy(outputs["pred_masks"]).astype(np.float32)

    batch_size = logits.shape[0]
    num_classes = logits.shape[2]
    probs = sigmoid(logits)

    results = []
    for i in range(batch_size):
        prob_i = probs[i]
        boxes_i = boxes[i]
        masks_i = mask_logits[i]

        flat_scores = prob_i.reshape(-1)
        num_select = min(num_top_queries, flat_scores.shape[0])
        topk_indices = np.argpartition(flat_scores, -num_select)[-num_select:]
        topk_indices = topk_indices[np.argsort(-flat_scores[topk_indices])]

        topk_scores = flat_scores[topk_indices]
        topk_query_indices = topk_indices // num_classes
        topk_labels = topk_indices % num_classes

        keep = topk_scores > threshold
        q_idx = topk_query_indices[keep]
        labels = topk_labels[keep]
        scores = topk_scores[keep]

        seen, sel = set(), []
        for j in range(len(q_idx)):
            if int(q_idx[j]) not in seen:
                seen.add(int(q_idx[j]))
                sel.append(j)
        q_idx = q_idx[sel]
        labels = labels[sel]
        scores = scores[sel]

        kept_boxes = boxes_i[q_idx]
        cx, cy, w, h = (
            kept_boxes[:, 0],
            kept_boxes[:, 1],
            kept_boxes[:, 2],
            kept_boxes[:, 3],
        )
        xyxy_boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)

        if target_sizes is not None:
            img_h, img_w = target_sizes[i]
            xyxy_boxes = xyxy_boxes * np.array(
                [img_w, img_h, img_w, img_h], dtype=np.float32
            )

        if q_idx.size > 0:
            km = masks_i[q_idx][..., None]  # (K, mh, mw, 1)
            if target_sizes is not None:
                img_h, img_w = target_sizes[i]
                km = keras.ops.convert_to_numpy(
                    keras.ops.image.resize(
                        km,
                        (img_h, img_w),
                        interpolation="bilinear",
                        data_format="channels_last",
                    )
                )
            km = km[..., 0]
            masks_bin = (1.0 / (1.0 + np.exp(-km))) > mask_threshold
        else:
            mh, mw = target_sizes[i] if target_sizes is not None else masks_i.shape[1:]
            masks_bin = np.zeros((0, mh, mw), dtype=bool)

        _names = label_names if label_names is not None else COCO_91_CLASSES
        mapped_names = [_names[l] if l < len(_names) else f"class_{l}" for l in labels]

        results.append(
            {
                "scores": scores,
                "labels": labels,
                "label_names": mapped_names,
                "boxes": xyxy_boxes,
                "masks": masks_bin,
            }
        )

    return results


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )
