from typing import Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from PIL import Image

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.image_util import get_data_format, load_image
from zeromodels.utils.labels_util import (
    COCO_PANOPTIC_133_CLASSES,
    COCO_PANOPTIC_STUFF_IDS,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_resized_size(orig_h: int, orig_w: int, target_size: int) -> Tuple[int, int]:
    """Compute the resized (h, w) that fits inside target_size square."""
    scale = target_size / max(orig_h, orig_w)
    return int(orig_h * scale), int(orig_w * scale)


@keras.saving.register_keras_serializable(package="zeromodels")
class EoMTImageProcessor(BaseImageProcessor):
    """Preprocess images for EoMT inference.

    Resizes, pads to square, rescales to [0, 1], and applies ImageNet
    normalization. Uses pure Keras 3 ops for all tensor operations.

    Args:
        target_size: Target square size (default 640).
        image_mean: Per-channel mean for normalization.
        image_std: Per-channel std for normalization.
        data_format: ``"channels_first"`` / ``"channels_last"``;
            ``None`` resolves to ``keras.backend.image_data_format()``.

    Example:
        ```python
        from zeromodels.models.eomt import EoMTLarge
        from zeromodels.models.eomt.eomt_image_processor import (
            EoMTImageProcessor,
        )

        model = EoMTLarge(weights="coco_panoptic_640")
        proc = EoMTImageProcessor(target_size=640)
        img = proc("photo.jpg")
        output = model(img, training=False)
        ```
    """

    def __init__(
        self,
        target_size: int = 640,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        data_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_size = target_size
        self.image_mean = image_mean if image_mean is not None else IMAGENET_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STD
        self.data_format = data_format

    def __call__(
        self, image: Union[str, np.ndarray, Image.Image]
    ) -> Dict[str, keras.KerasTensor]:
        return self.call(image)

    def call(
        self, image: Union[str, np.ndarray, Image.Image]
    ) -> Dict[str, keras.KerasTensor]:
        if isinstance(image, np.ndarray) and image.ndim == 4:
            image = image[0]
        image = load_image(image).astype(np.float32)

        h, w = image.shape[:2]
        new_h, new_w = get_resized_size(h, w, self.target_size)

        image = keras.ops.convert_to_tensor(image, dtype="float32")
        image = keras.ops.expand_dims(image, axis=0)

        image = keras.ops.image.resize(image, (new_h, new_w), interpolation="bilinear")

        image = image / 255.0

        padded = keras.ops.zeros(
            (1, self.target_size, self.target_size, 3), dtype="float32"
        )
        padded = keras.ops.slice_update(padded, (0, 0, 0, 0), image)

        mean = keras.ops.reshape(
            keras.ops.convert_to_tensor(self.image_mean, dtype="float32"),
            (1, 1, 1, 3),
        )
        std = keras.ops.reshape(
            keras.ops.convert_to_tensor(self.image_std, dtype="float32"),
            (1, 1, 1, 3),
        )
        padded = (padded - mean) / std

        if get_data_format(self.data_format) == "channels_first":
            padded = keras.ops.transpose(padded, (0, 3, 1, 2))

        return {"pixel_values": padded}

    def post_process_panoptic_segmentation(
        self,
        outputs: Dict[str, keras.KerasTensor],
        target_size: Tuple[int, int],
        threshold: float = 0.8,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        stuff_classes: Optional[List[int]] = None,
        label_names: Optional[List[str]] = None,
    ) -> Dict:
        return eomt_post_process_panoptic(
            outputs,
            target_size=target_size,
            threshold=threshold,
            mask_threshold=mask_threshold,
            overlap_mask_area_threshold=overlap_mask_area_threshold,
            model_size=self.target_size,
            stuff_classes=stuff_classes,
            label_names=label_names,
        )

    def post_process_semantic_segmentation(
        self,
        outputs: Dict[str, keras.KerasTensor],
        target_size: Tuple[int, int],
        label_names: Optional[List[str]] = None,
    ) -> Dict:
        return eomt_post_process_semantic(
            outputs,
            target_size=target_size,
            model_size=self.target_size,
            label_names=label_names,
        )

    def post_process_instance_segmentation(
        self,
        outputs: Dict[str, keras.KerasTensor],
        target_size: Tuple[int, int],
        threshold: float = 0.5,
        label_names: Optional[List[str]] = None,
    ) -> Dict:
        return eomt_post_process_instance(
            outputs,
            target_size=target_size,
            threshold=threshold,
            model_size=self.target_size,
            label_names=label_names,
        )


def unpad_and_resize_masks(
    mask_logits,
    model_size: int,
    target_h: int,
    target_w: int,
):
    """Unpad mask logits and resize to original image dimensions.

    Follows the reference approach: first resize to the model's input
    resolution, crop out the padding region, then resize to the target size.
    Uses pure Keras 3 ops.

    Args:
        mask_logits: Raw mask logits ``(1, Q, mH, mW)`` as tensor or array.
        model_size: The square input size used during preprocessing (e.g. 640).
        target_h: Original image height.
        target_w: Original image width.

    Returns:
        Mask logits numpy array of shape ``(1, Q, target_h, target_w)``.
    """
    resized_h, resized_w = get_resized_size(target_h, target_w, model_size)

    mask_logits = keras.ops.convert_to_tensor(mask_logits, dtype="float32")

    # (1, Q, mH, mW) -> (1, mH, mW, Q) for keras resize
    mask_4d = keras.ops.transpose(mask_logits, (0, 2, 3, 1))
    mask_full = keras.ops.image.resize(
        mask_4d, (model_size, model_size), interpolation="bilinear"
    )
    # (1, S, S, Q) -> (1, Q, S, S)
    mask_full = keras.ops.transpose(mask_full, (0, 3, 1, 2))

    mask_cropped = mask_full[:, :, :resized_h, :resized_w]

    # (1, Q, rH, rW) -> (1, rH, rW, Q)
    mask_cropped_4d = keras.ops.transpose(mask_cropped, (0, 2, 3, 1))
    mask_final = keras.ops.image.resize(
        mask_cropped_4d, (target_h, target_w), interpolation="bilinear"
    )
    # (1, tH, tW, Q) -> (1, Q, tH, tW)
    mask_final = keras.ops.transpose(mask_final, (0, 3, 1, 2))

    return keras.ops.convert_to_numpy(mask_final)


def eomt_post_process_panoptic(
    outputs: Dict[str, keras.KerasTensor],
    target_size: Tuple[int, int],
    threshold: float = 0.8,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    model_size: int = 640,
    stuff_classes: Optional[List[int]] = None,
    label_names: Optional[List[str]] = None,
) -> Dict:
    """Post-process EoMT outputs into panoptic segmentation predictions.

    Uses Keras 3 ops for softmax/sigmoid and numpy for segment iteration.

    Args:
        outputs: Model output dict with ``"class_logits"`` of shape
            ``(1, num_queries, num_classes+1)`` and ``"mask_logits"`` of shape
            ``(1, num_queries, mask_h, mask_w)``.
        target_size: Original image ``(height, width)`` for upsampling masks.
        threshold: Minimum confidence to keep a segment.
        mask_threshold: Threshold for binarizing mask probabilities.
        overlap_mask_area_threshold: Minimum overlap ratio to keep a segment.
        model_size: Square input size used during preprocessing (default 640).
        stuff_classes: List of class IDs considered "stuff" (merged if same class).
            Defaults to COCO panoptic stuff IDs (80-132).
        label_names: Class name list. Defaults to COCO panoptic classes.

    Returns:
        Dict with:
            - ``"segmentation"``: Integer array of shape ``(H, W)`` with segment IDs.
            - ``"segments_info"``: List of dicts with ``"id"``, ``"label_id"``,
              ``"label_name"``, and ``"score"`` for each segment.

    Example:
        ```python
        output = model(img, training=False)
        result = eomt_post_process_panoptic(
            output, target_size=(original_h, original_w)
        )
        ```
    """
    if stuff_classes is None:
        stuff_classes = COCO_PANOPTIC_STUFF_IDS
    if label_names is None:
        label_names = COCO_PANOPTIC_133_CLASSES

    class_logits = outputs["class_logits"]
    mask_logits = outputs["mask_logits"]

    num_classes = class_logits.shape[-1] - 1
    target_h, target_w = target_size

    mask_logits_resized = unpad_and_resize_masks(
        mask_logits, model_size, target_h, target_w
    )

    scores_all = keras.ops.convert_to_numpy(keras.ops.softmax(class_logits[0], axis=-1))
    pred_scores = np.max(scores_all, axis=-1)
    pred_labels = np.argmax(scores_all, axis=-1)

    mask_probs = mask_logits_resized[0]

    keep = (pred_labels != num_classes) & (pred_scores > threshold)
    mask_probs = mask_probs[keep]
    pred_scores = pred_scores[keep]
    pred_labels = pred_labels[keep]

    if mask_probs.shape[0] == 0:
        return {
            "segmentation": np.full(target_size, -1, dtype=np.int32),
            "segments_info": [],
        }

    mask_probs_sig = keras.ops.convert_to_numpy(
        keras.ops.sigmoid(keras.ops.convert_to_tensor(mask_probs, dtype="float32"))
    )
    mask_labels = (pred_scores[:, None, None] * mask_probs_sig).argmax(0)

    segmentation = np.full(target_size, -1, dtype=np.int32)
    segments_info = []
    current_id = 0
    stuff_memory = {}

    for k in range(pred_labels.shape[0]):
        pred_class = int(pred_labels[k])

        mask_k = mask_labels == k
        mask_k_area = mask_k.sum()
        original_mask = mask_probs_sig[k] >= mask_threshold
        original_area = original_mask.sum()
        final_mask = mask_k & original_mask
        final_area = final_mask.sum()

        if mask_k_area == 0 or original_area == 0 or final_area == 0:
            continue

        area_ratio = mask_k_area / original_area
        if area_ratio <= overlap_mask_area_threshold:
            continue

        if stuff_classes and pred_class in stuff_classes:
            if pred_class in stuff_memory:
                segmentation[final_mask] = stuff_memory[pred_class]
                continue
            else:
                stuff_memory[pred_class] = current_id

        segmentation[final_mask] = current_id
        name = (
            label_names[pred_class]
            if pred_class < len(label_names)
            else f"class_{pred_class}"
        )
        segments_info.append(
            {
                "id": current_id,
                "label_id": pred_class,
                "label_name": name,
                "score": round(float(pred_scores[k]), 6),
            }
        )
        current_id += 1

    return {"segmentation": segmentation, "segments_info": segments_info}


def eomt_post_process_semantic(
    outputs: Dict[str, keras.KerasTensor],
    target_size: Tuple[int, int],
    model_size: int = 640,
    label_names: Optional[List[str]] = None,
) -> Dict:
    """Post-process EoMT outputs into semantic segmentation predictions.

    Uses Keras 3 ops for softmax/sigmoid/einsum and numpy for final assembly.

    Args:
        outputs: Model output dict with ``"class_logits"`` and ``"mask_logits"``.
        target_size: Original image ``(height, width)`` for upsampling.
        model_size: Square input size used during preprocessing (default 640).
        label_names: Class name list.

    Returns:
        Dict with:
            - ``"segmentation"``: Integer array of shape ``(H, W)`` with class IDs.
            - ``"class_names"``: List of unique class names present in the segmentation.
    """
    if label_names is None:
        label_names = COCO_PANOPTIC_133_CLASSES

    class_logits = outputs["class_logits"]
    mask_logits = outputs["mask_logits"]

    target_h, target_w = target_size

    mask_resized = unpad_and_resize_masks(mask_logits, model_size, target_h, target_w)

    masks_classes = keras.ops.softmax(class_logits[0], axis=-1)[:, :-1]
    masks_probs = keras.ops.sigmoid(
        keras.ops.convert_to_tensor(mask_resized[0], dtype="float32")
    )

    seg_logits = keras.ops.einsum("qc,qhw->chw", masks_classes, masks_probs)
    segmentation = keras.ops.convert_to_numpy(keras.ops.argmax(seg_logits, axis=0))
    segmentation = segmentation.astype(np.int32)

    unique_ids = np.unique(segmentation)
    class_names = [
        label_names[i] if i < len(label_names) else f"class_{i}" for i in unique_ids
    ]

    return {"segmentation": segmentation, "class_names": class_names}


def eomt_post_process_instance(
    outputs: Dict[str, keras.KerasTensor],
    target_size: Tuple[int, int],
    threshold: float = 0.5,
    model_size: int = 640,
    label_names: Optional[List[str]] = None,
) -> Dict:
    """Post-process EoMT outputs into instance segmentation predictions.

    Uses Keras 3 ops for softmax/sigmoid and numpy for segment iteration.

    Args:
        outputs: Model output dict with ``"class_logits"`` and ``"mask_logits"``.
        target_size: Original image ``(height, width)`` for upsampling.
        threshold: Minimum score to keep an instance.
        model_size: Square input size used during preprocessing (default 640).
        label_names: Class name list.

    Returns:
        Dict with:
            - ``"segmentation"``: Integer array of shape ``(H, W)`` with instance IDs.
            - ``"segments_info"``: List of dicts per instance.
    """
    if label_names is None:
        label_names = COCO_PANOPTIC_133_CLASSES

    class_logits = outputs["class_logits"]
    mask_logits = outputs["mask_logits"]

    target_h, target_w = target_size

    mask_resized = unpad_and_resize_masks(mask_logits, model_size, target_h, target_w)

    class_probs = keras.ops.convert_to_numpy(
        keras.ops.softmax(class_logits[0], axis=-1)
    )[:, :-1]
    scores = np.max(class_probs, axis=-1)
    pred_classes = np.argmax(class_probs, axis=-1)

    mask_np = mask_resized[0]
    pred_masks = (mask_np > 0).astype(np.float32)
    mask_probs_sig = keras.ops.convert_to_numpy(
        keras.ops.sigmoid(keras.ops.convert_to_tensor(mask_np, dtype="float32"))
    )

    mask_scores = (mask_probs_sig * pred_masks).reshape(pred_masks.shape[0], -1).sum(1)
    mask_scores = mask_scores / (
        pred_masks.reshape(pred_masks.shape[0], -1).sum(1) + 1e-6
    )
    pred_scores = scores * mask_scores

    segmentation = np.full(target_size, -1, dtype=np.int32)
    segments_info = []
    current_id = 0

    for j in range(pred_scores.shape[0]):
        score = float(pred_scores[j])
        if pred_masks[j].sum() > 0 and score >= threshold:
            segmentation[pred_masks[j] == 1] = current_id
            name = (
                label_names[int(pred_classes[j])]
                if int(pred_classes[j]) < len(label_names)
                else f"class_{int(pred_classes[j])}"
            )
            segments_info.append(
                {
                    "id": current_id,
                    "label_id": int(pred_classes[j]),
                    "label_name": name,
                    "score": round(score, 6),
                }
            )
            current_id += 1

    return {"segmentation": segmentation, "segments_info": segments_info}
