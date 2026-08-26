from typing import Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from PIL import Image

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.image_util import get_data_format, load_image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@keras.saving.register_keras_serializable(package="zeromodels")
class MaskFormerImageProcessor(BaseImageProcessor):
    """Preprocess images for MaskFormer.

    Resizes the longest edge to ``target_size``, pads to a square,
    rescales to ``[0, 1]``, and applies ImageNet normalization. Uses pure
    Keras 3 ops for all tensor operations.

    Args:
        target_size: Target square edge length (matches the model's
            ``image_size``).
        image_mean: Per-channel mean for normalization.
        image_std: Per-channel std for normalization.
        data_format: ``"channels_first"`` / ``"channels_last"``; ``None``
            resolves to ``keras.config.image_data_format()``.
    """

    def __init__(
        self,
        target_size: Optional[int] = None,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        data_format: Optional[str] = None,
        variant: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.variant = variant
        self.target_size = (
            target_size if target_size is not None else self.variant_size(variant)
        )
        self.image_mean = image_mean if image_mean is not None else IMAGENET_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STD
        self.data_format = data_format

    @staticmethod
    def variant_size(variant):
        """Default square side (512).

        The COCO checkpoints build at 384 and the ADE ones at 512. The
        per-variant resolution is no longer kept in the package: it travels in
        the repo's ``zm_preprocessor.json`` and is applied when loading via
        ``from_weights("zeromodels/<variant>")``. Direct construction without
        an explicit ``target_size`` falls back to 512.
        """
        return 512

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
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

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

    def post_process_semantic_segmentation(
        self,
        outputs: Dict[str, keras.KerasTensor],
        target_sizes: Optional[List[Tuple[int, int]]] = None,
        label_names: Optional[List[str]] = None,
    ) -> List[np.ndarray]:
        return maskformer_post_process_semantic(
            outputs,
            target_sizes=target_sizes,
            model_size=self.target_size,
            label_names=label_names,
        )

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
        return maskformer_post_process_panoptic(
            outputs,
            target_size=target_size,
            threshold=threshold,
            mask_threshold=mask_threshold,
            overlap_mask_area_threshold=overlap_mask_area_threshold,
            model_size=self.target_size,
            stuff_classes=stuff_classes,
            label_names=label_names,
        )


def unpad_and_resize_masks(
    mask_logits, model_size: int, target_h: int, target_w: int
) -> np.ndarray:
    """Upscale mask logits, remove padding, and resize to the original image.

    The model predicts masks for a square ``model_size`` input that the
    processor produced by aspect-ratio resize + bottom/right padding. This
    upsamples the masks to ``model_size``, crops away the padded region, then
    resizes to the true ``(target_h, target_w)``.

    Args:
        mask_logits: Mask logits of shape ``(1, Q, h, w)``.
        model_size: Square edge length the model was run at.
        target_h: Original (unpadded) image height.
        target_w: Original (unpadded) image width.

    Returns:
        Numpy array of shape ``(1, Q, target_h, target_w)``.
    """
    scale = model_size / max(target_h, target_w)
    resized_h, resized_w = int(target_h * scale), int(target_w * scale)
    mask_logits = keras.ops.convert_to_tensor(mask_logits, dtype="float32")

    mask_4d = keras.ops.transpose(mask_logits, (0, 2, 3, 1))
    mask_full = keras.ops.image.resize(
        mask_4d, (model_size, model_size), interpolation="bilinear"
    )
    mask_full = keras.ops.transpose(mask_full, (0, 3, 1, 2))
    mask_cropped = mask_full[:, :, :resized_h, :resized_w]

    mask_cropped_4d = keras.ops.transpose(mask_cropped, (0, 2, 3, 1))
    mask_final = keras.ops.image.resize(
        mask_cropped_4d, (target_h, target_w), interpolation="bilinear"
    )
    mask_final = keras.ops.transpose(mask_final, (0, 3, 1, 2))
    return keras.ops.convert_to_numpy(mask_final)


def default_label_names(num_classes):
    """Pick a label set from the head width when the caller supplies none.

    The class count identifies the training set unambiguously across the
    checkpoints in this library, and every one of these post-processors is
    shared by MaskFormer, Mask2Former and OneFormer, so without this they all
    fall back to ``class_57`` style placeholders.
    """
    from zeromodels.utils.labels_util import (
        ADE20K_150_CLASSES,
        CITYSCAPES_19_CLASSES,
        COCO_PANOPTIC_133_CLASSES,
    )

    # class_queries_logits carries a trailing "no object" column, so the head
    # is one wider than the label list it corresponds to.
    return {
        len(COCO_PANOPTIC_133_CLASSES) + 1: COCO_PANOPTIC_133_CLASSES,
        len(ADE20K_150_CLASSES) + 1: ADE20K_150_CLASSES,
        len(CITYSCAPES_19_CLASSES) + 1: CITYSCAPES_19_CLASSES,
    }.get(num_classes)


def maskformer_post_process_semantic(
    outputs: Dict[str, keras.KerasTensor],
    target_sizes: Optional[List[Tuple[int, int]]] = None,
    model_size: int = 512,
    label_names: Optional[List[str]] = None,
) -> List[np.ndarray]:
    """Fuse per-query class and mask predictions into semantic label maps.

    For each image, softmaxes the class logits (dropping the no-object class),
    sigmoids the resized masks, combines them (``qc, qhw -> chw``), and takes
    the per-pixel argmax over classes.

    Args:
        outputs: Model output dict with ``class_queries_logits`` and
            ``masks_queries_logits``.
        target_sizes: Optional per-image ``(height, width)`` outputs; defaults
            to ``model_size`` square.
        model_size: Square edge length the model was run at.
        label_names: Optional class names (unused; kept for API parity).

    Returns:
        List of ``(H, W)`` integer label maps, one per image.
    """
    class_logits = outputs["class_queries_logits"]
    mask_logits = outputs["masks_queries_logits"]

    batch_size = class_logits.shape[0]
    results: List[np.ndarray] = []
    for i in range(batch_size):
        if target_sizes is None:
            target_h, target_w = model_size, model_size
        else:
            target_h, target_w = target_sizes[i]

        mask_resized = unpad_and_resize_masks(
            mask_logits[i : i + 1], model_size, target_h, target_w
        )
        masks_classes = keras.ops.softmax(class_logits[i], axis=-1)[:, :-1]
        masks_probs = keras.ops.sigmoid(
            keras.ops.convert_to_tensor(mask_resized[0], dtype="float32")
        )
        seg_logits = keras.ops.einsum("qc,qhw->chw", masks_classes, masks_probs)
        seg = keras.ops.convert_to_numpy(keras.ops.argmax(seg_logits, axis=0)).astype(
            np.int32
        )
        results.append(seg)
    return results


def maskformer_post_process_panoptic(
    outputs: Dict[str, keras.KerasTensor],
    target_size: Tuple[int, int],
    threshold: float = 0.8,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    model_size: int = 512,
    stuff_classes: Optional[List[int]] = None,
    label_names: Optional[List[str]] = None,
) -> Dict:
    """Build a single-image panoptic segmentation from raw model outputs.

    Keeps confident, non-no-object queries, assigns each pixel to its
    highest-scoring kept query, drops segments whose surviving area falls below
    ``overlap_mask_area_threshold``, and merges "stuff" classes into one segment
    each.

    Args:
        outputs: Model output dict with ``class_queries_logits`` and
            ``masks_queries_logits``.
        target_size: ``(height, width)`` of the output panoptic map.
        threshold: Minimum query confidence to keep a predicted segment.
        mask_threshold: Probability cutoff for binarising each mask.
        overlap_mask_area_threshold: Minimum kept-area fraction for a segment
            after resolving overlaps.
        model_size: Square edge length the model was run at.
        stuff_classes: Class ids treated as amorphous "stuff".
        label_names: Optional class names attached to each segment's info.

    Returns:
        Dict with the panoptic ``segmentation`` map and per-segment info list.
    """
    class_logits = outputs["class_queries_logits"]
    mask_logits = outputs["masks_queries_logits"]
    if label_names is None:
        label_names = default_label_names(int(class_logits.shape[-1]))

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
    segments_info: List[Dict] = []
    current_id = 0
    stuff_memory: Dict[int, int] = {}

    for k in range(pred_labels.shape[0]):
        pred_class = int(pred_labels[k])
        mask_k = mask_labels == k
        mask_k_area = int(mask_k.sum())
        original_mask = mask_probs_sig[k] >= mask_threshold
        original_area = int(original_mask.sum())
        final_mask = mask_k & original_mask
        final_area = int(final_mask.sum())

        if mask_k_area == 0 or original_area == 0 or final_area == 0:
            continue
        area_ratio = mask_k_area / original_area
        if area_ratio <= overlap_mask_area_threshold:
            continue

        if stuff_classes and pred_class in stuff_classes:
            if pred_class in stuff_memory:
                segmentation[final_mask] = stuff_memory[pred_class]
                continue
            stuff_memory[pred_class] = current_id

        segmentation[final_mask] = current_id
        name = (
            label_names[pred_class]
            if label_names is not None and pred_class < len(label_names)
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
