from typing import Dict, List, Optional, Tuple

import keras
from keras import ops

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.image_util import get_data_format, load_image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@keras.saving.register_keras_serializable(package="zeromodels")
class MaskFormerImageProcessor(BaseImageProcessor):
    """Preprocess images for MaskFormer.

    Resizes the longest edge to ``target_size``, pads to a square,
    rescales to ``[0, 1]``, and applies ImageNet normalization. Uses pure
    Keras 3 ops for all tensor operations. Accepts a path, a PIL image, or an
    array (a 4D array is treated as a single-image batch).

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

    def __call__(self, image) -> Dict[str, keras.KerasTensor]:
        return self.call(image)

    def call(self, image) -> Dict[str, keras.KerasTensor]:
        if hasattr(image, "ndim") and image.ndim == 4:
            image = image[0]
        image = load_image(image)

        h, w = image.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        image = ops.convert_to_tensor(image, dtype="float32")
        image = ops.expand_dims(image, axis=0)
        image = ops.image.resize(image, (new_h, new_w), interpolation="bilinear")
        image = image / 255.0

        padded = ops.zeros((1, self.target_size, self.target_size, 3), dtype="float32")
        padded = ops.slice_update(padded, (0, 0, 0, 0), image)

        mean = ops.reshape(
            ops.convert_to_tensor(self.image_mean, dtype="float32"),
            (1, 1, 1, 3),
        )
        std = ops.reshape(
            ops.convert_to_tensor(self.image_std, dtype="float32"),
            (1, 1, 1, 3),
        )
        padded = (padded - mean) / std

        if get_data_format(self.data_format) == "channels_first":
            padded = ops.transpose(padded, (0, 3, 1, 2))

        return {"pixel_values": padded}

    def post_process_semantic_segmentation(
        self,
        outputs: Dict[str, keras.KerasTensor],
        target_sizes: Optional[List[Tuple[int, int]]] = None,
        label_names: Optional[List[str]] = None,
    ) -> list:
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


def unpad_and_resize_masks(mask_logits, model_size: int, target_h: int, target_w: int):
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
        Tensor of shape ``(1, Q, target_h, target_w)``.
    """
    scale = model_size / max(target_h, target_w)
    resized_h, resized_w = int(target_h * scale), int(target_w * scale)
    mask_logits = ops.convert_to_tensor(mask_logits, dtype="float32")

    mask_4d = ops.transpose(mask_logits, (0, 2, 3, 1))
    mask_full = ops.image.resize(
        mask_4d, (model_size, model_size), interpolation="bilinear"
    )
    mask_full = ops.transpose(mask_full, (0, 3, 1, 2))
    mask_cropped = mask_full[:, :, :resized_h, :resized_w]

    mask_cropped_4d = ops.transpose(mask_cropped, (0, 2, 3, 1))
    mask_final = ops.image.resize(
        mask_cropped_4d, (target_h, target_w), interpolation="bilinear"
    )
    mask_final = ops.transpose(mask_final, (0, 3, 1, 2))
    return mask_final


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
) -> list:
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
    results = []
    for i in range(batch_size):
        if target_sizes is None:
            target_h, target_w = model_size, model_size
        else:
            target_h, target_w = target_sizes[i]

        mask_resized = unpad_and_resize_masks(
            mask_logits[i : i + 1], model_size, target_h, target_w
        )
        masks_classes = ops.softmax(class_logits[i], axis=-1)[:, :-1]
        masks_probs = ops.sigmoid(mask_resized[0])
        seg_logits = ops.einsum("qc,qhw->chw", masks_classes, masks_probs)
        seg = ops.convert_to_numpy(ops.argmax(seg_logits, axis=0)).astype("int32")
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
    num_queries = int(class_logits.shape[1])
    target_h, target_w = target_size

    mask_logits_resized = unpad_and_resize_masks(
        mask_logits, model_size, target_h, target_w
    )
    scores = ops.softmax(class_logits[0], axis=-1)
    pred_scores = ops.max(scores, axis=-1)
    pred_labels = ops.argmax(scores, axis=-1)
    mask_probs_sig = ops.sigmoid(mask_logits_resized[0])

    keep = ops.logical_and(
        ops.not_equal(pred_labels, num_classes),
        ops.greater(pred_scores, threshold),
    )
    if int(ops.sum(ops.cast(keep, "int32"))) == 0:
        return {
            "segmentation": ops.convert_to_numpy(
                ops.full(target_size, -1, dtype="int32")
            ),
            "segments_info": [],
        }
    weighted = ops.reshape(pred_scores, (-1, 1, 1)) * mask_probs_sig
    keep_bias = ops.where(
        keep,
        ops.zeros_like(pred_scores),
        ops.full_like(pred_scores, float("-inf")),
    )
    weighted = weighted + ops.reshape(keep_bias, (-1, 1, 1))
    mask_labels = ops.argmax(weighted, axis=0)

    segmentation = ops.full(target_size, -1, dtype="int32")
    segments_info: List[Dict] = []
    current_id = 0
    stuff_memory: Dict[int, int] = {}

    for k in range(num_queries):
        if not bool(keep[k]):
            continue
        pred_class = int(pred_labels[k])
        mask_k = ops.equal(mask_labels, k)
        mask_k_area = int(ops.sum(ops.cast(mask_k, "int32")))
        original_mask = ops.greater_equal(mask_probs_sig[k], mask_threshold)
        original_area = int(ops.sum(ops.cast(original_mask, "int32")))
        final_mask = ops.logical_and(mask_k, original_mask)
        final_area = int(ops.sum(ops.cast(final_mask, "int32")))

        if mask_k_area == 0 or original_area == 0 or final_area == 0:
            continue
        area_ratio = mask_k_area / original_area
        if area_ratio <= overlap_mask_area_threshold:
            continue

        if stuff_classes and pred_class in stuff_classes:
            if pred_class in stuff_memory:
                segmentation = ops.where(
                    final_mask,
                    ops.cast(stuff_memory[pred_class], "int32"),
                    segmentation,
                )
                continue
            stuff_memory[pred_class] = current_id

        segmentation = ops.where(
            final_mask, ops.cast(current_id, "int32"), segmentation
        )
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

    return {
        "segmentation": ops.convert_to_numpy(segmentation),
        "segments_info": segments_info,
    }
