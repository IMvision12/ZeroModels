import math
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from PIL import Image

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.image_util import get_data_format, load_image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_preprocess_shape(
    old_h: int, old_w: int, long_side_length: int
) -> Tuple[int, int]:
    """Compute resized (h, w) that scales the longest side to ``long_side_length``."""
    scale = long_side_length / max(old_h, old_w)
    new_h = int(old_h * scale + 0.5)
    new_w = int(old_w * scale + 0.5)
    return new_h, new_w


@keras.saving.register_keras_serializable(package="zeromodels")
class SAMImageProcessor(BaseImageProcessor):
    """Preprocess images for SAM inference.

    Resizes the image so its longest side equals ``target_length``,
    applies ImageNet normalization, and pads to a square. Also prepares
    default prompt inputs (empty points, labels, boxes) so the model
    can run with just an image.

    Args:
        target_length: Target size for the longest side (default 1024).
        image_mean: Per-channel mean for normalization.
        image_std: Per-channel std for normalization.
        data_format: ``"channels_first"`` / ``"channels_last"``;
            ``None`` resolves to ``keras.backend.image_data_format()``.
    """

    def __init__(
        self,
        target_length: int = 1024,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        data_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_length = target_length
        self.image_mean = image_mean if image_mean is not None else IMAGENET_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STD
        self.data_format = data_format

    def __call__(
        self, image: Union[str, np.ndarray, "Image.Image"]
    ) -> Dict[str, "keras.KerasTensor"]:
        return self.call(image)

    def call(
        self, image: Union[str, np.ndarray, "Image.Image"]
    ) -> Dict[str, "keras.KerasTensor"]:
        if isinstance(image, np.ndarray) and image.ndim == 4:
            image = image[0]
        image = load_image(image).astype(np.float32)

        orig_h, orig_w = image.shape[:2]
        new_h, new_w = get_preprocess_shape(orig_h, orig_w, self.target_length)

        image = keras.ops.convert_to_tensor(image, dtype="float32")
        image = keras.ops.expand_dims(image, axis=0)
        image = keras.ops.image.resize(image, (new_h, new_w), interpolation="bilinear")

        image = image / 255.0

        mean = keras.ops.reshape(
            keras.ops.convert_to_tensor(self.image_mean, dtype="float32"),
            (1, 1, 1, 3),
        )
        std = keras.ops.reshape(
            keras.ops.convert_to_tensor(self.image_std, dtype="float32"),
            (1, 1, 1, 3),
        )
        image = (image - mean) / std

        padded = keras.ops.zeros(
            (1, self.target_length, self.target_length, 3), dtype="float32"
        )
        padded = keras.ops.slice_update(padded, (0, 0, 0, 0), image)

        if get_data_format(self.data_format) == "channels_first":
            padded = keras.ops.transpose(padded, (0, 3, 1, 2))

        return {
            "pixel_values": padded,
            "original_size": (orig_h, orig_w),
            "reshaped_size": (new_h, new_w),
        }

    def post_process_masks(
        self, pred_masks, original_size, reshaped_size, target_length=None
    ):
        return sam_post_process_masks(
            pred_masks,
            original_size=original_size,
            reshaped_size=reshaped_size,
            target_length=target_length
            if target_length is not None
            else self.target_length,
        )


@keras.saving.register_keras_serializable(package="zeromodels")
class SAMProcessor(SAMImageProcessor):
    """Preprocess an image plus optional prompts for SAM inference.

    Extends :class:`SAMImageProcessor` by also encoding prompt inputs
    (points, labels, boxes) into the tensor format the model expects.
    Constructor kwargs match :class:`SAMImageProcessor`; prompts are
    passed at call time.

    Example:
        ```python
        from zeromodels.models.sam import SAMProcessor, SAMPromptableSegment

        model = SAMPromptableSegment.from_weights("zeromodels/sam_vit_huge")
        proc = SAMProcessor.from_weights("zeromodels/sam_vit_huge")
        inputs = proc(
            "photo.jpg",
            input_points=np.array([[[450, 600]]]),
            input_labels=np.array([[1]]),
        )
        outputs = model(inputs)
        ```
    """

    def __call__(
        self,
        image: Union[str, np.ndarray, "Image.Image"],
        input_points: Optional[np.ndarray] = None,
        input_labels: Optional[np.ndarray] = None,
        input_boxes: Optional[np.ndarray] = None,
    ) -> Dict[str, "keras.KerasTensor"]:
        return self.call(image, input_points, input_labels, input_boxes)

    def call(
        self,
        image: Union[str, np.ndarray, "Image.Image"],
        input_points: Optional[np.ndarray] = None,
        input_labels: Optional[np.ndarray] = None,
        input_boxes: Optional[np.ndarray] = None,
    ) -> Dict[str, "keras.KerasTensor"]:
        result = SAMImageProcessor.call(self, image)

        orig_h, orig_w = result["original_size"]
        scale = self.target_length / max(orig_h, orig_w)

        if input_points is not None:
            points = np.array(input_points, dtype=np.float64) * scale
            points = keras.ops.convert_to_tensor(points, dtype="float32")
            if keras.ops.ndim(points) == 2:
                points = keras.ops.expand_dims(points, axis=0)
            if keras.ops.ndim(points) == 3:
                points = keras.ops.expand_dims(points, axis=0)
            result["input_points"] = points

        if input_labels is not None:
            labels = keras.ops.convert_to_tensor(input_labels, dtype="int32")
            if keras.ops.ndim(labels) == 1:
                labels = keras.ops.expand_dims(labels, axis=0)
            if keras.ops.ndim(labels) == 2:
                labels = keras.ops.expand_dims(labels, axis=0)
            result["input_labels"] = labels

        if input_boxes is not None:
            boxes = np.array(input_boxes, dtype=np.float64) * scale
            boxes = keras.ops.convert_to_tensor(boxes, dtype="float32")
            if keras.ops.ndim(boxes) == 1:
                boxes = keras.ops.expand_dims(boxes, axis=0)
            if keras.ops.ndim(boxes) == 2:
                boxes = keras.ops.expand_dims(boxes, axis=0)
            result["input_boxes"] = boxes
            result["has_boxes_input"] = keras.ops.ones((1, 1), dtype="float32")
            if "input_points" not in result:
                result["input_points"] = keras.ops.zeros((1, 1, 1, 2), dtype="float32")
                result["input_labels"] = -keras.ops.ones((1, 1, 1), dtype="int32")

        return result


def sam_post_process_masks(
    pred_masks: "keras.KerasTensor",
    original_size: Tuple[int, int],
    reshaped_size: Tuple[int, int],
    target_length: int = 1024,
) -> "keras.KerasTensor":
    """Post-process predicted masks to original image resolution.

    Upsamples low-resolution masks to the model's input size, crops out the
    padding region, and resizes to the original image dimensions.

    Args:
        pred_masks: Predicted masks of shape
            ``(batch, point_batch, num_masks, mask_h, mask_w)``.
        original_size: Original image ``(height, width)``.
        reshaped_size: Resized image ``(height, width)`` before padding.
        target_length: Model input resolution (default 1024).

    Returns:
        Keras tensor of masks with shape
        ``(batch, point_batch, num_masks, orig_h, orig_w)``.

    Example:
        ```python
        masks_np = sam_post_process_masks(
            outputs["pred_masks"],
            original_size=inputs["original_size"],
            reshaped_size=inputs["reshaped_size"],
        )
        ```
    """
    pred_masks = keras.ops.convert_to_tensor(pred_masks, dtype="float32")

    batch = keras.ops.shape(pred_masks)[0]
    point_batch = keras.ops.shape(pred_masks)[1]
    num_masks = keras.ops.shape(pred_masks)[2]
    mask_h = keras.ops.shape(pred_masks)[3]
    mask_w = keras.ops.shape(pred_masks)[4]

    masks_4d = keras.ops.reshape(
        pred_masks, (batch * point_batch * num_masks, mask_h, mask_w, 1)
    )

    masks_upsampled = keras.ops.image.resize(
        masks_4d, (target_length, target_length), interpolation="bilinear"
    )

    new_h, new_w = reshaped_size
    masks_cropped = masks_upsampled[:, :new_h, :new_w, :]

    orig_h, orig_w = original_size
    masks_final = keras.ops.image.resize(
        masks_cropped, (orig_h, orig_w), interpolation="bilinear"
    )

    masks_final = keras.ops.reshape(
        masks_final, (batch, point_batch, num_masks, orig_h, orig_w)
    )

    return masks_final


def build_point_grid(n_per_side: int):
    """Regular ``n_per_side × n_per_side`` point grid in ``[0, 1] × [0, 1]``."""
    offset = 1.0 / (2 * n_per_side)
    points_one_side = keras.ops.linspace(offset, 1.0 - offset, n_per_side)
    points_one_side = keras.ops.cast(points_one_side, "float32")
    points_x = keras.ops.tile(
        keras.ops.expand_dims(points_one_side, 0), (n_per_side, 1)
    )
    points_y = keras.ops.tile(
        keras.ops.expand_dims(points_one_side, 1), (1, n_per_side)
    )
    points = keras.ops.stack([points_x, points_y], axis=-1)
    return keras.ops.reshape(points, (-1, 2))


def generate_per_layer_crops(
    crop_n_layers: int, overlap_ratio: float, original_size: Tuple[int, int]
) -> Tuple[List[List[int]], List[int]]:
    """Hierarchical crop boxes in XYXY, plus each crop's layer index.

    Layer 0 is the full image. Each additional layer subdivides the image
    into ``2**(i+1) × 2**(i+1)`` overlapping crops.
    """
    crop_boxes: List[List[int]] = []
    layer_idxs: List[int] = []
    im_height, im_width = original_size
    short_side = min(im_height, im_width)

    crop_boxes.append([0, 0, im_width, im_height])
    layer_idxs.append(0)

    for i_layer in range(crop_n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_width = int(
            math.ceil((overlap * (n_crops_per_side - 1) + im_width) / n_crops_per_side)
        )
        crop_height = int(
            math.ceil((overlap * (n_crops_per_side - 1) + im_height) / n_crops_per_side)
        )

        crop_box_x0 = [int((crop_width - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [
            int((crop_height - overlap) * i) for i in range(n_crops_per_side)
        ]

        for left, top in product(crop_box_x0, crop_box_y0):
            box = [
                left,
                top,
                min(left + crop_width, im_width),
                min(top + crop_height, im_height),
            ]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs


def normalize_coordinates(
    target_size: int,
    coords,
    original_size: Tuple[int, int],
    is_bounding_box: bool = False,
):
    """Rescale coordinates from the original image frame to the model input frame."""
    old_height, old_width = original_size

    scale = target_size * 1.0 / max(old_height, old_width)
    new_height = int(old_height * scale + 0.5)
    new_width = int(old_width * scale + 0.5)

    coords = keras.ops.cast(coords, "float32")

    if is_bounding_box:
        coords = keras.ops.reshape(coords, (-1, 2, 2))

    scale_xy = keras.ops.convert_to_tensor(
        [new_width / old_width, new_height / old_height], dtype="float32"
    )
    coords = coords * scale_xy

    if is_bounding_box:
        coords = keras.ops.reshape(coords, (-1, 4))

    return coords


def generate_crop_images(
    crop_boxes: List[List[int]],
    image,
    points_grid: List,
    layer_idxs: List[int],
    target_size: int,
    original_size: Tuple[int, int],
):
    """Slice the image into crops and compute per-crop point prompts."""
    cropped_images = []
    total_points_per_crop = []
    for i, crop_box in enumerate(crop_boxes):
        left, top, right, bottom = crop_box
        cropped_im = image[top:bottom, left:right, :]
        cropped_images.append(cropped_im)

        crop_h = bottom - top
        crop_w = right - left
        points_scale = keras.ops.convert_to_tensor([[crop_w, crop_h]], dtype="float32")

        points = points_grid[layer_idxs[i]] * points_scale
        normalized_points = normalize_coordinates(target_size, points, original_size)
        total_points_per_crop.append(normalized_points)

    return cropped_images, total_points_per_crop


def generate_crop_boxes(
    image: Union[np.ndarray, "Image.Image", str],
    target_size: int = 1024,
    crop_n_layers: int = 0,
    overlap_ratio: float = 512 / 1500,
    points_per_crop: int = 32,
    crop_n_points_downscale_factor: int = 1,
) -> Dict[str, Any]:
    """Generate hierarchical crops and per-crop point prompts for AMG.

    Reproduces ``SamProcessor.generate_crop_boxes`` behaviour.

    Args:
        image: Input image as a file path, numpy array ``(H, W, 3)``,
            or PIL Image.
        target_size: Model input size (longest side). Defaults to ``1024``.
        crop_n_layers: If ``>0``, additional crop layers are produced.
            Layer ``i`` contributes ``(2**(i+1))**2`` crops. Defaults to ``0``
            (full image only).
        overlap_ratio: Fraction of the short side by which crops overlap
            in layer 1. Defaults to ``512/1500`` (the original default).
        points_per_crop: Number of points per side in the layer-0 grid.
            Defaults to ``32`` (1024 total points for layer 0).
        crop_n_points_downscale_factor: Points-per-side in layer ``n`` is
            ``points_per_crop / factor**n``. Defaults to ``1``.

    Returns:
        Dict with:
            - ``"crop_boxes"``: ``(num_crops, 4)`` float32 XYXY in
              original-image pixel coordinates.
            - ``"points_per_crop"``: ``(num_crops, num_points, 2)`` float32
              point coordinates in the ``target_size`` model frame.
            - ``"cropped_images"``: list of ``num_crops`` numpy arrays at
              original-image pixel resolution (variable shapes).
            - ``"input_labels"``: ``(num_crops, num_points)`` int32 labels
              (all ones: foreground).
            - ``"original_size"``: ``(orig_h, orig_w)``.
    """
    if isinstance(image, np.ndarray) and image.ndim == 4:
        image = image[0]
    image = load_image(image).astype(np.float32, copy=False)
    image = keras.ops.convert_to_tensor(image, dtype="float32")

    if keras.ops.ndim(image) != 3 or int(keras.ops.shape(image)[-1]) != 3:
        raise ValueError(
            f"Expected image shape (H, W, 3), got {keras.ops.shape(image)}"
        )

    original_size = (int(keras.ops.shape(image)[0]), int(keras.ops.shape(image)[1]))

    points_grid: List = []
    for i in range(crop_n_layers + 1):
        n_points = int(points_per_crop / (crop_n_points_downscale_factor**i))
        points_grid.append(build_point_grid(n_points))

    crop_boxes, layer_idxs = generate_per_layer_crops(
        crop_n_layers, overlap_ratio, original_size
    )

    cropped_images, point_grid_per_crop = generate_crop_images(
        crop_boxes, image, points_grid, layer_idxs, target_size, original_size
    )

    crop_boxes_tensor = keras.ops.convert_to_tensor(crop_boxes, dtype="float32")
    points_per_crop_tensor = keras.ops.cast(
        keras.ops.stack(point_grid_per_crop, axis=0), "float32"
    )
    input_labels_tensor = keras.ops.ones(
        keras.ops.shape(points_per_crop_tensor)[:2], dtype="int32"
    )

    return {
        "crop_boxes": crop_boxes_tensor,
        "points_per_crop": points_per_crop_tensor,
        "cropped_images": cropped_images,
        "input_labels": input_labels_tensor,
        "original_size": original_size,
    }


def compute_stability_score(
    masks, mask_threshold: float, stability_score_offset: float
):
    """Ratio of mask areas at two thresholds: higher means more stable.

    ``masks`` is a float tensor; values above ``mask_threshold`` are
    considered foreground. Stability is defined as
    ``|mask > (t + offset)| / |mask > (t - offset)|``.
    """
    high = keras.ops.cast(masks > (mask_threshold + stability_score_offset), "int32")
    low = keras.ops.cast(masks > (mask_threshold - stability_score_offset), "int32")
    intersections = keras.ops.sum(high, axis=(-2, -1))
    unions = keras.ops.sum(low, axis=(-2, -1))
    unions = keras.ops.maximum(unions, 1)
    return keras.ops.cast(intersections, "float32") / keras.ops.cast(unions, "float32")


def batched_mask_to_box(masks):
    """Compute XYXY boxes for binary masks.

    Args:
        masks: bool tensor with shape ``(..., H, W)``.

    Returns:
        Float tensor with shape ``(..., 4)``. Empty masks produce
        ``[0, 0, 0, 0]``.
    """
    shape = keras.ops.shape(masks)
    if int(keras.ops.size(masks)) == 0:
        out_shape = tuple(int(s) for s in shape[:-2]) + (4,)
        return keras.ops.zeros(out_shape, dtype="float32")

    height = int(shape[-2])
    width = int(shape[-1])

    in_height = keras.ops.any(masks, axis=-1)  # (..., H)
    in_width = keras.ops.any(masks, axis=-2)  # (..., W)

    in_height_int = keras.ops.cast(in_height, "int32")
    in_width_int = keras.ops.cast(in_width, "int32")

    h_range = keras.ops.arange(height, dtype="int32")
    w_range = keras.ops.arange(width, dtype="int32")

    in_h_coords = in_height_int * h_range
    bottom_edges = keras.ops.max(in_h_coords, axis=-1)
    in_h_coords_for_top = in_h_coords + height * (1 - in_height_int)
    top_edges = keras.ops.min(in_h_coords_for_top, axis=-1)

    in_w_coords = in_width_int * w_range
    right_edges = keras.ops.max(in_w_coords, axis=-1)
    in_w_coords_for_left = in_w_coords + width * (1 - in_width_int)
    left_edges = keras.ops.min(in_w_coords_for_left, axis=-1)

    empty = keras.ops.logical_or(right_edges < left_edges, bottom_edges < top_edges)
    out = keras.ops.stack([left_edges, top_edges, right_edges, bottom_edges], axis=-1)
    out = keras.ops.cast(out, "float32")
    keep_mask = keras.ops.cast(keras.ops.logical_not(empty), "float32")
    out = out * keras.ops.expand_dims(keep_mask, axis=-1)
    return out


def is_box_near_crop_edge(
    boxes,
    crop_box: List[int],
    orig_box: List[int],
    atol: float = 20.0,
):
    """Mark boxes that touch a crop edge but not the original image edge.

    Boxes already in crop-local coordinates are translated into original
    image coordinates using ``crop_box``'s origin before the comparison.
    """
    crop_box_t = keras.ops.convert_to_tensor(crop_box, dtype="float32")
    orig_box_t = keras.ops.convert_to_tensor(orig_box, dtype="float32")

    left, top, _, _ = crop_box
    offset = keras.ops.convert_to_tensor([[left, top, left, top]], dtype="float32")
    if keras.ops.ndim(boxes) == 3:
        offset = keras.ops.expand_dims(offset, axis=1)
    boxes_shifted = keras.ops.cast(boxes, "float32") + offset

    near_crop = keras.ops.isclose(
        boxes_shifted, keras.ops.expand_dims(crop_box_t, 0), atol=atol, rtol=0.0
    )
    near_image = keras.ops.isclose(
        boxes_shifted, keras.ops.expand_dims(orig_box_t, 0), atol=atol, rtol=0.0
    )
    near_crop = keras.ops.logical_and(near_crop, keras.ops.logical_not(near_image))
    return keras.ops.any(near_crop, axis=-1)


def pad_masks(
    masks,
    crop_box: List[int],
    orig_height: int,
    orig_width: int,
):
    """Pad crop-local masks back to the original image frame."""
    left, top, right, bottom = crop_box
    if left == 0 and top == 0 and right == orig_width and bottom == orig_height:
        return masks

    pad_x = orig_width - (right - left)
    pad_y = orig_height - (bottom - top)
    rank = keras.ops.ndim(masks)
    pad_spec = [[0, 0]] * (rank - 2) + [
        [top, pad_y - top],
        [left, pad_x - left],
    ]
    return keras.ops.pad(masks, pad_spec, mode="constant", constant_values=0)


def filter_masks(
    masks,
    iou_scores,
    original_size: Tuple[int, int],
    cropped_box_image: List[int],
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    mask_threshold: float = 0.0,
    stability_score_offset: float = 1.0,
):
    """Filter per-crop predictions by quality, stability, and crop-edge.

    Args:
        masks: Float mask logits of shape
            ``(batch, point_batch, H, W)`` (or a flattened 3-D tensor).
        iou_scores: Predicted IoU scores of shape
            ``(batch, point_batch)``.
        original_size: Original image ``(height, width)``.
        cropped_box_image: Crop box in the original image frame
            ``[left, top, right, bottom]``.
        pred_iou_thresh: Drop masks with predicted IoU below this.
        stability_score_thresh: Drop masks with stability score below
            this.
        mask_threshold: Logit threshold used for binarization.
        stability_score_offset: Offset around ``mask_threshold`` when
            computing the stability score.

    Returns:
        ``(rle_masks, scores, boxes)`` where ``rle_masks`` is a list of
        uncompressed RLE dicts ``{"size", "counts"}``, ``scores`` is a
        ``(N,)`` float32 tensor, and ``boxes`` is ``(N, 4)`` XYXY in
        original-image coordinates.
    """
    original_height, original_width = original_size

    iou_scores = keras.ops.reshape(keras.ops.cast(iou_scores, "float32"), (-1,))
    masks_shape = keras.ops.shape(masks)
    mask_h = int(masks_shape[-2])
    mask_w = int(masks_shape[-1])
    masks = keras.ops.reshape(keras.ops.cast(masks, "float32"), (-1, mask_h, mask_w))

    if int(keras.ops.shape(masks)[0]) != int(keras.ops.shape(iou_scores)[0]):
        raise ValueError("masks and iou_scores must have the same batch size.")

    keep_mask = keras.ops.ones(keras.ops.shape(masks)[0], dtype="bool")
    if pred_iou_thresh > 0.0:
        keep_mask = keras.ops.logical_and(keep_mask, iou_scores > pred_iou_thresh)

    if stability_score_thresh > 0.0:
        stability_scores = compute_stability_score(
            masks, mask_threshold, stability_score_offset
        )
        keep_mask = keras.ops.logical_and(
            keep_mask, stability_scores > stability_score_thresh
        )

    keep_idx = keras.ops.nonzero(keep_mask)[0]
    scores = keras.ops.take(iou_scores, keep_idx, axis=0)
    masks = keras.ops.take(masks, keep_idx, axis=0)

    if int(keras.ops.shape(masks)[0]) == 0:
        return [], scores, keras.ops.zeros((0, 4), dtype="float32")

    masks_bool = masks > mask_threshold
    converted_boxes = batched_mask_to_box(masks_bool)

    near_edge = is_box_near_crop_edge(
        converted_boxes,
        cropped_box_image,
        [0, 0, original_width, original_height],
    )
    keep_edge_idx = keras.ops.nonzero(keras.ops.logical_not(near_edge))[0]
    scores = keras.ops.take(scores, keep_edge_idx, axis=0)
    masks_bool = keras.ops.take(masks_bool, keep_edge_idx, axis=0)
    converted_boxes = keras.ops.take(converted_boxes, keep_edge_idx, axis=0)

    masks_padded = pad_masks(
        masks_bool, cropped_box_image, original_height, original_width
    )
    rle_masks = mask_to_rle(masks_padded)

    return rle_masks, scores, converted_boxes


def mask_to_rle(input_mask) -> List[Dict[str, Any]]:
    """Encode a batch of binary masks as uncompressed COCO-style RLE.

    Args:
        input_mask: Bool tensor of shape ``(batch, H, W)``.

    Returns:
        List of ``batch`` dicts with keys ``"size": [H, W]`` and
        ``"counts": [int, ...]``.

    Notes:
        RLE encoding produces variable-length ``counts`` per mask, so
        the final Python list construction uses ``numpy`` after the
        tensor-level XOR + ``nonzero`` step.
    """
    if keras.ops.ndim(input_mask) == 2:
        input_mask = keras.ops.expand_dims(input_mask, 0)
    input_mask = keras.ops.cast(input_mask, "bool")

    shape = keras.ops.shape(input_mask)
    batch_size = int(shape[0])
    height = int(shape[1])
    width = int(shape[2])

    if batch_size == 0:
        return []

    # Fortran-order flatten: transpose to (batch, W, H) then flatten trailing dims.
    flat = keras.ops.reshape(
        keras.ops.transpose(input_mask, (0, 2, 1)),
        (batch_size, height * width),
    )
    diff = keras.ops.not_equal(flat[:, 1:], flat[:, :-1])

    # Variable-length per-row extraction of change indices: numpy boundary.
    flat_np = keras.ops.convert_to_numpy(flat)
    diff_np = keras.ops.convert_to_numpy(diff)

    out: List[Dict[str, Any]] = []
    for i in range(batch_size):
        row = flat_np[i]
        changes = np.nonzero(diff_np[i])[0] + 1
        if len(changes) == 0:
            if not row[0]:
                out.append({"size": [height, width], "counts": [height * width]})
            else:
                out.append({"size": [height, width], "counts": [0, height * width]})
            continue

        counts: List[int] = [] if not row[0] else [0]
        counts.append(int(changes[0]))
        if len(changes) > 1:
            counts.extend((changes[1:] - changes[:-1]).astype(int).tolist())
        counts.append(int(height * width - changes[-1]))
        out.append({"size": [height, width], "counts": counts})

    return out


def rle_to_mask(rle: Dict[str, Any]):
    """Decode an uncompressed RLE dict into a bool ``(H, W)`` keras tensor.

    The per-count Python loop is inherently sequential; the final
    reshape/transpose uses ``keras.ops``.
    """
    height, width = rle["size"]
    flat = np.empty(height * width, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        flat[idx : idx + count] = parity
        idx += count
        parity = not parity
    mask = keras.ops.convert_to_tensor(flat)
    mask = keras.ops.reshape(mask, (width, height))
    mask = keras.ops.transpose(mask, (1, 0))
    return mask


def box_iou_matrix(boxes_a, boxes_b):
    """Pairwise IoU between two sets of XYXY boxes."""
    area_a = keras.ops.maximum(boxes_a[:, 2] - boxes_a[:, 0], 0) * keras.ops.maximum(
        boxes_a[:, 3] - boxes_a[:, 1], 0
    )
    area_b = keras.ops.maximum(boxes_b[:, 2] - boxes_b[:, 0], 0) * keras.ops.maximum(
        boxes_b[:, 3] - boxes_b[:, 1], 0
    )

    lt = keras.ops.maximum(
        keras.ops.expand_dims(boxes_a[:, :2], 1),
        keras.ops.expand_dims(boxes_b[:, :2], 0),
    )
    rb = keras.ops.minimum(
        keras.ops.expand_dims(boxes_a[:, 2:], 1),
        keras.ops.expand_dims(boxes_b[:, 2:], 0),
    )
    wh = keras.ops.maximum(rb - lt, 0)
    inter = wh[..., 0] * wh[..., 1]
    union = keras.ops.expand_dims(area_a, 1) + keras.ops.expand_dims(area_b, 0) - inter
    return inter / keras.ops.maximum(union, 1e-10)


def batched_nms(boxes, scores, iou_threshold: float):
    """Single-class greedy NMS. Returns kept indices sorted by score desc.

    The outer greedy loop is inherently sequential: IoU computation
    uses ``keras.ops``, the loop state uses plain Python lists.
    """
    n = int(keras.ops.shape(boxes)[0])
    if n == 0:
        return keras.ops.zeros((0,), dtype="int64")

    order = keras.ops.argsort(-keras.ops.cast(scores, "float32"))
    order_np = keras.ops.convert_to_numpy(order).tolist()
    suppressed = [False] * n
    keep: List[int] = []

    for i in order_np:
        if suppressed[i]:
            continue
        keep.append(int(i))
        remaining = [j for j in range(n) if not suppressed[j]]
        if not remaining:
            break
        remaining_t = keras.ops.convert_to_tensor(remaining, dtype="int32")
        ious = box_iou_matrix(
            keras.ops.take(boxes, [int(i)], axis=0),
            keras.ops.take(boxes, remaining_t, axis=0),
        )[0]
        ious_np = keras.ops.convert_to_numpy(ious)
        for j_idx, j in enumerate(remaining):
            if ious_np[j_idx] > iou_threshold:
                suppressed[j] = True

    return keras.ops.convert_to_tensor(keep, dtype="int64")


def post_process_for_mask_generation(
    all_rle_masks: List[Dict[str, Any]],
    all_scores,
    all_boxes,
    crops_nms_thresh: float = 0.7,
):
    """Deduplicate AMG outputs across crops with NMS on the predicted boxes.

    Args:
        all_rle_masks: List of uncompressed RLE dicts.
        all_scores: ``(N,)`` float tensor of predicted IoU scores.
        all_boxes: ``(N, 4)`` XYXY bounding box tensor.
        crops_nms_thresh: IoU threshold for NMS. Defaults to ``0.7``.

    Returns:
        ``(masks, scores, rle_masks, boxes)``: ``masks`` is a list of
        bool ``(H, W)`` keras tensors decoded from the kept RLEs,
        ``scores`` and ``boxes`` are filtered tensors, and
        ``rle_masks`` is the filtered RLE list.
    """
    all_scores = keras.ops.cast(all_scores, "float32")
    all_boxes = keras.ops.cast(all_boxes, "float32")

    keep = batched_nms(all_boxes, all_scores, iou_threshold=crops_nms_thresh)

    kept_scores = keras.ops.take(all_scores, keep, axis=0)
    kept_boxes = keras.ops.take(all_boxes, keep, axis=0)
    keep_list = keras.ops.convert_to_numpy(keep).tolist()
    kept_rles = [all_rle_masks[int(i)] for i in keep_list]
    masks = [rle_to_mask(rle) for rle in kept_rles]

    return masks, kept_scores, kept_rles, kept_boxes


def preprocess_image_for_sam(
    image,
    target_length: int,
    image_mean: Tuple[float, ...],
    image_std: Tuple[float, ...],
    data_format: Optional[str] = None,
):
    """Resize (longest side → ``target_length``), normalize, pad to square.

    Returns ``(padded_image, original_size, reshaped_size)``.
    """
    image = keras.ops.cast(image, "float32")
    shape = keras.ops.shape(image)
    orig_h, orig_w = int(shape[0]), int(shape[1])
    new_h, new_w = get_preprocess_shape(orig_h, orig_w, target_length)

    tensor = keras.ops.expand_dims(image, axis=0)
    resized = keras.ops.image.resize(tensor, (new_h, new_w), interpolation="bilinear")
    resized = resized / 255.0

    mean = keras.ops.reshape(
        keras.ops.convert_to_tensor(image_mean, dtype="float32"), (1, 1, 1, 3)
    )
    std = keras.ops.reshape(
        keras.ops.convert_to_tensor(image_std, dtype="float32"), (1, 1, 1, 3)
    )
    normalized = (resized - mean) / std

    padded = keras.ops.zeros((1, target_length, target_length, 3), dtype="float32")
    padded = keras.ops.slice_update(padded, (0, 0, 0, 0), normalized)
    if get_data_format(data_format) == "channels_first":
        padded = keras.ops.transpose(padded, (0, 3, 1, 2))
    return padded, (orig_h, orig_w), (new_h, new_w)


def SAMGenerateMasks(
    model,
    image: Union[str, np.ndarray, "Image.Image"],
    points_per_side: int = 32,
    points_per_batch: int = 64,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    stability_score_offset: float = 1.0,
    mask_threshold: float = 0.0,
    crops_nms_thresh: float = 0.7,
    crop_n_layers: int = 0,
    crop_overlap_ratio: float = 512 / 1500,
    crop_n_points_downscale_factor: int = 1,
    target_length: int = 1024,
    image_mean: Optional[Tuple[float, ...]] = None,
    image_std: Optional[Tuple[float, ...]] = None,
    data_format: Optional[str] = None,
) -> Dict[str, Any]:
    """Run SAM's prompt-free "segment everything" pipeline.

    Samples a grid of points over the image (optionally over multiple
    image crops), runs the SAM mask decoder for each point, filters the
    resulting masks by quality and stability, converts to bounding
    boxes, and deduplicates across crops with NMS.

    Args:
        model: A built :class:`zeromodels.models.sam.SAM` model. Must
            expose ``vision_encoder_model`` and ``prompt_decoder_model``.
        image: Input image as a file path, numpy ``(H, W, 3)`` array,
            or PIL Image.
        points_per_side: Side length of the layer-0 point grid.
            Defaults to ``32`` (1024 points).
        points_per_batch: Number of prompt points processed per model
            call. Controls memory usage. Defaults to ``64``.
        pred_iou_thresh: Minimum predicted IoU score to keep a mask.
            Defaults to ``0.88``.
        stability_score_thresh: Minimum stability score to keep a mask.
            Defaults to ``0.95``.
        stability_score_offset: Offset used when computing stability.
            Defaults to ``1.0``.
        mask_threshold: Logit threshold for mask binarization.
            Defaults to ``0.0``.
        crops_nms_thresh: IoU threshold for final NMS across all
            crops. Defaults to ``0.7``.
        crop_n_layers: Number of additional crop layers (multi-scale).
            Defaults to ``0`` (full image only).
        crop_overlap_ratio: Crop overlap fraction (layer 1). Defaults
            to ``512/1500``.
        crop_n_points_downscale_factor: Downscale factor per layer for
            points-per-side. Defaults to ``1``.
        target_length: Model input resolution. Defaults to ``1024``.
        image_mean: Per-channel normalization mean. Defaults to
            ImageNet.
        image_std: Per-channel normalization std. Defaults to ImageNet.

    Returns:
        Dict with:
            - ``"masks"``: list of bool ``(orig_h, orig_w)`` arrays.
            - ``"iou_scores"``: ``(N,)`` float32 IoU predictions.
            - ``"boxes"``: ``(N, 4)`` XYXY boxes in original-image
              pixel coordinates.
            - ``"rle_masks"``: list of RLE dicts.

    Notes:
        - This driver runs the model with ``multimask_output=True`` so
          three masks are produced per point; all three are filtered
          independently.
        - The SAM model must be constructed for the same
          ``target_length`` used here.

    Example:
        ```python
        from zeromodels.models.sam import SAMGenerateMasks, SAMPromptableSegment
        model = SAMPromptableSegment.from_weights("zeromodels/sam_vit_base")
        out = SAMGenerateMasks(model, "photo.jpg", points_per_side=16)
        print(len(out["masks"]))
        ```
    """
    if image_mean is None:
        image_mean = IMAGENET_MEAN
    if image_std is None:
        image_std = IMAGENET_STD

    crop_data = generate_crop_boxes(
        image,
        target_size=target_length,
        crop_n_layers=crop_n_layers,
        overlap_ratio=crop_overlap_ratio,
        points_per_crop=points_per_side,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
    )
    crop_boxes = crop_data["crop_boxes"]
    cropped_images = crop_data["cropped_images"]
    points_per_crop = crop_data["points_per_crop"]
    original_size = crop_data["original_size"]

    all_rles: List[Dict[str, Any]] = []
    all_scores_list: List = []
    all_boxes_list: List = []

    mask_input_side = target_length // 4
    empty_masks = keras.ops.zeros(
        (1, mask_input_side, mask_input_side, 1), dtype="float32"
    )
    off_flag = keras.ops.zeros((1, 1), dtype="float32")

    model_enable_boxes = bool(getattr(model, "enable_boxes", False))
    model_enable_masks = bool(getattr(model, "enable_masks", False))

    num_crops = int(keras.ops.shape(crop_boxes)[0])
    crop_boxes_np = keras.ops.convert_to_numpy(crop_boxes)

    for crop_idx in range(num_crops):
        left, top, right, bottom = [int(x) for x in crop_boxes_np[crop_idx]]
        crop_box_list = [left, top, right, bottom]

        cropped_image = cropped_images[crop_idx]
        padded, _orig_size, _reshaped_size = preprocess_image_for_sam(
            cropped_image,
            target_length,
            image_mean,
            image_std,
            data_format=data_format,
        )

        image_embeddings = model.vision_encoder_model(padded)

        grid = points_per_crop[crop_idx]  # (num_points, 2)
        num_points = int(keras.ops.shape(grid)[0])

        crop_rles: List[Dict[str, Any]] = []
        crop_scores: List = []
        crop_boxes_collected: List = []

        for start in range(0, num_points, points_per_batch):
            end = min(start + points_per_batch, num_points)
            pb = end - start
            batch_points = keras.ops.expand_dims(
                keras.ops.expand_dims(grid[start:end], axis=1), axis=0
            )
            batch_labels = keras.ops.ones((1, pb, 1), dtype="int32")
            batch_boxes_zero = keras.ops.zeros((1, pb, 4), dtype="float32")

            decoder_inputs = {
                "image_embeddings": image_embeddings,
                "input_points": batch_points,
                "input_labels": batch_labels,
            }
            if model_enable_boxes:
                decoder_inputs["input_boxes"] = batch_boxes_zero
                decoder_inputs["has_boxes_input"] = off_flag
            if model_enable_masks:
                decoder_inputs["input_masks"] = empty_masks
                decoder_inputs["has_mask_input"] = off_flag
            out = model.prompt_decoder_model(decoder_inputs)
            pred_masks = out["pred_masks"]  # (1, pb, num_masks, H', W')
            iou_scores = out["iou_scores"]  # (1, pb, num_masks)

            crop_h, crop_w = int(cropped_image.shape[0]), int(cropped_image.shape[1])
            upsampled = sam_post_process_masks(
                pred_masks,
                original_size=(crop_h, crop_w),
                reshaped_size=_reshaped_size,
                target_length=target_length,
            )
            up_shape = keras.ops.shape(upsampled)
            batch_size = int(up_shape[0])
            pb_out = int(up_shape[1])
            nm_out = int(up_shape[2])
            masks_flat = keras.ops.reshape(
                upsampled, (batch_size * pb_out * nm_out, crop_h, crop_w)
            )
            scores_flat = keras.ops.reshape(iou_scores, (-1,))

            rles, scores, boxes = filter_masks(
                keras.ops.expand_dims(masks_flat, axis=0),
                keras.ops.expand_dims(scores_flat, axis=0),
                original_size=original_size,
                cropped_box_image=crop_box_list,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                mask_threshold=mask_threshold,
                stability_score_offset=stability_score_offset,
            )

            if len(rles) == 0:
                continue
            crop_rles.extend(rles)
            crop_scores.append(scores)
            crop_boxes_collected.append(boxes)

        if len(crop_rles) == 0:
            continue

        crop_scores_t = keras.ops.concatenate(crop_scores, axis=0)
        crop_boxes_t = keras.ops.concatenate(crop_boxes_collected, axis=0)

        # Shift box coordinates from crop-local to original-image frame.
        offset = keras.ops.convert_to_tensor([[left, top, left, top]], dtype="float32")
        crop_boxes_t = crop_boxes_t + offset

        all_rles.extend(crop_rles)
        all_scores_list.append(crop_scores_t)
        all_boxes_list.append(crop_boxes_t)

    if len(all_rles) == 0:
        return {
            "masks": [],
            "iou_scores": keras.ops.zeros((0,), dtype="float32"),
            "boxes": keras.ops.zeros((0, 4), dtype="float32"),
            "rle_masks": [],
        }

    all_scores_t = keras.ops.concatenate(all_scores_list, axis=0)
    all_boxes_t = keras.ops.concatenate(all_boxes_list, axis=0)

    masks, final_scores, final_rles, final_boxes = post_process_for_mask_generation(
        all_rles, all_scores_t, all_boxes_t, crops_nms_thresh=crops_nms_thresh
    )

    return {
        "masks": masks,
        "iou_scores": final_scores,
        "boxes": final_boxes,
        "rle_masks": final_rles,
    }
