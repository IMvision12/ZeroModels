from typing import Any, Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from PIL import Image

from zeromodels.base import BaseImageProcessor
from zeromodels.models.sam.sam_image_processor import (
    build_point_grid,
    filter_masks,
    generate_per_layer_crops,
    post_process_for_mask_generation,
)
from zeromodels.utils.image_util import get_data_format, load_image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM2ImageProcessor(BaseImageProcessor):
    """Preprocess images for Sam2 inference.

    Resizes the image to ``(target_length, target_length)`` with
    antialiased bilinear interpolation (no aspect-ratio preservation,
    matching the reference ``SAM2ImageProcessor``), applies ImageNet
    normalization, and prepares default prompt placeholders so the
    model can run with just an image.

    Args:
        target_length: Target spatial size for both axes (default 1024).
        image_mean: Per-channel mean for normalization. Defaults to
            ImageNet statistics.
        image_std: Per-channel std for normalization. Defaults to
            ImageNet statistics.
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

        image = keras.ops.convert_to_tensor(image, dtype="float32")
        image = keras.ops.expand_dims(image, axis=0)
        image = keras.ops.image.resize(
            image,
            (self.target_length, self.target_length),
            interpolation="bilinear",
            antialias=True,
            data_format="channels_last",
        )

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

        if get_data_format(self.data_format) == "channels_first":
            image = keras.ops.transpose(image, (0, 3, 1, 2))

        empty_points = keras.ops.zeros((1, 1, 0, 2), dtype="float32")
        empty_labels = keras.ops.zeros((1, 1, 0), dtype="int32")

        return {
            "pixel_values": image,
            "input_points": empty_points,
            "input_labels": empty_labels,
            "original_size": (orig_h, orig_w),
            "reshaped_size": (self.target_length, self.target_length),
        }

    def post_process_masks(self, pred_masks, original_size, target_length=None):
        return sam2_post_process_masks(
            pred_masks,
            original_size=original_size,
            target_length=target_length
            if target_length is not None
            else self.target_length,
        )


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM2Processor(SAM2ImageProcessor):
    """Preprocess an image plus optional point or box prompts for Sam2 inference.

    Extends :class:`SAM2ImageProcessor` by also encoding point and box
    prompts. Since SAM 2 stretches images independently per axis, prompt
    coordinates are scaled per axis as well
    (``x_new = x * target / orig_w``, ``y_new = y * target / orig_h``).

    ``input_boxes`` only feeds a model built with
    ``include_box_input=True``; the default graph declares point inputs
    only.
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
        result = SAM2ImageProcessor.call(self, image)

        orig_h, orig_w = result["original_size"]
        scale_x = self.target_length / float(orig_w)
        scale_y = self.target_length / float(orig_h)

        if input_points is not None:
            points = np.array(input_points, dtype=np.float64)
            points[..., 0] = points[..., 0] * scale_x
            points[..., 1] = points[..., 1] * scale_y
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
            boxes = np.array(input_boxes, dtype=np.float64)
            boxes[..., 0::2] = boxes[..., 0::2] * scale_x
            boxes[..., 1::2] = boxes[..., 1::2] * scale_y
            boxes = keras.ops.convert_to_tensor(boxes, dtype="float32")
            if keras.ops.ndim(boxes) == 1:
                boxes = keras.ops.expand_dims(boxes, axis=0)
            if keras.ops.ndim(boxes) == 2:
                boxes = keras.ops.expand_dims(boxes, axis=0)
            result["input_boxes"] = boxes
            if input_points is None:
                result["input_points"] = keras.ops.zeros(
                    (1, keras.ops.shape(boxes)[1], 1, 2), dtype="float32"
                )
                result["input_labels"] = -keras.ops.ones(
                    (1, keras.ops.shape(boxes)[1], 1), dtype="int32"
                )

        return result


def sam2_post_process_masks(
    pred_masks: "keras.KerasTensor",
    original_size: Tuple[int, int],
    target_length: int = 1024,
) -> "keras.KerasTensor":
    """Resize predicted Sam2 masks back to original image resolution.

    Since SAM 2 stretches images directly to the target square (no
    aspect-ratio preservation, no padding), reversing it is just a
    bilinear resize from the decoder mask resolution to
    ``original_size``.

    Args:
        pred_masks: Predicted masks of shape
            ``(batch, point_batch, num_masks, mask_h, mask_w)``.
        original_size: Original image ``(height, width)``.
        target_length: Model input resolution. Unused by this
            implementation but kept for API parity with
            :func:`sam_post_process_masks`. Defaults to ``1024``.

    Returns:
        Keras tensor of masks shaped
        ``(batch, point_batch, num_masks, orig_h, orig_w)``.

    Example:
        ```python
        from zeromodels.models.sam2 import sam2_post_process_masks

        masks = sam2_post_process_masks(
            outputs["pred_masks"],
            original_size=inputs["original_size"],
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

    orig_h, orig_w = original_size
    masks_final = keras.ops.image.resize(
        masks_4d,
        (orig_h, orig_w),
        interpolation="bilinear",
        antialias=True,
        data_format="channels_last",
    )

    masks_final = keras.ops.reshape(
        masks_final, (batch, point_batch, num_masks, orig_h, orig_w)
    )

    return masks_final


def load_image_to_numpy(image: Union[str, np.ndarray, "Image.Image"]) -> np.ndarray:
    """Decode various image inputs into a NumPy ``(H, W, 3)`` float32 array.

    Thin wrapper around :func:`zeromodels.utils.image_util.load_image` that strips an
    optional leading batch axis and casts to float32, matching the dtype the
    rest of the Sam2 pipeline expects.
    """
    if isinstance(image, np.ndarray) and image.ndim == 4:
        image = image[0]
    return load_image(image).astype(np.float32, copy=False)


def stretch_preprocess_crop(
    crop: np.ndarray,
    target_length: int,
    image_mean: Tuple[float, ...],
    image_std: Tuple[float, ...],
    data_format: Optional[str] = None,
) -> "keras.KerasTensor":
    """Stretch-resize a crop to ``(target_length, target_length)`` and normalize.

    Matches the SAM2 convention (per-axis stretch, no aspect-ratio
    preservation, no padding). Output is a batched
    ``(1, target_length, target_length, 3)`` Keras tensor with
    ImageNet normalization applied.
    """
    tensor = keras.ops.convert_to_tensor(crop, dtype="float32")
    tensor = keras.ops.expand_dims(tensor, axis=0)
    tensor = keras.ops.image.resize(
        tensor,
        (target_length, target_length),
        interpolation="bilinear",
        antialias=True,
    )
    tensor = tensor / 255.0
    mean = keras.ops.reshape(
        keras.ops.convert_to_tensor(image_mean, dtype="float32"), (1, 1, 1, 3)
    )
    std = keras.ops.reshape(
        keras.ops.convert_to_tensor(image_std, dtype="float32"), (1, 1, 1, 3)
    )
    normalized = (tensor - mean) / std
    if get_data_format(data_format) == "channels_first":
        normalized = keras.ops.transpose(normalized, (0, 3, 1, 2))
    return normalized


def SAM2GenerateMasks(
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
    """Prompt-free "segment everything" AMG pipeline for SAM2.

    Mirrors ``SAMGenerateMasks`` from ``zeromodels.models.sam`` but uses
    SAM2's stretch-to-``target_length`` preprocessing and calls the
    :attr:`SAM2.vision_encoder_model` + :attr:`SAM2.prompt_decoder_model`
    sub-models so the backbone runs once per crop and the mask
    decoder runs once per point batch.

    Samples a regular grid of foreground points across the image (and
    optionally across hierarchical crops), runs the mask decoder for
    every point, filters the predictions by predicted IoU and
    stability, converts each mask to its XYXY bounding box, and
    deduplicates across crops via NMS on those boxes.

    Args:
        model: A built :class:`zeromodels.models.sam2.SAM2PromptableSegment` instance.
            Must expose ``vision_encoder_model`` and
            ``prompt_decoder_model`` (both are attached automatically
            in :meth:`SAM2PromptableSegment.__init__`). The model should be built with
            the default ``include_box_input=False``,
            ``include_mask_input=False`` flags so the decoder
            sub-model accepts the point-only prompt interface used
            here.
        image: Input image as a file path, numpy ``(H, W, 3)`` array,
            or PIL Image.
        points_per_side: Side length of the layer-0 point grid.
            ``32`` → 1024 points per crop. Defaults to ``32``.
        points_per_batch: Number of points processed per decoder
            call. Controls peak memory. Defaults to ``64``.
        pred_iou_thresh: Minimum predicted IoU score to keep a mask.
            Defaults to ``0.88``.
        stability_score_thresh: Minimum stability score to keep a
            mask. Defaults to ``0.95``.
        stability_score_offset: Offset around ``mask_threshold`` when
            computing the stability score. Defaults to ``1.0``.
        mask_threshold: Logit threshold for mask binarization.
            Defaults to ``0.0``.
        crops_nms_thresh: IoU threshold for the final NMS across all
            crops. Defaults to ``0.7``.
        crop_n_layers: Number of extra crop layers for multi-scale
            AMG. Layer ``i`` contributes ``(2**(i+1))**2`` crops.
            Defaults to ``0`` (full image only).
        crop_overlap_ratio: Crop overlap fraction (layer 1). Defaults
            to ``512 / 1500``.
        crop_n_points_downscale_factor: Points-per-side scaling in
            deeper crop layers. Defaults to ``1``.
        target_length: Model input resolution. Defaults to ``1024``.
        image_mean: Per-channel normalization mean. Defaults to
            ImageNet.
        image_std: Per-channel normalization std. Defaults to
            ImageNet.

    Returns:
        Dict with:
            - ``"masks"``: list of bool ``(orig_h, orig_w)`` arrays,
              one per surviving mask.
            - ``"iou_scores"``: ``(N,)`` float32 predicted IoU scores.
            - ``"boxes"``: ``(N, 4)`` XYXY boxes in the original-image
              pixel frame.
            - ``"rle_masks"``: list of uncompressed COCO-style RLE
              dicts for the surviving masks.

    Example:
        ```python
        from zeromodels.models.sam2 import SAM2PromptableSegment, SAM2GenerateMasks

        model = SAM2PromptableSegment.from_weights("sam2_hiera_tiny")
        out = SAM2GenerateMasks(model, "photo.jpg", points_per_side=16)
        print(len(out["masks"]), "masks found")
        ```
    """
    if image_mean is None:
        image_mean = IMAGENET_MEAN
    if image_std is None:
        image_std = IMAGENET_STD

    if model.include_mask_input or model.include_box_input:
        raise ValueError(
            "SAM2GenerateMasks requires a SAM2 model built with the "
            "default point-only prompt interface "
            "(include_box_input=False, include_mask_input=False). "
            "The AMG driver samples grid points and does not supply "
            "box or mask inputs."
        )

    image_np = load_image_to_numpy(image)
    original_size = (image_np.shape[0], image_np.shape[1])

    # Build the layer-0..N point grids in [0, 1] x [0, 1] space.
    points_grid: List = []
    for i in range(crop_n_layers + 1):
        n_points = int(points_per_side / (crop_n_points_downscale_factor**i))
        points_grid.append(build_point_grid(n_points))

    crop_boxes, layer_idxs = generate_per_layer_crops(
        crop_n_layers, crop_overlap_ratio, original_size
    )

    all_rles: List[Dict[str, Any]] = []
    all_scores_list: List = []
    all_boxes_list: List = []

    for crop_idx, crop_box in enumerate(crop_boxes):
        left, top, right, bottom = [int(x) for x in crop_box]
        crop_h = bottom - top
        crop_w = right - left
        if crop_h <= 0 or crop_w <= 0:
            continue
        cropped_image = image_np[top:bottom, left:right, :]

        pixel_values = stretch_preprocess_crop(
            cropped_image,
            target_length,
            image_mean,
            image_std,
            data_format=data_format,
        )
        encoder_outputs = model.vision_encoder_model(pixel_values)
        norm_grid = points_grid[layer_idxs[crop_idx]]
        crop_scale = keras.ops.convert_to_tensor(
            [[float(target_length), float(target_length)]], dtype="float32"
        )
        grid = norm_grid * crop_scale

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

            decoder_inputs = {
                "image_embeddings": encoder_outputs["image_embeddings"],
                "high_res_feat_s0": encoder_outputs["high_res_feat_s0"],
                "high_res_feat_s1": encoder_outputs["high_res_feat_s1"],
                "input_points": batch_points,
                "input_labels": batch_labels,
            }
            out = model.prompt_decoder_model(decoder_inputs)
            pred_masks = out["pred_masks"]
            iou_scores = out["iou_scores"]

            upsampled = sam2_post_process_masks(
                pred_masks, original_size=(crop_h, crop_w), target_length=target_length
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
                cropped_box_image=[left, top, right, bottom],
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
