from typing import Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from keras import ops
from PIL import Image

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.image_util import get_data_format, load_image
from zeromodels.utils.labels_util import COCO_80_CLASSES

_PIL_RESAMPLE = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
}


@keras.saving.register_keras_serializable(package="zeromodels")
class DFineImageProcessor(BaseImageProcessor):
    """Preprocess images for D-FINE inference.

    Loads the image (if needed), resizes to ``size`` using PIL bilinear
    interpolation (to match the reference DFineImageProcessor exactly), and
    rescales pixel values to ``[0, 1]``. D-FINE's published checkpoints
    do **not** apply ImageNet normalisation, but ``do_normalize=True``
    is exposed for fine-tuning on custom datasets.

    Args:
        size: Target size as ``{"height": H, "width": W}``.
            Default: ``{"height": 640, "width": 640}``.
        resample: Interpolation (``"nearest"``, ``"bilinear"``,
            ``"bicubic"``).
        do_rescale: Whether to multiply by ``rescale_factor``.
        rescale_factor: Multiplicative rescale factor.
            Defaults to ``1/255``.
        do_normalize: Whether to apply ImageNet normalization.
            Defaults to ``False`` (D-FINE published checkpoints do not
            normalize).
        image_mean: Per-channel mean for normalization. Used only when
            ``do_normalize=True``. Defaults to ImageNet mean.
        image_std: Per-channel std for normalization. Used only when
            ``do_normalize=True``. Defaults to ImageNet std.
        return_tensor: Return Keras tensor (True) or numpy array.
        data_format: ``"channels_first"`` / ``"channels_last"``;
            ``None`` resolves to ``keras.backend.image_data_format()``.
    """

    def __init__(
        self,
        size: Optional[Dict[str, int]] = None,
        resample: str = "bilinear",
        do_rescale: bool = True,
        rescale_factor: float = 1.0 / 255.0,
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

        if self.resample not in _PIL_RESAMPLE:
            raise ValueError(
                f"resample must be one of {list(_PIL_RESAMPLE)}, got {resample!r}"
            )

    def __call__(self, image: Union[str, np.ndarray, "Image.Image"]):
        return self.call(image)

    def call(self, image: Union[str, np.ndarray, "Image.Image", List]):
        if isinstance(image, (list, tuple)):
            return self.stack_images(image)
        if isinstance(image, np.ndarray) and image.ndim == 4:
            image = image[0]
        arr = load_image(image)
        pil_img = Image.fromarray(arr)
        target_wh = (self.size["width"], self.size["height"])
        if pil_img.size != target_wh:
            pil_img = pil_img.resize(target_wh, _PIL_RESAMPLE[self.resample])
        image = np.array(pil_img, dtype=np.float32)

        image = ops.convert_to_tensor(image, dtype="float32")
        image = ops.expand_dims(image, axis=0)
        if self.do_rescale:
            image = image * self.rescale_factor
        if self.do_normalize:
            mean = ops.reshape(
                ops.convert_to_tensor(self.image_mean, dtype="float32"),
                (1, 1, 1, 3),
            )
            std = ops.reshape(
                ops.convert_to_tensor(self.image_std, dtype="float32"),
                (1, 1, 1, 3),
            )
            image = (image - mean) / std
        if get_data_format(self.data_format) == "channels_first":
            image = ops.transpose(image, (0, 3, 1, 2))
        if not self.return_tensor:
            image = ops.convert_to_numpy(image)
        return {"pixel_values": image}

    def post_process_object_detection(
        self,
        outputs,
        threshold=0.5,
        target_sizes=None,
        num_top_queries=300,
        label_names=None,
    ):
        return dfine_post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
            num_top_queries=num_top_queries,
            label_names=label_names,
        )


def dfine_post_process_object_detection(
    outputs: Dict[str, "keras.KerasTensor"],
    threshold: float = 0.5,
    target_sizes: Optional[List[Tuple[int, int]]] = None,
    num_top_queries: int = 300,
    label_names: Optional[List[str]] = None,
):
    """Post-process raw D-FINE outputs into detection results.

    Applies sigmoid to logits, selects the top scoring ``(query, class)``
    pairs, converts boxes from ``cxcywh`` normalised format to ``xyxy``
    pixel coordinates, and filters by confidence threshold. All operations
    use ``keras.ops`` for backend portability.

    Args:
        outputs: Raw model output dict with ``"logits"`` of shape
            ``(B, Q, C)`` and ``"pred_boxes"`` of shape ``(B, Q, 4)``
            in normalised ``cxcywh`` format.
        threshold: Minimum confidence score to keep a detection.
            Defaults to ``0.5``.
        target_sizes: Optional list of ``(height, width)`` tuples, one
            per image in the batch, used to scale boxes to pixel
            coordinates. If ``None`` boxes stay normalised.
        num_top_queries: Number of top ``(query, class)`` pairs to
            consider before thresholding. Defaults to ``300``.
        label_names: Custom class name list for mapping label indices to
            names. If ``None``, defaults to COCO class names. Provide this
            when using a model fine-tuned on a custom dataset.

    Returns:
        List of dicts (one per image) with keys:

        - ``"scores"``: 1-D tensor of confidence scores.
        - ``"labels"``: 1-D integer tensor of class indices.
        - ``"label_names"``: List of class name strings.
        - ``"boxes"``: Tensor of shape ``(N, 4)`` in ``xyxy`` format.
    """
    logits = outputs["logits"]
    pred_boxes = outputs["pred_boxes"]

    batch_size = ops.shape(logits)[0]
    num_classes = ops.shape(logits)[2]

    scores = ops.sigmoid(logits)

    cx = pred_boxes[..., 0]
    cy = pred_boxes[..., 1]
    w = pred_boxes[..., 2]
    h = pred_boxes[..., 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    boxes_xyxy = ops.stack([x1, y1, x2, y2], axis=-1)

    results = []
    batch_size_int = int(ops.convert_to_numpy(batch_size))
    num_classes_int = int(ops.convert_to_numpy(num_classes))

    for i in range(batch_size_int):
        scores_i = scores[i]
        boxes_i = boxes_xyxy[i]

        flat_scores = ops.reshape(scores_i, [-1])

        num_select = min(num_top_queries, int(ops.shape(flat_scores)[0]))
        topk_scores, topk_indices = ops.top_k(flat_scores, k=num_select)

        topk_labels = topk_indices % num_classes_int
        topk_box_idx = ops.cast(topk_indices // num_classes_int, "int32")

        topk_boxes = ops.take(boxes_i, topk_box_idx, axis=0)

        if target_sizes is not None:
            img_h, img_w = target_sizes[i]
            scale = ops.convert_to_tensor([img_w, img_h, img_w, img_h], dtype="float32")
            topk_boxes = topk_boxes * scale

        keep = topk_scores > threshold
        kept_scores = topk_scores[keep]
        kept_labels = topk_labels[keep]
        kept_boxes = topk_boxes[keep]

        labels_np = ops.convert_to_numpy(kept_labels).astype(int)
        _names = label_names if label_names is not None else COCO_80_CLASSES
        mapped_names = [
            _names[l] if l < len(_names) else f"class_{l}" for l in labels_np
        ]

        results.append(
            {
                "scores": kept_scores,
                "labels": kept_labels,
                "label_names": mapped_names,
                "boxes": kept_boxes,
            }
        )

    return results
