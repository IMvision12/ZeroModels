from typing import Dict, List, Optional, Tuple

import keras
from keras import ops

from zeromodels.base import BaseImageProcessor

TABLE_DETECTION_LABELS = ("table", "table rotated")
TABLE_STRUCTURE_LABELS = (
    "table",
    "table column",
    "table row",
    "table column header",
    "table projected row header",
    "table spanning cell",
)


@keras.saving.register_keras_serializable(package="zeromodels")
class TableTransformerImageProcessor(BaseImageProcessor):
    """Preprocess images for Table Transformer inference.

    The model takes already-normalized input, so run pixels through this processor first.
    Mirrors the reference Detr image processor the Table Transformer checkpoints
    ship with: rescale to `[0, 1]`, resize to a square `size`, and apply
    ImageNet normalization.

    Args:
        size: Target size as ``{"height": H, "width": W}``. Default:
            ``{"height": 800, "width": 800}``.
        resample: Interpolation method (``"nearest"``, ``"bilinear"``, or
            ``"bicubic"``).
        do_rescale: Whether to divide pixel values by 255.
        rescale_factor: Rescale factor (default ``1/255``).
        do_normalize: Whether to apply ImageNet normalization.
        image_mean: Per-channel mean for normalization. Default:
            ``(0.485, 0.456, 0.406)``.
        image_std: Per-channel std for normalization. Default:
            ``(0.229, 0.224, 0.225)``.
        return_tensor: If True return a Keras tensor, otherwise a numpy array.
        data_format: ``"channels_first"`` / ``"channels_last"``; ``None``
            resolves to ``keras.backend.image_data_format()``.
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

    def __call__(self, image) -> Dict[str, keras.KerasTensor]:
        return self.call(image)

    def call(self, image) -> Dict[str, keras.KerasTensor]:
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
            image = ops.convert_to_numpy(image)

        return {"pixel_values": image}

    def post_process_object_detection(
        self, outputs, threshold=0.7, target_sizes=None, label_names=None
    ):
        return table_transformer_post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
            label_names=label_names,
        )


def table_transformer_post_process_object_detection(
    outputs: Dict[str, keras.KerasTensor],
    threshold: float = 0.7,
    target_sizes: Optional[List[Tuple[int, int]]] = None,
    label_names: Optional[List[str]] = None,
) -> list:
    logits = ops.convert_to_tensor(outputs["logits"])
    boxes = ops.convert_to_tensor(outputs["pred_boxes"])
    batch_size = logits.shape[0]

    # Drop the trailing no-object class, then reduce to one score and label per query.
    probs = ops.softmax(logits, axis=-1)[:, :, :-1]
    scores_all = ops.max(probs, axis=-1)
    labels_all = ops.argmax(probs, axis=-1)

    cx, cy, w, h = boxes[:, :, 0], boxes[:, :, 1], boxes[:, :, 2], boxes[:, :, 3]
    xyxy_all = ops.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)

    results = []
    for i in range(batch_size):
        keep = ops.nonzero(ops.greater(scores_all[i], threshold))[0]
        scores = ops.take(scores_all[i], keep, axis=0)
        labels = ops.take(labels_all[i], keep, axis=0)
        xyxy_boxes = ops.take(xyxy_all[i], keep, axis=0)

        if target_sizes is not None:
            img_h, img_w = target_sizes[i]
            scale = ops.convert_to_tensor([img_w, img_h, img_w, img_h], dtype="float32")
            xyxy_boxes = xyxy_boxes * scale

        scores = ops.convert_to_numpy(scores)
        labels = ops.convert_to_numpy(labels)
        xyxy_boxes = ops.convert_to_numpy(xyxy_boxes)

        names = label_names if label_names is not None else TABLE_STRUCTURE_LABELS
        mapped_names = [
            names[label] if label < len(names) else f"class_{label}" for label in labels
        ]

        results.append(
            {
                "scores": scores,
                "labels": labels,
                "label_names": mapped_names,
                "boxes": xyxy_boxes,
            }
        )

    return results
