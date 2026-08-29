from typing import Dict, Optional, Tuple

import keras
from keras import ops

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.image_util import get_data_format, load_image
from zeromodels.utils.labels_util import COCO_91_CLASSES

from .efficientdet_layers import EfficientDetNMS


@keras.saving.register_keras_serializable(package="zeromodels")
class EfficientDetImageProcessor(BaseImageProcessor):
    """Preprocess images for EfficientDet inference.

    EfficientDet uses an aspect-preserving letterbox: the image is scaled by
    ``image_size / max(height, width)``, ImageNet-normalized, then zero-padded on the
    bottom / right to a square ``image_size x image_size``. The scale factor is
    returned so detections can be mapped back to the original image (also recomputed
    from ``target_sizes`` in
    :meth:`EfficientDetDetect.post_process_object_detection`).

    Args:
        image_size: Square target size. Defaults to ``512`` (EfficientDet-D0).
        resample: Interpolation method. Defaults to ``"bilinear"``.
        image_mean: Per-channel mean. Defaults to ImageNet mean.
        image_std: Per-channel std. Defaults to ImageNet std.
        rescale_factor: Scale applied before normalization. Defaults to ``1/255``.
        return_tensor: If True return a Keras tensor, else a numpy array.
        data_format: ``"channels_first"`` / ``"channels_last"``; ``None`` resolves to
            the Keras default.
    """

    def __init__(
        self,
        image_size: int = 512,
        resample: str = "bilinear",
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        rescale_factor: float = 1 / 255,
        return_tensor: bool = True,
        data_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.resample = resample
        self.image_mean = (
            image_mean if image_mean is not None else (0.485, 0.456, 0.406)
        )
        self.image_std = image_std if image_std is not None else (0.229, 0.224, 0.225)
        self.rescale_factor = rescale_factor
        self.return_tensor = return_tensor
        self.data_format = data_format

    def __call__(self, image):
        return self.call(image)

    def preprocess_one(self, image):
        arr = load_image(image)
        orig_h, orig_w = int(arr.shape[0]), int(arr.shape[1])
        scale = self.image_size / float(max(orig_h, orig_w))
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)

        # Preprocess entirely in channels_last (NHWC), then transpose at the end, so
        # the base resize/normalize helpers are not misled by a channels_first global.
        t = ops.convert_to_tensor(arr, dtype="float32")
        t = ops.expand_dims(t, axis=0)
        t = self.resize(
            t,
            (new_h, new_w),
            interpolation=self.resample,
            antialias=True,
            data_format="channels_last",
        )
        t = self.rescale_and_normalize(
            t,
            do_rescale=True,
            scale=self.rescale_factor,
            do_normalize=True,
            mean=self.image_mean,
            std=self.image_std,
            data_format="channels_last",
        )
        pad_h = self.image_size - new_h
        pad_w = self.image_size - new_w
        t = ops.pad(t, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)], constant_values=0.0)
        # data_format=None follows keras.config.image_data_format(), so the output
        # layout matches a model built under the same global setting.
        if get_data_format(self.data_format) == "channels_first":
            t = ops.transpose(t, (0, 3, 1, 2))
        return t, scale, (orig_h, orig_w)

    def call(self, image) -> Dict:
        items = list(image) if isinstance(image, (list, tuple)) else [image]
        tensors, scales, sizes = [], [], []
        for item in items:
            t, scale, size = self.preprocess_one(item)
            tensors.append(t)
            scales.append(scale)
            sizes.append(size)
        pixel_values = ops.concatenate(tensors, axis=0)
        if not self.return_tensor:
            pixel_values = ops.convert_to_numpy(pixel_values)
        return {
            "pixel_values": pixel_values,
            "scales": scales,
            "original_sizes": sizes,
        }

    def post_process_object_detection(
        self,
        outputs,
        threshold: float = 0.3,
        iou_threshold: float = 0.5,
        max_detections: int = 100,
        class_agnostic: bool = True,
        target_sizes=None,
        label_names=None,
    ):
        """Turn raw :class:`EfficientDetDetect` outputs into per-image detections.

        Applies per-class hard NMS (via :class:`EfficientDetNMS`) to the decoded
        boxes and sigmoid scores, then, when ``target_sizes`` is given, undoes the
        letterbox (divides by ``image_size / max(orig_h, orig_w)``) and clips boxes
        to each original image.

        Args:
            outputs: The detector output dict ``{"boxes": (B, N, 4),
                "scores": (B, N, num_classes)}``. Boxes are ``[ymin, xmin, ymax,
                xmax]`` in ``image_size`` pixel coordinates.
            threshold: Minimum score to keep a detection.
            iou_threshold: NMS IoU threshold.
            max_detections: Maximum detections kept per image.
            class_agnostic: If True (default, Google's ``postprocess_global``) run one
                NMS across all classes so an object yields a single box; if False, run
                NMS per class.
            target_sizes: List of ``(height, width)`` original image sizes, one per
                batch item, used to rescale boxes back to the original image. If
                ``None``, boxes stay in ``image_size`` coordinates.
            label_names: Class-name list indexed by detector class id. Defaults to
                the 90 COCO categories (class id ``c`` -> ``COCO_91_CLASSES[c + 1]``).

        Returns:
            A list (one dict per image) with ``"scores"``, ``"labels"``,
            ``"label_names"`` and ``"boxes"`` (``[xmin, ymin, xmax, ymax]``).
        """
        nms = EfficientDetNMS(
            iou_threshold=iou_threshold,
            score_threshold=threshold,
            max_detections=max_detections,
            class_agnostic=class_agnostic,
        )
        detections = nms(outputs["boxes"], outputs["scores"])
        if label_names is None:
            names, offset = COCO_91_CLASSES, 1
        else:
            names, offset = label_names, 0

        results = []
        for i in range(int(detections.shape[0])):
            det = detections[i]
            num_valid = int(ops.sum(ops.cast(det[:, 4] > 0.0, "int32")))
            det = det[:num_valid]
            ymin, xmin, ymax, xmax = det[:, 0], det[:, 1], det[:, 2], det[:, 3]
            scores = det[:, 4]
            labels = ops.cast(det[:, 5], "int32")
            if target_sizes is not None:
                orig_h, orig_w = target_sizes[i]
                scale = self.image_size / float(max(orig_h, orig_w))
                ymin, xmin, ymax, xmax = (
                    ymin / scale,
                    xmin / scale,
                    ymax / scale,
                    xmax / scale,
                )
                ymin = ops.clip(ymin, 0, orig_h)
                ymax = ops.clip(ymax, 0, orig_h)
                xmin = ops.clip(xmin, 0, orig_w)
                xmax = ops.clip(xmax, 0, orig_w)
            boxes_xyxy = ops.stack([xmin, ymin, xmax, ymax], axis=-1)
            mapped = [
                names[c + offset] if (c + offset) < len(names) else f"class_{c}"
                for c in (int(x) for x in labels)
            ]
            results.append(
                {
                    "scores": ops.convert_to_numpy(scores),
                    "labels": ops.convert_to_numpy(labels),
                    "label_names": mapped,
                    "boxes": ops.convert_to_numpy(boxes_xyxy),
                }
            )
        return results

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "resample": self.resample,
                "image_mean": self.image_mean,
                "image_std": self.image_std,
                "rescale_factor": self.rescale_factor,
                "return_tensor": self.return_tensor,
                "data_format": self.data_format,
            }
        )
        return config
