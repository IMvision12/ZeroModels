from typing import Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from keras import ops
from PIL import Image

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.image_util import load_image


@keras.saving.register_keras_serializable(package="zeromodels")
class Owlv2ImageProcessor(BaseImageProcessor):
    """Preprocess images for OWLv2 inference.

    Reproduces ``Owlv2ImageProcessor``'s unusual order:
    rescale → pad-to-square → resize → normalize. The pad-to-square
    step pads the bottom/right of each image with zeros so the input
    becomes square before resizing: this preserves the aspect ratio
    of the original image, unlike OWL-ViT's straight resize.

    Args:
        size: Target size as ``{"height": H, "width": W}``.
            Default: ``{"height": 960, "width": 960}``.
        resample: Interpolation method (``"nearest"``, ``"bilinear"``,
            or ``"bicubic"``). Defaults to ``"bicubic"``.
        do_rescale: Whether to divide pixel values by 255.
        rescale_factor: Rescale factor (default ``1/255``).
        do_pad: Whether to pad the image to a square with zeros.
            Defaults to ``True``.
        do_normalize: Whether to apply CLIP normalization.
        image_mean: Per-channel mean for normalization.
            Default: ``(0.48145466, 0.4578275, 0.40821073)``.
        image_std: Per-channel std for normalization.
            Default: ``(0.26862954, 0.26130258, 0.27577711)``.
        return_tensor: If True return a Keras tensor, otherwise numpy
            array.
        data_format: ``"channels_first"`` / ``"channels_last"``;
            ``None`` resolves to ``keras.backend.image_data_format()``.
    """

    def __init__(
        self,
        size: Optional[Dict[str, int]] = None,
        resample: str = "bicubic",
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_pad: bool = True,
        do_normalize: bool = True,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        return_tensor: bool = True,
        data_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = size if size is not None else {"height": 960, "width": 960}
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_pad = do_pad
        self.do_normalize = do_normalize
        self.image_mean = (
            image_mean
            if image_mean is not None
            else (0.48145466, 0.4578275, 0.40821073)
        )
        self.image_std = (
            image_std if image_std is not None else (0.26862954, 0.26130258, 0.27577711)
        )
        self.return_tensor = return_tensor
        self.data_format = data_format or keras.config.image_data_format()

    def __call__(
        self, images: Union[str, np.ndarray, Image.Image, List]
    ) -> Dict[str, Union[keras.KerasTensor, np.ndarray]]:
        return self.call(images)

    def call(
        self, images: Union[str, np.ndarray, Image.Image, List]
    ) -> Dict[str, Union[keras.KerasTensor, np.ndarray]]:
        if not isinstance(images, (list, tuple)):
            images = [images]

        processed = []
        for img in images:
            arr = self._to_hwc_float32(img)
            if self.do_rescale:
                arr = arr * (255.0 * self.rescale_factor)
            if self.do_pad:
                arr = self._pad_to_square(arr, constant_value=0.0)
            arr = self._resize(
                arr, (self.size["height"], self.size["width"]), self.resample
            )
            if self.do_normalize:
                mean = np.array(self.image_mean, dtype=np.float32)
                std = np.array(self.image_std, dtype=np.float32)
                arr = (arr - mean) / std
            processed.append(arr)

        x = np.stack(processed, axis=0).astype(np.float32)
        if self.data_format == "channels_first":
            x = np.transpose(x, (0, 3, 1, 2))

        out = ops.convert_to_tensor(x) if self.return_tensor else x
        return {"pixel_values": out}

    @staticmethod
    def _to_hwc_float32(image) -> np.ndarray:
        if isinstance(image, str):
            image = load_image(image)
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            arr = np.asarray(image, dtype=np.float32) / 255.0
        else:
            arr = np.asarray(image)
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            else:
                arr = arr.astype(np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        return arr

    @staticmethod
    def _pad_to_square(arr: np.ndarray, constant_value: float = 0.0) -> np.ndarray:
        h, w = arr.shape[:2]
        side = max(h, w)
        pad_h = side - h
        pad_w = side - w
        if pad_h == 0 and pad_w == 0:
            return arr
        return np.pad(
            arr,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=constant_value,
        )

    @staticmethod
    def _resize(
        arr: np.ndarray, target_hw: Tuple[int, int], interpolation: str
    ) -> np.ndarray:
        pil_resample = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
        }[interpolation]
        # PIL needs uint8 or float images via fromarray. We're carrying
        # float32 with values roughly in [0, 1] after rescale; scale back
        # to uint8 only if it's still raw uint8 range: otherwise PIL's
        # bicubic preserves precision on float arrays via mode 'F' per
        # channel. To keep parity with timm's anti-aliased bicubic, use
        # PIL with float channels.
        channels = []
        for c in range(arr.shape[-1]):
            ch = Image.fromarray(arr[..., c], mode="F")
            ch = ch.resize((target_hw[1], target_hw[0]), resample=pil_resample)
            channels.append(np.asarray(ch, dtype=np.float32))
        return np.stack(channels, axis=-1)

    def post_process_object_detection(
        self,
        outputs,
        threshold: float = 0.1,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
        text_labels: Optional[List[List[str]]] = None,
    ):
        return owlv2_post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
            text_labels=text_labels,
        )


def owlv2_post_process_object_detection(
    outputs: Dict,
    threshold: float = 0.1,
    target_sizes: Optional[List[Tuple[int, int]]] = None,
    text_labels: Optional[List[List[str]]] = None,
) -> List[Dict[str, np.ndarray]]:
    """Post-process raw OWLv2 outputs into per-image detection dicts.

    Converts raw model outputs (logits + normalized boxes) into filtered
    detections with the best query per box, confidence scores, and
    bounding boxes in ``[x_min, y_min, x_max, y_max]`` pixel coordinates.

    OWLv2 also produces ``objectness_logits`` (a per-patch, query-
    independent object score); this post-processor exposes them on the
    result dict but uses the same logits-based scoring as OWL-ViT for
    threshold filtering. Callers can multiply ``sigmoid(scores) *
    sigmoid(objectness)`` themselves if they want the combined score.

    Args:
        outputs: Raw model output dict with keys ``"logits"`` of shape
            ``(B, num_patches, num_queries)``, ``"pred_boxes"`` of
            shape ``(B, num_patches, 4)`` in normalized
            ``(cx, cy, w, h)`` format, and optionally
            ``"objectness_logits"`` of shape ``(B, num_patches)``.
        threshold: Minimum sigmoid score to keep a detection.
        target_sizes: List of ``(height, width)`` for each image. If
            ``None``, boxes stay in normalized ``[0, 1]`` coordinates.
        text_labels: ``[[...] per image]`` of text query strings used
            to populate the ``"text_labels"`` field.

    Returns:
        List of dicts (one per image) with keys ``"scores"``,
        ``"labels"`` (winning query index per detection),
        ``"text_labels"`` (only if ``text_labels`` is given),
        ``"boxes"``, and ``"objectness_scores"`` (only if
        ``objectness_logits`` was in ``outputs``).
    """
    logits = ops.convert_to_numpy(outputs["logits"])
    boxes = ops.convert_to_numpy(outputs["pred_boxes"])
    objectness = (
        ops.convert_to_numpy(outputs["objectness_logits"])
        if "objectness_logits" in outputs
        else None
    )

    probs = 1.0 / (1.0 + np.exp(-logits))
    scores_max = probs.max(axis=-1)
    labels_max = probs.argmax(axis=-1)

    results = []
    for i in range(logits.shape[0]):
        keep = scores_max[i] > threshold
        scores = scores_max[i][keep]
        labels = labels_max[i][keep]
        kept_boxes = boxes[i][keep]

        cx, cy, w, h = (
            kept_boxes[:, 0],
            kept_boxes[:, 1],
            kept_boxes[:, 2],
            kept_boxes[:, 3],
        )
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        xyxy = np.stack([x1, y1, x2, y2], axis=-1)

        if target_sizes is not None:
            img_h, img_w = target_sizes[i]
            scale = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
            xyxy = xyxy * scale

        result = {"scores": scores, "labels": labels, "boxes": xyxy}
        if text_labels is not None:
            queries = text_labels[i]
            result["text_labels"] = [
                queries[int(l)] if int(l) < len(queries) else f"query_{int(l)}"
                for l in labels
            ]
        if objectness is not None:
            obj_probs = 1.0 / (1.0 + np.exp(-objectness[i]))
            result["objectness_scores"] = obj_probs[keep]
        results.append(result)

    return results
