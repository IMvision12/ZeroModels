from typing import Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from PIL import Image

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.image_util import get_data_format, load_image
from zeromodels.utils.labels_util import PASCAL_VOC_CLASSES


@keras.saving.register_keras_serializable(package="zeromodels")
class MobileViTImageProcessor(BaseImageProcessor):
    """Preprocess images for MobileViT (V1 / V2) inference.

    Matches the reference ``MobileViTImageProcessor`` pipeline:

    1. Resize so the shortest edge equals ``size["shortest_edge"]`` while
       preserving aspect ratio (bilinear).
    2. Center crop to ``crop_size["height"] x crop_size["width"]``.
    3. Rescale by ``rescale_factor`` (defaults to ``1/255``).
    4. (Optional) Flip channel order RGB → BGR: the MobileViT training
       recipe used BGR input, so the checkpoints expect this flip.
    5. Return as a Keras tensor in the requested data format.

    Note that MobileViT checkpoints are **not** mean/std normalized: the
    only intensity transform is the rescale step.

    Args:
        size: ``{"shortest_edge": int}``. Defaults to ``{"shortest_edge":
            288}`` for the classification recipe; segmentation recipes use
            ``544``.
        crop_size: ``{"height": int, "width": int}``. Defaults to
            ``{"height": 256, "width": 256}``; segmentation uses
            ``512x512``.
        resample: Interpolation mode (``"bilinear"``, ``"bicubic"``,
            ``"nearest"``). Defaults to ``"bilinear"``.
        do_resize: Whether to resize. Defaults to ``True``.
        do_center_crop: Whether to center-crop. Defaults to ``True``.
        do_rescale: Whether to rescale pixel values. Defaults to ``True``.
        rescale_factor: Multiplier for rescaling. Defaults to ``1/255``.
        do_flip_channel_order: Whether to flip RGB to BGR. Defaults to
            ``True`` to match the reference defaults.
        return_tensor: If ``True`` return a Keras tensor, else numpy.
            Defaults to ``True``.
        data_format: ``"channels_first"`` / ``"channels_last"``. ``None``
            uses ``keras.config.image_data_format()``.

    Example:
        ```python
        from zeromodels.models.mobilevit import (
            MobileViTImageClassify, MobileViTImageProcessor,
        )

        model = MobileViTImageClassify.from_weights("hf:apple/mobilevit-small")
        processor = MobileViTImageProcessor()
        x = processor("photo.jpg")["pixel_values"]
        logits = model(x, training=False)
        ```
    """

    def __init__(
        self,
        size: Optional[Dict[str, int]] = None,
        crop_size: Optional[Dict[str, int]] = None,
        resample: str = "bilinear",
        do_resize: bool = True,
        do_center_crop: bool = True,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_flip_channel_order: bool = True,
        return_tensor: bool = True,
        data_format: Optional[str] = None,
        variant: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.variant = variant
        default_size, default_crop = self.variant_sizes(variant)
        self.size = size if size is not None else default_size
        self.crop_size = crop_size if crop_size is not None else default_crop
        self.resample = resample
        self.do_resize = do_resize
        self.do_center_crop = do_center_crop
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_flip_channel_order = do_flip_channel_order
        self.return_tensor = return_tensor
        self.data_format = data_format

    @staticmethod
    def variant_sizes(variant):
        """``(size, crop_size)`` for a release variant.

        The classification and segmentation checkpoints train at different
        resolutions (256 vs 512), so a single default is wrong for one of them.
        The side is read from the model config and the resize target is the
        crop plus 32, matching the reference preprocessor configs (288/256 for
        classification, 544/512 for the DeepLabV3 heads).
        """
        side = 256
        if variant is not None:
            import importlib

            family = type_module = None
            for family in ("mobilevit", "mobilevitv2"):
                try:
                    cfg = importlib.import_module(
                        f"zeromodels.models.{family}.{family}_config"
                    )
                except ModuleNotFoundError:
                    continue
                for name in dir(cfg):
                    if not name.isupper() or "CONFIG" not in name:
                        continue
                    entry = getattr(cfg, name).get(variant)
                    if entry and entry.get("image_size"):
                        side = entry["image_size"]
                        type_module = family
                        break
                if type_module:
                    break
        return {"shortest_edge": side + 32}, {"height": side, "width": side}

    def __call__(self, image):
        return self.call(image)

    def call(self, image: Union[str, np.ndarray, "Image.Image"]):
        arr = load_image(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        arr = arr.astype(np.float32)

        if self.do_resize:
            short = int(self.size["shortest_edge"])
            h, w = arr.shape[:2]
            if h < w:
                new_h = short
                new_w = int(round(w * short / h))
            else:
                new_w = short
                new_h = int(round(h * short / w))
            pil = Image.fromarray(arr.astype(np.uint8))
            interp = {
                "bilinear": Image.BILINEAR,
                "bicubic": Image.BICUBIC,
                "nearest": Image.NEAREST,
            }[self.resample]
            arr = np.asarray(pil.resize((new_w, new_h), interp)).astype(np.float32)

        if self.do_center_crop:
            crop_h = int(self.crop_size["height"])
            crop_w = int(self.crop_size["width"])
            h, w = arr.shape[:2]
            top = max((h - crop_h) // 2, 0)
            left = max((w - crop_w) // 2, 0)
            arr = arr[top : top + crop_h, left : left + crop_w]
            if arr.shape[0] < crop_h or arr.shape[1] < crop_w:
                padded = np.zeros((crop_h, crop_w, arr.shape[-1]), dtype=np.float32)
                padded[: arr.shape[0], : arr.shape[1]] = arr
                arr = padded

        if self.do_rescale:
            arr = arr * float(self.rescale_factor)

        if self.do_flip_channel_order:
            arr = arr[..., ::-1].copy()

        arr = arr[np.newaxis, ...]

        df = get_data_format(self.data_format)
        if df == "channels_first":
            arr = np.transpose(arr, (0, 3, 1, 2))

        pixel_values = (
            keras.ops.convert_to_tensor(arr, dtype="float32")
            if self.return_tensor
            else arr
        )
        return {"pixel_values": pixel_values}

    def post_process_semantic_segmentation(
        self,
        outputs,
        target_size: Optional[Tuple[int, int]] = None,
        label_names: Optional[List[str]] = None,
        data_format: Optional[str] = None,
    ) -> Dict:
        """Argmax + optional resize for MobileViT segmentation outputs."""
        _names = label_names if label_names is not None else PASCAL_VOC_CLASSES

        logits = keras.ops.convert_to_numpy(outputs)
        channel_axis = 0 if get_data_format(data_format) == "channels_first" else -1
        # Upsample the logits, then take the argmax. Doing it the other way
        # round quantises the decision to the model's output stride and gives
        # visibly blocky boundaries, which matters most for heads that predict
        # at a coarse stride.
        if target_size is not None:
            if channel_axis == 0:
                logits = np.transpose(logits, (0, 2, 3, 1))
            resized = keras.ops.image.resize(
                logits, (target_size[0], target_size[1]), interpolation="bilinear"
            )
            pred_mask = np.argmax(keras.ops.convert_to_numpy(resized)[0], axis=-1)
        else:
            pred_mask = np.argmax(logits[0], axis=channel_axis)

        unique_classes = np.unique(pred_mask)
        class_names = [
            _names[c] if c < len(_names) else f"class_{c}" for c in unique_classes
        ]

        return {
            "segmentation": pred_mask,
            "class_names": class_names,
            "unique_classes": unique_classes,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "size": self.size,
                "crop_size": self.crop_size,
                "resample": self.resample,
                "do_resize": self.do_resize,
                "do_center_crop": self.do_center_crop,
                "do_rescale": self.do_rescale,
                "rescale_factor": self.rescale_factor,
                "do_flip_channel_order": self.do_flip_channel_order,
                "return_tensor": self.return_tensor,
                "data_format": self.data_format,
            }
        )
        return config
