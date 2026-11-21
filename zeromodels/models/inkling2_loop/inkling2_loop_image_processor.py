import math

import keras
import numpy as np
from PIL import Image

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.image_util import load_image


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2ImageProcessor(BaseImageProcessor):
    """Image processor for Inkling-2-Mini (matches HF ``Inkling2ImageProcessor``).

    - Transparent RGBA images are composited onto an 8x8 gray checkerboard (204 / 153).
    - Pads each image to multiples of ``size`` (default ``40x40``) with ``-1.0`` on the
      0–255 scale, rescales by ``1/255``, normalizes with ``mean=0.5, std=0.5``, and
      cuts into ``(num_tokens, height, width, 3)`` patches with ``image_grid_thw``.
    """

    DEFAULT_SIZE = {"height": 40, "width": 40}
    DEFAULT_MEAN = (0.5, 0.5, 0.5)
    DEFAULT_STD = (0.5, 0.5, 0.5)

    def __init__(
        self,
        size=None,
        do_convert_rgb=True,
        do_rescale=True,
        rescale_factor=1 / 255.0,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        **kwargs,
    ):
        size = size if size is not None else self.DEFAULT_SIZE
        image_mean = image_mean if image_mean is not None else self.DEFAULT_MEAN
        image_std = image_std if image_std is not None else self.DEFAULT_STD
        super().__init__(
            size=size,
            do_resize=False,
            do_center_crop=False,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            **kwargs,
        )
        self.do_convert_rgb = bool(do_convert_rgb)

    @staticmethod
    def _composite_rgba(rgba_uint8):
        """Composite an ``(H, W, 4)`` uint8 RGBA image onto an 8x8 checkerboard."""
        h, w = rgba_uint8.shape[:2]
        rows = np.arange(h, dtype=np.int32)[:, None] // 8
        cols = np.arange(w, dtype=np.int32)[None, :] // 8
        bg = np.where((rows + cols) % 2 == 0, 204, 153)[..., None]
        rgba = rgba_uint8.astype(np.int32)
        alpha = rgba[..., 3:4]
        rgb = (rgba[..., :3] * alpha + bg * (255 - alpha) + 127) // 255
        return rgb.astype(np.uint8)

    def _to_rgb_array(self, image):
        if isinstance(image, Image.Image):
            if self.do_convert_rgb:
                if getattr(image, "has_transparency_data", False) or image.mode in (
                    "RGBA",
                    "LA",
                    "PA",
                ):
                    return self._composite_rgba(np.asarray(image.convert("RGBA")))
                return np.asarray(image.convert("RGB"))
            return np.asarray(image)
        arr = np.asarray(load_image(image))
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 4 and self.do_convert_rgb:
            arr = self._composite_rgba(arr.astype(np.uint8))
        return arr

    def call(self, images):
        if not isinstance(images, (list, tuple)):
            images = [images]
        patch_h = int(self.size["height"])
        patch_w = int(self.size["width"])
        mean = np.asarray(self.image_mean, dtype=np.float32)
        std = np.asarray(self.image_std, dtype=np.float32)

        all_patches = []
        grids = []
        for img in images:
            arr = self._to_rgb_array(img).astype(np.float32)  # (H, W, C)
            h, w, c = arr.shape
            rows = math.ceil(h / patch_h)
            cols = math.ceil(w / patch_w)
            pad_h = rows * patch_h - h
            pad_w = cols * patch_w - w
            if pad_h > 0 or pad_w > 0:
                arr = np.pad(
                    arr,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant",
                    constant_values=-1.0,
                )
            if self.do_rescale:
                arr = arr * float(self.rescale_factor)
            if self.do_normalize:
                arr = (arr - mean) / std
            # (rows, patch_h, cols, patch_w, C) -> (rows*cols, patch_h, patch_w, C)
            patches = arr.reshape(rows, patch_h, cols, patch_w, c).transpose(
                0, 2, 1, 3, 4
            )
            all_patches.append(patches.reshape(rows * cols, patch_h, patch_w, c))
            grids.append([1, rows, cols])

        return {
            "pixel_values": np.concatenate(all_patches, axis=0).astype(np.float32),
            "image_grid_thw": np.asarray(grids, dtype=np.int64),
        }

    def get_config(self):
        config = super().get_config()
        config.update({"do_convert_rgb": self.do_convert_rgb})
        return config
