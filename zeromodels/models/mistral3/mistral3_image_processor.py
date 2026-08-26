import math

import keras
import numpy as np

from zeromodels.base import BaseImageProcessor

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@keras.saving.register_keras_serializable(package="zeromodels")
class Mistral3ImageProcessor(BaseImageProcessor):
    """Pixtral variable-resolution image processor (Mistral 3 recipe).

    Each image is scaled down (floor) so its longest edge fits
    ``longest_edge``, then bicubic-resized so each side becomes
    ``ceil(side / patch_size) * patch_size`` (rounded *up* to patch
    multiples), rescaled to ``[0, 1]`` and CLIP-normalized. The batch is
    zero-padded to the largest image and per-image ``(height, width)`` sizes
    are returned alongside.

    Returns ``{"pixel_values": (num_images, max_H, max_W, 3) float32,
    "image_sizes": (num_images, 2) int}``.

    Args:
        longest_edge: Maximum image side in pixels. Defaults to ``1540``.
        patch_size: Patch size each side is rounded up to. Defaults to ``14``.
        image_mean / image_std: Normalization constants (CLIP).
    """

    def __init__(
        self,
        longest_edge=1540,
        patch_size=14,
        image_mean=CLIP_MEAN,
        image_std=CLIP_STD,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.longest_edge = longest_edge
        self.patch_size = patch_size
        self.image_mean = tuple(image_mean)
        self.image_std = tuple(image_std)

    def to_pil(self, image):
        from PIL import Image

        if isinstance(image, Image.Image):
            return image.convert("RGB")
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = (
                (arr * 255.0).clip(0, 255).astype("uint8")
                if arr.max() <= 1.0
                else arr.clip(0, 255).astype("uint8")
            )
        return Image.fromarray(arr).convert("RGB")

    def target_size(self, height, width):
        ratio = max(height / self.longest_edge, width / self.longest_edge)
        if ratio > 1:
            height = int(math.floor(height / ratio))
            width = int(math.floor(width / ratio))
        target_h = math.ceil(height / self.patch_size) * self.patch_size
        target_w = math.ceil(width / self.patch_size) * self.patch_size
        return target_h, target_w

    def call(self, images):
        from PIL import Image

        if not isinstance(images, (list, tuple)):
            images = [images]
        arrays = []
        sizes = []
        for image in images:
            pil = self.to_pil(image)
            target_h, target_w = self.target_size(pil.height, pil.width)
            resized = pil.resize((target_w, target_h), Image.Resampling.BICUBIC)
            arr = np.asarray(resized, dtype=np.float32) / 255.0
            arr = (arr - np.asarray(self.image_mean, dtype=np.float32)) / np.asarray(
                self.image_std, dtype=np.float32
            )
            arrays.append(arr)
            sizes.append((target_h, target_w))
        max_h = max(h for h, _ in sizes)
        max_w = max(w for _, w in sizes)
        batch = np.zeros((len(arrays), max_h, max_w, 3), dtype=np.float32)
        for i, arr in enumerate(arrays):
            batch[i, : arr.shape[0], : arr.shape[1], :] = arr
        return {
            "pixel_values": batch,
            "image_sizes": np.asarray(sizes, dtype=np.int32),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "longest_edge": self.longest_edge,
                "patch_size": self.patch_size,
                "image_mean": self.image_mean,
                "image_std": self.image_std,
            }
        )
        return config
