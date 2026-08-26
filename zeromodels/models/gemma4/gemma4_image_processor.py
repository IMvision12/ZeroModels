import math

import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseImageProcessor

SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)


def get_aspect_ratio_preserving_size(
    height, width, patch_size, max_patches, pooling_kernel_size
):
    """Largest (height, width) that stays under the patch budget and is
    divisible by ``pooling_kernel_size * patch_size``."""
    total_px = height * width
    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / total_px)
    side_mult = pooling_kernel_size * patch_size
    target_height = int(math.floor(factor * height / side_mult)) * side_mult
    target_width = int(math.floor(factor * width / side_mult)) * side_mult

    if target_height == 0 and target_width == 0:
        raise ValueError(
            "Attempting to resize to a 0 x 0 image. Resized height should be "
            f"divisible by pooling_kernel_size * patch_size = {side_mult}."
        )
    max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(
            int(math.floor(height / width)) * side_mult, max_side_length
        )
    return target_height, target_width


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma4ImageProcessor(BaseImageProcessor):
    """Gemma 4 (NaViT) image processor, in pure Keras ops (PIL only decodes and
    resizes the image).

    Aspect-ratio-preserving resize into a patch budget, rescale to ``[0, 1]`` (no
    mean/std normalization), patchify, and pad to
    ``max_soft_tokens * pooling_kernel_size**2`` patches. Each image keeps its
    aspect ratio and yields a variable patch count; the batch is padded to the
    shared maximum and real patches are marked by non-negative
    ``image_position_ids``. Returns
    ``{"pixel_values": (num_images, max_patches, patch_size**2 * 3),
    "image_position_ids": (num_images, max_patches, 2),
    "num_soft_tokens_per_image": [int, ...]}``.

    Args:
        patch_size: Square patch side in pixels. Defaults to ``16``.
        max_soft_tokens: Vision tokens per image after pooling; one of
            ``(70, 140, 280, 560, 1120)``. Defaults to ``280``.
        pooling_kernel_size: Spatial pooling kernel applied after patchify.
            Defaults to ``3``.
        rescale_factor: Pixel rescale. Defaults to ``1/255``.
        image_mean / image_std: Normalization constants (identity by default, as
            Gemma 4 trains on pixels in ``[0, 1]``).
    """

    def __init__(
        self,
        patch_size=16,
        max_soft_tokens=280,
        pooling_kernel_size=3,
        rescale_factor=1 / 255,
        image_mean=(0.0, 0.0, 0.0),
        image_std=(1.0, 1.0, 1.0),
        **kwargs,
    ):
        super().__init__(**kwargs)
        if max_soft_tokens not in SUPPORTED_SOFT_TOKENS:
            raise ValueError(
                f"max_soft_tokens must be one of {SUPPORTED_SOFT_TOKENS}, "
                f"got {max_soft_tokens}."
            )
        self.patch_size = patch_size
        self.max_soft_tokens = max_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size
        self.rescale_factor = rescale_factor
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

    def _patchify(self, image):
        """``(C, H, W)`` tensor -> ``(num_patches, patch_size**2 * C)``."""
        c, h, w = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
        p = self.patch_size
        nph, npw = h // p, w // p
        patched = ops.reshape(image, (c, nph, p, npw, p))
        patched = ops.transpose(patched, (1, 3, 2, 4, 0))
        return ops.reshape(patched, (nph * npw, p * p * c))

    def call(self, images):
        from PIL import Image

        if not isinstance(images, (list, tuple)):
            images = [images]
        p = self.patch_size
        k = self.pooling_kernel_size
        max_patches = self.max_soft_tokens * k**2
        mean = ops.convert_to_tensor(self.image_mean, "float32")
        std = ops.convert_to_tensor(self.image_std, "float32")
        identity_norm = all(m == 0.0 for m in self.image_mean) and all(
            s == 1.0 for s in self.image_std
        )

        pixel_values = []
        position_ids = []
        num_soft_tokens_per_image = []
        for image in images:
            pil = self.to_pil(image)
            width, height = pil.size
            th, tw = get_aspect_ratio_preserving_size(height, width, p, max_patches, k)
            if (th, tw) != (height, width):
                pil = pil.resize((tw, th), Image.Resampling.BICUBIC)

            arr = (
                ops.convert_to_tensor(np.asarray(pil), "float32") * self.rescale_factor
            )
            if not identity_norm:
                arr = (arr - mean) / std
            arr = ops.transpose(arr, (2, 0, 1))  # (H, W, C) -> (C, H, W)

            patches = self._patchify(arr)
            num_patches = int(patches.shape[0])
            num_soft_tokens_per_image.append(num_patches // k**2)

            patch_h, patch_w = th // p, tw // p
            grid_x, grid_y = ops.meshgrid(
                ops.arange(patch_w), ops.arange(patch_h), indexing="xy"
            )
            positions = ops.reshape(
                ops.stack([grid_x, grid_y], axis=-1), (num_patches, 2)
            )

            pad = max_patches - num_patches
            if pad > 0:
                patches = ops.pad(patches, [[0, pad], [0, 0]])
                positions = ops.pad(positions, [[0, pad], [0, 0]], constant_values=-1)
            pixel_values.append(patches)
            position_ids.append(positions)

        return {
            "pixel_values": ops.stack(pixel_values, axis=0),
            "image_position_ids": ops.stack(position_ids, axis=0),
            "num_soft_tokens_per_image": num_soft_tokens_per_image,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "max_soft_tokens": self.max_soft_tokens,
                "pooling_kernel_size": self.pooling_kernel_size,
                "rescale_factor": self.rescale_factor,
                "image_mean": self.image_mean,
                "image_std": self.image_std,
            }
        )
        return config
