import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseImageProcessor
from zeromodels.models.gemma4.gemma4_image_processor import (
    get_aspect_ratio_preserving_size,
)

SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma4UnifiedImageProcessor(BaseImageProcessor):
    """Gemma 4 unified image processor, in pure Keras ops (PIL only decodes and
    resizes the image).

    Like the "gemma4" NaViT processor it does an aspect-ratio-preserving resize
    into a patch budget and rescales to ``[0, 1]``, but then it merges each
    ``pooling_kernel_size`` x ``pooling_kernel_size`` block of teacher patches into
    one ``model_patch_size`` (``patch_size * pooling_kernel_size``, 48px) merged
    patch, because the unified model is encoder-free and consumes raw merged pixel
    patches directly. Each image keeps its aspect ratio (variable patch count); the
    batch is padded to ``max_soft_tokens`` merged patches and real patches are
    marked by non-negative ``image_position_ids``. Returns
    ``{"pixel_values": (num_images, max_soft_tokens, model_patch_size**2 * 3),
    "image_position_ids": (num_images, max_soft_tokens, 2),
    "num_soft_tokens_per_image": [int, ...]}``.

    Args:
        patch_size: Teacher patch side in pixels (before merging). Defaults to ``16``.
        max_soft_tokens: Merged patches per image; one of
            ``(70, 140, 280, 560, 1120)``. Defaults to ``280``.
        pooling_kernel_size: Merge kernel side. Defaults to ``3``.
        rescale_factor: Pixel rescale. Defaults to ``1/255``.
        image_mean / image_std: Normalization constants (identity by default).
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

    def patchify(self, image):
        """``(C, H, W)`` tensor -> ``(num_patches, patch_size**2 * C)``."""
        c, h, w = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
        p = self.patch_size
        nph, npw = h // p, w // p
        patched = ops.reshape(image, (c, nph, p, npw, p))
        patched = ops.transpose(patched, (1, 3, 2, 4, 0))
        return ops.reshape(patched, (nph * npw, p * p * c))

    def merge_patches(self, patches, positions, patch_width, length):
        """Merge ``k x k`` teacher patches (row-major) into ``length`` merged
        patches of ``(k * patch_size)**2 * 3`` channels, mirroring transformers'
        ``patches_merge`` (single image, no padding yet)."""
        p = self.patch_size
        k = self.pooling_kernel_size
        kernel_idx = positions // k
        num_from_tl = k * k * kernel_idx[:, 0] + k * patch_width * kernel_idx[:, 1]
        within = positions % k
        num_within = within[:, 0] + within[:, 1] * k
        order = ops.cast(num_within + num_from_tl, "int32")
        perm = ops.argsort(order)

        ordered = ops.take(patches, perm, axis=0)
        ordered = ops.reshape(ordered, (length, k, k, p, p, 3))
        ordered = ops.transpose(ordered, (0, 1, 3, 2, 4, 5))
        merged = ops.reshape(ordered, (length, k * p * k * p * 3))

        ordered_pos = ops.take(positions, perm, axis=0)
        ordered_pos = ops.reshape(ordered_pos, (length, k * k, 2))
        merged_pos = ops.min(ordered_pos // k, axis=1)
        return merged, ops.cast(merged_pos, "int32")

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

            patches = self.patchify(arr)
            num_patches = int(patches.shape[0])
            patch_h, patch_w = th // p, tw // p
            grid_x, grid_y = ops.meshgrid(
                ops.arange(patch_w), ops.arange(patch_h), indexing="xy"
            )
            positions = ops.cast(
                ops.reshape(ops.stack([grid_x, grid_y], axis=-1), (num_patches, 2)),
                "int32",
            )

            num_merged = num_patches // k**2
            merged, merged_pos = self.merge_patches(
                patches, positions, patch_w, num_merged
            )
            num_soft_tokens_per_image.append(num_merged)

            pad = self.max_soft_tokens - num_merged
            if pad > 0:
                merged = ops.pad(merged, [[0, pad], [0, 0]])
                merged_pos = ops.pad(merged_pos, [[0, pad], [0, 0]], constant_values=-1)
            pixel_values.append(merged)
            position_ids.append(merged_pos)

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
