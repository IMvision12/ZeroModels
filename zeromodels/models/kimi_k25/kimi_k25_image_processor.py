import math

import keras
import numpy as np
from PIL import Image

from zeromodels.base import BaseImageProcessor


def navit_resize(width, height, patch_size, merge_kernel_size, max_patches, max_side):
    """Native-resolution target size, then the size padded to a whole merge cell.

    Never upscales: the scale is the most restrictive of the total-patch budget
    (a sqrt, since it acts on both axes) and the per-side budget. The padded size
    rounds up to a multiple of ``merge_kernel_size * patch_size`` so the 2x2
    patch merge divides evenly.
    """
    patches_w = max(1.0, width // patch_size)
    patches_h = max(1.0, height // patch_size)
    scale = min(
        1.0,
        math.sqrt(max_patches / (patches_w * patches_h)),
        (max_side * patch_size) / width,
        (max_side * patch_size) / height,
    )
    new_width = min(max(1, int(width * scale)), max_side * patch_size)
    new_height = min(max(1, int(height * scale)), max_side * patch_size)
    factor = merge_kernel_size * patch_size
    pad_height = (factor - new_height % factor) % factor + new_height
    pad_width = (factor - new_width % factor) % factor + new_width
    return (new_height, new_width), (pad_height, pad_width)


def resize_and_pad(image, patch_size, merge_kernel_size, max_patches, max_side):
    """Bicubic-resize a PIL image, then zero-pad bottom/right to the merge grid."""
    width, height = image.size
    (new_height, new_width), (pad_height, pad_width) = navit_resize(
        width, height, patch_size, merge_kernel_size, max_patches, max_side
    )
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    if (pad_height, pad_width) != (new_height, new_width):
        padded = Image.new(image.mode, (pad_width, pad_height), 0)
        padded.paste(image, (0, 0))
        image = padded
    return image


def patchify(array, patch_size):
    """``(C, H, W)`` -> ``(grid_h * grid_w, C, patch, patch)`` in row-major order."""
    channels, height, width = array.shape
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = array.reshape(channels, grid_h, patch_size, grid_w, patch_size)
    patches = patches.transpose(1, 3, 0, 2, 4)
    return patches.reshape(-1, channels, patch_size, patch_size), (grid_h, grid_w)


@keras.saving.register_keras_serializable(package="zeromodels")
class KimiK25ImageProcessor(BaseImageProcessor):
    """Native-resolution (NaViT) patch preprocessor for Kimi K2.5's MoonViT.

    Bicubic-resizes each image under a total-patch and a per-side budget, zero-pads
    to a whole ``merge_size x merge_size`` cell, scales to [0,1] and normalizes
    with mean/std 0.5, then flattens to ``(num_patches, 3, patch, patch)``. The
    padding is applied *before* normalization, so padded pixels land at -1.0 --
    matching the reference. Emits the ``(num_images, 3)`` ``(t, h, w)`` grid the
    vision tower consumes, with ``t = 1`` for stills.

    Args:
        patch_size: MoonViT patch (14).
        merge_size: Spatial patch-merge kernel (2).
        max_patches: Total patch budget per image (16384).
        max_side: Per-side patch budget (512).
        image_mean / image_std: Per-channel normalization.
    """

    def __init__(
        self,
        patch_size=14,
        merge_size=2,
        max_patches=16384,
        max_side=512,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.max_patches = max_patches
        self.max_side = max_side
        self.image_mean = tuple(image_mean)
        self.image_std = tuple(image_std)

    @classmethod
    def from_hf(cls, repo, **kwargs):
        # Kimi nests its preprocessor config under `media_proc_cfg` with its own
        # key names, so the generic BaseImageProcessor.from_hf mapping misses it.
        import json

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo.removeprefix("hf:"), "preprocessor_config.json")
        with open(path) as handle:
            cfg = json.load(handle).get("media_proc_cfg", {})
        defaults = {
            "patch_size": cfg.get("patch_size", 14),
            "merge_size": cfg.get("merge_kernel_size", 2),
            "max_patches": cfg.get("in_patch_limit", 16384),
            "max_side": cfg.get("patch_limit_on_one_side", 512),
            "image_mean": tuple(cfg.get("image_mean", (0.5, 0.5, 0.5))),
            "image_std": tuple(cfg.get("image_std", (0.5, 0.5, 0.5))),
        }
        defaults.update(kwargs)
        return cls(**defaults)

    def normalize(self, image):
        mean = np.array(self.image_mean, dtype="float32")[:, None, None]
        std = np.array(self.image_std, dtype="float32")[:, None, None]
        array = np.asarray(image, dtype="float32").transpose(2, 0, 1) / 255.0
        return (array - mean) / std

    def call(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        pixel_values, grids = [], []
        for image in images:
            image = resize_and_pad(
                image.convert("RGB"),
                self.patch_size,
                self.merge_size,
                self.max_patches,
                self.max_side,
            )
            patches, (grid_h, grid_w) = patchify(self.normalize(image), self.patch_size)
            pixel_values.append(patches)
            grids.append((1, grid_h, grid_w))
        return {
            "pixel_values": np.concatenate(pixel_values, axis=0).astype("float32"),
            "image_grid_thw": np.array(grids, dtype="int32"),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "merge_size": self.merge_size,
                "max_patches": self.max_patches,
                "max_side": self.max_side,
                "image_mean": self.image_mean,
                "image_std": self.image_std,
            }
        )
        return config
