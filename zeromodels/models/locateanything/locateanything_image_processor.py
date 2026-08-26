import math

import keras
import numpy as np
from PIL import Image

from zeromodels.base import BaseImageProcessor


@keras.saving.register_keras_serializable(package="zeromodels")
class LocateAnythingImageProcessor(BaseImageProcessor):
    """Native-resolution patch preprocessor for LocateAnything / MoonViT.

    Resizes each image so the patch count stays under ``in_token_limit`` and the
    dimensions are a multiple of ``merge_kernel * patch_size`` (bicubic), scales
    to [0,1], normalizes with mean/std 0.5, and patchifies into
    ``(num_patches, 3, patch, patch)`` plus a ``(num_images, 2)`` grid of
    (h_patches, w_patches). Output feeds ``LocateAnythingModel`` directly.
    """

    def __init__(
        self,
        patch_size=14,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        in_token_limit=4096,
        merge_kernel_size=(2, 2),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.image_mean = tuple(image_mean)
        self.image_std = tuple(image_std)
        self.in_token_limit = in_token_limit
        self.merge_kernel_size = tuple(merge_kernel_size)

    def rescale_image(self, image):
        w, h = image.size
        p = self.patch_size
        if (w // p) * (h // p) > self.in_token_limit:
            scale = math.sqrt(self.in_token_limit / ((w // p) * (h // p)))
            image = image.resize(
                (int(w * scale), int(h * scale)), Image.Resampling.BICUBIC
            )
        new_w, new_h = image.size
        pad_h = self.merge_kernel_size[0] * p
        pad_w = self.merge_kernel_size[1] * p
        target_w = math.ceil(new_w / pad_w) * pad_w
        target_h = math.ceil(new_h / pad_h) * pad_h
        if target_w != new_w or target_h != new_h:
            image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        w, h = image.size
        if w // p >= 512 or h // p >= 512:
            raise ValueError(
                "image too large for MoonViT position embedding (>=512 patches)"
            )
        return image

    def patchify(self, arr):
        p = self.patch_size
        c, h, w = arr.shape
        patches = arr.reshape(c, h // p, p, w // p, p)
        patches = patches.transpose(1, 3, 0, 2, 4).reshape(-1, c, p, p)
        return patches, (h // p, w // p)

    def preprocess(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        mean = np.array(self.image_mean, dtype="float32")[:, None, None]
        std = np.array(self.image_std, dtype="float32")[:, None, None]
        pixel_values, grids = [], []
        for image in images:
            image = self.rescale_image(image.convert("RGB"))
            arr = np.asarray(image, dtype="float32").transpose(2, 0, 1) / 255.0
            arr = (arr - mean) / std
            patches, grid = self.patchify(arr)
            pixel_values.append(patches)
            grids.append(grid)
        return {
            "pixel_values": np.concatenate(pixel_values, axis=0).astype("float32"),
            "image_grid_hws": np.array(grids, dtype="int64"),
        }

    def __call__(self, images):
        return self.preprocess(images)
