import math

import keras
import numpy as np

from zeromodels.base import BaseImageProcessor

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def smart_resize(height, width, factor, min_pixels, max_pixels):
    """Round (h, w) to multiples of ``factor`` keeping pixels in range + aspect."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen2VLImageProcessor(BaseImageProcessor):
    """Turn PIL/array images into ``{"pixel_values", "image_grid_thw"}``."""

    def __init__(
        self,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        min_pixels=56 * 56,
        max_pixels=28 * 28 * 1280,
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_mean = np.array(image_mean, dtype="float32")
        self.image_std = np.array(image_std, dtype="float32")

    @classmethod
    def from_hf(cls, repo, **kwargs):
        import json

        hf = {}
        try:
            from huggingface_hub import hf_hub_download

            with open(
                hf_hub_download(repo, "preprocessor_config.json"), encoding="utf-8"
            ) as f:
                hf = json.load(f)
        except Exception:  # noqa: BLE001
            pass
        for param, key in (
            ("patch_size", "patch_size"),
            ("temporal_patch_size", "temporal_patch_size"),
            ("spatial_merge_size", "merge_size"),
            ("min_pixels", "min_pixels"),
            ("max_pixels", "max_pixels"),
        ):
            if hf.get(key) is not None:
                kwargs.setdefault(param, hf[key])
        return super().from_hf(repo, **kwargs)

    def _to_rgb_array(self, image):
        import os

        from PIL import Image

        if isinstance(image, (str, os.PathLike)):
            image = Image.open(image)
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image).astype("uint8"))
        return image.convert("RGB")

    def _preprocess_one(self, image):
        from PIL import Image

        img = self._to_rgb_array(image)
        w, h = img.size
        factor = self.patch_size * self.spatial_merge_size
        rh, rw = smart_resize(h, w, factor, self.min_pixels, self.max_pixels)
        img = img.resize((rw, rh), resample=Image.BICUBIC)

        x = np.asarray(img, dtype="float32") / 255.0
        x = (x - self.image_mean) / self.image_std
        x = np.transpose(x, (2, 0, 1))

        t = self.temporal_patch_size
        m = self.spatial_merge_size
        p = self.patch_size
        frames = np.stack([x] * t, axis=0)
        grid_t, grid_h, grid_w = 1, rh // p, rw // p

        patches = frames.reshape(grid_t, t, 3, grid_h // m, m, p, grid_w // m, m, p)
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flat = patches.reshape(grid_t * grid_h * grid_w, 3 * t * p * p)
        return flat.astype("float32"), [grid_t, grid_h, grid_w]

    def call(self, images):
        if not isinstance(images, (list, tuple)):
            images = [images]
        all_patches, grids = [], []
        for image in images:
            flat, grid = self._preprocess_one(image)
            all_patches.append(flat)
            grids.append(grid)
        return {
            "pixel_values": np.concatenate(all_patches, axis=0),
            "image_grid_thw": np.array(grids, dtype="int64"),
        }
