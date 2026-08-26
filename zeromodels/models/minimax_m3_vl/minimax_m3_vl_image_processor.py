import math

import keras
import numpy as np

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.image_util import load_image

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def smart_resize(height, width, factor=28, min_pixels=4 * 28 * 28, max_pixels=451584):
    """Round (h, w) to multiples of ``factor`` keeping the pixel budget."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


@keras.saving.register_keras_serializable(package="zeromodels")
class MiniMaxM3VLImageProcessor(BaseImageProcessor):
    """Preprocess images (or video frames) for MiniMax-M3 VL.

    Qwen-VL-style pipeline: bicubic ``smart_resize`` to multiples of
    ``patch_size * merge_size`` within the pixel budget, rescale + CLIP
    normalization, then pack into temporal-patch rows
    ``(num_patches, C * temporal_patch_size * patch_size**2)`` with the
    matching ``(t, h, w)`` grid per image.

    Args:
        patch_size / temporal_patch_size / merge_size: Patch geometry.
        min_pixels / max_pixels: Resize budget (M3: 4*28*28 / 451584).
        image_mean / image_std: Normalization constants (CLIP).

    Call args:
        images: One image or a list (PIL / array / path / url).

    Returns:
        ``{"pixel_values": (total_patches, 1176), "image_grid_thw": (N, 3)}``.
    """

    def __init__(
        self,
        patch_size=14,
        temporal_patch_size=2,
        merge_size=2,
        min_pixels=4 * 28 * 28,
        max_pixels=451584,
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_mean = tuple(image_mean)
        self.image_std = tuple(image_std)

    def resize_frame(self, image):
        from PIL import Image

        image = load_image(image)
        img = Image.fromarray(image.astype("uint8")).convert("RGB")
        h, w = image.shape[:2]
        factor = self.patch_size * self.merge_size
        rh, rw = smart_resize(h, w, factor, self.min_pixels, self.max_pixels)
        img = img.resize((rw, rh), resample=Image.BICUBIC)
        x = np.asarray(img, dtype="float32") / 255.0
        x = (x - np.asarray(self.image_mean, dtype="float32")) / np.asarray(
            self.image_std, dtype="float32"
        )
        return np.transpose(x, (2, 0, 1))  # (C, H, W)

    def pack_frames(self, frames):
        """Pack ``(T, C, H, W)`` frames into patch rows + the (t, h, w) grid."""
        t_len = frames.shape[0]
        tps = self.temporal_patch_size
        if t_len % tps:
            pad = tps - t_len % tps
            frames = np.concatenate([frames, np.repeat(frames[-1:], pad, axis=0)], 0)
        m, p = self.merge_size, self.patch_size
        grid_t = frames.shape[0] // tps
        grid_h, grid_w = frames.shape[2] // p, frames.shape[3] // p
        patches = frames.reshape(grid_t, tps, 3, grid_h // m, m, p, grid_w // m, m, p)
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flat = patches.reshape(grid_t * grid_h * grid_w, 3 * tps * p * p)
        return flat.astype("float32"), [grid_t, grid_h, grid_w]

    def call(self, images):
        if not isinstance(images, (list, tuple)):
            images = [images]
        all_patches, grids = [], []
        for image in images:
            frame = self.resize_frame(image)
            flat, grid = self.pack_frames(frame[None])
            all_patches.append(flat)
            grids.append(grid)
        return {
            "pixel_values": np.concatenate(all_patches, axis=0),
            "image_grid_thw": np.asarray(grids, dtype="int64"),
        }

    def process_video(self, frames):
        """Pack a list of frames (one video) into patch rows + grid."""
        stacked = np.stack([self.resize_frame(f) for f in frames], axis=0)
        flat, grid = self.pack_frames(stacked)
        return {
            "pixel_values_videos": flat,
            "video_grid_thw": np.asarray([grid], dtype="int64"),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "temporal_patch_size": self.temporal_patch_size,
                "merge_size": self.merge_size,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
                "image_mean": self.image_mean,
                "image_std": self.image_std,
            }
        )
        return config
