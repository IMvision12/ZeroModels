import keras
import numpy as np

from zeromodels.base import BaseImageProcessor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def supported_aspect_ratios(min_patches, max_patches):
    # Enumeration order matters for tie-breaking: mirror HF's
    # get_all_supported_aspect_ratios (width-major loops).
    ratios = []
    for width in range(1, max_patches + 1):
        for height in range(1, max_patches + 1):
            if min_patches <= width * height <= max_patches:
                ratios.append((width, height))
    return ratios


def optimal_tiled_canvas(orig_height, orig_width, tile_size, min_patches, max_patches):
    """Pick the ``(num_columns, num_rows)`` tile grid whose aspect ratio is
    closest to the image's (HF ``get_optimal_tiled_canvas`` port, including the
    favor-more-tiles tie-break while the image area exceeds half the canvas).
    """
    aspect_ratio = orig_width / orig_height
    area = orig_width * orig_height
    best_diff = float("inf")
    best_grid = (1, 1)
    for grid in supported_aspect_ratios(min_patches, max_patches):
        grid_ratio = grid[0] / grid[1]
        diff = abs(aspect_ratio - grid_ratio)
        if diff < best_diff:
            best_diff = diff
            best_grid = grid
        elif (
            diff == best_diff and area > 0.5 * tile_size * tile_size * grid[0] * grid[1]
        ):
            best_grid = grid
    return best_grid


@keras.saving.register_keras_serializable(package="zeromodels")
class InternVLImageProcessor(BaseImageProcessor):
    """InternVL dynamic-tiling image processor (HF GotOcr2 recipe).

    Each image is matched to the tile grid (between ``min_patches`` and
    ``max_patches`` 448x448 tiles) whose aspect ratio is closest to its own,
    bicubic-resized onto that canvas, cropped into tiles, and: when more than
    one tile is produced: a full-image thumbnail tile is appended. Tiles are
    rescaled to ``[0, 1]`` and ImageNet-normalized.

    Returns ``{"pixel_values": (total_tiles, size, size, 3) float32,
    "num_patches": [tiles_per_image, ...]}``.

    Args:
        size: Tile side in pixels. Defaults to ``448``.
        min_patches: Minimum tiles per image. Defaults to ``1``.
        max_patches: Maximum tiles per image (excluding the thumbnail).
            Defaults to ``12``.
        crop_to_patches: Whether to tile at all (single resized tile
            otherwise). Defaults to ``True`` (the InternVL processor default).
        use_thumbnail: Whether to append the thumbnail tile when tiling
            produced more than one tile. Defaults to ``True``.
        image_mean / image_std: Normalization constants (ImageNet).
    """

    def __init__(
        self,
        size=448,
        min_patches=1,
        max_patches=12,
        crop_to_patches=True,
        use_thumbnail=True,
        image_mean=IMAGENET_MEAN,
        image_std=IMAGENET_STD,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = size
        self.min_patches = min_patches
        self.max_patches = max_patches
        self.crop_to_patches = crop_to_patches
        self.use_thumbnail = use_thumbnail
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

    def tile_image(self, image):
        from PIL import Image

        s = self.size
        if not self.crop_to_patches:
            tile = image.resize((s, s), Image.Resampling.BICUBIC)
            return [tile]
        cols, rows = optimal_tiled_canvas(
            image.height, image.width, s, self.min_patches, self.max_patches
        )
        resized = image.resize((cols * s, rows * s), Image.Resampling.BICUBIC)
        tiles = []
        for i in range(cols * rows):
            col, row = i % cols, i // cols
            tiles.append(resized.crop((col * s, row * s, (col + 1) * s, (row + 1) * s)))
        if self.use_thumbnail and len(tiles) != 1:
            tiles.append(image.resize((s, s), Image.Resampling.BICUBIC))
        return tiles

    def normalize(self, tiles):
        arr = np.stack([np.asarray(t, dtype=np.float32) for t in tiles], axis=0)
        arr = arr / 255.0
        mean = np.asarray(self.image_mean, dtype=np.float32)
        std = np.asarray(self.image_std, dtype=np.float32)
        return (arr - mean) / std

    def call(self, images):
        if not isinstance(images, (list, tuple)):
            images = [images]
        all_tiles = []
        num_patches = []
        for image in images:
            tiles = self.tile_image(self.to_pil(image))
            num_patches.append(len(tiles))
            all_tiles.extend(tiles)
        return {
            "pixel_values": self.normalize(all_tiles),
            "num_patches": num_patches,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "size": self.size,
                "min_patches": self.min_patches,
                "max_patches": self.max_patches,
                "crop_to_patches": self.crop_to_patches,
                "use_thumbnail": self.use_thumbnail,
                "image_mean": self.image_mean,
                "image_std": self.image_std,
            }
        )
        return config
