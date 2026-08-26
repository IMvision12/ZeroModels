import keras
import numpy as np

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.image_util import get_data_format, load_image


@keras.saving.register_keras_serializable(package="zeromodels")
class DeepseekVLImageProcessor(BaseImageProcessor):
    """Preprocess images for DeepSeek-VL.

    Per image: bicubic-resize so the longest edge equals ``size`` (preserving
    aspect ratio, each side at least ``min_size``), center-pad to a square
    ``size`` x ``size`` canvas with ``background_color`` (in raw 0-255 space),
    rescale to ``[0, 1]``, and normalize with mean/std ``0.5``: the HF
    ``DeepseekVLImageProcessor`` recipe.

    Args:
        size: Target square edge length (384).
        min_size: Per-side floor after the aspect-preserving resize.
        background_color: RGB pad fill, applied before rescaling.
        image_mean / image_std: Normalization constants.
        data_format: ``"channels_first"`` / ``"channels_last"``; ``None``
            resolves to ``keras.config.image_data_format()``.

    Call args:
        images: One image or a list (PIL / array / path / url).

    Returns:
        ``{"pixel_values": (num_images, size, size, 3)}`` (or channels-first).
    """

    def __init__(
        self,
        size=384,
        min_size=14,
        background_color=(127, 127, 127),
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = size
        self.min_size = min_size
        self.background_color = tuple(background_color)
        self.image_mean = tuple(image_mean)
        self.image_std = tuple(image_std)
        self.data_format = data_format

    def process_one(self, image):
        image = load_image(image).astype(np.float32)
        h, w = image.shape[:2]
        delta = self.size / max(h, w)
        new_h = max(round(h * delta), self.min_size)
        new_w = max(round(w * delta), self.min_size)

        x = keras.ops.convert_to_tensor(image, dtype="float32")[None]
        x = keras.ops.image.resize(
            x, (new_h, new_w), interpolation="bicubic", antialias=True
        )

        canvas = keras.ops.full(
            (1, self.size, self.size, 3),
            0.0,
            dtype="float32",
        ) + keras.ops.reshape(
            keras.ops.convert_to_tensor(self.background_color, dtype="float32"),
            (1, 1, 1, 3),
        )
        top = (self.size - new_h) // 2
        left = (self.size - new_w) // 2
        canvas = keras.ops.slice_update(canvas, (0, top, left, 0), x)
        return canvas

    def call(self, images):
        if not isinstance(images, (list, tuple)):
            images = [images]
        batch = keras.ops.concatenate([self.process_one(im) for im in images], axis=0)
        batch = batch / 255.0
        mean = keras.ops.reshape(
            keras.ops.convert_to_tensor(self.image_mean, dtype="float32"), (1, 1, 1, 3)
        )
        std = keras.ops.reshape(
            keras.ops.convert_to_tensor(self.image_std, dtype="float32"), (1, 1, 1, 3)
        )
        batch = (batch - mean) / std
        if get_data_format(self.data_format) == "channels_first":
            batch = keras.ops.transpose(batch, (0, 3, 1, 2))
        return {"pixel_values": batch}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "size": self.size,
                "min_size": self.min_size,
                "background_color": self.background_color,
                "image_mean": self.image_mean,
                "image_std": self.image_std,
                "data_format": self.data_format,
            }
        )
        return config
