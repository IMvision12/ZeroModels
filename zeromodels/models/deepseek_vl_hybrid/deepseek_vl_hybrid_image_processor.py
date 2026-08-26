import keras
import numpy as np

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.image_util import get_data_format, load_image

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@keras.saving.register_keras_serializable(package="zeromodels")
class DeepseekVLHybridImageProcessor(BaseImageProcessor):
    """Dual-resolution preprocessing for DeepSeek-VL Hybrid (7B).

    Produces TWO tensors per image, matching HF's
    ``DeepseekVLHybridImageProcessor``:

    * ``high_res_pixel_values`` (``1024``): bicubic-resize so the longest edge is
      ``high_res_size`` (each side >= ``min_size``), center-pad to a square with
      ``high_res_background_color`` (raw 0-255), rescale, normalize with
      OpenAI-CLIP mean/std -> the SAM/ViTDet tower input.
    * ``pixel_values`` (``384``): the **high-res padded** image (0-255) is then
      bilinear-resized to ``size`` and normalized with mean/std ``0.5`` -> the
      SigLIP tower input. (Deriving low-res from the padded high-res image,
      not the original, is what HF does.)

    Returns ``{"pixel_values", "high_res_pixel_values"}`` (channels-last by
    default; ``data_format="channels_first"`` transposes both).
    """

    def __init__(
        self,
        size=384,
        high_res_size=1024,
        min_size=14,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        high_res_image_mean=OPENAI_CLIP_MEAN,
        high_res_image_std=OPENAI_CLIP_STD,
        background_color=(127, 127, 127),
        high_res_background_color=(122, 116, 104),
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = size
        self.high_res_size = high_res_size
        self.min_size = min_size
        self.image_mean = tuple(image_mean)
        self.image_std = tuple(image_std)
        self.high_res_image_mean = tuple(high_res_image_mean)
        self.high_res_image_std = tuple(high_res_image_std)
        self.background_color = tuple(background_color)
        self.high_res_background_color = tuple(high_res_background_color)
        self.data_format = data_format

    def resize_pad(self, image, target, background, interpolation):
        # Resize so the longest edge == target (each side >= min_size), then
        # center-pad to a square `target` canvas in raw 0-255 space.
        h, w = int(image.shape[1]), int(image.shape[2])
        delta = target / max(h, w)
        new_h = max(round(h * delta), self.min_size)
        new_w = max(round(w * delta), self.min_size)
        x = keras.ops.image.resize(
            image, (new_h, new_w), interpolation=interpolation, antialias=True
        )
        canvas = keras.ops.full((1, target, target, 3), 0.0, dtype="float32")
        canvas = canvas + keras.ops.reshape(
            keras.ops.convert_to_tensor(background, dtype="float32"), (1, 1, 1, 3)
        )
        top = (target - new_h) // 2
        left = (target - new_w) // 2
        return keras.ops.slice_update(canvas, (0, top, left, 0), x)

    def process_one(self, image):
        image = load_image(image).astype(np.float32)
        x = keras.ops.convert_to_tensor(image, dtype="float32")[None]
        high = self.resize_pad(
            x, self.high_res_size, self.high_res_background_color, "bicubic"
        )
        # Low-res is derived from the (square) high-res padded image.
        low = self.resize_pad(high, self.size, self.background_color, "bilinear")
        return low, high

    def normalize(self, batch, mean, std):
        batch = batch / 255.0
        mean = keras.ops.reshape(
            keras.ops.convert_to_tensor(mean, dtype="float32"), (1, 1, 1, 3)
        )
        std = keras.ops.reshape(
            keras.ops.convert_to_tensor(std, dtype="float32"), (1, 1, 1, 3)
        )
        return (batch - mean) / std

    def call(self, images):
        if not isinstance(images, (list, tuple)):
            images = [images]
        lows, highs = [], []
        for im in images:
            low, high = self.process_one(im)
            lows.append(low)
            highs.append(high)
        low = self.normalize(
            keras.ops.concatenate(lows, axis=0), self.image_mean, self.image_std
        )
        high = self.normalize(
            keras.ops.concatenate(highs, axis=0),
            self.high_res_image_mean,
            self.high_res_image_std,
        )
        if get_data_format(self.data_format) == "channels_first":
            low = keras.ops.transpose(low, (0, 3, 1, 2))
            high = keras.ops.transpose(high, (0, 3, 1, 2))
        return {"pixel_values": low, "high_res_pixel_values": high}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "size": self.size,
                "high_res_size": self.high_res_size,
                "min_size": self.min_size,
                "image_mean": self.image_mean,
                "image_std": self.image_std,
                "high_res_image_mean": self.high_res_image_mean,
                "high_res_image_std": self.high_res_image_std,
                "background_color": self.background_color,
                "high_res_background_color": self.high_res_background_color,
                "data_format": self.data_format,
            }
        )
        return config
