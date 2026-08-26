import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseImageProcessor


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3nImageProcessor(BaseImageProcessor):
    """Gemma 3n (SigLIP-style) image processor, in pure Keras ops (PIL only
    decodes and resizes).

    Fixed-size square resize, rescale to ``[0, 1]`` then normalize with
    ``image_mean`` / ``image_std``. Each image yields
    ``vision_soft_tokens_per_image`` soft tokens from the MobileNet-V5 tower.
    Returns ``{"pixel_values": (num_images, size, size, 3)}`` channels-last (the
    layout the Keras :class:`MobileNetV5Encoder` consumes).

    Args:
        size: Target square side in pixels. Defaults to ``768``.
        resample: PIL resample filter. Defaults to bicubic.
        rescale_factor: Pixel rescale. Defaults to ``1/255``.
        image_mean / image_std: Per-channel normalization. Default ``0.5`` each
            (maps ``[0, 1]`` pixels to ``[-1, 1]``).
    """

    def __init__(
        self,
        size=768,
        rescale_factor=1 / 255,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = size
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

    def call(self, images):
        from PIL import Image

        if not isinstance(images, (list, tuple)):
            images = [images]
        mean = ops.convert_to_tensor(self.image_mean, "float32")
        std = ops.convert_to_tensor(self.image_std, "float32")
        pixel_values = []
        for image in images:
            pil = self.to_pil(image)
            if pil.size != (self.size, self.size):
                pil = pil.resize((self.size, self.size), Image.Resampling.BILINEAR)
            arr = ops.convert_to_tensor(np.asarray(pil), "float32")
            arr = arr * self.rescale_factor
            arr = (arr - mean) / std  # (H, W, 3), channels-last
            pixel_values.append(arr)
        return {"pixel_values": ops.stack(pixel_values, axis=0)}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "size": self.size,
                "rescale_factor": self.rescale_factor,
                "image_mean": self.image_mean,
                "image_std": self.image_std,
            }
        )
        return config
