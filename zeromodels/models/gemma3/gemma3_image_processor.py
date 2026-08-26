import keras
import numpy as np

from zeromodels.base import BaseImageProcessor


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3ImageProcessor(BaseImageProcessor):
    """Gemma 3 image processor: bicubic resize to a fixed square (896) +
    rescale to ``[0, 1]`` + normalize with mean/std 0.5 (SigLIP convention).

    Pan-and-scan cropping (off by default in the released processors) is not
    implemented. Returns
    ``{"pixel_values": (num_images, size, size, 3) float32}``.

    Args:
        size: Square side in pixels. Defaults to ``896``.
        image_mean / image_std: Normalization constants. Default ``0.5``.
    """

    def __init__(
        self,
        size=896,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = size
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
        out = []
        for image in images:
            pil = self.to_pil(image).resize(
                (self.size, self.size), Image.Resampling.BICUBIC
            )
            arr = np.asarray(pil, dtype=np.float32) / 255.0
            arr = (arr - np.asarray(self.image_mean, dtype=np.float32)) / np.asarray(
                self.image_std, dtype=np.float32
            )
            out.append(arr)
        return {"pixel_values": np.stack(out, axis=0)}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "size": self.size,
                "image_mean": self.image_mean,
                "image_std": self.image_std,
            }
        )
        return config
