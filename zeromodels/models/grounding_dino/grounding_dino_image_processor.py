import keras
import numpy as np

from zeromodels.base import BaseImageProcessor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def size_with_aspect_ratio(height, width, shortest_edge, longest_edge):
    """DETR resize target: scale so the short side hits ``shortest_edge`` while
    the long side stays under ``longest_edge`` (aspect ratio preserved)."""
    min_orig, max_orig = float(min(height, width)), float(max(height, width))
    scale = shortest_edge / min_orig
    if max_orig * scale > longest_edge:
        scale = longest_edge / max_orig
    new_h = int(round(height * scale))
    new_w = int(round(width * scale))
    return new_h, new_w


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoImageProcessor(BaseImageProcessor):
    """DETR-style preprocessing -> ``{"pixel_values", "pixel_mask"}`` (channels-last).

    Resizes the shortest edge to ``shortest_edge`` (capped by ``longest_edge``),
    rescales to ``[0, 1]`` and normalizes with the ImageNet statistics. Single
    image per call (no batch padding); returns channels-last pixel values.
    """

    def __init__(
        self,
        shortest_edge=800,
        longest_edge=1333,
        image_mean=IMAGENET_MEAN,
        image_std=IMAGENET_STD,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.shortest_edge = shortest_edge
        self.longest_edge = longest_edge
        self.image_mean = np.array(image_mean, dtype="float32")
        self.image_std = np.array(image_std, dtype="float32")
        self.data_format = data_format

    def _to_rgb(self, image):
        import os

        from PIL import Image

        if isinstance(image, (str, os.PathLike)):
            image = Image.open(image)
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image).astype("uint8"))
        return image.convert("RGB")

    def _preprocess_one(self, image):
        from PIL import Image

        img = self._to_rgb(image)
        w, h = img.size
        new_h, new_w = size_with_aspect_ratio(
            h, w, self.shortest_edge, self.longest_edge
        )
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        x = np.asarray(img, dtype="float32") / 255.0
        x = (x - self.image_mean) / self.image_std
        return x  # (H, W, 3) channels-last

    def call(self, images):
        if not isinstance(images, (list, tuple)):
            images = [images]
        processed = [self._preprocess_one(im) for im in images]
        max_h = max(x.shape[0] for x in processed)
        max_w = max(x.shape[1] for x in processed)
        pixel_values = np.zeros((len(processed), max_h, max_w, 3), dtype="float32")
        pixel_mask = np.zeros((len(processed), max_h, max_w), dtype="int64")
        for i, x in enumerate(processed):
            pixel_values[i, : x.shape[0], : x.shape[1]] = x
            pixel_mask[i, : x.shape[0], : x.shape[1]] = 1
        data_format = self.data_format or keras.config.image_data_format()
        if data_format == "channels_first":
            pixel_values = np.transpose(pixel_values, (0, 3, 1, 2))
        return {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
