from typing import Any, Dict, List, Optional, Union

import keras
import numpy as np
from keras import ops
from PIL import Image

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.image_util import get_data_format, load_image


@keras.saving.register_keras_serializable(package="zeromodels")
class DinoV3ImageProcessor(BaseImageProcessor):
    """Image processor for DINOv3 (ViT / ConvNeXt) feature extractors.

    Matches transformers' ``DINOv3ViTImageProcessor`` for ``facebook/dinov3-*``: a
    square resize to ``image_resolution`` (bilinear, through PIL on the raw uint8
    image so the interpolation matches the reference and is backend-independent),
    rescale to ``[0, 1]``, and ImageNet-standard normalization. No center crop.

    Args:
        image_resolution: Square target side in pixels. Defaults to ``224``.
        mean / std: Per-channel normalization. Defaults to ImageNet standard.
        do_normalize / do_resize: Toggle the normalize / resize steps.
    """

    def __init__(
        self,
        image_resolution: int = 224,
        mean=BaseImageProcessor.IMAGENET_STANDARD_MEAN,
        std=BaseImageProcessor.IMAGENET_STANDARD_STD,
        do_normalize: bool = True,
        do_resize: bool = True,
        data_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_resolution = image_resolution
        self.mean = ops.array(mean, dtype="float32")
        self.std = ops.array(std, dtype="float32")
        self.do_normalize = do_normalize
        self.do_resize = do_resize
        self.data_format = get_data_format(data_format)

    def resize_to_target(self, arr: np.ndarray) -> np.ndarray:
        size = self.image_resolution
        if arr.shape[:2] == (size, size):
            return arr
        pil = Image.fromarray(arr).resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(pil)

    def preprocess(self, image: Any) -> Any:
        arr = load_image(image)
        if self.do_resize:
            arr = self.resize_to_target(arr)
        image = self.to_unit_range(arr)
        if self.do_normalize:
            image = self.normalize_image(
                image, self.mean, self.std, data_format="channels_last"
            )
        return image

    def process_path(self, image_path: str) -> Any:
        return self.preprocess(image_path)

    def __call__(
        self,
        image: Union[str, List[str], np.ndarray, Image.Image, "keras.KerasTensor"],
    ) -> Dict[str, Any]:
        return self.call(image)

    def call(
        self,
        image: Union[str, List[str], np.ndarray, Image.Image, "keras.KerasTensor"],
    ) -> Dict[str, Any]:
        if isinstance(image, str):
            images = ops.expand_dims(self.preprocess(image), axis=0)
        elif isinstance(image, (list, tuple)):
            if len(image) == 0:
                raise ValueError("image list cannot be empty")
            if not all(isinstance(p, str) for p in image):
                raise ValueError("List inputs must be a list of file paths")
            images = ops.stack([self.preprocess(p) for p in image])
        elif isinstance(image, Image.Image):
            images = ops.expand_dims(self.preprocess(image), axis=0)
        else:
            rank = len(ops.shape(image))
            if rank == 3:
                images = ops.expand_dims(self.preprocess(image), axis=0)
            elif rank == 4:
                batch = np.asarray(ops.convert_to_numpy(image))
                images = ops.stack([self.preprocess(frame) for frame in batch])
            else:
                raise ValueError(
                    "Input images must have 3 dimensions (H, W, C) or 4 dimensions "
                    f"(B, H, W, C), got shape: {ops.shape(image)}"
                )
        if self.data_format == "channels_first":
            images = ops.transpose(images, (0, 3, 1, 2))
        return {"pixel_values": images}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_resolution": self.image_resolution,
                "mean": self.mean.tolist()
                if hasattr(self.mean, "tolist")
                else self.mean,
                "std": self.std.tolist() if hasattr(self.std, "tolist") else self.std,
                "do_normalize": self.do_normalize,
                "do_resize": self.do_resize,
                "data_format": self.data_format,
            }
        )
        return config
