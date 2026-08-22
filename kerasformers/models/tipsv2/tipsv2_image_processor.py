from typing import Any, Dict, List, Optional, Union

import keras
import numpy as np
from keras import ops
from PIL import Image

from kerasformers.base import BaseImageProcessor
from kerasformers.utils.image_util import get_data_format, load_image


@keras.saving.register_keras_serializable(package="kerasformers")
class Tipsv2ImageProcessor(BaseImageProcessor):
    """Image processor for TIPSv2.

    Resizes to ``image_resolution`` (bilinear) and rescales to ``[0, 1]``. TIPSv2 does
    NOT apply mean/std normalization or center cropping (``do_normalize=False``,
    ``do_center_crop=False``); the vision tower consumes ``[0, 1]`` pixels directly.
    """

    def __init__(
        self,
        image_resolution: int = 448,
        resample: str = "bilinear",
        do_normalize: bool = False,
        do_resize: bool = True,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        data_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_resolution = image_resolution
        self.resample = resample
        self.do_normalize = do_normalize
        self.do_resize = do_resize
        self.mean = ops.array(mean, dtype="float32")
        self.std = ops.array(std, dtype="float32")
        self.data_format = get_data_format(data_format)

    def preprocess(self, image: Any) -> Any:
        image = self.to_3_channels(image)
        image = self.to_unit_range(image)
        if self.do_resize:
            image = ops.image.resize(
                image,
                (self.image_resolution, self.image_resolution),
                interpolation=self.resample,
                antialias=True,
            )
        if self.do_normalize:
            image = self.normalize_image(
                image, self.mean, self.std, data_format="channels_last"
            )
        return image

    def process_path(self, image_path: str) -> Any:
        arr = load_image(image_path)
        if self.do_resize:
            pil = Image.fromarray(arr.astype(np.uint8)).resize(
                (self.image_resolution, self.image_resolution), Image.BILINEAR
            )
            arr = np.array(pil)
        image = self.to_unit_range(arr)
        if self.do_normalize:
            image = self.normalize_image(
                image, self.mean, self.std, data_format="channels_last"
            )
        return image

    def __call__(self, image) -> Dict[str, Any]:
        return self.call(image)

    def call(
        self,
        image: Union[str, List[str], np.ndarray, Image.Image, "keras.KerasTensor"],
    ) -> Dict[str, Any]:
        if isinstance(image, str):
            images = ops.expand_dims(self.process_path(image), axis=0)
        elif isinstance(image, (list, tuple)):
            if len(image) == 0:
                raise ValueError("image list cannot be empty")
            if not all(isinstance(p, str) for p in image):
                raise ValueError("List inputs must be a list of file paths")
            images = ops.stack([self.process_path(p) for p in image])
        elif isinstance(image, Image.Image):
            images = ops.expand_dims(self.preprocess(np.array(image)), axis=0)
        else:
            if len(ops.shape(image)) == 3:
                images = ops.expand_dims(self.preprocess(image), axis=0)
            elif len(ops.shape(image)) == 4:
                images = ops.vectorized_map(self.preprocess, image)
            else:
                raise ValueError(
                    "Input images must have 3 or 4 dimensions, got shape: "
                    f"{ops.shape(image)}"
                )
        if self.data_format == "channels_first":
            images = ops.transpose(images, (0, 3, 1, 2))
        return {"pixel_values": images}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_resolution": self.image_resolution,
                "resample": self.resample,
                "do_normalize": self.do_normalize,
                "do_resize": self.do_resize,
                "mean": self.mean.tolist()
                if hasattr(self.mean, "tolist")
                else self.mean,
                "std": self.std.tolist() if hasattr(self.std, "tolist") else self.std,
                "data_format": self.data_format,
            }
        )
        return config
