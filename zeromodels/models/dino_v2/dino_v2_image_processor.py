from typing import Any, Dict, List, Optional, Union

import keras
import numpy as np
from keras import ops
from PIL import Image

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.image_util import get_data_format, load_image


@keras.saving.register_keras_serializable(package="zeromodels")
class DinoV2ImageProcessor(BaseImageProcessor):
    """Image processor for DINOv2 feature extractors.

    Matches transformers' ``BitImageProcessor`` for ``facebook/dinov2-*``: an
    aspect-preserving shortest-edge resize to ``resize_size`` (bicubic, through PIL
    on the raw uint8 image), a center crop to ``crop_size``, rescale to ``[0, 1]``,
    and ImageNet-standard normalization.

    Args:
        resize_size: Shortest-edge target of the resize. Defaults to ``256``.
        crop_size: Square center-crop side. Defaults to ``224``.
        mean / std: Per-channel normalization. Defaults to ImageNet standard.
        do_center_crop / do_normalize / do_resize: Toggle the steps.
    """

    def __init__(
        self,
        resize_size: int = 256,
        crop_size: int = 224,
        mean=BaseImageProcessor.IMAGENET_STANDARD_MEAN,
        std=BaseImageProcessor.IMAGENET_STANDARD_STD,
        do_center_crop: bool = True,
        do_normalize: bool = True,
        do_resize: bool = True,
        data_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.mean = ops.array(mean, dtype="float32")
        self.std = ops.array(std, dtype="float32")
        self.do_center_crop = do_center_crop
        self.do_normalize = do_normalize
        self.do_resize = do_resize
        self.data_format = get_data_format(data_format)

    def target_size(self, height: int, width: int) -> tuple:
        """Shortest edge to ``resize_size`` (the long side derived by truncation,
        matching the reference)."""
        s = self.resize_size
        if width <= height:
            return int(s * height / width), s
        return s, int(s * width / height)

    def resize_to_target(self, arr: np.ndarray) -> np.ndarray:
        new_height, new_width = self.target_size(arr.shape[0], arr.shape[1])
        if (new_height, new_width) == arr.shape[:2]:
            return arr
        pil = Image.fromarray(arr).resize(
            (new_width, new_height), Image.Resampling.BICUBIC
        )
        return np.asarray(pil)

    def _center_crop(self, image: Any) -> Any:
        shape = ops.shape(image)
        height, width = shape[0], shape[1]
        target = self.crop_size
        y_start = (height - target) // 2
        x_start = (width - target) // 2
        can_crop = ops.logical_and(
            ops.logical_and(y_start >= 0, x_start >= 0),
            ops.logical_and(y_start + target <= height, x_start + target <= width),
        )
        simple = ops.slice(image, [y_start, x_start, 0], [target, target, 3])

        new_h = ops.maximum(target, height)
        new_w = ops.maximum(target, width)
        pad_top = (new_h - height) // 2
        pad_left = (new_w - width) // 2
        padded = ops.pad(
            image,
            [
                (pad_top, new_h - height - pad_top),
                (pad_left, new_w - width - pad_left),
                (0, 0),
            ],
        )
        padded = ops.slice(
            padded,
            [(new_h - target) // 2, (new_w - target) // 2, 0],
            [target, target, 3],
        )
        return ops.where(can_crop, simple, padded)

    def preprocess(self, image: Any) -> Any:
        arr = load_image(image)
        if self.do_resize:
            arr = self.resize_to_target(arr)
        image = self.to_unit_range(arr)
        if self.do_center_crop:
            image = self._center_crop(image)
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
                "resize_size": self.resize_size,
                "crop_size": self.crop_size,
                "mean": self.mean.tolist()
                if hasattr(self.mean, "tolist")
                else self.mean,
                "std": self.std.tolist() if hasattr(self.std, "tolist") else self.std,
                "do_center_crop": self.do_center_crop,
                "do_normalize": self.do_normalize,
                "do_resize": self.do_resize,
                "data_format": self.data_format,
            }
        )
        return config
