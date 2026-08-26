from typing import Any, Dict, List, Optional, Union

import keras
import numpy as np
from keras import ops
from PIL import Image

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.image_util import get_data_format, load_image


@keras.saving.register_keras_serializable(package="zeromodels")
class DinoImageProcessor(BaseImageProcessor):
    """Image processor for DINO (v1) feature extractors.

    Two recipes, selected by ``model_type`` (the same tag the model config uses):

    * ``"dino_vit"`` (default) matches transformers' ``ViTImageProcessor`` for the
      ``facebook/dino-*`` ViTs: a square resize to ``image_resolution`` (bilinear),
      rescale to ``[0, 1]``, and ImageNet-standard normalization. No center crop.
    * ``"dino_resnet"`` matches the torch.hub ``facebookresearch/dino`` ResNet-50
      eval transform: an aspect-preserving shortest-edge resize to ``resize_size``
      (bicubic), a center crop to ``image_resolution``, rescale, and the same
      normalization.

    Both resize through PIL on the raw uint8 image so the interpolation matches the
    reference and is backend-independent.

    Args:
        image_resolution: Final square side in pixels (crop side for ResNet).
            Defaults to ``224``.
        model_type: ``"dino_vit"`` or ``"dino_resnet"``. Defaults to ``"dino_vit"``.
        resize_size: Shortest-edge target of the pre-crop resize, ResNet only.
            Defaults to ``256``.
        mean / std: Per-channel normalization. Defaults to ImageNet standard.
        do_normalize / do_resize: Toggle the normalize / resize steps.
    """

    def __init__(
        self,
        image_resolution: int = 224,
        model_type: str = "dino_vit",
        resize_size: int = 256,
        mean=BaseImageProcessor.IMAGENET_STANDARD_MEAN,
        std=BaseImageProcessor.IMAGENET_STANDARD_STD,
        do_normalize: bool = True,
        do_resize: bool = True,
        data_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_resolution = image_resolution
        self.model_type = model_type
        self.resize_size = resize_size
        self.mean = ops.array(mean, dtype="float32")
        self.std = ops.array(std, dtype="float32")
        self.do_normalize = do_normalize
        self.do_resize = do_resize
        self.data_format = get_data_format(data_format)

    @property
    def is_resnet(self) -> bool:
        return self.model_type == "dino_resnet"

    def resize_to_target(self, arr: np.ndarray) -> np.ndarray:
        """Square resize to ``image_resolution`` (the ViT recipe, bilinear)."""
        size = self.image_resolution
        if arr.shape[:2] == (size, size):
            return arr
        pil = Image.fromarray(arr).resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(pil)

    def target_size(self, height: int, width: int) -> tuple:
        """Shortest edge to ``resize_size`` (long side by truncation, matching the
        reference), for the ResNet recipe."""
        s = self.resize_size
        if width <= height:
            return int(s * height / width), s
        return s, int(s * width / height)

    def resize_shortest_edge(self, arr: np.ndarray) -> np.ndarray:
        """Aspect-preserving shortest-edge resize (the ResNet recipe, bicubic)."""
        new_height, new_width = self.target_size(arr.shape[0], arr.shape[1])
        if (new_height, new_width) == arr.shape[:2]:
            return arr
        pil = Image.fromarray(arr).resize(
            (new_width, new_height), Image.Resampling.BICUBIC
        )
        return np.asarray(pil)

    def center_crop(self, image: Any) -> Any:
        shape = ops.shape(image)
        height, width = shape[0], shape[1]
        target = self.image_resolution
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
        # Resize first on the raw uint8 image (the reference interpolates before
        # rescaling; bicubic/bilinear on uint8 differs from doing it on floats).
        arr = load_image(image)
        if self.do_resize:
            if self.is_resnet:
                arr = self.resize_shortest_edge(arr)
            else:
                arr = self.resize_to_target(arr)
        image = self.to_unit_range(arr)
        if self.is_resnet:
            image = self.center_crop(image)
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
                # A loop, not vectorized_map: PIL resizing cannot be traced.
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
                "model_type": self.model_type,
                "resize_size": self.resize_size,
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
