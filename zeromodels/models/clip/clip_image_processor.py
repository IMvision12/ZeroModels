from typing import Any, Dict, List, Optional, Union

import keras
import numpy as np
from keras import ops
from PIL import Image

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.image_util import get_data_format, load_image


@keras.saving.register_keras_serializable(package="zeromodels")
class CLIPImageProcessor(BaseImageProcessor):
    """
    Image processor for CLIP (Contrastive Language-Image Pre-training) models.

    This processor handles various preprocessing steps for images to be used with CLIP models,
    including resizing, center cropping, and normalization.

    Attributes:
        image_resolution (int): Target resolution for the processed images.
        mean (keras.ops.Tensor): Mean values for RGB channels used in normalization.
        std (keras.ops.Tensor): Standard deviation values for RGB channels used in normalization.
        do_center_crop (bool): Whether to perform center cropping.
        do_normalize (bool): Whether to normalize the image using mean and std values.
        do_resize (bool): Whether to resize the image to the target resolution.

    Examples:
        Basic usage with an image tensor:
        ```python
        import keras
        from keras import ops

        # Create the processor
        processor = CLIPImageProcessor(image_resolution=224)

        # Process a single image
        image = keras.utils.load_img("path/to/image.jpg")
        image_array = keras.utils.img_to_array(image)
        result = processor(image_array)
        processed_image = result["pixel_values"]  # Shape: (1, 224, 224, 3)

        # Process a batch of images
        batch_size = 4
        random_images = ops.random.uniform((batch_size, 256, 256, 3))
        result = processor(random_images)
        processed_batch = result["pixel_values"]  # Shape: (4, 224, 224, 3)
        ```

        Process images from file paths:
        ```python
        # Process a single image path
        result = processor("path/to/image.jpg")
        processed_image = result["pixel_values"]  # Shape: (1, 224, 224, 3)

        # Process multiple image paths
        result = processor(["path/to/image1.jpg", "path/to/image2.jpg", "path/to/image3.jpg"])
        processed_images = result["pixel_values"]  # Shape: (3, 224, 224, 3)
        ```

        Custom processing configuration:
        ```python
        # Create processor with custom settings
        custom_processor = CLIPImageProcessor(
            image_resolution=336,  # Higher resolution
            mean=[0.5, 0.5, 0.5],  # Different normalization
            std=[0.5, 0.5, 0.5],
            do_center_crop=False,  # Skip center cropping
        )

        # Use with CLIP model
        clip_model = load_clip_model()
        image = keras.utils.load_img("path/to/image.jpg")
        image_array = keras.utils.img_to_array(image)
        processed = custom_processor(image_array)
        image_features = clip_model(processed)
        ```

        Integration with image augmentation:
        ```python
        # Define augmentation layer
        augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
        ])

        # Apply augmentation before CLIP processing
        image = keras.utils.load_img("path/to/image.jpg")
        image_array = keras.utils.img_to_array(image)
        image_array = image_array / 255.0  # Normalize to [0,1]

        # Augment and add batch dimension
        augmented = augmentation(ops.expand_dims(image_array, 0))

        # Process augmented image
        processor = CLIPImageProcessor()
        result = processor(augmented)
        processed_image = result["pixel_values"]
        ```
    """

    def __init__(
        self,
        image_resolution: int = 224,
        mean=BaseImageProcessor.OPENAI_CLIP_MEAN,
        std=BaseImageProcessor.OPENAI_CLIP_STD,
        do_center_crop: bool = True,
        do_normalize: bool = True,
        do_resize: bool = True,
        data_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_resolution = image_resolution
        self.mean = ops.array(mean, dtype="float32")
        self.std = ops.array(std, dtype="float32")
        self.do_center_crop = do_center_crop
        self.do_normalize = do_normalize
        self.do_resize = do_resize
        self.data_format = get_data_format(data_format)

    def preprocess(self, image: Any) -> Any:
        # Resize first, on the raw uint8 image: the reference interpolates
        # before rescaling, and bicubic on uint8 (clamped and rounded) is not
        # the same as bicubic on floats.
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

    def target_size(self, height: int, width: int) -> tuple:
        """Resized ``(height, width)``: shortest edge to ``image_resolution``.

        Mirrors the reference's shortest-edge rule, which sets the short side
        exactly and derives the long side, rather than scaling both by a float.
        """
        if width <= height:
            return int(self.image_resolution * height / width), self.image_resolution
        return self.image_resolution, int(self.image_resolution * width / height)

    def resize_to_target(self, image: np.ndarray) -> np.ndarray:
        """Bicubic resize through PIL, matching the reference.

        ``ops.image.resize``'s bicubic agrees with neither PIL nor itself
        across backends (max|diff| vs the reference ~0.7 on torch, ~0.5 on
        jax), so the interpolation has to go through PIL to stay correct and
        backend-independent.
        """
        new_height, new_width = self.target_size(image.shape[0], image.shape[1])
        if (new_height, new_width) == image.shape[:2]:
            return image
        pil = Image.fromarray(image).resize(
            (new_width, new_height), Image.Resampling.BICUBIC
        )
        return np.asarray(pil)

    def _center_crop(self, image: Any) -> Any:
        shape = ops.shape(image)
        height, width = shape[0], shape[1]
        target_size = self.image_resolution

        y_start = (height - target_size) // 2
        x_start = (width - target_size) // 2
        y_end = y_start + target_size
        x_end = x_start + target_size

        can_crop = ops.logical_and(
            ops.logical_and(y_start >= 0, x_start >= 0),
            ops.logical_and(y_end <= height, x_end <= width),
        )

        simple_cropped = ops.slice(
            image, [y_start, x_start, 0], [target_size, target_size, 3]
        )

        new_height = ops.maximum(target_size, height)
        new_width = ops.maximum(target_size, width)

        pad_top = (new_height - height) // 2
        pad_bottom = new_height - height - pad_top
        pad_left = (new_width - width) // 2
        pad_right = new_width - width - pad_left

        paddings = [(pad_top, pad_bottom), (pad_left, pad_right), (0, 0)]

        padded_image = ops.pad(image, paddings, constant_values=0)
        crop_y_start = (new_height - target_size) // 2
        crop_x_start = (new_width - target_size) // 2

        padded_cropped = ops.slice(
            padded_image,
            [crop_y_start, crop_x_start, 0],
            [target_size, target_size, 3],
        )

        return ops.where(can_crop, simple_cropped, padded_cropped)

    def process_path(self, image_path: str) -> Any:
        # One implementation for every input type: a second copy here is what
        # let the array path drift away from this one in the first place.
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
                # A loop, not vectorized_map: preprocessing resizes through PIL
                # to match the reference, which cannot be traced.
                batch = np.asarray(ops.convert_to_numpy(image))
                images = ops.stack([self.preprocess(frame) for frame in batch])
            else:
                raise ValueError(
                    f"Input images must have 3 dimensions (H, W, C) or 4 dimensions (B, H, W, C), "
                    f"got shape: {ops.shape(image)}"
                )
        if self.data_format == "channels_first":
            images = ops.transpose(images, (0, 3, 1, 2))
        return {"pixel_values": images}
