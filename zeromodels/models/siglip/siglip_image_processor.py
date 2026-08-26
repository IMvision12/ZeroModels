from typing import Any, Dict, List, Optional, Union

import keras
import numpy as np
from keras import ops
from PIL import Image

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.image_util import get_data_format, load_image


@keras.saving.register_keras_serializable(package="zeromodels")
class SigLIPImageProcessor(BaseImageProcessor):
    """
    Image processor for SigLIP (Sigmoid Loss for Language Image Pre-training) models.
    This processor handles various preprocessing steps for images to be used with SigLIP models,
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
        processor = SigLIPImageProcessor(image_resolution=224)

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
        custom_processor = SigLIPImageProcessor(
            image_resolution=384,  # Higher resolution
            mean=[0.5, 0.5, 0.5],  # Different normalization
            std=[0.5, 0.5, 0.5],
            do_center_crop=False,  # Skip center cropping
        )

        # Use with SigLIP model
        siglip_model = load_siglip_model()
        image = keras.utils.load_img("path/to/image.jpg")
        image_array = keras.utils.img_to_array(image)
        processed = custom_processor(image_array)
        image_features = siglip_model(processed)
        ```

        Integration with image augmentation:
        ```python
        # Define augmentation layer
        augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
        ])

        # Apply augmentation before SigLIP processing
        image = keras.utils.load_img("path/to/image.jpg")
        image_array = keras.utils.img_to_array(image)
        image_array = image_array / 255.0  # Normalize to [0,1]

        # Augment and add batch dimension
        augmented = augmentation(ops.expand_dims(image_array, 0))

        # Process augmented image
        processor = SigLIPImageProcessor()
        result = processor(augmented)
        processed_image = result["pixel_values"]
        ```
    """

    def __init__(
        self,
        image_resolution: int = 224,
        mean=BaseImageProcessor.IMAGENET_INCEPTION_MEAN,
        std=BaseImageProcessor.IMAGENET_INCEPTION_STD,
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
        image = self.to_3_channels(image)
        image = self.to_unit_range(image)

        if self.do_resize:
            image = ops.image.resize(
                image,
                (self.image_resolution, self.image_resolution),
                interpolation="bicubic",
                antialias=True,
            )

        if self.do_center_crop:
            image = self._center_crop(image)

        if self.do_normalize:
            image = self.normalize_image(
                image, self.mean, self.std, data_format="channels_last"
            )

        return image

    def _center_crop(self, image: Any) -> Any:
        shape = ops.shape(image)
        height, width = shape[0], shape[1]
        target_size = self.image_resolution

        scale_h = ops.cast(target_size, "float32") / ops.cast(height, "float32")
        scale_w = ops.cast(target_size, "float32") / ops.cast(width, "float32")
        scale = ops.maximum(scale_h, scale_w)

        new_height = ops.cast(ops.cast(height, "float32") * scale, "int32")
        new_width = ops.cast(ops.cast(width, "float32") * scale, "int32")

        new_height = ops.maximum(new_height, target_size)
        new_width = ops.maximum(new_width, target_size)

        resized_image = ops.image.resize(
            image,
            (new_height, new_width),
            interpolation="bicubic",
            antialias=True,
        )

        y_start = (new_height - target_size) // 2
        x_start = (new_width - target_size) // 2

        cropped_image = ops.slice(
            resized_image,
            [y_start, x_start, 0],
            [target_size, target_size, 3],
        )

        return cropped_image

    def process_path(self, image_path: str) -> Any:
        arr = load_image(image_path)
        if self.do_resize:
            pil = Image.fromarray(arr.astype(np.uint8))
            pil = pil.resize(
                (self.image_resolution, self.image_resolution), Image.BICUBIC
            )
            arr = np.array(pil)
        image = self.to_unit_range(arr)
        if self.do_center_crop:
            image = self._center_crop(image)
        if self.do_normalize:
            image = self.normalize_image(
                image, self.mean, self.std, data_format="channels_last"
            )
        return image

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
            processed_image = self.process_path(image)
            images = ops.expand_dims(processed_image, axis=0)
        elif isinstance(image, (list, tuple)):
            if len(image) == 0:
                raise ValueError("image list cannot be empty")
            if not all(isinstance(p, str) for p in image):
                raise ValueError("List inputs must be a list of file paths")
            processed_images = [self.process_path(p) for p in image]
            images = ops.stack(processed_images)
        elif isinstance(image, Image.Image):
            arr = np.array(image)
            processed_image = self.preprocess(arr)
            images = ops.expand_dims(processed_image, axis=0)
        else:
            if len(ops.shape(image)) == 3:
                processed_image = self.preprocess(image)
                images = ops.expand_dims(processed_image, axis=0)
            elif len(ops.shape(image)) == 4:
                images = ops.vectorized_map(self.preprocess, image)
            else:
                raise ValueError(
                    f"Input images must have 3 dimensions (H, W, C) or 4 dimensions (B, H, W, C), "
                    f"got shape: {ops.shape(image)}"
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
                "do_center_crop": self.do_center_crop,
                "do_normalize": self.do_normalize,
                "do_resize": self.do_resize,
                "data_format": self.data_format,
            }
        )
        return config
