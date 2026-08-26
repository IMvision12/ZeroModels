from typing import Optional, Tuple

import keras
import numpy as np

from zeromodels.base import BaseImageProcessor

SAM3_IMAGE_MEAN = (0.5, 0.5, 0.5)
SAM3_IMAGE_STD = (0.5, 0.5, 0.5)


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM3ImageProcessor(BaseImageProcessor):
    """Preprocess images for SAM3 inference.

    Stretches the image to a square ``image_resolution`` with bilinear
    interpolation (no aspect-ratio preservation, matching the reference
    ``Sam3ImageProcessor``), rescales via a float64 intermediate to match the
    reference precision, and normalizes with mean/std ``0.5``. Returns
    ``pixel_values`` ``(1, H, W, 3)`` plus the ``original_size`` needed to scale
    masks / boxes back to the input image.

    The default config is what ``zeromodels/sam3`` hosts in its
    ``zm_preprocessor.json``; ``image_resolution`` doubles as the model input
    size, so set it (and build the model at the same size) for custom-resolution
    inference.

    Args:
        image_resolution: Square target size for both axes (default 1008).
        image_mean: Per-channel normalization mean. Defaults to ``(0.5, 0.5, 0.5)``.
        image_std: Per-channel normalization std. Defaults to ``(0.5, 0.5, 0.5)``.
        rescale_factor: Pixel rescale factor applied before normalization
            (default ``1/255``).
    """

    def __init__(
        self,
        image_resolution: int = 1008,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        rescale_factor: float = 1.0 / 255.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_resolution = image_resolution
        self.image_mean = image_mean if image_mean is not None else SAM3_IMAGE_MEAN
        self.image_std = image_std if image_std is not None else SAM3_IMAGE_STD
        self.rescale_factor = rescale_factor

    def __call__(self, image):
        return self.call(image)

    def call(self, image):
        from .sam3_processor import preprocess_image

        pixel_values, original_size = preprocess_image(
            image,
            target_size=self.image_resolution,
            image_mean=self.image_mean,
            image_std=self.image_std,
            rescale_factor=self.rescale_factor,
        )
        return {
            "pixel_values": np.asarray(pixel_values),
            "original_size": original_size,
        }

    def post_process_object_detection(self, outputs, threshold=0.3, target_sizes=None):
        from .sam3_processor import post_process_object_detection

        return post_process_object_detection(outputs, threshold, target_sizes)

    def post_process_instance_segmentation(
        self, outputs, threshold=0.3, mask_threshold=0.5, target_sizes=None
    ):
        from .sam3_processor import post_process_instance_segmentation

        return post_process_instance_segmentation(
            outputs, threshold, mask_threshold, target_sizes
        )

    def post_process_semantic_segmentation(
        self, outputs, target_sizes=None, threshold=0.5
    ):
        from .sam3_processor import post_process_semantic_segmentation

        return post_process_semantic_segmentation(outputs, target_sizes, threshold)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_resolution": self.image_resolution,
                "image_mean": tuple(self.image_mean),
                "image_std": tuple(self.image_std),
                "rescale_factor": self.rescale_factor,
            }
        )
        return config
