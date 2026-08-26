from typing import Dict, Optional, Tuple, Union

import keras
import numpy as np
from PIL import Image

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.image_util import get_data_format

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@keras.saving.register_keras_serializable(package="zeromodels")
class DepthAnythingV1ImageProcessor(BaseImageProcessor):
    """Preprocess images for DepthAnythingV1 inference.

    Resizes the image to ``target_size`` with bicubic interpolation
    (using the standard DPT resample setting),
    rescales to ``[0, 1]``, and applies ImageNet normalization.

    Args:
        target_size: Target spatial size. Either a single ``int``
            (square output) or a ``(height, width)`` tuple. Defaults
            to ``518``.
        image_mean: Per-channel mean for normalization. Defaults to
            ImageNet statistics.
        image_std: Per-channel std for normalization. Defaults to
            ImageNet statistics.
        data_format: ``"channels_first"`` / ``"channels_last"``;
            ``None`` resolves to ``keras.backend.image_data_format()``.

    Example:
        ```python
        from zeromodels.models.depth_anything_v1 import (
            DepthAnythingV1Small, DepthAnythingV1ImageProcessor,
        )

        model = DepthAnythingV1Small(
            input_shape=(392, 784, 3), weights="lihe_yang"
        )
        proc = DepthAnythingV1ImageProcessor(target_size=(392, 784))
        inputs = proc("photo.jpg")
        depth = model(inputs["pixel_values"])
        ```
    """

    def __init__(
        self,
        target_size: Union[int, Tuple[int, int]] = 518,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        data_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(target_size, int):
            self.target_h, self.target_w = target_size, target_size
        else:
            self.target_h = int(target_size[0])
            self.target_w = int(target_size[1])
        self.target_size = target_size
        self.image_mean = image_mean if image_mean is not None else IMAGENET_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STD
        self.data_format = data_format

    def __call__(
        self, image: Union[str, np.ndarray, "Image.Image"]
    ) -> Dict[str, "keras.KerasTensor"]:
        return self.call(image)

    def call(
        self, image: Union[str, np.ndarray, "Image.Image"]
    ) -> Dict[str, "keras.KerasTensor"]:
        pixel_values, original_sizes, reshaped_hw, _ = self.preprocess_image(
            image,
            target_size=(self.target_h, self.target_w),
            image_mean=self.image_mean,
            image_std=self.image_std,
            rescale=True,
            interpolation="bicubic",
            antialias=True,
            data_format=self.data_format,
        )

        return {
            "pixel_values": pixel_values,
            "original_size": original_sizes[0],
            "reshaped_size": reshaped_hw,
        }

    def post_process_depth_estimation(
        self, predicted_depth, original_size, data_format=None
    ):
        return depth_anything_v1_post_process_depth(
            predicted_depth, original_size=original_size, data_format=data_format
        )


def depth_anything_v1_post_process_depth(
    predicted_depth: "keras.KerasTensor",
    original_size: Tuple[int, int],
    data_format: Optional[str] = None,
) -> "keras.KerasTensor":
    """Resize predicted DepthAnythingV1 depth back to original image resolution.

    Post-processes the predicted depth:
    bilinear-interpolate the predicted depth map to the original image
    size and squeeze the channel dim. Accepts Keras model output in
    either channels-last ``(B, H, W, 1)`` or channels-first
    ``(B, 1, H, W)`` format, or a 3D depth tensor ``(B, H, W)``.

    Args:
        predicted_depth: Depth tensor shaped ``(B, H, W)``,
            ``(B, H, W, 1)``, or ``(B, 1, H, W)``.
        original_size: Original image ``(height, width)``.
        data_format: Layout of ``predicted_depth`` when rank-4. ``None``
            resolves to the global setting from
            ``keras.config.image_data_format()``. Ignored for rank-3
            inputs.

    Returns:
        Keras tensor of shape ``(B, orig_h, orig_w)``: the depth map
        resampled to the original image resolution.

    Example:
        ```python
        depth = model(inputs["pixel_values"])
        depth_full = depth_anything_v1_post_process_depth(
            depth, original_size=inputs["original_size"]
        )
        ```
    """
    pd = keras.ops.convert_to_tensor(predicted_depth, dtype="float32")
    ndim = keras.ops.ndim(pd)

    if ndim == 3:
        pd = keras.ops.expand_dims(pd, axis=-1)
    elif ndim == 4:
        shape = keras.ops.shape(pd)
        if get_data_format(data_format) == "channels_first" or shape[1] == 1:
            pd = keras.ops.transpose(pd, (0, 2, 3, 1))
    else:
        raise ValueError(f"Expected predicted_depth with 3 or 4 dims, got ndim={ndim}")

    orig_h, orig_w = original_size
    pd = keras.ops.image.resize(
        pd,
        (orig_h, orig_w),
        interpolation="bilinear",
        antialias=False,
    )
    pd = keras.ops.squeeze(pd, axis=-1)
    return pd
