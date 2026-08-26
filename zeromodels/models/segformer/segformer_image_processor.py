from typing import Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from PIL import Image

from zeromodels.base import BaseImageProcessor
from zeromodels.utils.image_util import get_data_format
from zeromodels.utils.labels_util import ADE20K_150_CLASSES


@keras.saving.register_keras_serializable(package="zeromodels")
class SegFormerImageProcessor(BaseImageProcessor):
    """Image preprocessing for SegFormer using Keras ops.

    Optional resize,
    rescale to ``[0, 1]``, and ImageNet normalization.

    Args:
        do_resize: Whether to resize the image.
        size: Dict with ``"height"`` / ``"width"`` keys for target size
            (default ``{"height": 512, "width": 512}``).
        resample: Interpolation method (``"nearest"``, ``"bilinear"``,
            or ``"bicubic"``).
        do_rescale: Whether to rescale pixel values.
        rescale_factor: Factor to rescale pixel values by.
        do_normalize: Whether to normalize with mean and std.
        image_mean: RGB mean values for normalization
            (default ``(0.485, 0.456, 0.406)``).
        image_std: RGB std values for normalization
            (default ``(0.229, 0.224, 0.225)``).
        return_tensor: Return a Keras tensor (True) or numpy array
            (False).
        data_format: ``"channels_first"`` / ``"channels_last"``;
            ``None`` resolves to ``keras.backend.image_data_format()``.
    """

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: str = "bilinear",
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, tuple]] = None,
        image_std: Optional[Union[float, tuple]] = None,
        return_tensor: bool = True,
        data_format: Optional[str] = None,
        variant: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.variant = variant
        self.do_resize = do_resize
        self.size = size if size is not None else self.variant_size(variant)
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = (
            image_mean if image_mean is not None else (0.485, 0.456, 0.406)
        )
        self.image_std = image_std if image_std is not None else (0.229, 0.224, 0.225)
        self.return_tensor = return_tensor
        self.data_format = data_format

        if self.size["height"] <= 0 or self.size["width"] <= 0:
            raise ValueError("Size dimensions must be positive")
        if self.resample not in ["nearest", "bilinear", "bicubic"]:
            raise ValueError(
                "Resample method must be 'nearest', 'bilinear', or 'bicubic'"
            )
        if self.rescale_factor < 0:
            raise ValueError("Rescale factor must be non-negative")

    @staticmethod
    def variant_size(variant):
        """Default square target size (512).

        The variants train at 512 (ADE), 640 (b5 ADE), 768 or 1024
        (Cityscapes). The per-variant resolution is no longer kept in the
        package: it travels in the repo's ``zm_preprocessor.json`` and is
        applied when loading via ``from_weights("zeromodels/<variant>")``.
        Direct construction without an explicit ``size`` falls back to 512.
        """
        return {"height": 512, "width": 512}

    def __call__(
        self, image: Union[str, np.ndarray, Image.Image, keras.KerasTensor]
    ) -> Union[keras.KerasTensor, np.ndarray]:
        return self.call(image)

    def call(
        self, image: Union[str, np.ndarray, Image.Image, keras.KerasTensor]
    ) -> Union[keras.KerasTensor, np.ndarray]:
        # Keras-tensor input keeps its original bespoke path (range
        # validation + eager scaling). Every other input type routes
        # through the shared `preprocess_image` helper.
        is_keras_tensor = (
            not isinstance(image, (str, np.ndarray, Image.Image))
            and hasattr(image, "shape")
            and hasattr(image, "dtype")
        )
        if is_keras_tensor:
            if len(image.shape) == 4:
                image = image[0]
            if len(image.shape) != 3:
                raise ValueError("Input tensor must have shape (H, W, C)")

            image_float = keras.ops.cast(image, dtype="float32")
            max_val_py = keras.ops.convert_to_numpy(keras.ops.max(image_float)).item()
            min_val_py = keras.ops.convert_to_numpy(keras.ops.min(image_float)).item()

            if max_val_py <= 1.0 and min_val_py >= 0.0:
                image = image_float * 255.0
            elif not (min_val_py >= 0 and max_val_py <= 255):
                raise ValueError("Tensor values must be in [0,1] or [0,255] range")
            else:
                image = image_float

            image = keras.ops.expand_dims(image, axis=0)
            if self.do_resize:
                target_size = (self.size["height"], self.size["width"])
                if image.shape[1:3] != target_size:
                    image = keras.ops.image.resize(
                        image, size=target_size, interpolation=self.resample
                    )
            if self.do_rescale:
                image = image * self.rescale_factor
            if self.do_normalize:
                mean = keras.ops.reshape(
                    keras.ops.convert_to_tensor(self.image_mean, dtype="float32"),
                    (1, 1, 1, 3),
                )
                std = keras.ops.reshape(
                    keras.ops.convert_to_tensor(self.image_std, dtype="float32"),
                    (1, 1, 1, 3),
                )
                image = (image - mean) / std
        else:
            image, _, _, _ = self.preprocess_image(
                image,
                target_size=(self.size["height"], self.size["width"]),
                image_mean=self.image_mean if self.do_normalize else None,
                image_std=self.image_std if self.do_normalize else None,
                rescale=self.do_rescale,
                interpolation=self.resample,
                antialias=False,
                data_format=self.data_format,
            )
            if self.do_rescale and self.rescale_factor != 1 / 255:
                image = image * (self.rescale_factor * 255)

        if is_keras_tensor and get_data_format(self.data_format) == "channels_first":
            image = keras.ops.transpose(image, (0, 3, 1, 2))

        if not self.return_tensor:
            image = keras.ops.convert_to_numpy(image)

        return {"pixel_values": image}

    def post_process_semantic_segmentation(
        self, outputs, target_size=None, label_names=None, data_format=None
    ):
        return segformer_post_process_semantic_segmentation(
            outputs,
            target_size=target_size,
            label_names=label_names,
            data_format=data_format,
        )


def segformer_post_process_semantic_segmentation(
    outputs: "keras.KerasTensor",
    target_size: Optional[Tuple[int, int]] = None,
    label_names: Optional[List[str]] = None,
    data_format: Optional[str] = None,
) -> Dict:
    """Post-process raw SegFormer outputs into semantic segmentation results.

    Takes the raw logits from SegFormer, computes the argmax class map,
    optionally resizes to the original image size, and maps class indices
    to human-readable names.

    Args:
        outputs: Raw model output tensor of shape ``(1, H, W, num_classes)``
            when ``data_format="channels_last"`` or
            ``(1, num_classes, H, W)`` when ``data_format="channels_first"``.
        target_size: Original image ``(height, width)`` for resizing the
            prediction mask. If ``None``, the mask is returned at model
            output resolution.
        label_names: Custom class name list for mapping label indices to
            names. If ``None``, defaults to ADE20K class names (150
            classes). Provide this when using a model fine-tuned on a
            custom dataset (e.g. Cityscapes names via
            ``zeromodels.utils.labels_util.CITYSCAPES_19_CLASSES``).
        data_format: Layout of the channel axis in ``outputs``. ``None``
            resolves to the global setting from
            ``keras.config.image_data_format()``.

    Returns:
        Dict with:
            - ``"segmentation"``: Integer array of shape ``(H, W)`` with
              class indices.
            - ``"class_names"``: List of unique class names detected in the
              image.
            - ``"unique_classes"``: Array of unique class indices.

    Example:
        ```python
        from zeromodels.models.segformer import (
            SegFormerB0, SegFormerImageProcessor, segformer_post_process_semantic_segmentation,
        )

        model = SegFormerB0(weights="ade20k_512", input_shape=(512, 512, 3))
        proc = SegFormerImageProcessor()
        img = proc("photo.jpg")
        output = model(img, training=False)
        result = segformer_post_process_semantic_segmentation(output, target_size=(orig_h, orig_w))
        print(result["class_names"])
        ```
    """
    _names = label_names if label_names is not None else ADE20K_150_CLASSES

    logits = keras.ops.convert_to_numpy(outputs)
    channel_axis = 0 if get_data_format(data_format) == "channels_first" else -1
    # Upsample the logits, then take the argmax. Doing it the other way round
    # quantises the decision to the model's output stride and gives visibly
    # blocky boundaries.
    if target_size is not None:
        if channel_axis == 0:
            logits = np.transpose(logits, (0, 2, 3, 1))
        resized = keras.ops.image.resize(
            logits, (target_size[0], target_size[1]), interpolation="bilinear"
        )
        pred_mask = np.argmax(keras.ops.convert_to_numpy(resized)[0], axis=-1)
    else:
        pred_mask = np.argmax(logits[0], axis=channel_axis)  # (H, W)

    unique_classes = np.unique(pred_mask)
    class_names = [
        _names[c] if c < len(_names) else f"class_{c}" for c in unique_classes
    ]

    return {
        "segmentation": pred_mask,
        "class_names": class_names,
        "unique_classes": unique_classes,
    }
