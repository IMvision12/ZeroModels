from typing import Dict, List, Optional, Tuple, Union

import keras
from keras import ops
from PIL import Image

from zeromodels.base import BaseImageProcessor
from zeromodels.base.base_image_processor import load_image
from zeromodels.utils.image_util import get_data_format

RESAMPLE_NAMES = {0: "nearest", 2: "bilinear", 3: "bicubic"}


@keras.saving.register_keras_serializable(package="zeromodels")
class BeitImageProcessor(BaseImageProcessor):
    """Preprocess images for BEiT, mirroring transformers' ``BeitImageProcessor``.

    Resizes to a square ``size``, optionally center-crops to ``crop_size``, rescales
    to ``[0, 1]``, and normalizes with 0.5/0.5 statistics (``IMAGENET_STANDARD``).
    Because it normalizes, pair it with a model built with
    ``include_normalization=False``; the models otherwise normalize internally and
    take raw ``[0, 255]`` pixels with no processor. Runs entirely on ``keras.ops``.

    Also provides ``post_process_semantic_segmentation``, which upsamples the
    quarter-resolution segmentation logits to the requested sizes and takes the
    argmax over the class axis, matching the reference implementation.

    Args:
        size: Target size as ``{"height": H, "width": W}`` (or an int). Default:
            ``{"height": 224, "width": 224}``. The ADE20K segmentation checkpoints
            use 640.
        resample: Interpolation (``"nearest"``, ``"bilinear"``, ``"bicubic"``).
            Default ``"bicubic"``.
        do_resize: Whether to resize. Default ``True``.
        do_center_crop: Whether to center-crop to ``crop_size`` after resizing.
            Default ``False``.
        crop_size: Center-crop size. Default ``{"height": 224, "width": 224}``.
        do_rescale: Whether to divide by 255. Default ``True``.
        rescale_factor: Rescale factor. Default ``1/255``.
        do_normalize: Whether to normalize. Default ``True``.
        image_mean: Per-channel mean. Default ``(0.5, 0.5, 0.5)``.
        image_std: Per-channel std. Default ``(0.5, 0.5, 0.5)``.
        antialias: Antialias the resize. Default ``True`` to match transformers'
            fast (torchvision) processor (max|d| ~7e-3 vs ~0.7 without).
        return_tensor: Return a Keras tensor (True) or numpy array.
        data_format: ``"channels_first"`` / ``"channels_last"``; ``None`` resolves
            to ``keras.backend.image_data_format()``.
    """

    def __init__(
        self,
        size: Optional[Union[Dict[str, int], int]] = None,
        resample: str = "bicubic",
        do_resize: bool = True,
        do_center_crop: bool = False,
        crop_size: Optional[Union[Dict[str, int], int]] = None,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        antialias: bool = True,
        return_tensor: bool = True,
        data_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = self.as_size(size, 224)
        self.resample = resample
        self.do_resize = do_resize
        self.do_center_crop = do_center_crop
        self.crop_size = self.as_size(crop_size, 224)
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else (0.5, 0.5, 0.5)
        self.image_std = image_std if image_std is not None else (0.5, 0.5, 0.5)
        self.antialias = antialias
        self.return_tensor = return_tensor
        self.data_format = data_format

    @staticmethod
    def as_size(size, default):
        if size is None:
            return {"height": default, "width": default}
        if isinstance(size, int):
            return {"height": size, "width": size}
        return dict(size)

    def __call__(self, image):
        return self.call(image)

    def call(self, image):
        if isinstance(image, (list, tuple)):
            batch = ops.concatenate(
                [ops.convert_to_tensor(self.call(im)) for im in image], axis=0
            )
            return ops.convert_to_numpy(batch) if not self.return_tensor else batch

        if isinstance(image, (str, Image.Image)):
            if self.do_resize:
                image, _, _, _ = self.preprocess_image(
                    image,
                    target_size=(self.size["height"], self.size["width"]),
                    image_mean=self.image_mean if self.do_normalize else None,
                    image_std=self.image_std if self.do_normalize else None,
                    rescale=self.do_rescale,
                    interpolation=self.resample,
                    antialias=self.antialias,
                    data_format=self.data_format,
                )
            else:
                image = ops.cast(
                    ops.expand_dims(ops.convert_to_tensor(load_image(image)), 0),
                    "float32",
                )
                if self.do_rescale:
                    image = image / 255.0
                if self.do_normalize:
                    image = self.normalize_image(
                        image,
                        self.image_mean,
                        self.image_std,
                        data_format="channels_last",
                    )
                if get_data_format(self.data_format) == "channels_first":
                    image = ops.transpose(image, (0, 3, 1, 2))
        else:
            image = ops.convert_to_tensor(image)
            if len(image.shape) == 4:
                image = image[0]
            if len(image.shape) != 3:
                raise ValueError("Input tensor must have shape (H, W, C)")
            image = ops.cast(image, "float32")
            max_v = float(ops.convert_to_numpy(ops.max(image)))
            min_v = float(ops.convert_to_numpy(ops.min(image)))
            if max_v <= 1.0 and min_v >= 0.0:
                image = image * 255.0
            elif min_v < 0 or max_v > 255:
                raise ValueError("Tensor values must be in [0, 1] or [0, 255] range")
            image = ops.expand_dims(image, axis=0)
            if self.do_resize:
                target = (self.size["height"], self.size["width"])
                if tuple(image.shape[1:3]) != target:
                    image = ops.image.resize(
                        image,
                        size=target,
                        interpolation=self.resample,
                        antialias=self.antialias,
                    )
            if self.do_rescale:
                image = image * self.rescale_factor
            if self.do_normalize:
                mean = ops.reshape(
                    ops.convert_to_tensor(self.image_mean, "float32"), (1, 1, 1, 3)
                )
                std = ops.reshape(
                    ops.convert_to_tensor(self.image_std, "float32"), (1, 1, 1, 3)
                )
                image = (image - mean) / std
            if get_data_format(self.data_format) == "channels_first":
                image = ops.transpose(image, (0, 3, 1, 2))

        if self.do_center_crop:
            image = self.center_crop(image)
        if not self.return_tensor:
            image = ops.convert_to_numpy(image)
        return image

    def center_crop(self, image):
        crop_h, crop_w = self.crop_size["height"], self.crop_size["width"]
        if get_data_format(self.data_format) == "channels_first":
            h, w = image.shape[2], image.shape[3]
            top, left = (h - crop_h) // 2, (w - crop_w) // 2
            return image[:, :, top : top + crop_h, left : left + crop_w]
        h, w = image.shape[1], image.shape[2]
        top, left = (h - crop_h) // 2, (w - crop_w) // 2
        return image[:, top : top + crop_h, left : left + crop_w, :]

    def post_process_semantic_segmentation(self, outputs, target_sizes=None):
        return beit_post_process_semantic_segmentation(outputs, target_sizes)

    @classmethod
    def from_hf(cls, hf_preprocessor_config: Dict):
        c = hf_preprocessor_config
        resample = c.get("resample", 3)
        return cls(
            size=c.get("size"),
            resample=RESAMPLE_NAMES.get(resample, "bicubic"),
            do_resize=c.get("do_resize", True),
            do_center_crop=c.get("do_center_crop", False),
            crop_size=c.get("crop_size"),
            do_rescale=c.get("do_rescale", True),
            rescale_factor=c.get("rescale_factor", 1 / 255),
            do_normalize=c.get("do_normalize", True),
            image_mean=tuple(c["image_mean"]) if c.get("image_mean") else None,
            image_std=tuple(c["image_std"]) if c.get("image_std") else None,
        )


def beit_post_process_semantic_segmentation(
    outputs,
    target_sizes: Optional[List[Tuple[int, int]]] = None,
    data_format: Optional[str] = None,
) -> list:
    """Turn BEiT segmentation logits into per-pixel label maps, matching HF.

    ``BeitSemanticSegment`` returns logits at a quarter of the input resolution.
    For each image this upsamples the logits to the requested target size
    (bilinear, ``align_corners=False``) and takes the argmax over the class axis;
    upsampling before the argmax keeps class boundaries smooth. When
    ``target_sizes`` is ``None`` the maps stay at the model's output resolution.
    Runs on ``keras.ops`` and materializes each map to numpy at the end.

    Args:
        outputs: Model logits, ``(B, H, W, num_classes)`` for ``channels_last`` or
            ``(B, num_classes, H, W)`` for ``channels_first``.
        target_sizes: Per-image ``(height, width)`` to resize each map to (one per
            batch element), matching transformers' ``BeitImageProcessor``.
        data_format: Channel layout of ``outputs``. ``None`` resolves to the global
            ``keras.config.image_data_format()``.

    Returns:
        A list of length ``batch_size``, each a ``(height, width)`` array of class
        ids.
    """
    logits = ops.convert_to_tensor(outputs)
    if get_data_format(data_format) == "channels_first":
        logits = ops.transpose(logits, (0, 2, 3, 1))  # -> (B, H, W, C)

    results = []
    for i in range(int(logits.shape[0])):
        single = logits[i : i + 1]  # (1, H, W, C)
        if target_sizes is not None:
            single = ops.image.resize(
                single,
                (int(target_sizes[i][0]), int(target_sizes[i][1])),
                interpolation="bilinear",
            )
        results.append(ops.convert_to_numpy(ops.argmax(single[0], axis=-1)))
    return results
