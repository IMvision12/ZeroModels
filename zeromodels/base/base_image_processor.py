import numpy as np
from keras import ops
from PIL import Image

from zeromodels.base.base_mixin import PreprocessorMixin
from zeromodels.utils.image_util import get_data_format, load_image

PIL_RESAMPLE_TO_INTERPOLATION = {0: "nearest", 2: "bilinear", 3: "bicubic"}

HF_PREPROCESSOR_DIRECT_PARAMS = (
    "do_resize",
    "do_center_crop",
    "do_rescale",
    "do_normalize",
    "do_pad",
    "do_convert_rgb",
    "do_flip_channel_order",
    "rescale_factor",
)


class BaseImageProcessor(PreprocessorMixin):
    """Abstract base for zeromodels image preprocessors.

    Subclasses implement ``call(images)`` returning the model-ready pixel tensor
    (or a dict that includes one). The loading API (``from_weights`` /
    ``from_variant``) and the ``__call__`` -> ``call`` forwarder are inherited
    from :class:`PreprocessorMixin`; ``from_hf`` is overridden here to map the
    repo's ``preprocessor_config.json`` onto whatever constructor params the
    subclass exposes: ``image_mean`` / ``image_std`` (also as ``mean`` / ``std``),
    ``size`` / ``crop_size`` (any HF form: int, ``[h, w]``, ``height``/``width``,
    ``shortest_edge``/``longest_edge``: reshaped to the form the subclass's own
    default uses, or reduced to a scalar for ``image_resolution`` /
    ``target_size`` / ``target_length``), ``resample`` (PIL code -> keras
    interpolation name) and the ``do_*`` / ``rescale_factor`` passthroughs.
    Explicit caller kwargs always win; a missing config falls back to the
    subclass defaults. Concrete subclasses define their own constructor kwargs
    (resolution, normalization stats, interpolation mode, patch size, …) and
    ``get_config`` payload: the base bakes in no defaults.

    Provides backend-agnostic (``keras.ops``) building blocks shared by the pixel
    pipelines so processors don't re-derive them: the dtype/channel atoms
    :meth:`to_3_channels` (grayscale/RGBA -> RGB) and :meth:`to_unit_range` (cast +
    0-255 -> [0, 1]); the transform atoms :meth:`rescale` (x * scale), :meth:`resize`,
    :meth:`center_crop` (center crop, zero-padded if smaller), :meth:`pad`,
    :meth:`normalize_image` (data-format-aware ``(x - mean) / std``) and the composite
    :meth:`rescale_and_normalize`; the per-channel normalization constants
    (``OPENAI_CLIP_*``, ``IMAGENET_STANDARD_*``, ``IMAGENET_INCEPTION_*``); and
    :meth:`preprocess_image`, the one-shot ``load -> resize -> rescale -> normalize ->
    transpose`` pipeline (detection / segmentation / depth processors). The per-model
    *resize policy* (aspect-preserving vs square / shortest-edge / smart-resize) stays
    in the concrete processor; only the primitives live here.
    """

    OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
    IMAGENET_STANDARD_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STANDARD_STD = (0.229, 0.224, 0.225)
    IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
    IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

    DEFAULT_SIZE = 224
    DEFAULT_CROP_SIZE = 224
    DEFAULT_RESAMPLE = "bicubic"
    DEFAULT_DO_CENTER_CROP = False
    DEFAULT_MEAN = IMAGENET_STANDARD_MEAN
    DEFAULT_STD = IMAGENET_STANDARD_STD

    def __init__(
        self,
        size=None,
        crop_size=None,
        resample=None,
        do_resize=True,
        do_center_crop=None,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        antialias=True,
        return_tensor=True,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = self.as_size(size if size is not None else self.DEFAULT_SIZE)
        self.crop_size = self.as_size(
            crop_size if crop_size is not None else self.DEFAULT_CROP_SIZE
        )
        self.resample = resample if resample is not None else self.DEFAULT_RESAMPLE
        self.do_resize = do_resize
        self.do_center_crop = (
            do_center_crop
            if do_center_crop is not None
            else self.DEFAULT_DO_CENTER_CROP
        )
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = (
            tuple(image_mean) if image_mean is not None else self.DEFAULT_MEAN
        )
        self.image_std = tuple(image_std) if image_std is not None else self.DEFAULT_STD
        self.antialias = antialias
        self.return_tensor = return_tensor
        self.data_format = data_format

    @staticmethod
    def as_size(size):
        """Normalize ``size`` (int or dict) to ``{"height": H, "width": W}``."""
        if isinstance(size, int):
            return {"height": size, "width": size}
        return dict(size)

    def call(self, image):
        """Generalized classification preprocessing: resize -> optional
        center-crop -> rescale -> normalize, entirely on ``keras.ops``.

        Accepts a file path, PIL image, ``uint8`` array, or a ``keras``/float pixel
        tensor (a list gives a batch); returns a ``(B, H, W, C)`` tensor (or numpy
        if ``return_tensor=False``). Detection / segmentation / multimodal
        processors override this with their own pipeline.
        """
        if isinstance(image, (list, tuple)):
            batch = ops.concatenate(
                [ops.convert_to_tensor(self.call(im)) for im in image], axis=0
            )
            return ops.convert_to_numpy(batch) if not self.return_tensor else batch

        if isinstance(image, (str, Image.Image)):
            image = ops.cast(ops.convert_to_tensor(load_image(image)), "float32")
        else:
            image = ops.convert_to_tensor(image)
            if len(image.shape) == 4:
                image = image[0]
            image = ops.cast(image, "float32")
            max_v = float(ops.convert_to_numpy(ops.max(image)))
            min_v = float(ops.convert_to_numpy(ops.min(image)))
            if max_v <= 1.0 and min_v >= 0.0:
                image = image * 255.0
            elif min_v < 0 or max_v > 255:
                raise ValueError("Tensor values must be in [0, 1] or [0, 255] range")
        if len(image.shape) != 3:
            raise ValueError("Input image must have shape (H, W, C)")

        image = ops.expand_dims(image, axis=0)
        if self.do_resize:
            image = ops.image.resize(
                image,
                size=(self.size["height"], self.size["width"]),
                interpolation=self.resample,
                antialias=self.antialias,
            )
        if self.do_center_crop:
            height, width = image.shape[1], image.shape[2]
            crop_h, crop_w = self.crop_size["height"], self.crop_size["width"]
            top, left = (height - crop_h) // 2, (width - crop_w) // 2
            image = image[:, top : top + crop_h, left : left + crop_w, :]
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
        if not self.return_tensor:
            image = ops.convert_to_numpy(image)
        return image

    def stack_images(self, images):
        """Preprocess each image and concatenate along the batch axis.

        For processors whose ``call`` returns ``{"pixel_values": (1, ...)}``
        for a single image: delegating a list here is what gives them batch
        support. Safe because these resize to one fixed size, so the parts
        always share a shape.
        """
        if len(images) == 0:
            raise ValueError(
                f"{type(self).__name__} received an empty image list; pass at "
                f"least one image."
            )
        values = [self.call(image)["pixel_values"] for image in images]
        if getattr(self, "return_tensor", True):
            return {"pixel_values": ops.concatenate(values, axis=0)}
        return {"pixel_values": np.concatenate(values, axis=0)}

    @classmethod
    def from_hf(cls, repo, **kwargs):
        import inspect
        import json

        params = set(inspect.signature(cls).parameters)
        mappable = params & {
            "mean",
            "std",
            "image_mean",
            "image_std",
            "image_resolution",
            "target_size",
            "target_length",
            "size",
            "crop_size",
            "resample",
            *HF_PREPROCESSOR_DIRECT_PARAMS,
        }
        if not mappable:
            return cls(**kwargs)
        try:
            from huggingface_hub import hf_hub_download

            with open(
                hf_hub_download(repo, "preprocessor_config.json"), encoding="utf-8"
            ) as f:
                hf = json.load(f)
        except Exception:
            return cls(**kwargs)

        for param, key in (
            ("mean", "image_mean"),
            ("std", "image_std"),
            ("image_mean", "image_mean"),
            ("image_std", "image_std"),
        ):
            if param in params and hf.get(key) is not None:
                kwargs.setdefault(param, hf[key])

        size = cls.normalize_hf_size(hf.get("size"))
        crop_size = cls.normalize_hf_size(hf.get("crop_size"))
        # When the HF pipeline center-crops, ``size`` is only the pre-crop
        # resize and ``crop_size`` is what actually reaches the model, so a
        # class with a single size knob must take the crop, not the resize.
        # Classes exposing both ``size`` and ``crop_size`` mirror HF directly.
        if crop_size is not None and hf.get("do_center_crop", True):
            final_size = crop_size
        else:
            final_size = size
        if final_size is not None:
            if "image_resolution" in params:
                kwargs.setdefault("image_resolution", final_size["shortest_edge"])
            if "target_size" in params:
                kwargs.setdefault("target_size", final_size["shortest_edge"])
            if "target_length" in params:
                kwargs.setdefault("target_length", final_size["longest_edge"])
        size_for_param = size if "crop_size" in params else final_size
        if ("size" in params and size_for_param is not None) or (
            "crop_size" in params and crop_size is not None
        ):
            try:
                probe = cls()
            except Exception:
                probe = None
            if probe is not None:
                if "size" in params and size_for_param is not None:
                    shaped = cls.shape_hf_size_like(
                        size_for_param, getattr(probe, "size", None)
                    )
                    if shaped is not None:
                        kwargs.setdefault("size", shaped)
                if "crop_size" in params and crop_size is not None:
                    shaped = cls.shape_hf_size_like(
                        crop_size, getattr(probe, "crop_size", None)
                    )
                    if shaped is not None:
                        kwargs.setdefault("crop_size", shaped)

        if "resample" in params and hf.get("resample") is not None:
            resample = hf["resample"]
            interpolation = (
                resample
                if isinstance(resample, str)
                else PIL_RESAMPLE_TO_INTERPOLATION.get(resample)
            )
            if interpolation is not None:
                kwargs.setdefault("resample", interpolation)

        for param in HF_PREPROCESSOR_DIRECT_PARAMS:
            if param in params and hf.get(param) is not None:
                kwargs.setdefault(param, hf[param])

        return cls(**kwargs)

    @staticmethod
    def normalize_hf_size(value):
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            edge = int(value)
            return {
                "height": edge,
                "width": edge,
                "shortest_edge": edge,
                "longest_edge": edge,
            }
        if isinstance(value, (list, tuple)) and len(value) == 2:
            height, width = int(value[0]), int(value[1])
            return {
                "height": height,
                "width": width,
                "shortest_edge": min(height, width),
                "longest_edge": max(height, width),
            }
        if isinstance(value, dict):
            out = {
                k: int(v)
                for k, v in value.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            height, width = out.get("height"), out.get("width")
            shortest, longest = out.get("shortest_edge"), out.get("longest_edge")
            if height is not None and width is not None:
                out.setdefault("shortest_edge", min(height, width))
                out.setdefault("longest_edge", max(height, width))
            elif shortest is not None:
                out.setdefault("height", shortest)
                out.setdefault("width", shortest)
                out.setdefault("longest_edge", shortest)
            elif longest is not None:
                out.setdefault("height", longest)
                out.setdefault("width", longest)
                out.setdefault("shortest_edge", longest)
            else:
                return None
            return out
        return None

    @staticmethod
    def shape_hf_size_like(canonical, template):
        if isinstance(template, dict):
            if all(k in canonical for k in template):
                return {k: canonical[k] for k in template}
            return None
        if isinstance(template, (int, float)) and not isinstance(template, bool):
            return canonical["shortest_edge"]
        return None

    @staticmethod
    def to_3_channels(image):
        num_channels = ops.shape(image)[-1]
        if num_channels == 1:
            return ops.repeat(image, 3, axis=-1)
        if num_channels == 4:
            return image[..., :3]
        if num_channels == 3:
            return image
        raise ValueError(f"Unsupported number of image channels: {num_channels}")

    @staticmethod
    def to_unit_range(image):
        image = ops.cast(image, "float32")
        return ops.where(ops.greater(ops.max(image), 1.0), image / 255.0, image)

    @staticmethod
    def normalize_image(x, mean, std, data_format=None):
        data_format = get_data_format(data_format)
        mean = ops.convert_to_tensor(mean, dtype="float32")
        std = ops.convert_to_tensor(std, dtype="float32")
        rank = len(x.shape)
        if data_format == "channels_first":
            shape = (1, -1, 1, 1) if rank == 4 else (-1, 1, 1)
        else:
            shape = (1, 1, 1, -1) if rank == 4 else (1, 1, -1)
        mean = ops.reshape(mean, shape)
        std = ops.reshape(std, shape)
        return (x - mean) / std

    @staticmethod
    def rescale(image, scale=1.0 / 255.0):
        return ops.cast(image, "float32") * scale

    @staticmethod
    def resize(image, size, interpolation="bilinear", antialias=True, data_format=None):
        return ops.image.resize(
            image,
            size,
            interpolation=interpolation,
            antialias=antialias,
            data_format=get_data_format(data_format),
        )

    @staticmethod
    def center_crop(image, size, data_format=None):
        data_format = get_data_format(data_format)
        crop_h, crop_w = int(size[0]), int(size[1])
        rank = len(image.shape)
        h_axis, w_axis = (
            (rank - 2, rank - 1)
            if data_format == "channels_first"
            else (rank - 3, rank - 2)
        )
        orig_h, orig_w = int(image.shape[h_axis]), int(image.shape[w_axis])
        top, left = (orig_h - crop_h) // 2, (orig_w - crop_w) // 2

        if orig_h < crop_h or orig_w < crop_w:
            pad_h, pad_w = max(crop_h - orig_h, 0), max(crop_w - orig_w, 0)
            top_pad, left_pad = (pad_h + 1) // 2, (pad_w + 1) // 2  # ceil(pad / 2)
            pad_width = [(0, 0)] * rank
            pad_width[h_axis] = (top_pad, pad_h - top_pad)
            pad_width[w_axis] = (left_pad, pad_w - left_pad)
            image = ops.pad(image, pad_width, constant_values=0)
            top, left = top + top_pad, left + left_pad

        top, left = max(top, 0), max(left, 0)
        starts = [0] * rank
        sizes = [int(d) for d in image.shape]
        starts[h_axis], sizes[h_axis] = top, crop_h
        starts[w_axis], sizes[w_axis] = left, crop_w
        return ops.slice(image, starts, sizes)

    @staticmethod
    def pad(image, padding, mode="constant", constant_values=0.0, data_format=None):
        data_format = get_data_format(data_format)
        if isinstance(padding, int):
            hw = ((padding, padding), (padding, padding))
        elif len(padding) == 2 and isinstance(padding[0], int):
            hw = (tuple(padding), tuple(padding))
        else:
            hw = (tuple(padding[0]), tuple(padding[1]))
        rank = len(image.shape)
        if data_format == "channels_first":
            pad_width = [(0, 0)] * (rank - 2) + [hw[0], hw[1]]
        else:
            pad_width = [(0, 0)] * (rank - 3) + [hw[0], hw[1], (0, 0)]
        if mode == "constant":
            return ops.pad(
                image, pad_width, mode="constant", constant_values=constant_values
            )
        return ops.pad(image, pad_width, mode=mode)

    @staticmethod
    def rescale_and_normalize(
        image,
        do_rescale=True,
        scale=1.0 / 255.0,
        do_normalize=True,
        mean=None,
        std=None,
        data_format=None,
    ):
        if do_rescale:
            image = BaseImageProcessor.rescale(image, scale)
        if do_normalize:
            image = BaseImageProcessor.normalize_image(
                image, mean, std, data_format=data_format
            )
        return image

    @staticmethod
    def preprocess_image(
        images,
        target_size,
        image_mean=None,
        image_std=None,
        rescale=True,
        interpolation="bilinear",
        antialias=True,
        data_format=None,
    ):
        data_format = get_data_format(data_format)

        if isinstance(images, (list, tuple)):
            items = list(images)
        elif isinstance(images, np.ndarray) and images.ndim == 4:
            items = [images[i] for i in range(images.shape[0])]
        else:
            items = [images]
        if not items:
            raise ValueError("`images` must contain at least one image.")

        loaded = [load_image(img) for img in items]
        original_sizes = [(int(arr.shape[0]), int(arr.shape[1])) for arr in loaded]

        if isinstance(target_size, int):
            target_h = target_w = target_size
        else:
            target_h, target_w = target_size

        per_image = []
        for arr in loaded:
            t = ops.convert_to_tensor(arr, dtype="float32")
            t = ops.expand_dims(t, axis=0)
            t = ops.image.resize(
                t,
                size=(target_h, target_w),
                interpolation=interpolation,
                antialias=antialias,
                data_format="channels_last",
            )
            per_image.append(t)

        x = ops.concatenate(per_image, axis=0)

        if rescale:
            x = x / 255.0

        if image_mean is not None:
            if image_std is None:
                raise ValueError("image_std must be provided when image_mean is set.")
            x = BaseImageProcessor.normalize_image(
                x, image_mean, image_std, data_format="channels_last"
            )

        if data_format == "channels_first":
            x = ops.transpose(x, (0, 3, 1, 2))

        return x, original_sizes, (target_h, target_w), data_format
