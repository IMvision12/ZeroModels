import keras

from zeromodels.models.clip.clip_image_processor import CLIPImageProcessor


@keras.saving.register_keras_serializable(package="zeromodels")
class MetaClip2ImageProcessor(CLIPImageProcessor):
    """Image processor for MetaCLIP 2: direct square bicubic resize.

    Subclass of :class:`CLIPImageProcessor`, differing only in the resize
    geometry: the reference publishes ``size={"height": N, "width": N}``, i.e.
    the image is stretched straight onto the square rather than resized on its
    shortest edge and cropped (OpenAI CLIP's rule, which the parent keeps).
    Center-cropping to the same square afterwards is then a no-op, and mean /
    std are unchanged from CLIP.

    The variants are not unanimous: ``metaclip-2-worldwide-huge-quickgelu``
    publishes ``size={"shortest_edge": 224}`` and so wants the parent's rule,
    while every other variant checked (``l14``, ``huge-378``) is square. Pass
    ``square_resize=False`` for that one.

    Pixel values are rescaled to ``[0, 1]`` and normalized with the
    OpenAI-CLIP mean / std (MetaCLIP 2 keeps these unchanged from CLIP).

    Args:
        image_resolution: Target square resolution. Defaults to ``224``.
        mean: Per-channel mean for normalization. Defaults to OpenAI
            CLIP's ``(0.48145466, 0.4578275, 0.40821073)``.
        std: Per-channel std for normalization. Defaults to OpenAI
            CLIP's ``(0.26862954, 0.26130258, 0.27577711)``.
        do_center_crop: Whether to center-crop to ``image_resolution`` after
            the resize. Defaults to ``True``.
        do_normalize: Whether to apply mean/std normalization.
            Defaults to ``True``.
        do_resize: Whether to resize images to ``image_resolution``.
            Defaults to ``True``.
        square_resize: Stretch straight onto the square (``True``, the common
            case) or resize on the shortest edge like OpenAI CLIP (``False``,
            for ``huge-quickgelu``). Defaults to ``True``.
        data_format: ``"channels_last"`` / ``"channels_first"`` /
            ``None`` (auto from ``keras.config.image_data_format()``).
        **kwargs: Forwarded to :class:`CLIPImageProcessor`.

    Example:
        >>> from zeromodels.models.metaclip2 import (
        ...     MetaClip2ImageProcessor, MetaClip2ZeroShotClassify,
        ... )
        >>> processor = MetaClip2ImageProcessor(image_resolution=224)
        >>> inputs = processor("photo.jpg")
        >>> inputs["pixel_values"].shape   # (1, 224, 224, 3): channels_last
    """

    def __init__(
        self,
        image_resolution: int = 224,
        mean=CLIPImageProcessor.OPENAI_CLIP_MEAN,
        std=CLIPImageProcessor.OPENAI_CLIP_STD,
        do_center_crop: bool = True,
        do_normalize: bool = True,
        do_resize: bool = True,
        square_resize: bool = True,
        data_format=None,
        **kwargs,
    ):
        super().__init__(
            image_resolution=image_resolution,
            mean=list(mean),
            std=list(std),
            do_center_crop=do_center_crop,
            do_normalize=do_normalize,
            do_resize=do_resize,
            data_format=data_format,
            **kwargs,
        )
        self.square_resize = square_resize

    def target_size(self, height: int, width: int) -> tuple:
        """Square: MetaCLIP 2 stretches onto the target rather than preserving aspect.

        ``square_resize=False`` falls back to the parent's shortest-edge rule,
        which is what the ``huge-quickgelu`` variant publishes.
        """
        if not self.square_resize:
            return super().target_size(height, width)
        return self.image_resolution, self.image_resolution
