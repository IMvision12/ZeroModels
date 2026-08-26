import keras

from zeromodels.base import BaseProcessor

from .metaclip2_image_processor import MetaClip2ImageProcessor
from .metaclip2_tokenizer import MetaClip2Tokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class MetaClip2Processor(BaseProcessor):
    """Combined image + text processor for MetaCLIP 2.

    Bundles :class:`MetaClip2ImageProcessor` (direct square resize +
    OpenAI-CLIP normalization) and :class:`MetaClip2Tokenizer` (XLM-R
    SentencePiece) into a single object so callers can prepare the
    full ``{images, token_ids, padding_mask}`` input dict expected by
    :class:`MetaClip2Model` / :class:`MetaClip2ZeroShotClassify` in
    one call.

    For the mT5-tokenizer MetaCLIP 2 variants (the ``metaclip2_mt5_*``
    family), swap the tokenizer attribute manually after construction
    or instantiate :class:`MetaClip2ImageProcessor` and
    :class:`MetaClip2Mt5Tokenizer` directly.

    Args:
        image_resolution: Square image side length. Defaults to ``224``.
        mean: Per-channel mean for normalization. Defaults to OpenAI
            CLIP's ``(0.48145466, 0.4578275, 0.40821073)``.
        std: Per-channel std for normalization. Defaults to OpenAI
            CLIP's ``(0.26862954, 0.26130258, 0.27577711)``.
        do_center_crop: Whether to center-crop. **MetaCLIP 2 does not
            center-crop**: :class:`MetaClip2ImageProcessor` ignores
            this and always sets it to ``False`` internally. Kept here
            for signature parity with :class:`CLIPProcessor`.
        do_normalize: Whether to apply mean/std normalization.
            Defaults to ``True``.
        do_resize: Whether to resize images to ``image_resolution``.
            Defaults to ``True``.
        data_format: ``"channels_last"`` / ``"channels_first"`` /
            ``None`` (auto from ``keras.config.image_data_format()``).
        sentencepiece_model_file: Path to ``sentencepiece.bpe.model``.
            ``None`` triggers a one-time download from the MetaCLIP 2
            release.
        max_seq_len: Tokenizer max sequence length. Defaults to ``77``.
        **kwargs: Forwarded to :class:`BaseProcessor`.

    Example:
        >>> from zeromodels.models.metaclip2 import (
        ...     MetaClip2Processor, MetaClip2ZeroShotClassify,
        ... )
        >>> processor = MetaClip2Processor.from_weights("metaclip2_worldwide_b32_224")
        >>> model = MetaClip2ZeroShotClassify.from_weights(
        ...     "metaclip2_worldwide_b32_224"
        ... )
        >>> inputs = processor(
        ...     text=["a cat", "un chat", "ein Kater"],
        ...     image_paths="cat.jpg",
        ... )
        >>> out = model({
        ...     "images": inputs["images"],
        ...     "token_ids": inputs["token_ids"],
        ...     "padding_mask": inputs["padding_mask"],
        ... })
        >>> out["image_logits"].shape    # (1, 3)
    """

    TOKENIZER_CLS = MetaClip2Tokenizer
    IMAGE_PROCESSOR_CLS = MetaClip2ImageProcessor

    def __init__(
        self,
        image_resolution: int = 224,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
        do_center_crop: bool = True,
        do_normalize: bool = True,
        do_resize: bool = True,
        data_format=None,
        variant: str = None,
        tokenizer_file: str = None,
        max_seq_len: int = 77,
        tokenizer=None,
        image_processor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_processor = image_processor or MetaClip2ImageProcessor(
            image_resolution=image_resolution,
            mean=mean,
            std=std,
            do_normalize=do_normalize,
            do_resize=do_resize,
            data_format=data_format,
        )
        self.tokenizer = tokenizer or MetaClip2Tokenizer(
            variant=variant,
            tokenizer_file=tokenizer_file,
            max_seq_len=max_seq_len,
        )

    def call(self, text=None, images=None, image_paths=None):
        if text is None and images is None and image_paths is None:
            raise ValueError(
                "At least one of 'text', 'images', or 'image_paths' must be provided"
            )
        if images is not None and image_paths is not None:
            raise ValueError("Cannot specify both 'images' and 'image_paths'")

        encoding = {}
        if text is not None:
            encoding.update(self.tokenizer(inputs=text))
        if images is not None:
            encoding["images"] = self.image_processor(images)["pixel_values"]
        if image_paths is not None:
            encoding["images"] = self.image_processor(image_paths)["pixel_values"]
        return encoding
