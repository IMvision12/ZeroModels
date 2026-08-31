import keras

from zeromodels.base import BaseProcessor
from zeromodels.models.clip.clip_image_processor import CLIPImageProcessor
from zeromodels.models.clip.clip_tokenizer import CLIPTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class CLIPProcessor(BaseProcessor):
    """Combined processor for CLIP: image processing + text tokenization.

    Composes a :class:`CLIPImageProcessor` and a :class:`CLIPTokenizer` behind one
    callable. ``processor(text=..., images=...)`` returns the tokenizer outputs
    (``input_ids`` / ``attention_mask``) plus ``images`` (pixel tensor). Pass
    ``image_paths=`` to load images from disk.

    Construction:

    * ``CLIPProcessor.from_weights("clip_vit_base_16")``: zeromodels release.
    * ``CLIPProcessor.from_weights("hf:openai/clip-vit-base-patch16")``: pulls the
      tokenizer files **and** builds the image processor from the HF repo.
    * ``CLIPProcessor(variant="clip_vit_base_16")``: build from a zeromodels variant
      (no default variant); or pass pre-built ``tokenizer=`` / ``image_processor=``,
      or per-component build kwargs (``image_resolution=`` …).
    """

    TOKENIZER_CLS = CLIPTokenizer
    IMAGE_PROCESSOR_CLS = CLIPImageProcessor

    def __init__(
        self,
        image_resolution=224,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
        do_center_crop=True,
        do_normalize=True,
        do_resize=True,
        variant=None,
        tokenizer_file=None,
        max_seq_len=77,
        unk_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        tokenizer=None,
        image_processor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_processor = image_processor or CLIPImageProcessor(
            image_resolution=image_resolution,
            mean=mean,
            std=std,
            do_center_crop=do_center_crop,
            do_normalize=do_normalize,
            do_resize=do_resize,
        )
        self.tokenizer = tokenizer or CLIPTokenizer(
            variant=variant,
            tokenizer_file=tokenizer_file,
            max_seq_len=max_seq_len,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
        )

    def call(self, text=None, images=None, image_paths=None):
        if text is None and images is None and image_paths is None:
            raise ValueError(
                "At least one of 'text', 'images', or 'image_paths' must be provided"
            )
        if images is not None and image_paths is not None:
            raise ValueError("Cannot specify both 'images' and 'image_paths'")
        if image_paths is not None and isinstance(image_paths, (list, tuple)):
            if len(image_paths) == 0:
                raise ValueError("image_paths cannot be an empty list")

        encoding = {}
        if text is not None:
            encoding.update(self.tokenizer(inputs=text))
        if images is not None:
            encoding["images"] = self.image_processor(images)["pixel_values"]
        if image_paths is not None:
            encoding["images"] = self.image_processor(image_paths)["pixel_values"]
        return encoding
