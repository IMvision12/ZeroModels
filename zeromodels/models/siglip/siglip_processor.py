from typing import List, Optional, Union

import keras

from zeromodels.base import BaseProcessor
from zeromodels.models.siglip.siglip_image_processor import SigLIPImageProcessor
from zeromodels.models.siglip.siglip_tokenizer import SigLIPTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class SigLIPProcessor(BaseProcessor):
    """Combined image + text processor for SigLIP.

    Composes :class:`SigLIPImageProcessor` and :class:`SigLIPTokenizer`.
    ``processor(text=..., images=...)`` returns the tokenizer outputs plus
    ``images`` (pixel tensor); pass ``image_paths=`` to load from disk.

    Construction:

    * ``SigLIPProcessor.from_weights("siglip_base_p16_224")``: zeromodels release.
    * ``SigLIPProcessor.from_weights("hf:google/siglip-base-patch16-224")``: pulls the
      SentencePiece tokenizer **and** builds the image processor from the HF repo.
    * ``SigLIPProcessor(variant="siglip_base_p16_224")``: build from a zeromodels
      variant (no default variant); or pass pre-built ``tokenizer=`` /
      ``image_processor=``, or per-component build kwargs. Set ``multilingual=True``
      for the multilingual checkpoints.
    """

    TOKENIZER_CLS = SigLIPTokenizer
    IMAGE_PROCESSOR_CLS = SigLIPImageProcessor

    def __init__(
        self,
        image_resolution: int = 224,
        mean: tuple = (0.5, 0.5, 0.5),
        std: tuple = (0.5, 0.5, 0.5),
        do_center_crop: bool = True,
        do_normalize: bool = True,
        do_resize: bool = True,
        variant: Optional[str] = None,
        tokenizer_file: Optional[str] = None,
        max_seq_len: int = 64,
        unk_token: str = "<unk>",
        pad_token: str = "</s>",
        eos_token: str = "</s>",
        tokenizer=None,
        image_processor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_processor = image_processor or SigLIPImageProcessor(
            image_resolution=image_resolution,
            mean=mean,
            std=std,
            do_center_crop=do_center_crop,
            do_normalize=do_normalize,
            do_resize=do_resize,
        )
        self.tokenizer = tokenizer or SigLIPTokenizer(
            variant=variant,
            tokenizer_file=tokenizer_file,
            max_seq_len=max_seq_len,
            unk_token=unk_token,
            pad_token=pad_token,
            eos_token=eos_token,
        )

    def call(
        self,
        text: Optional[Union[str, List[str]]] = None,
        images: Optional[Union[keras.KerasTensor, List]] = None,
        image_paths: Optional[Union[str, List[str]]] = None,
    ):
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

    def decode_text(
        self, token_ids: keras.KerasTensor, skip_special_tokens: bool = True
    ) -> List[str]:
        return self.tokenizer.batch_decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def get_sequence_length(self, input_ids: keras.KerasTensor) -> keras.KerasTensor:
        return self.tokenizer.get_sequence_length(input_ids)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def unk_token_id(self) -> int:
        return self.tokenizer.unk_token_id
