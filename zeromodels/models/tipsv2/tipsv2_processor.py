from typing import List, Optional, Union

import keras

from zeromodels.base import BaseProcessor
from zeromodels.models.tipsv2.tipsv2_image_processor import Tipsv2ImageProcessor
from zeromodels.models.tipsv2.tipsv2_tokenizer import Tipsv2Tokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class Tipsv2Processor(BaseProcessor):
    """Combined image + text processor for TIPSv2.

    Composes :class:`Tipsv2ImageProcessor` and :class:`Tipsv2Tokenizer`.
    ``processor(text=..., images=...)`` returns ``{"images", "token_ids",
    "padding_mask"}`` matching :class:`Tipsv2Model`'s input dict; pass
    ``image_paths=`` to load images from disk.
    """

    TOKENIZER_CLS = Tipsv2Tokenizer
    IMAGE_PROCESSOR_CLS = Tipsv2ImageProcessor

    def __init__(
        self,
        image_resolution: int = 448,
        resample: str = "bilinear",
        do_normalize: bool = False,
        do_resize: bool = True,
        variant: Optional[str] = None,
        tokenizer_file: Optional[str] = None,
        max_seq_len: int = 64,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        tokenizer=None,
        image_processor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_processor = image_processor or Tipsv2ImageProcessor(
            image_resolution=image_resolution,
            resample=resample,
            do_normalize=do_normalize,
            do_resize=do_resize,
        )
        self.tokenizer = tokenizer or Tipsv2Tokenizer(
            variant=variant,
            tokenizer_file=tokenizer_file,
            max_seq_len=max_seq_len,
            unk_token=unk_token,
            pad_token=pad_token,
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

        encoding = {}
        if text is not None:
            tok = self.tokenizer(inputs=text)
            encoding["token_ids"] = tok["input_ids"]
            encoding["padding_mask"] = tok["attention_mask"]
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

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id
