from typing import List, Optional, Union

import keras

from zeromodels.base import BaseProcessor
from zeromodels.models.siglip2.siglip2_image_processor import SigLIP2ImageProcessor
from zeromodels.models.siglip2.siglip2_tokenizer import SigLIP2Tokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class SigLIP2Processor(BaseProcessor):
    """Combined processor for SigLIP 2 models: image + Gemma text.

    Pairs :class:`SigLIP2ImageProcessor` (resize / center crop / normalize) with
    :class:`SigLIP2Tokenizer` (Gemma SentencePiece, vocab 256000).
    ``from_weights("hf:google/siglip2-...")`` pulls the Gemma tokenizer **and**
    builds the image processor from the repo; ``from_weights("siglip2_...")`` uses
    the zeromodels release.
    """

    TOKENIZER_CLS = SigLIP2Tokenizer
    IMAGE_PROCESSOR_CLS = SigLIP2ImageProcessor

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
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        tokenizer=None,
        image_processor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_processor = image_processor or SigLIP2ImageProcessor(
            image_resolution=image_resolution,
            mean=mean,
            std=std,
            do_center_crop=do_center_crop,
            do_normalize=do_normalize,
            do_resize=do_resize,
        )
        self.tokenizer = tokenizer or SigLIP2Tokenizer(
            variant=variant,
            tokenizer_file=tokenizer_file,
            max_seq_len=max_seq_len,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
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

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id
