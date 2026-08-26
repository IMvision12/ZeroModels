import keras
import numpy as np

from zeromodels.base import BaseProcessor

from .kimi_k25_image_processor import KimiK25ImageProcessor
from .kimi_k25_tokenizer import KimiK25Tokenizer

IMAGE_TOKEN = "<|media_pad|>"


@keras.saving.register_keras_serializable(package="zeromodels")
class KimiK25Processor(BaseProcessor):
    """Text + image processor for Kimi K2.5 / K2.6 / K2.7-Code.

    Each ``IMAGE_TOKEN`` in the prompt is expanded to one token per *merged* patch
    (``t * h * w / merge_size**2``). The model zeroes the placeholder before the
    embedding lookup and scatters the projected patches back in.

    Args:
        tokenizer / image_processor: Pre-built components, or omit them to construct
            the defaults.
    """

    TOKENIZER_CLS = KimiK25Tokenizer
    IMAGE_PROCESSOR_CLS = KimiK25ImageProcessor
    COMPONENTS = ("tokenizer", "image_processor")

    def __init__(
        self,
        tokenizer=None,
        image_processor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or KimiK25Tokenizer()
        self.image_processor = image_processor or KimiK25ImageProcessor()

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(
            tokenizer=KimiK25Tokenizer.from_hf(repo),
            image_processor=KimiK25ImageProcessor.from_hf(repo),
            **kwargs,
        )

    def merged_tokens(self, grid):
        merge = self.image_processor.merge_size**2
        return int(np.prod(grid)) // merge

    def expand_images(self, text, grids):
        # Split rather than repeated replace(): once the first marker expands, the
        # next replace() would land inside that span instead of the next marker.
        parts = text.split(IMAGE_TOKEN)
        if len(parts) - 1 != len(grids):
            raise ValueError(
                f"{len(parts) - 1} {IMAGE_TOKEN} placeholders but "
                f"{len(grids)} images were given."
            )
        expanded = parts[0]
        for grid, part in zip(grids, parts[1:]):
            expanded += IMAGE_TOKEN * self.merged_tokens(grid) + part
        return expanded

    def call(self, text, images=None):
        texts = self.tokenizer.normalize_texts(text)
        inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images)
            inputs.update(image_inputs)
            grids = np.asarray(image_inputs["image_grid_thw"]).tolist()
            per_text = self.deal_per_text(texts, IMAGE_TOKEN, grids)
            texts = [self.expand_images(t, g) for t, g in zip(texts, per_text)]

        sequences = [self.tokenizer.encode(t) for t in texts]
        input_ids, attention_mask = self.tokenizer.pad_batch(
            sequences, self.tokenizer.pad_token_id
        )
        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = attention_mask
        return inputs
