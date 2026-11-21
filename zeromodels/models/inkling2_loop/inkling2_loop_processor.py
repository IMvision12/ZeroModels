import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseProcessor

from .inkling2_loop_image_processor import Inkling2ImageProcessor
from .inkling2_loop_tokenizer import IMAGE_TOKEN, Inkling2LoopTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2LoopProcessor(BaseProcessor):
    """Multimodal processor (``Inkling2ImageProcessor`` + ``Inkling2LoopTokenizer``)."""

    TOKENIZER_CLS = Inkling2LoopTokenizer
    IMAGE_PROCESSOR_CLS = Inkling2ImageProcessor
    COMPONENTS = ("tokenizer", "image_processor")

    def __init__(
        self,
        hf_id=None,
        tokenizer=None,
        image_processor=None,
        image_token=IMAGE_TOKEN,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hf_id = hf_id
        self.image_token = image_token
        self.image_processor = image_processor or Inkling2ImageProcessor()
        self.tokenizer = tokenizer or Inkling2LoopTokenizer(hf_id=hf_id)

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(hf_id=repo, **kwargs)

    def expand_image_tokens(self, text, image_grids):
        parts = text.split(self.image_token)
        if len(parts) - 1 != len(image_grids):
            raise ValueError(
                f"{len(parts) - 1} {self.image_token} placeholders but "
                f"{len(image_grids)} images were given."
            )
        out = parts[0]
        for grid, part in zip(image_grids, parts[1:]):
            n_tokens = int(np.prod(grid))
            out += (self.image_token * n_tokens) + part
        return out

    def apply_chat_template(
        self, messages, add_generation_prompt=True, reasoning_effort="high"
    ):
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            reasoning_effort=reasoning_effort,
        )

    def call(
        self,
        text=None,
        images=None,
        messages=None,
        add_generation_prompt=True,
        reasoning_effort="high",
    ):
        if messages is not None:
            text = self.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                reasoning_effort=reasoning_effort,
            )
            if images is None:
                extracted = []
                for msg in messages:
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "image":
                                img = item.get("image") or item.get("url")
                                if img is not None:
                                    extracted.append(img)
                if extracted:
                    images = extracted

        out = {}
        image_grids = None
        if images is not None:
            img_out = self.image_processor(images)
            out["pixel_values"] = ops.convert_to_tensor(
                img_out["pixel_values"], dtype="float32"
            )
            out["image_grid_thw"] = ops.convert_to_tensor(
                img_out["image_grid_thw"], dtype="int32"
            )
            image_grids = img_out["image_grid_thw"]

        if text is not None:
            texts = [text] if isinstance(text, str) else list(text)
            if image_grids is not None:
                if len(texts) != 1:
                    raise ValueError(
                        "Batch text with images requires expanding per-sample grids."
                    )
                texts = [self.expand_image_tokens(texts[0], image_grids)]
            seqs = [self.tokenizer.encode(t) for t in texts]
            pad_id = self.tokenizer.pad_token_id or 0
            input_ids, attention_mask = self.tokenizer.pad_batch(seqs, pad_value=pad_id)
            out["input_ids"] = ops.convert_to_tensor(input_ids, dtype="int32")
            out["attention_mask"] = ops.convert_to_tensor(attention_mask, dtype="int32")
        return out

    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, sequences, skip_special_tokens=True):
        return self.tokenizer.batch_decode(
            sequences, skip_special_tokens=skip_special_tokens
        )

    def get_config(self):
        config = super().get_config()
        config.update({"hf_id": self.hf_id, "image_token": self.image_token})
        return config
