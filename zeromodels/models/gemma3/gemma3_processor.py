import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseProcessor

from .gemma3_image_processor import Gemma3ImageProcessor
from .gemma3_tokenizer import Gemma3Tokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3Processor(BaseProcessor):
    """Image + text -> model inputs for the Gemma 3 multimodal models.

    Composes the tokenizer and image processor. ``call`` renders the
    ``<start_of_turn>`` chat template (each ``{"type": "image"}`` content item
    becomes one ``<start_of_image>`` marker), resizes the images, expands
    every marker to the full image sequence: two newlines +
    ``<start_of_image>`` + ``<image_soft_token>`` x ``mm_tokens_per_image`` +
    ``<end_of_image>`` + two newlines, and tokenizes to padded
    ``{"input_ids", "attention_mask"}`` (bos prepended) alongside
    ``pixel_values``.

    Args:
        hf_id: Hub repo for the tokenizer's ``tokenizer.json``.
        mm_tokens_per_image: Soft tokens per image (256).
        tokenizer / image_processor: Optional pre-built components.
    """

    TOKENIZER_CLS = Gemma3Tokenizer
    IMAGE_PROCESSOR_CLS = Gemma3ImageProcessor
    COMPONENTS = ("tokenizer", "image_processor")

    def __init__(
        self,
        hf_id=None,
        mm_tokens_per_image=256,
        tokenizer=None,
        image_processor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hf_id = hf_id
        self.mm_tokens_per_image = mm_tokens_per_image
        self.image_processor = image_processor or Gemma3ImageProcessor()
        self.tokenizer = tokenizer or Gemma3Tokenizer(hf_id=hf_id)

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(hf_id=repo, **kwargs)

    def full_image_sequence(self):
        tok = self.tokenizer
        return (
            "\n\n"
            + tok.boi_token
            + tok.image_token * self.mm_tokens_per_image
            + tok.eoi_token
            + "\n\n"
        )

    def load_image(self, item):
        from PIL import Image

        if item.get("image") is not None:
            return item["image"]
        if item.get("path") is not None:
            return Image.open(item["path"])
        if item.get("url") is not None:
            import io
            import urllib.request

            with urllib.request.urlopen(item["url"]) as resp:
                return Image.open(io.BytesIO(resp.read()))
        raise ValueError("Image content item needs a 'path', 'url', or 'image'.")

    def extract_images(self, conversation):
        images = []
        for msg in conversation:
            content = msg.get("content")
            if isinstance(content, (list, tuple)):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        images.append(self.load_image(item))
        return images or None

    def call(
        self,
        conversation=None,
        text=None,
        images=None,
        messages=None,
        add_generation_prompt=True,
    ):
        if conversation is not None:
            messages = conversation
            if images is None:
                images = self.extract_images(conversation)
        if messages is not None:
            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt)
        if text is None:
            raise ValueError("Provide a `conversation`, `messages`, or `text`.")
        texts = [text] if isinstance(text, str) else list(text)

        out = {}
        if images is not None:
            image_inputs = self.image_processor(images)
            out["pixel_values"] = ops.convert_to_tensor(image_inputs["pixel_values"])
            seq = self.full_image_sequence()
            texts = [t.replace(self.tokenizer.boi_token, seq) for t in texts]

        bos = self.tokenizer.bos_token_id
        ids = [[bos] + self.tokenizer.encode(t) for t in texts]
        max_len = max(len(x) for x in ids)
        input_ids = np.zeros((len(ids), max_len), dtype="int32")
        attention_mask = np.zeros((len(ids), max_len), dtype="int32")
        for i, s in enumerate(ids):
            input_ids[i, : len(s)] = s
            attention_mask[i, : len(s)] = 1
        out["input_ids"] = ops.convert_to_tensor(input_ids)
        out["attention_mask"] = ops.convert_to_tensor(attention_mask)
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {"hf_id": self.hf_id, "mm_tokens_per_image": self.mm_tokens_per_image}
        )
        return config
