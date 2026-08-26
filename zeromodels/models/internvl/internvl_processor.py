import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseProcessor

from .internvl_image_processor import InternVLImageProcessor
from .internvl_tokenizer import InternVLTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class InternVLProcessor(BaseProcessor):
    """Image + text -> model inputs for the InternVL3 models.

    Composes the tokenizer and the dynamic-tiling image processor. ``call``
    renders the chat template (each ``{"type": "image"}`` content item becomes
    one ``<IMG_CONTEXT>`` placeholder, matching the HF template), tiles the
    images, expands every placeholder to
    ``<img>`` + ``<IMG_CONTEXT>`` x (``image_seq_length`` x tiles) + ``</img>``,
    and tokenizes to padded ``{"input_ids", "attention_mask"}`` alongside
    ``pixel_values``.

    Args:
        hf_id: Hub repo for the tokenizer's ``tokenizer.json``.
        image_seq_length: Vision tokens per tile,
            ``(size // patch_size)**2 * downsample_ratio**2`` (448/14 -> 256).
        tokenizer / image_processor: Optional pre-built components.
    """

    TOKENIZER_CLS = InternVLTokenizer
    IMAGE_PROCESSOR_CLS = InternVLImageProcessor
    COMPONENTS = ("tokenizer", "image_processor")

    def __init__(
        self,
        hf_id=None,
        image_seq_length=256,
        tokenizer=None,
        image_processor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hf_id = hf_id
        self.image_seq_length = image_seq_length
        self.image_processor = image_processor or InternVLImageProcessor()
        self.tokenizer = tokenizer or InternVLTokenizer(hf_id=hf_id)
        self.image_token = self.tokenizer.image_token
        self.start_image_token = self.tokenizer.start_image_token
        self.end_image_token = self.tokenizer.end_image_token

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(hf_id=repo, **kwargs)

    def apply_chat_template(self, messages, add_generation_prompt=True):
        """Render OpenAI-style ``messages`` to the InternVL3 ChatML prompt.

        Mirrors the checkpoint's Jinja template: each ``{"type": "image"}``
        content item becomes a single ``<IMG_CONTEXT>`` placeholder line
        (expanded to the full tile token count later, once the tiling is
        known).
        """
        text = ""
        for msg in messages:
            content = msg["content"]
            text += f"<|im_start|>{msg['role']}\n"
            if isinstance(content, str):
                text += content
            else:
                for item in content:
                    if item.get("type") == "image":
                        text += f"{self.image_token}\n"
                    elif item.get("type") == "text":
                        text += item["text"]
            text += "<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def expand_image_tokens(self, text, num_patches):
        parts = text.split(self.image_token)
        n = len(parts) - 1
        if n != len(num_patches):
            raise ValueError(
                f"{n} {self.image_token} placeholders but {len(num_patches)} "
                f"images were given."
            )
        out = parts[0]
        for i, tiles in enumerate(num_patches):
            block = (
                self.start_image_token
                + self.image_token * (self.image_seq_length * int(tiles))
                + self.end_image_token
            )
            out += block + parts[i + 1]
        return out

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
            texts, extracted = self.render_conversations(
                conversation, add_generation_prompt
            )
            if images is None:
                images = extracted
        elif messages is not None:
            texts = [self.apply_chat_template(messages, add_generation_prompt)]
        elif text is not None:
            texts = [text] if isinstance(text, str) else list(text)
        else:
            raise ValueError("Provide a `conversation`, `messages`, or `text`.")

        out = {}
        if images is not None:
            image_inputs = self.image_processor(images)
            out["pixel_values"] = ops.convert_to_tensor(image_inputs["pixel_values"])
            num_patches = list(image_inputs["num_patches"])
            per_text = self.deal_per_text(texts, self.image_token, num_patches)
            texts = [self.expand_image_tokens(t, n) for t, n in zip(texts, per_text)]

        ids = [self.tokenizer.encode(t) for t in texts]
        max_len = max(len(x) for x in ids)
        input_ids = np.zeros((len(ids), max_len), dtype="int32")
        attention_mask = np.zeros((len(ids), max_len), dtype="int32")
        for i, seq in enumerate(ids):
            input_ids[i, : len(seq)] = seq
            attention_mask[i, : len(seq)] = 1
        out["input_ids"] = ops.convert_to_tensor(input_ids)
        out["attention_mask"] = ops.convert_to_tensor(attention_mask)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({"hf_id": self.hf_id, "image_seq_length": self.image_seq_length})
        return config
