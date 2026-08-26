import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseProcessor

from .deepseek_vl_image_processor import DeepseekVLImageProcessor
from .deepseek_vl_tokenizer import DeepseekVLTokenizer

SYSTEM_PROMPT = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language."
)


@keras.saving.register_keras_serializable(package="zeromodels")
class DeepseekVLProcessor(BaseProcessor):
    """Image + text -> model inputs for DeepSeek-VL.

    Composes the tokenizer and image processor. Like the HF processor, every
    ``<image_placeholder>`` in the prompt is expanded to ``num_image_tokens``
    (576) copies before tokenizing (BOS prepended). ``call`` also accepts an
    OpenAI-style ``conversation``, rendered with the DeepSeek-VL chat format
    (``"{system}\\n\\nUser: ...\\n\\nAssistant:"``; each image content item
    becomes one ``<image_placeholder>``).

    Args:
        hf_id: Hub repo for the tokenizer's ``tokenizer.json``.
        num_image_tokens: Vision tokens per image ((384 / 16)**2 = 576).
        tokenizer / image_processor: Optional pre-built components.
    """

    TOKENIZER_CLS = DeepseekVLTokenizer
    IMAGE_PROCESSOR_CLS = DeepseekVLImageProcessor
    COMPONENTS = ("tokenizer", "image_processor")

    def __init__(
        self,
        variant=None,
        hf_id=None,
        num_image_tokens=576,
        tokenizer=None,
        image_processor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.variant = variant
        self.hf_id = hf_id
        self.num_image_tokens = num_image_tokens
        self.image_processor = image_processor or DeepseekVLImageProcessor()
        self.tokenizer = tokenizer or DeepseekVLTokenizer(variant=variant, hf_id=hf_id)
        self.image_token = self.tokenizer.image_token

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(hf_id=repo, **kwargs)

    def apply_chat_template(self, messages, add_generation_prompt=True):
        roles = {"user": "User", "assistant": "Assistant"}
        system = SYSTEM_PROMPT
        turns = []
        for msg in messages:
            content = msg["content"]
            if not isinstance(content, str):
                parts = []
                for item in content:
                    if item.get("type") == "image":
                        parts.append(self.image_token)
                    elif item.get("type") == "text":
                        parts.append(item["text"])
                content = "".join(parts)
            if msg["role"] == "system":
                system = content
                continue
            turns.append(f"{roles[msg['role']]}: {content}")
        text = "\n\n".join([system] + turns)
        if add_generation_prompt:
            text += "\n\nAssistant:"
        return text

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

    def expand_image_tokens(self, text):
        return text.replace(self.image_token, self.image_token * self.num_image_tokens)

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
        texts = [self.expand_image_tokens(t) for t in texts]

        out = {}
        if images is not None:
            out["pixel_values"] = ops.convert_to_tensor(
                self.image_processor(images)["pixel_values"]
            )
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
        config.update(
            {
                "variant": self.variant,
                "hf_id": self.hf_id,
                "num_image_tokens": self.num_image_tokens,
            }
        )
        return config
