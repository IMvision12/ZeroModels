import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseProcessor

from .mistral3_image_processor import Mistral3ImageProcessor
from .mistral3_tokenizer import Mistral3Tokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class Mistral3Processor(BaseProcessor):
    """Image + text -> model inputs for the Mistral 3 (Small 3.1/3.2) models.

    Composes the tokenizer and the variable-resolution image processor.
    ``call`` renders the ``[INST]`` chat template (each ``{"type": "image"}``
    content item becomes one ``[IMG]`` placeholder), resizes the images,
    expands every placeholder to the merged-grid token rows:
    ``[IMG] * (w // merged_patch)`` + ``[IMG_BREAK]`` per row, with the final
    ``[IMG_BREAK]`` replaced by ``[IMG_END]``, and tokenizes to padded
    ``{"input_ids", "attention_mask"}`` (bos prepended) alongside
    ``pixel_values`` / ``image_sizes``.

    Args:
        hf_id: Hub repo for the tokenizer's ``tokenizer.json``.
        patch_size: Vision patch size in pixels (14).
        spatial_merge_size: Projector patch-merge factor (2), the effective
            token stride is ``patch_size * spatial_merge_size``.
        tokenizer / image_processor: Optional pre-built components.
    """

    TOKENIZER_CLS = Mistral3Tokenizer
    IMAGE_PROCESSOR_CLS = Mistral3ImageProcessor
    COMPONENTS = ("tokenizer", "image_processor")

    def __init__(
        self,
        hf_id=None,
        patch_size=14,
        spatial_merge_size=2,
        tokenizer=None,
        image_processor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hf_id = hf_id
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.image_processor = image_processor or Mistral3ImageProcessor(
            patch_size=patch_size
        )
        self.tokenizer = tokenizer or Mistral3Tokenizer(hf_id=hf_id)
        self.image_token = self.tokenizer.image_token
        self.image_break_token = self.tokenizer.image_break_token
        self.image_end_token = self.tokenizer.image_end_token

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(hf_id=repo, **kwargs)

    def apply_chat_template(self, messages, add_generation_prompt=True, system=None):
        if messages and messages[0].get("role") == "system":
            system = messages[0]["content"]
            messages = messages[1:]
        text = ""
        user_index = 0
        for m in messages:
            if m["role"] == "user":
                content = m["content"]
                if isinstance(content, str):
                    body = content
                else:
                    body = ""
                    for item in content:
                        if item.get("type") == "image":
                            body += self.image_token
                        elif item.get("type") == "text":
                            body += item["text"]
                if user_index == 0 and system is not None:
                    body = f"{system}\n\n{body}"
                text += f"[INST]{body}[/INST]"
                user_index += 1
            else:
                text += f"{m['content']}{self.tokenizer.eos_token}"
        return text

    def expand_image_tokens(self, text, image_sizes):
        merged = self.patch_size * self.spatial_merge_size
        parts = text.split(self.image_token)
        n = len(parts) - 1
        if n != len(image_sizes):
            raise ValueError(
                f"{n} {self.image_token} placeholders but {len(image_sizes)} "
                f"images were given."
            )
        out = parts[0]
        for i, (height, width) in enumerate(image_sizes):
            rows = int(height) // merged
            cols = int(width) // merged
            tokens = ([self.image_token] * cols + [self.image_break_token]) * rows
            tokens[-1] = self.image_end_token
            out += "".join(tokens) + parts[i + 1]
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
            out["image_sizes"] = ops.convert_to_tensor(image_inputs["image_sizes"])
            sizes = np.asarray(image_inputs["image_sizes"]).tolist()
            per_text = self.deal_per_text(texts, self.image_token, sizes)
            texts = [self.expand_image_tokens(t, s) for t, s in zip(texts, per_text)]

        bos = self.tokenizer.bos_token_id
        ids = [[bos] + self.tokenizer.encode(t) for t in texts]
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
                "hf_id": self.hf_id,
                "patch_size": self.patch_size,
                "spatial_merge_size": self.spatial_merge_size,
            }
        )
        return config
