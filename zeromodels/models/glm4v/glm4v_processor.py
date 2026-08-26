import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseProcessor

from .glm4v_image_processor import Glm4vImageProcessor
from .glm4v_tokenizer import Glm4vTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class Glm4vProcessor(BaseProcessor):
    """Image + text -> model inputs for GLM-4V.

    Composes the tokenizer and image processor. ``call`` renders the GLM chat
    template, runs the image processor, expands each ``<|image|>`` placeholder
    (wrapped in ``<|begin_of_image|>`` / ``<|end_of_image|>``) to the right
    number of merged vision tokens, and tokenizes.
    """

    TOKENIZER_CLS = Glm4vTokenizer
    IMAGE_PROCESSOR_CLS = Glm4vImageProcessor
    COMPONENTS = ("tokenizer", "image_processor")

    def __init__(
        self,
        hf_id=None,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokenizer=None,
        image_processor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hf_id = hf_id
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.image_processor = image_processor or Glm4vImageProcessor(
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
        )
        self.tokenizer = tokenizer or Glm4vTokenizer(hf_id=hf_id)
        self.image_token = self.tokenizer.image_token

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(hf_id=repo, **kwargs)

    def apply_chat_template(self, messages, add_generation_prompt=True):
        text = "[gMASK]<sop>"
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text += f"<|{role}|>\n"
            if isinstance(content, str):
                text += content
            else:
                for item in content:
                    if item.get("type") == "image":
                        text += "<|begin_of_image|><|image|><|end_of_image|>"
                    elif item.get("type") == "text":
                        text += item["text"]
        if add_generation_prompt:
            text += "<|assistant|>\n"
        return text

    def expand_pads(self, text, token, grids):
        parts = text.split(token)
        n = len(parts) - 1
        if n != len(grids):
            raise ValueError(
                f"{n} {token} placeholders but {len(grids)} images were given."
            )
        m2 = self.spatial_merge_size**2
        out = parts[0]
        for i, g in enumerate(grids):
            out += token * (int(np.prod(g)) // m2) + parts[i + 1]
        return out

    def _load_image(self, item):
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
                        images.append(self._load_image(item))
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
        image_grids = None
        if images is not None:
            image_inputs = self.image_processor(images)
            out["pixel_values"] = ops.convert_to_tensor(image_inputs["pixel_values"])
            out["image_grid_thw"] = ops.convert_to_tensor(
                image_inputs["image_grid_thw"]
            )
            image_grids = np.asarray(image_inputs["image_grid_thw"])

        if image_grids is not None:
            per_text = self.deal_per_text(texts, self.image_token, image_grids)
            texts = [
                self.expand_pads(t, self.image_token, g)
                for t, g in zip(texts, per_text)
            ]

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
                "hf_id": self.hf_id,
                "patch_size": self.patch_size,
                "spatial_merge_size": self.spatial_merge_size,
                "temporal_patch_size": self.temporal_patch_size,
            }
        )
        return config
