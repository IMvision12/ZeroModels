import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseProcessor

from .locateanything_image_processor import LocateAnythingImageProcessor
from .locateanything_tokenizer import LocateAnythingTokenizer

DEFAULT_SYSTEM = "You are a helpful assistant."

TASK_PROMPTS = {
    "detection": (
        "Locate all the instances that matches the following description: {text}."
    ),
    "phrase_grounding": (
        "Locate a single instance that matches the following description: {text}."
    ),
    "referring": (
        "Locate all the instances that match the following description: {text}."
    ),
    "text_grounding": "Please locate the text referred as {text}.",
    "ocr": "Detect all the text in box format.",
    "layout": "Locate the region that matches the following description: {text}.",
    "pointing": "Point to: {text}.",
}


def locate_prompt(task, text=""):
    """Build a LocateAnything instruction for one of its grounding tasks.

    The templates are the verbatim instruction strings used by the official
    model-card worker (``LocateAnythingWorker``). ``task`` is one of:
    ``detection`` (object detection / layout over a category list),
    ``referring`` (multi-instance phrase grounding), ``phrase_grounding``
    (a single instance), ``text_grounding`` and ``ocr`` (OCR: locate named text
    / detect all text), ``layout`` (region / layout grounding), or ``pointing``.
    ``text`` fills the category list or phrase and is ignored by ``ocr``; for
    ``detection`` pass a list of categories (joined with the official ``</c>``
    separator) or a pre-joined string. Put the returned string in the user
    message alongside the image, then parse the answer with the tokenizer's
    ``parse_boxes`` / ``parse_points`` / ``parse_grounding``."""
    if task not in TASK_PROMPTS:
        raise ValueError(f"Unknown task {task!r}; choose from {sorted(TASK_PROMPTS)}")
    if not isinstance(text, str):
        text = "</c>".join(text)
    return TASK_PROMPTS[task].format(text=text)


@keras.saving.register_keras_serializable(package="zeromodels")
class LocateAnythingProcessor(BaseProcessor):
    """Image + text -> model inputs for LocateAnything-3B.

    Composes the tokenizer and the native-resolution MoonViT image processor.
    ``call`` renders the ChatML template (each image content item is one
    ``<IMG_CONTEXT>`` placeholder), preprocesses the images to get each one's
    patch grid, expands every placeholder to ``<img>`` +
    ``<IMG_CONTEXT>`` x (``h*w // merge**2``) + ``</img>`` (so the count matches
    MoonViT's merged-token output), and tokenizes to padded
    ``{"input_ids", "attention_mask"}`` alongside ``pixel_values`` /
    ``image_grid_hws``. Build per-task instructions with :func:`locate_prompt`,
    and decode answers with ``parse_boxes`` / ``parse_points`` /
    ``parse_grounding``.
    """

    TOKENIZER_CLS = LocateAnythingTokenizer
    IMAGE_PROCESSOR_CLS = LocateAnythingImageProcessor
    COMPONENTS = ("tokenizer", "image_processor")

    def __init__(
        self,
        variant=None,
        hf_id=None,
        tokenizer=None,
        image_processor=None,
        merge_kernel_size=(2, 2),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.variant = variant
        self.hf_id = hf_id
        self.image_processor = image_processor or LocateAnythingImageProcessor()
        self.tokenizer = tokenizer or LocateAnythingTokenizer(
            variant=variant, hf_id=hf_id
        )
        self.merge_kernel_size = tuple(merge_kernel_size)
        self.image_token = self.tokenizer.image_token
        self.image_start_token = self.tokenizer.image_start_token
        self.image_end_token = self.tokenizer.image_end_token

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(hf_id=repo, **kwargs)

    def apply_chat_template(
        self, messages, add_generation_prompt=True, system=DEFAULT_SYSTEM
    ):
        text = ""
        if system is not None and not any(m.get("role") == "system" for m in messages):
            text += f"<|im_start|>system\n{system}<|im_end|>\n"
        for msg in messages:
            text += f"<|im_start|>{msg['role']}\n"
            content = msg["content"]
            if isinstance(content, str):
                text += content
            else:
                for item in content:
                    if item.get("type") == "image" or "image" in item:
                        text += self.image_token
                    elif item.get("type") == "text" or "text" in item:
                        text += item.get("text", "")
            text += "<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def expand_image_tokens(self, text, grid_hws):
        parts = text.split(self.image_token)
        n = len(parts) - 1
        if n != len(grid_hws):
            raise ValueError(
                f"{n} image placeholders but {len(grid_hws)} images were given."
            )
        kh, kw = self.merge_kernel_size
        out = parts[0]
        for i, (h, w) in enumerate(grid_hws):
            num_tokens = (int(h) * int(w)) // (kh * kw)
            block = (
                f"<image {i + 1}>"
                + self.image_start_token
                + self.image_token * num_tokens
                + self.image_end_token
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
                    if isinstance(item, dict) and (
                        item.get("type") == "image" or "image" in item
                    ):
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
            out["image_grid_hws"] = ops.convert_to_tensor(
                image_inputs["image_grid_hws"]
            )
            grid = [tuple(g) for g in np.asarray(image_inputs["image_grid_hws"])]
            per_text = self.deal_per_text(texts, self.image_token, grid)
            texts = [self.expand_image_tokens(t, g) for t, g in zip(texts, per_text)]

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

    def parse_boxes(self, ids):
        return self.tokenizer.parse_boxes(ids)

    def parse_points(self, ids):
        return self.tokenizer.parse_points(ids)

    def parse_grounding(self, ids):
        return self.tokenizer.parse_grounding(ids)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "variant": self.variant,
                "hf_id": self.hf_id,
                "merge_kernel_size": list(self.merge_kernel_size),
            }
        )
        return config
