import keras
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
    ``image_grid_hws``.

    Grounding is task-driven: ``processor(images=img, task="detection", text="zebra")``
    builds the instruction internally, and :meth:`post_process_generation` turns the
    generated ids into structured (optionally pixel-space) results, so the task drives
    both ends (the Florence-2 pattern). Pass a ``conversation`` instead for a custom or
    multi-turn prompt.
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

    def build_task_conversation(self, task, text=None, images=None):
        """One-shot conversation for a grounding task: the image(s) plus the task
        instruction. ``task`` is one of :data:`TASK_PROMPTS`; ``text`` is the category
        or phrase (a list is joined with the official ``</c>`` separator) and is
        ignored by the ``ocr`` task."""
        if task not in TASK_PROMPTS:
            raise ValueError(
                f"Unknown task {task!r}; choose from {sorted(TASK_PROMPTS)}"
            )
        if text is None:
            text = ""
        elif not isinstance(text, str):
            text = "</c>".join(text)
        prompt = TASK_PROMPTS[task].format(text=text)
        if images is None:
            image_items = []
        elif isinstance(images, (list, tuple)):
            image_items = list(images)
        else:
            image_items = [images]
        content = [{"type": "image", "image": im} for im in image_items]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def call(
        self,
        images=None,
        task=None,
        text=None,
        conversation=None,
        add_generation_prompt=True,
    ):
        if task is not None:
            # Task-driven (Florence-2 style): build the conversation from the task +
            # text + image(s); parse the answer with post_process_generation.
            conversation = self.build_task_conversation(task, text, images)
            images = None  # extracted from the conversation below
        if conversation is None:
            raise ValueError(
                "Provide `task` (with `text` / `images`) or a `conversation`."
            )
        texts, extracted = self.render_conversations(
            conversation, add_generation_prompt
        )
        if images is None:
            images = extracted

        out = {}
        if images is not None:
            image_inputs = self.image_processor(images)
            out["pixel_values"] = ops.convert_to_tensor(image_inputs["pixel_values"])
            out["image_grid_hws"] = ops.convert_to_tensor(
                image_inputs["image_grid_hws"]
            )
            grid = [
                tuple(g) for g in ops.convert_to_numpy(image_inputs["image_grid_hws"])
            ]
            per_text = self.deal_per_text(texts, self.image_token, grid)
            texts = [self.expand_image_tokens(t, g) for t, g in zip(texts, per_text)]

        ids = [self.tokenizer.encode(t) for t in texts]
        max_len = max(len(x) for x in ids)
        input_ids = [list(seq) + [0] * (max_len - len(seq)) for seq in ids]
        attention_mask = [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in ids]
        out["input_ids"] = ops.convert_to_tensor(input_ids, dtype="int32")
        out["attention_mask"] = ops.convert_to_tensor(attention_mask, dtype="int32")
        return out

    def post_process_generation(self, generated, task=None, image_size=None, text=None):
        """Parse a ``model.generate`` output into grounding results (Florence-2 style).

        ``generated`` is the generated id sequence (one sequence or a batch). Parsing
        is universal via the tokenizer's ``parse_grounding`` (the box / ref / coord
        tokens are self-describing), so ``task`` only labels the result. With
        ``image_size=(width, height)`` the ``[0, 1000]`` coordinates are rescaled to
        pixels; without it they stay in the grid. When ``text`` is a single string it
        fills the ``label`` of any object the model left unlabeled (detection /
        pointing name their target in the prompt, not the answer).

        Returns ``{"task": task, "objects": [...]}`` for one sequence, or a list of
        those for a batch. Each object is ``{"label": str | None, "box":
        [x1, y1, x2, y2]}`` or ``{"label": ..., "point": [x, y]}``.
        """
        try:
            arr = ops.convert_to_numpy(generated)
        except (TypeError, ValueError):
            arr = None
        if arr is not None:
            sequences = (
                [arr.tolist()] if arr.ndim == 1 else [row.tolist() for row in arr]
            )
        elif generated and isinstance(generated[0], (list, tuple)):
            sequences = [list(row) for row in generated]
        else:
            sequences = [list(generated)]

        results = []
        for seq in sequences:
            objects = self.tokenizer.parse_grounding(seq)
            if image_size is not None:
                w, h = image_size
                for obj in objects:
                    if "box" in obj:
                        x1, y1, x2, y2 = obj["box"]
                        obj["box"] = [
                            x1 / 1000 * w,
                            y1 / 1000 * h,
                            x2 / 1000 * w,
                            y2 / 1000 * h,
                        ]
                    elif "point" in obj:
                        x, y = obj["point"]
                        obj["point"] = [x / 1000 * w, y / 1000 * h]
            if isinstance(text, str):
                for obj in objects:
                    if obj.get("label") is None:
                        obj["label"] = text
            results.append({"task": task, "objects": objects})
        return results[0] if len(results) == 1 else results

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
