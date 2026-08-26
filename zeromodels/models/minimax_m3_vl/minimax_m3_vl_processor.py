import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseProcessor, BaseTokenizer

from .minimax_m3_vl_image_processor import MiniMaxM3VLImageProcessor

IMAGE_TOKEN = "]<]image[>["
VIDEO_TOKEN = "]<]video[>["
VISION_START_TOKEN = "]<]start of image[>["
VISION_END_TOKEN = "]<]end of image[>["


@keras.saving.register_keras_serializable(package="zeromodels")
class MiniMaxM3VLTokenizer(BaseTokenizer):
    """MiniMax-M3 BPE tokenizer (``tokenizers`` backend, ~200k vocab).

    Loads the model's ``tokenizer.json`` (downloaded on the fly from ``hf_id``
    when no explicit file is given) and exposes the vision special tokens
    (``]<]image[>[`` / ``]<]video[>[`` / start / end). No BOS/EOS is added.

    Args:
        hf_id: Hub repo to pull ``tokenizer.json`` from.
        tokenizer_file: Explicit path to a ``tokenizer.json`` (overrides the
            download).
    """

    def __init__(self, hf_id=None, tokenizer_file=None, **kwargs):
        super().__init__(**kwargs)
        from tokenizers import Tokenizer

        tokenizer_file = self.resolve_tokenizer_json_from_hf(hf_id, tokenizer_file)
        self.hf_id = hf_id
        self.tokenizer_file = tokenizer_file
        self._tok = Tokenizer.from_file(tokenizer_file)
        self.image_token = IMAGE_TOKEN
        self.video_token = VIDEO_TOKEN
        self.vision_start_token = VISION_START_TOKEN
        self.vision_end_token = VISION_END_TOKEN
        self.eos_token = "[e~["
        self.image_token_id = self._tok.token_to_id(IMAGE_TOKEN)
        self.video_token_id = self._tok.token_to_id(VIDEO_TOKEN)
        self.eos_token_id = self._tok.token_to_id(self.eos_token)

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text, add_special_tokens=False).ids

    def call(self, inputs):
        texts = self.normalize_texts(inputs)
        return {"input_ids": [self.encode(t) for t in texts]}

    def decode(self, ids, skip_special_tokens=True):
        return self._tok.decode(
            self.to_id_list(ids), skip_special_tokens=skip_special_tokens
        )

    def get_config(self):
        config = super().get_config()
        config.update({"hf_id": self.hf_id, "tokenizer_file": self.tokenizer_file})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MiniMaxM3VLProcessor(BaseProcessor):
    """Image / video + text -> model inputs for MiniMax-M3 VL.

    Each ``]<]image[>[`` marker in the prompt is expanded to
    ``]<]start of image[>[`` + one placeholder per merged patch
    (``grid.prod() / merge_size**2``) + ``]<]end of image[>[``, and likewise
    per video frame for ``]<]video[>[`` (with an optional
    ``]<]{t:.1f} seconds[>[`` timestamp prefix when ``fps`` is given).
    ``call`` also accepts an OpenAI-style ``conversation`` rendered with the
    MiniMax chat format (``]~b]role\\n...[e~[``-free minimal form is not
    published; user/assistant turns are joined with newlines).

    Args:
        hf_id: Hub repo for the tokenizer's ``tokenizer.json``.
        tokenizer / image_processor: Optional pre-built components.
    """

    TOKENIZER_CLS = MiniMaxM3VLTokenizer
    IMAGE_PROCESSOR_CLS = MiniMaxM3VLImageProcessor
    COMPONENTS = ("tokenizer", "image_processor")

    def __init__(
        self,
        hf_id=None,
        tokenizer=None,
        image_processor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hf_id = hf_id
        self.image_processor = image_processor or MiniMaxM3VLImageProcessor()
        self.tokenizer = tokenizer or MiniMaxM3VLTokenizer(hf_id=hf_id)

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(hf_id=repo, **kwargs)

    def expand_image_tokens(self, text, image_grids):
        merge = self.image_processor.merge_size**2
        parts = text.split(IMAGE_TOKEN)
        if len(parts) - 1 != len(image_grids):
            raise ValueError(
                f"{len(parts) - 1} {IMAGE_TOKEN} placeholders but "
                f"{len(image_grids)} images were given."
            )
        out = parts[0]
        for grid, part in zip(image_grids, parts[1:]):
            n = int(np.prod(grid)) // merge
            out += VISION_START_TOKEN + IMAGE_TOKEN * n + VISION_END_TOKEN + part
        return out

    def expand_video_tokens(self, text, video_grids, fps=None):
        merge = self.image_processor.merge_size**2
        tps = self.image_processor.temporal_patch_size
        parts = text.split(VIDEO_TOKEN)
        if len(parts) - 1 != len(video_grids):
            raise ValueError(
                f"{len(parts) - 1} {VIDEO_TOKEN} placeholders but "
                f"{len(video_grids)} videos were given."
            )
        out = parts[0]
        for grid, part in zip(video_grids, parts[1:]):
            grid_t = int(grid[0])
            frame_len = int(grid[1] * grid[2]) // merge
            chunk = ""
            for frame in range(grid_t):
                if fps is not None:
                    ts = frame * tps / fps
                    chunk += f"]<]{ts:.1f} seconds[>["
                chunk += VISION_START_TOKEN + VIDEO_TOKEN * frame_len + VISION_END_TOKEN
            out += chunk + part
        return out

    def apply_chat_template(self, messages, add_generation_prompt=True):
        turns = []
        for msg in messages:
            content = msg["content"]
            if not isinstance(content, str):
                parts = []
                for item in content:
                    if item.get("type") == "image":
                        parts.append(IMAGE_TOKEN)
                    elif item.get("type") == "video":
                        parts.append(VIDEO_TOKEN)
                    elif item.get("type") == "text":
                        parts.append(item["text"])
                content = "".join(parts)
            turns.append(f"{msg['role']}: {content}")
        text = "\n".join(turns)
        if add_generation_prompt:
            text += "\nassistant:"
        return text

    def extract_images(self, conversation):
        from PIL import Image

        images = []
        for msg in conversation:
            content = msg.get("content")
            if isinstance(content, (list, tuple)):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        if item.get("image") is not None:
                            images.append(item["image"])
                        elif item.get("path") is not None:
                            images.append(Image.open(item["path"]))
        return images or None

    def call(
        self,
        conversation=None,
        text=None,
        images=None,
        videos=None,
        fps=None,
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
            out["image_grid_thw"] = image_inputs["image_grid_thw"]
            grids = list(image_inputs["image_grid_thw"])
            per_text = self.deal_per_text(texts, IMAGE_TOKEN, grids)
            texts = [self.expand_image_tokens(t, g) for t, g in zip(texts, per_text)]
        if videos is not None:
            video_inputs = self.image_processor.process_video(videos)
            out["pixel_values_videos"] = ops.convert_to_tensor(
                video_inputs["pixel_values_videos"]
            )
            out["video_grid_thw"] = video_inputs["video_grid_thw"]
            grids = list(video_inputs["video_grid_thw"])
            per_text = self.deal_per_text(texts, VIDEO_TOKEN, grids)
            texts = [
                self.expand_video_tokens(t, g, fps=fps) for t, g in zip(texts, per_text)
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
        config.update({"hf_id": self.hf_id})
        return config
