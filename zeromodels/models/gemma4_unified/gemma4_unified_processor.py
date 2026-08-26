import inspect

import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseProcessor
from zeromodels.conversion.kf_config import load_kf_preprocessor

from .gemma4_unified_audio_feature_extractor import Gemma4UnifiedAudioFeatureExtractor
from .gemma4_unified_image_processor import Gemma4UnifiedImageProcessor
from .gemma4_unified_tokenizer import Gemma4UnifiedTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma4UnifiedProcessor(BaseProcessor):
    """Text + image + audio -> model inputs for the Gemma 4 unified models.

    Composes the tokenizer, the encoder-free image processor (raw 48px merged
    pixel patches) and the encoder-free audio feature extractor (raw 640-sample
    waveform frames). ``call`` renders the ``<start_of_turn>`` chat template (each
    image content item becomes one ``<|image|>`` marker, each audio item one
    ``<|audio|>`` marker), preprocesses the images and audio, expands every marker
    to its full soft-token run (``<|image>`` + ``<|image|>`` x n + ``<image|>`` and
    the audio equivalent), and tokenizes to padded ``{"input_ids",
    "attention_mask"}`` (bos prepended) alongside ``pixel_values`` /
    ``pixel_position_ids`` and ``input_features`` / ``input_features_mask``.

    Unlike the "gemma4" processor, the audio pipeline has no downsampling: the
    per-audio soft-token count equals the number of valid raw waveform frames.

    Args:
        hf_id: Hub repo for the tokenizer's ``tokenizer.json``.
        tokenizer / image_processor / feature_extractor: Optional pre-built parts.
    """

    TOKENIZER_CLS = Gemma4UnifiedTokenizer
    IMAGE_PROCESSOR_CLS = Gemma4UnifiedImageProcessor
    FEATURE_EXTRACTOR_CLS = Gemma4UnifiedAudioFeatureExtractor
    COMPONENTS = ("tokenizer", "image_processor", "feature_extractor")

    def __init__(
        self,
        hf_id=None,
        tokenizer=None,
        image_processor=None,
        feature_extractor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hf_id = hf_id
        self.tokenizer = tokenizer or Gemma4UnifiedTokenizer(hf_id=hf_id)
        self.image_processor = image_processor or Gemma4UnifiedImageProcessor()
        self.feature_extractor = (
            feature_extractor or Gemma4UnifiedAudioFeatureExtractor()
        )

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(hf_id=repo, **kwargs)

    @classmethod
    def from_hub_repo(cls, repo_id, **kwargs):
        repo_id = repo_id.rstrip("/")
        tokenizer = cls.TOKENIZER_CLS.from_weights(repo_id)
        spec = load_kf_preprocessor(repo_id) or {}
        img_params = set(inspect.signature(cls.IMAGE_PROCESSOR_CLS.__init__).parameters)
        img_kwargs = {k: v for k, v in spec.items() if k in img_params}
        return cls(
            tokenizer=tokenizer,
            image_processor=cls.IMAGE_PROCESSOR_CLS(**img_kwargs),
            feature_extractor=cls.FEATURE_EXTRACTOR_CLS(),
            **kwargs,
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

    def load_audio(self, item):
        if item.get("audio") is not None:
            return np.asarray(item["audio"], dtype="float32").reshape(-1)
        if item.get("path") is not None:
            import soundfile as sf

            data, _ = sf.read(item["path"], dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            return data
        raise ValueError("Audio content item needs an 'audio' array or a 'path'.")

    def extract_media(self, conversation):
        images, audio = [], []
        for msg in conversation:
            content = msg.get("content")
            if isinstance(content, (list, tuple)):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") in ("image", "image_url"):
                        images.append(self.load_image(item))
                    elif item.get("type") in ("audio", "input_audio"):
                        audio.append(self.load_audio(item))
        return images or None, audio or None

    @staticmethod
    def _expand_markers(text, marker, replacements):
        parts = text.split(marker)
        if len(parts) - 1 != len(replacements):
            raise ValueError(
                f"Found {len(parts) - 1} '{marker}' markers but {len(replacements)} "
                f"media items."
            )
        out = parts[0]
        for replacement, tail in zip(replacements, parts[1:]):
            out += replacement + tail
        return out

    def call(
        self,
        conversation=None,
        text=None,
        images=None,
        audio=None,
        messages=None,
        add_generation_prompt=True,
    ):
        if conversation is not None:
            messages = conversation
            extracted_images, extracted_audio = self.extract_media(conversation)
            images = images if images is not None else extracted_images
            audio = audio if audio is not None else extracted_audio
        if messages is not None:
            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt)
        if text is None:
            raise ValueError("Provide a `conversation`, `messages`, or `text`.")
        texts = [text] if isinstance(text, str) else list(text)

        out = {}
        tok = self.tokenizer
        if images is not None:
            image_inputs = self.image_processor(images)
            out["pixel_values"] = ops.convert_to_tensor(image_inputs["pixel_values"])
            out["pixel_position_ids"] = ops.convert_to_tensor(
                image_inputs["image_position_ids"]
            )
            counts = image_inputs["num_soft_tokens_per_image"]
            replacements = [
                f"{tok.boi_token}{tok.image_token * n}{tok.eoi_token}" for n in counts
            ]
            texts = [
                self._expand_markers(t, tok.image_token, replacements) for t in texts
            ]
        if audio is not None:
            audio_inputs = self.feature_extractor(audio)
            out["input_features"] = ops.convert_to_tensor(
                audio_inputs["input_features"]
            )
            out["input_features_mask"] = ops.convert_to_tensor(
                audio_inputs["input_features_mask"]
            )
            mask_t = audio_inputs["input_features_mask"]
            counts = [
                int(ops.sum(ops.cast(mask_t[i], "int32")))
                for i in range(int(ops.shape(mask_t)[0]))
            ]
            replacements = [
                f"{tok.boa_token}{tok.audio_token * m}{tok.eoa_token}" for m in counts
            ]
            texts = [
                self._expand_markers(t, tok.audio_token, replacements) for t in texts
            ]

        bos = tok.bos_token_id
        ids = [[bos] + tok.encode(t) for t in texts]
        max_len = max(len(x) for x in ids)
        input_ids = [x + [0] * (max_len - len(x)) for x in ids]
        attention_mask = [[1] * len(x) + [0] * (max_len - len(x)) for x in ids]
        out["input_ids"] = ops.convert_to_tensor(input_ids, "int32")
        out["attention_mask"] = ops.convert_to_tensor(attention_mask, "int32")
        return out

    def get_config(self):
        config = super().get_config()
        config.update({"hf_id": self.hf_id})
        return config
