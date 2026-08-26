import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseProcessor

from .gemma3n_audio_feature_extractor import Gemma3nAudioFeatureExtractor
from .gemma3n_image_processor import Gemma3nImageProcessor
from .gemma3n_tokenizer import Gemma3nTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3nProcessor(BaseProcessor):
    """Text + image + audio -> model inputs for Gemma 3n.

    Composes the tokenizer, the SigLIP-style image processor, and the USM audio
    feature extractor. ``call`` renders the ``<start_of_turn>`` chat template
    (each image content item becomes one ``<image_soft_token>`` marker, each audio
    item one ``<audio_soft_token>`` marker), preprocesses the media, expands every
    marker to its fixed soft-token run (``<start_of_image>`` +
    ``<image_soft_token>`` x ``image_seq_length`` + ``<end_of_image>``, wrapped in
    newlines, and the audio equivalent), and tokenizes to padded
    ``{"input_ids", "attention_mask"}`` (bos prepended) alongside ``pixel_values``
    and ``input_features`` / ``input_features_mask``.

    Unlike Gemma 4's NaViT tower, the counts are fixed: every image yields
    ``image_seq_length`` (256) soft tokens and every audio clip is padded to
    ``audio_seq_length`` (188) soft tokens.

    Args:
        hf_id: Hub repo for the tokenizer's ``tokenizer.json``.
        image_seq_length / audio_seq_length: Fixed soft-token counts.
        tokenizer / image_processor / feature_extractor: Optional pre-built parts.
    """

    TOKENIZER_CLS = Gemma3nTokenizer
    IMAGE_PROCESSOR_CLS = Gemma3nImageProcessor
    FEATURE_EXTRACTOR_CLS = Gemma3nAudioFeatureExtractor
    COMPONENTS = ("tokenizer", "image_processor", "feature_extractor")

    def __init__(
        self,
        hf_id=None,
        image_seq_length=256,
        audio_seq_length=188,
        tokenizer=None,
        image_processor=None,
        feature_extractor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hf_id = hf_id
        self.image_seq_length = image_seq_length
        self.audio_seq_length = audio_seq_length
        self.tokenizer = tokenizer or Gemma3nTokenizer(hf_id=hf_id)
        self.image_processor = image_processor or Gemma3nImageProcessor()
        self.feature_extractor = feature_extractor or Gemma3nAudioFeatureExtractor()

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(hf_id=repo, **kwargs)

    @classmethod
    def from_hub_repo(cls, repo_id, **kwargs):
        import inspect

        from zeromodels.conversion.kf_config import load_kf_preprocessor

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
            seq = (
                f"\n\n{tok.boi_token}"
                f"{tok.image_token * self.image_seq_length}{tok.eoi_token}\n\n"
            )
            replacements = [seq] * len(images)
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
            seq = (
                f"\n\n{tok.boa_token}"
                f"{tok.audio_token * self.audio_seq_length}{tok.eoa_token}\n\n"
            )
            replacements = [seq] * len(audio)
            texts = [
                self._expand_markers(t, tok.audio_token, replacements) for t in texts
            ]

        bos = tok.bos_token_id
        ids = [[bos] + tok.encode(t) for t in texts]
        max_len = max(len(x) for x in ids)
        input_ids = np.zeros((len(ids), max_len), dtype="int32")
        attention_mask = np.zeros((len(ids), max_len), dtype="int32")
        for i, seq_ids in enumerate(ids):
            input_ids[i, : len(seq_ids)] = seq_ids
            attention_mask[i, : len(seq_ids)] = 1
        out["input_ids"] = ops.convert_to_tensor(input_ids)
        out["attention_mask"] = ops.convert_to_tensor(attention_mask)
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hf_id": self.hf_id,
                "image_seq_length": self.image_seq_length,
                "audio_seq_length": self.audio_seq_length,
            }
        )
        return config
