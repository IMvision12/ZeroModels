import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseProcessor

from .gemma4_audio_feature_extractor import Gemma4AudioFeatureExtractor
from .gemma4_image_processor import Gemma4ImageProcessor
from .gemma4_tokenizer import Gemma4Tokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma4Processor(BaseProcessor):
    """Text + image + audio -> model inputs for the Gemma 4 "Any-to-Any" models.

    Composes the tokenizer, the NaViT image processor, and the USM audio feature
    extractor. ``call`` renders the ``<start_of_turn>`` chat template (each image
    content item becomes one ``<|image|>`` marker, each audio item one ``<|audio|>``
    marker), preprocesses the images and audio, expands every marker to its full
    soft-token run (``<|image>`` + ``<|image|>`` x n + ``<image|>`` and the audio
    equivalent), and tokenizes to padded ``{"input_ids", "attention_mask"}`` (bos
    prepended) alongside ``pixel_values`` / ``pixel_position_ids`` and
    ``input_features`` / ``input_features_mask``.

    The per-image soft-token count comes from the image processor (NaViT keeps each
    image's aspect ratio, so counts vary); the per-audio count replicates the audio
    tower's two stride-2 subsampling convolutions over the feature mask.

    Args:
        hf_id: Hub repo for the tokenizer's ``tokenizer.json``.
        tokenizer / image_processor / feature_extractor: Optional pre-built parts.
    """

    TOKENIZER_CLS = Gemma4Tokenizer
    IMAGE_PROCESSOR_CLS = Gemma4ImageProcessor
    FEATURE_EXTRACTOR_CLS = Gemma4AudioFeatureExtractor
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
        self.tokenizer = tokenizer or Gemma4Tokenizer(hf_id=hf_id)
        self.image_processor = image_processor or Gemma4ImageProcessor()
        self.feature_extractor = feature_extractor or Gemma4AudioFeatureExtractor()

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(hf_id=repo, **kwargs)

    @classmethod
    def from_hub_repo(cls, repo_id, **kwargs):
        # Gemma 4 has two preprocessors (image + audio) but one
        # kf_preprocessor.json. The base loader would feed that single file to
        # both. Instead: the tokenizer loads from tokenizer.json, the image
        # processor takes its params from kf_preprocessor.json (or defaults when
        # absent), and the audio extractor uses the fixed USM defaults.
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

    def _audio_num_tokens(self, mask):
        """Valid audio soft tokens after the tower's two stride-2 subsampling
        convolutions (kernel 3, pad 1) over the feature mask."""
        m = np.asarray(ops.convert_to_numpy(mask)).astype(bool).reshape(-1)
        t = len(m)
        for _ in range(2):
            t_out = (t + 2 - 3) // 2 + 1
            m = m[::2][:t_out]
            t = len(m)
        return int(m.sum())

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
            masks = audio_inputs["input_features_mask"]
            counts = [self._audio_num_tokens(masks[i]) for i in range(len(masks))]
            replacements = [
                f"{tok.boa_token}{tok.audio_token * m}{tok.eoa_token}" for m in counts
            ]
            texts = [
                self._expand_markers(t, tok.audio_token, replacements) for t in texts
            ]

        bos = tok.bos_token_id
        ids = [[bos] + tok.encode(t) for t in texts]
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
