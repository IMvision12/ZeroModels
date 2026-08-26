import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseProcessor

from .granite_speech_feature_extractor import GraniteSpeechFeatureExtractor
from .granite_speech_tokenizer import GraniteSpeechTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class GraniteSpeechProcessor(BaseProcessor):
    """Audio + text -> model inputs for Granite Speech.

    Composes the tokenizer and the mel feature extractor. ``call`` runs the
    feature extractor on the audio, expands each ``<|audio|>`` placeholder in the
    text to the projector output length for that clip (``audio_embed_sizes``),
    tokenizes the expanded text, and returns ``input_ids`` / ``attention_mask``
    together with ``input_features`` / ``input_features_mask`` for the model.
    """

    TOKENIZER_CLS = GraniteSpeechTokenizer
    FEATURE_EXTRACTOR_CLS = GraniteSpeechFeatureExtractor

    def __init__(
        self,
        projector_window_size=15,
        projector_downsample_rate=5,
        tokenizer=None,
        feature_extractor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_extractor = feature_extractor or GraniteSpeechFeatureExtractor(
            projector_window_size=projector_window_size,
            projector_downsample_rate=projector_downsample_rate,
        )
        self.tokenizer = tokenizer or self.TOKENIZER_CLS()
        self.audio_token = self.tokenizer.audio_token

    def apply_chat_template(self, messages, add_generation_prompt=True):
        """Render a simple Granite chat prompt; each ``{"type": "audio"}`` content
        item becomes a single ``<|audio|>`` placeholder (expanded later)."""
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text += f"<|start_of_role|>{role}<|end_of_role|>"
            if isinstance(content, str):
                text += content
            else:
                for item in content:
                    if item.get("type") == "audio":
                        text += self.audio_token
                    elif item.get("type") == "text":
                        text += item["text"]
            text += "<|end_of_text|>\n"
        if add_generation_prompt:
            text += "<|start_of_role|>assistant<|end_of_role|>"
        return text

    def expand_audio(self, text, embed_sizes, counter):
        out = text
        while self.audio_token in out:
            n = embed_sizes[counter[0]]
            out = out.replace(self.audio_token, "<placeholder>" * n, 1)
            counter[0] += 1
        return out.replace("<placeholder>", self.audio_token)

    def call(
        self,
        text=None,
        audio=None,
        conversation=None,
        messages=None,
        sampling_rate=16000,
        add_generation_prompt=True,
    ):
        if conversation is not None:
            messages = conversation
        if messages is not None:
            text = self.apply_chat_template(messages, add_generation_prompt)
        if text is None:
            raise ValueError("Provide `text`, `messages`, or `conversation`.")
        texts = [text] if isinstance(text, str) else list(text)

        out = {}
        if audio is not None:
            audio_inputs = self.feature_extractor(audio, sampling_rate=sampling_rate)
            embed_sizes = audio_inputs["audio_embed_sizes"]
            out["input_features"] = ops.convert_to_tensor(
                audio_inputs["input_features"]
            )
            out["input_features_mask"] = ops.convert_to_tensor(
                audio_inputs["input_features_mask"]
            )
            counter = [0]
            texts = [self.expand_audio(t, embed_sizes, counter) for t in texts]

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
