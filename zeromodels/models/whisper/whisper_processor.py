from typing import List, Optional, Union

import keras

from zeromodels.base import BaseProcessor

from .whisper_feature_extractor import WhisperFeatureExtractor
from .whisper_tokenizer import WhisperTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class WhisperProcessor(BaseProcessor):
    """Combined audio + text processor for Whisper.

    Composes :class:`WhisperFeatureExtractor` (log-mel features) and
    :class:`WhisperTokenizer`. ``decode`` / ``batch_decode`` and the loaders
    (``from_weights`` / ``from_hf``) come from :class:`BaseProcessor`.

    Whisper-specific helpers stay here: :meth:`get_decoder_prompt_ids`
    (``("en", "transcribe")`` -> ``forced_decoder_ids``) and
    :attr:`decoder_start_token_id`, with the language / task / ``<|notimestamps|>``
    ids looked up from the tokenizer's ``added_tokens.json`` so the same processor
    works for both v1 (51865 vocab) and v3 (51866 vocab).
    """

    TOKENIZER_CLS = WhisperTokenizer
    FEATURE_EXTRACTOR_CLS = WhisperFeatureExtractor

    def __init__(
        self,
        variant: str = "whisper_tiny",
        n_mels: int = 80,
        sampling_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        chunk_length: int = 30,
        tokenizer_file: Optional[str] = None,
        bos_token_id: int = 50257,
        eos_token_id: int = 50257,
        pad_token_id: int = 50257,
        tokenizer=None,
        feature_extractor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_extractor = feature_extractor or WhisperFeatureExtractor(
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            chunk_length=chunk_length,
        )
        self.tokenizer = tokenizer or WhisperTokenizer(
            variant=variant,
            tokenizer_file=tokenizer_file,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        self.variant = self.tokenizer.variant

    @property
    def added_tokens(self) -> dict:
        return self.tokenizer.added_tokens

    def _special_id(self, token: str) -> int:
        try:
            return int(self.added_tokens[token])
        except KeyError as e:
            raise KeyError(
                f"Special token {token!r} not found in Whisper "
                f"{self.variant!r} added_tokens.json"
            ) from e

    @property
    def decoder_start_token_id(self) -> int:
        return self._special_id("<|startoftranscript|>")

    def get_decoder_prompt_ids(
        self,
        language: Optional[str] = "en",
        task: str = "transcribe",
        no_timestamps: bool = True,
    ) -> List[tuple]:
        if task not in ("transcribe", "translate"):
            raise ValueError(f"task must be 'transcribe' or 'translate', got {task!r}")

        prompt = []
        pos = 1
        if language is not None:
            tok = language if language.startswith("<|") else f"<|{language}|>"
            prompt.append((pos, self._special_id(tok)))
            pos += 1
        prompt.append((pos, self._special_id(f"<|{task}|>")))
        pos += 1
        if no_timestamps:
            prompt.append((pos, self._special_id("<|notimestamps|>")))
        return prompt

    def call(
        self,
        audio=None,
        text: Union[str, List[str], None] = None,
        sampling_rate: int = 16000,
    ):
        if audio is None and text is None:
            raise ValueError(
                "At least one of 'audio' or 'text' must be provided to WhisperProcessor"
            )

        out = {}
        if audio is not None:
            out["input_features"] = self.feature_extractor(
                audio, sampling_rate=sampling_rate
            )
        if text is not None:
            tok_out = self.tokenizer(inputs=text)
            out["input_ids"] = tok_out["input_ids"]
            out["attention_mask"] = tok_out["attention_mask"]
        return out
