from typing import List, Optional, Union

import keras

from zeromodels.base import BaseProcessor

from .speech2text_feature_extractor import Speech2TextFeatureExtractor
from .speech2text_tokenizer import Speech2TextTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class Speech2TextProcessor(BaseProcessor):
    """Combined audio + text processor for Speech2Text.

    Wraps :class:`Speech2TextFeatureExtractor` and
    :class:`Speech2TextTokenizer`, matching the reference ``Speech2TextProcessor``
    API:

    * ``processor(audio=..., sampling_rate=16000)`` -> fbank ``input_features``.
    * ``processor(text=...)`` -> label token ids (fine-tuning path).
    * ``processor.batch_decode`` / ``processor.decode`` -> proxy to the tokenizer.

    ``decoder_start_token_id`` is ``</s>`` (id 2) - the Bart-style convention
    Speech2Text uses to seed autoregressive decoding.

    Args:
        vocab_file / spm_file: Tokenizer files. Downloaded from the HF repo
            when ``None``.
        sampling_rate / num_mel_bins: Forwarded to the feature extractor.
        do_upper_case / do_lower_case: Forwarded to the tokenizer.
        decoder_start_token_id: Seed token id for generation (``</s>`` = 2).
    """

    TOKENIZER_CLS = Speech2TextTokenizer
    FEATURE_EXTRACTOR_CLS = Speech2TextFeatureExtractor

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        spm_file: Optional[str] = None,
        sampling_rate: int = 16000,
        num_mel_bins: int = 80,
        do_upper_case: bool = False,
        do_lower_case: bool = False,
        decoder_start_token_id: int = 2,
        tokenizer=None,
        feature_extractor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_extractor = feature_extractor or Speech2TextFeatureExtractor(
            sampling_rate=sampling_rate, num_mel_bins=num_mel_bins
        )
        self.tokenizer = tokenizer or Speech2TextTokenizer(
            vocab_file=vocab_file,
            spm_file=spm_file,
            do_upper_case=do_upper_case,
            do_lower_case=do_lower_case,
        )
        self.decoder_start_token_id = decoder_start_token_id

    def call(
        self,
        audio=None,
        text: Union[str, List[str], None] = None,
        sampling_rate: int = 16000,
    ):
        if audio is None and text is None:
            raise ValueError(
                "At least one of 'audio' or 'text' must be provided to "
                "Speech2TextProcessor"
            )
        out = {}
        if audio is not None:
            out["input_features"] = self.feature_extractor(
                audio, sampling_rate=sampling_rate
            )
        if text is not None:
            out.update(self.tokenizer(text))
        return out

    def get_config(self):
        config = super().get_config()
        config["decoder_start_token_id"] = self.decoder_start_token_id
        return config
