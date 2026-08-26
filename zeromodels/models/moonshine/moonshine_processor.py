import keras

from zeromodels.base import BaseProcessor

from .moonshine_feature_extractor import MoonshineFeatureExtractor
from .moonshine_tokenizer import MoonshineTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class MoonshineProcessor(BaseProcessor):
    """Combined audio + text processor for Moonshine.

    Composes :class:`MoonshineFeatureExtractor` (raw-waveform ``input_values``)
    and :class:`MoonshineTokenizer`. ``decode`` / ``batch_decode`` and the loaders
    (``from_weights`` / ``from_hf``) are inherited from :class:`BaseProcessor`.

    ``decoder_start_token_id`` is ``<s>`` (id 1): the token
    :class:`~zeromodels.models.moonshine.MoonshineConditionalGenerate` seeds greedy
    decoding with.
    """

    TOKENIZER_CLS = MoonshineTokenizer
    FEATURE_EXTRACTOR_CLS = MoonshineFeatureExtractor

    def __init__(
        self,
        variant=None,
        tokenizer_file=None,
        sampling_rate=16000,
        decoder_start_token_id=1,
        bos_token_id=1,
        eos_token_id=2,
        unk_token_id=0,
        tokenizer=None,
        feature_extractor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_extractor = feature_extractor or MoonshineFeatureExtractor(
            sampling_rate=sampling_rate
        )
        self.tokenizer = tokenizer or MoonshineTokenizer(
            variant=variant,
            tokenizer_file=tokenizer_file,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            unk_token_id=unk_token_id,
        )
        self.decoder_start_token_id = decoder_start_token_id

    def call(self, audio=None, text=None, sampling_rate=16000):
        if audio is None and text is None:
            raise ValueError(
                "At least one of 'audio' or 'text' must be provided to "
                "MoonshineProcessor"
            )
        out = {}
        if audio is not None:
            out["input_values"] = self.feature_extractor(
                audio, sampling_rate=sampling_rate
            )
        if text is not None:
            out.update(self.tokenizer(text))
        return out

    def get_config(self):
        config = super().get_config()
        config["decoder_start_token_id"] = self.decoder_start_token_id
        return config
