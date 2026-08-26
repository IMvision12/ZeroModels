import keras

from zeromodels.models.granite_speech.granite_speech_tokenizer import (
    GraniteSpeechTokenizer,
)


@keras.saving.register_keras_serializable(package="zeromodels")
class GraniteSpeechPlusTokenizer(GraniteSpeechTokenizer):
    """Granite Speech 4.1-plus tokenizer (granite-4.0 BPE, 100353-token vocab).

    Identical machinery to :class:`GraniteSpeechTokenizer`; downloads the
    per-variant ``tokenizer.json`` from ``zeromodels/<variant>`` (``<|audio|>`` =
    100352, eos = 100257). Load by repo id:
    ``GraniteSpeechPlusTokenizer.from_weights("zeromodels/granite_speech_4_1_2b_plus")``.
    """

    DEFAULT_VARIANT = "granite_speech_4_1_2b_plus"
