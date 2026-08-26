import keras

from zeromodels.models.granite_speech.granite_speech_processor import (
    GraniteSpeechProcessor,
)

from .granite_speech_plus_tokenizer import GraniteSpeechPlusTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class GraniteSpeechPlusProcessor(GraniteSpeechProcessor):
    """Granite Speech 4.1-plus processor: the GraniteSpeechProcessor pipeline with
    the plus tokenizer (granite-4.0 vocab). The mel feature extractor is shared.
    """

    TOKENIZER_CLS = GraniteSpeechPlusTokenizer
