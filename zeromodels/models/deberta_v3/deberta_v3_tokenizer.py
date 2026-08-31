import keras

from zeromodels.models.deberta_v2.deberta_v2_tokenizer import DebertaV2Tokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class DebertaV3Tokenizer(DebertaV2Tokenizer):
    """DeBERTa-v3 SentencePiece tokenizer.

    Identical machinery to :class:`DebertaV2Tokenizer`; only the per-variant
    ``tokenizer.json`` (a different SentencePiece vocab) differs. The inherited
    ``__init__`` resolves it from the ``zeromodels/<variant>`` Hub repo.
    """
