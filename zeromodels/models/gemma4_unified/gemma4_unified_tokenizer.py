import keras

from zeromodels.models.gemma4.gemma4_tokenizer import Gemma4Tokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma4UnifiedTokenizer(Gemma4Tokenizer):
    """Gemma 4 unified tokenizer.

    Identical to :class:`Gemma4Tokenizer` (same SentencePiece-BPE ``tokenizer.json``
    and the same image / audio soft-token markers); a distinct class so the unified
    checkpoints resolve their own tokenizer. :class:`Gemma4UnifiedProcessor` expands
    each ``<|image|>`` / ``<|audio|>`` marker to its full soft-token run.
    """
