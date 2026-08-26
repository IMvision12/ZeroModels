import keras

from zeromodels.models.qwen2_vl.qwen2_vl_tokenizer import Qwen2VLTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen3VLMoeTokenizer(Qwen2VLTokenizer):
    """Qwen3-VL-MoE BPE tokenizer (``tokenizers`` backend).

    Identical wiring to :class:`Qwen2VLTokenizer` (shared Qwen-VL vision placeholder /
    ChatML token strings); only the ``tokenizer.json`` differs, so the token ids are
    resolved from the file.
    """

    def __init__(
        self, hf_id="Qwen/Qwen3-VL-30B-A3B-Instruct", tokenizer_file=None, **kwargs
    ):
        super().__init__(hf_id=hf_id, tokenizer_file=tokenizer_file, **kwargs)
