import keras

from zeromodels.models.qwen2_vl.qwen2_vl_tokenizer import Qwen2VLTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen3_5MoeTokenizer(Qwen2VLTokenizer):
    """Qwen3.5-MoE BPE tokenizer (``tokenizers`` backend).

    Identical wiring to :class:`Qwen2VLTokenizer` (the vision placeholder / ChatML
    token strings are shared across the Qwen-VL line); only the ``tokenizer.json`` and
    its vocabulary differ, so the token ids are resolved from the file.
    """

    def __init__(self, hf_id="Qwen/Qwen3.5-35B-A3B", tokenizer_file=None, **kwargs):
        super().__init__(hf_id=hf_id, tokenizer_file=tokenizer_file, **kwargs)
