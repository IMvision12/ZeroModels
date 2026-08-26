import keras

from zeromodels.models.glm4_moe.glm4_moe_tokenizer import Glm4MoeTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class Glm4MoeLiteTokenizer(Glm4MoeTokenizer):
    """GLM-4.7-Flash BPE tokenizer (``tokenizers`` backend, ~155k vocab).

    Same loader as the GLM-4.5 tokenizer (pulls ``tokenizer.json`` from ``hf_id``);
    only the default end-of-text id differs.
    """

    def __init__(self, hf_id=None, tokenizer_file=None, **kwargs):
        super().__init__(hf_id=hf_id, tokenizer_file=tokenizer_file, **kwargs)
        self.eos_token_id = 154829  # <|endoftext|>
