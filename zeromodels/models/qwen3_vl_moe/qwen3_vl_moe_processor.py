import keras

from zeromodels.models.qwen3_vl.qwen3_vl_processor import Qwen3VLProcessor

from .qwen3_vl_moe_tokenizer import Qwen3VLMoeTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen3VLMoeProcessor(Qwen3VLProcessor):
    """Qwen3-VL-MoE image+text processor: identical to :class:`Qwen3VLProcessor`
    (16px patch, ChatML + image-pad expansion) with the Qwen3-VL-MoE tokenizer. The
    image processor is reused from Qwen2-VL."""

    TOKENIZER_CLS = Qwen3VLMoeTokenizer

    def __init__(self, hf_id="Qwen/Qwen3-VL-30B-A3B-Instruct", **kwargs):
        super().__init__(hf_id=hf_id, **kwargs)
