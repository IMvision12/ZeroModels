import keras

from zeromodels.models.qwen2_vl.qwen2_vl_processor import Qwen2VLProcessor

from .qwen3_5_moe_tokenizer import Qwen3_5MoeTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen3_5MoeProcessor(Qwen2VLProcessor):
    """Qwen3.5-MoE image+text processor.

    Like :class:`Qwen2VLProcessor` but with a 16px patch and the Qwen3.5-MoE tokenizer.
    The image processor is reused from Qwen2-VL: only the patch size differs (16 vs 14).
    """

    TOKENIZER_CLS = Qwen3_5MoeTokenizer

    def __init__(
        self,
        hf_id="Qwen/Qwen3.5-35B-A3B",
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        **kwargs,
    ):
        super().__init__(
            hf_id=hf_id,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
            **kwargs,
        )
