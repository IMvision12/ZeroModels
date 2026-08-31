import keras

from zeromodels.models.qwen2_vl.qwen2_vl_processor import Qwen2VLProcessor


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen3VLProcessor(Qwen2VLProcessor):
    """Qwen3-VL image+text processor: like :class:`Qwen2VLProcessor` but with a 16px
    patch and the Qwen3-VL image normalization."""

    def __init__(
        self,
        hf_id=None,
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
