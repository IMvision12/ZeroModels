from zeromodels.base import BaseConfig


class SamConfig(BaseConfig):
    r"""Configuration for [`SAMPromptableSegment`], the Segment Anything model.

    The defaults describe the SAM ViT-Base image encoder. Other variants override
    the vision-encoder dimensions. Fields serialize flat to a repo's
    `kf_config.json`.

    Args:
        vision_hidden_size (`int`, *optional*, defaults to 768):
            Hidden size of the ViT image encoder.
        vision_num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of transformer layers in the image encoder.
        vision_num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads in the image encoder.
        vision_mlp_dim (`int`, *optional*, defaults to 3072):
            Feed-forward dimension of the image encoder.
        vision_global_attn_indexes (`tuple`, *optional*, defaults to `(2, 5, 8, 11)`):
            Encoder layer indices that use global (non-windowed) attention.
        num_multimask_outputs (`int`, *optional*, defaults to 3):
            Number of masks predicted per prompt in multimask mode.
        multimask_output (`bool`, *optional*, defaults to `True`):
            Whether to output multiple masks (vs a single mask) per prompt.
        enable_boxes (`bool`, *optional*, defaults to `False`):
            Whether to build the box-prompt input branch.
        enable_masks (`bool`, *optional*, defaults to `False`):
            Whether to build the dense mask-prompt input branch.
        image_size (`int`, *optional*, defaults to 1024):
            Square input resolution the model is built for.

    Examples:

    ```python
    >>> from zeromodels.models.sam import SamConfig, SAMPromptableSegment

    >>> configuration = SamConfig()
    >>> model = SAMPromptableSegment(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "sam"

    vision_hidden_size: int = 768
    vision_num_hidden_layers: int = 12
    vision_num_attention_heads: int = 12
    vision_mlp_dim: int = 3072
    vision_global_attn_indexes: tuple = (2, 5, 8, 11)
    num_multimask_outputs: int = 3
    multimask_output: bool = True
    enable_boxes: bool = False
    enable_masks: bool = False
    image_size: int = 1024
