from zeromodels.base import BaseConfig


class Sam2Config(BaseConfig):
    r"""Configuration for [`SAM2PromptableSegment`], the SAM 2 model.

    The defaults describe the SAM 2 Hiera-Tiny image encoder. Other variants
    override the Hiera backbone dimensions. Fields serialize flat to a repo's
    `zm_config.json`.

    Args:
        hidden_dim (`int`, *optional*, defaults to 96):
            Embedding dimension of the first Hiera backbone stage.
        blocks_per_stage (`tuple`, *optional*, defaults to `(1, 2, 7, 2)`):
            Number of blocks in each of the four Hiera stages.
        embed_dim_per_stage (`tuple`, *optional*, defaults to `(96, 192, 384, 768)`):
            Embedding dimension of each Hiera stage.
        num_attention_heads_per_stage (`tuple`, *optional*, defaults to `(1, 2, 4, 8)`):
            Number of attention heads per Hiera stage.
        window_size_per_stage (`tuple`, *optional*, defaults to `(8, 4, 14, 7)`):
            Local-attention window size per Hiera stage.
        global_attention_blocks (`tuple`, *optional*, defaults to `(5, 7, 9)`):
            Block indices that use global attention.
        backbone_channel_list (`tuple`, *optional*, defaults to `(768, 384, 192, 96)`):
            Channel counts of the backbone feature maps fed to the FPN neck.
        window_pos_embed_bg_size (`tuple`, *optional*, defaults to `None`):
            Background grid size for the windowed position embedding (base_plus).
        num_multimask_outputs (`int`, *optional*, defaults to 3):
            Number of masks predicted per prompt in multimask mode.
        include_box_input (`bool`, *optional*, defaults to `False`):
            Whether to build the box-prompt input branch.
        include_mask_input (`bool`, *optional*, defaults to `False`):
            Whether to build the dense mask-prompt input branch.
        multimask_output (`bool`, *optional*, defaults to `True`):
            Whether to output multiple masks (vs a single mask) per prompt.
        image_size (`int`, *optional*, defaults to 1024):
            Square input resolution the model is built for.

    Examples:

    ```python
    >>> from zeromodels.models.sam2 import Sam2Config, SAM2PromptableSegment

    >>> configuration = Sam2Config()
    >>> model = SAM2PromptableSegment(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "sam2"

    hidden_dim: int = 96
    blocks_per_stage: tuple = (1, 2, 7, 2)
    embed_dim_per_stage: tuple = (96, 192, 384, 768)
    num_attention_heads_per_stage: tuple = (1, 2, 4, 8)
    window_size_per_stage: tuple = (8, 4, 14, 7)
    global_attention_blocks: tuple = (5, 7, 9)
    backbone_channel_list: tuple = (768, 384, 192, 96)
    window_pos_embed_bg_size: tuple = None
    num_multimask_outputs: int = 3
    include_box_input: bool = False
    include_mask_input: bool = False
    multimask_output: bool = True
    image_size: int = 1024
