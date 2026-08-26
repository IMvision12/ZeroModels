from zeromodels.base import BaseConfig


class NextViTConfig(BaseConfig):
    r"""Configuration for [`NextViTModel`] / [`NextViTImageClassify`].

    Next-ViT interleaves efficient convolutional (NCB) and transformer (NTB) blocks in
    a hardware-friendly hybrid backbone. One `kf_config.json` (declaring the canonical
    [`NextViTImageClassify`]) sits on each variant's repo, and both the backbone and
    classifier load from it. Fields mirror the model constructor and serialize flat.

    Args:
        depths (`tuple`, *optional*, defaults to `(3, 4, 10, 3)`):
            Number of blocks per stage.
        stem_chs (`tuple`, *optional*, defaults to `(64, 32, 64)`):
            Channel widths of the three stem convolutions.
        head_dim (`int`, *optional*, defaults to 32):
            Per-head channel dimension in the transformer blocks.
        mix_block_ratio (`float`, *optional*, defaults to 0.75):
            Fraction of channels routed through the convolutional path in NTB blocks.
        sr_ratios (`tuple`, *optional*, defaults to `(8, 4, 2, 1)`):
            Spatial-reduction ratio for attention per stage.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            Maximum stochastic-depth drop rate.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.nextvit import NextViTConfig, NextViTImageClassify

    >>> configuration = NextViTConfig()
    >>> model = NextViTImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "nextvit"

    depths: tuple = (3, 4, 10, 3)
    stem_chs: tuple = (64, 32, 64)
    head_dim: int = 32
    mix_block_ratio: float = 0.75
    sr_ratios: tuple = (8, 4, 2, 1)
    drop_path_rate: float = 0.1
    image_size: int = 224
    num_classes: int = 1000
