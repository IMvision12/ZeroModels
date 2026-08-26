from zeromodels.base import BaseConfig


class MaxViTConfig(BaseConfig):
    r"""Configuration for [`MaxViTModel`] / [`MaxViTImageClassify`].

    MaxViT stacks MBConv blocks with block (local window) and grid (dilated) attention,
    giving a multi-axis hybrid backbone with global receptive field at linear cost. One
    `kf_config.json` (declaring the canonical [`MaxViTImageClassify`]) sits on each
    variant's repo, and both the backbone and classifier load from it. Fields mirror
    the model constructor and serialize flat.

    Args:
        stem_width (`int`, *optional*, defaults to 64):
            Channel width of the convolutional stem.
        depths (`tuple`, *optional*, defaults to `(2, 2, 5, 2)`):
            Number of MaxViT blocks per stage.
        embed_dim (`tuple`, *optional*, defaults to `(64, 128, 256, 512)`):
            Channel width per stage.
        num_heads (`tuple`, *optional*, defaults to `(2, 4, 8, 16)`):
            Number of attention heads per stage.
        window_size (`int`, *optional*, defaults to 7):
            Side length of the block/grid attention window.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.maxvit import MaxViTConfig, MaxViTImageClassify

    >>> configuration = MaxViTConfig()
    >>> model = MaxViTImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "maxvit"

    stem_width: int = 64
    depths: tuple = (2, 2, 5, 2)
    embed_dim: tuple = (64, 128, 256, 512)
    num_heads: tuple = (2, 4, 8, 16)
    window_size: int = 7
    image_size: int = 224
    num_classes: int = 1000
