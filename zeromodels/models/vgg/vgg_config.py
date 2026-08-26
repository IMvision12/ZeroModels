from zeromodels.base import BaseConfig


class VGGConfig(BaseConfig):
    r"""Configuration for [`VGGModel`] / [`VGGImageClassify`].

    VGG is a classic deep CNN built from stacks of 3x3 convolutions and max-pooling.
    One `kf_config.json` (declaring the canonical [`VGGImageClassify`]) sits on each
    variant's repo, and both the backbone and classifier load from it. Fields mirror
    the model constructor and serialize flat.

    Args:
        num_filters (`tuple`, *optional*, defaults to the VGG-16 layout):
            Per-layer channel counts; the string `"M"` marks a max-pool stage.
        batch_norm (`bool`, *optional*, defaults to `False`):
            Whether convolutions are followed by BatchNorm (the ``*_bn`` variants).
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.vgg import VGGConfig, VGGImageClassify

    >>> configuration = VGGConfig()
    >>> model = VGGImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "vgg"

    num_filters: tuple = (
        64,
        "M",
        128,
        "M",
        256,
        256,
        "M",
        512,
        512,
        "M",
        512,
        512,
        "M",
    )
    batch_norm: bool = False
    image_size: int = 224
    num_classes: int = 1000
