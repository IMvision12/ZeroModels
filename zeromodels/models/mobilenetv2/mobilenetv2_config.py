from zeromodels.base import BaseConfig


class MobileNetV2Config(BaseConfig):
    r"""Configuration for [`MobileNetV2Model`] / [`MobileNetV2ImageClassify`].

    MobileNetV2 is an efficient mobile backbone built from inverted-residual blocks
    with linear bottlenecks, scaled by width and depth multipliers. One
    `kf_config.json` (declaring the canonical [`MobileNetV2ImageClassify`]) sits on
    each variant's repo, and both the backbone and classifier load from it. Fields
    mirror the model constructor and serialize flat.

    Args:
        width_multiplier (`float`, *optional*, defaults to 1.0):
            Channel-width scaling factor.
        depth_multiplier (`float`, *optional*, defaults to 1.0):
            Block-repeat (depth) scaling factor.
        fix_channels (`bool`, *optional*, defaults to `False`):
            Whether to keep channel counts fixed rather than rounding after scaling
            (the timm ``*d`` variants).
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.mobilenetv2 import (
    ...     MobileNetV2Config,
    ...     MobileNetV2ImageClassify,
    ... )

    >>> configuration = MobileNetV2Config()
    >>> model = MobileNetV2ImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "mobilenetv2"

    width_multiplier: float = 1.0
    depth_multiplier: float = 1.0
    fix_channels: bool = False
    image_size: int = 224
    num_classes: int = 1000
