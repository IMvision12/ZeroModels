from zeromodels.base import BaseConfig


class EfficientNetV2Config(BaseConfig):
    r"""Configuration for [`EfficientNetV2Model`] / [`EfficientNetV2ImageClassify`].

    EfficientNetV2 refines EfficientNet with fused-MBConv blocks in the early stages
    and a training-aware compound scaling. One `kf_config.json` (declaring the
    canonical [`EfficientNetV2ImageClassify`]) sits on each variant's repo, and both
    the backbone and classifier load from it. Fields mirror the model constructor and
    serialize flat.

    Args:
        width_coefficient (`float`, *optional*, defaults to 1.0):
            Channel-width scaling coefficient.
        depth_coefficient (`float`, *optional*, defaults to 1.0):
            Block-repeat (depth) scaling coefficient.
        default_size (`int`, *optional*, defaults to 300):
            Reference resolution the scaling was defined at.
        block_arch (`str`, *optional*, defaults to `"EfficientNetV2S"`):
            Named block-schedule preset (`"EfficientNetV2S/M/L/XL/B"`).
        head_filters (`int`, *optional*, defaults to 1280):
            Channel count of the final 1x1 head convolution.
        image_size (`int`, *optional*, defaults to 300):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.efficientnetv2 import (
    ...     EfficientNetV2Config,
    ...     EfficientNetV2ImageClassify,
    ... )

    >>> configuration = EfficientNetV2Config()
    >>> model = EfficientNetV2ImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "efficientnetv2"

    width_coefficient: float = 1.0
    depth_coefficient: float = 1.0
    default_size: int = 300
    block_arch: str = "EfficientNetV2S"
    head_filters: int = 1280
    image_size: int = 300
    num_classes: int = 1000
