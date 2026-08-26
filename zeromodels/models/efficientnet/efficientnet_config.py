from zeromodels.base import BaseConfig


class EfficientNetConfig(BaseConfig):
    r"""Configuration for [`EfficientNetModel`] / [`EfficientNetImageClassify`].

    EfficientNet scales network width, depth, and input resolution together with a
    compound coefficient on top of a mobile inverted-bottleneck (MBConv) backbone. One
    `kf_config.json` (declaring the canonical [`EfficientNetImageClassify`]) sits on
    each variant's repo, and both the backbone and classifier load from it. Fields
    mirror the model constructor and serialize flat.

    Args:
        width_coefficient (`float`, *optional*, defaults to 1.0):
            Channel-width scaling coefficient.
        depth_coefficient (`float`, *optional*, defaults to 1.0):
            Block-repeat (depth) scaling coefficient.
        dropout_rate (`float`, *optional*, defaults to 0.2):
            Dropout rate before the classifier head.
        default_size (`int`, *optional*, defaults to 224):
            Reference resolution the scaling was defined at.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.efficientnet import (
    ...     EfficientNetConfig,
    ...     EfficientNetImageClassify,
    ... )

    >>> configuration = EfficientNetConfig()
    >>> model = EfficientNetImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "efficientnet"

    width_coefficient: float = 1.0
    depth_coefficient: float = 1.0
    dropout_rate: float = 0.2
    default_size: int = 224
    image_size: int = 224
    num_classes: int = 1000
