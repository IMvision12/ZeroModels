from zeromodels.base import BaseConfig


class EfficientNetLiteConfig(BaseConfig):
    r"""Configuration for [`EfficientNetLiteModel`] / [`EfficientNetLiteImageClassify`].

    EfficientNet-Lite is a mobile-friendly EfficientNet variant that drops squeeze-and-
    excitation and swish (replaced by ReLU6) for edge deployment. One `kf_config.json`
    (declaring the canonical [`EfficientNetLiteImageClassify`]) sits on each variant's
    repo, and both the backbone and classifier load from it. Fields mirror the model
    constructor and serialize flat.

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
    >>> from zeromodels.models.efficientnet_lite import (
    ...     EfficientNetLiteConfig,
    ...     EfficientNetLiteImageClassify,
    ... )

    >>> configuration = EfficientNetLiteConfig()
    >>> model = EfficientNetLiteImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "efficientnet_lite"

    width_coefficient: float = 1.0
    depth_coefficient: float = 1.0
    dropout_rate: float = 0.2
    default_size: int = 224
    image_size: int = 224
    num_classes: int = 1000
