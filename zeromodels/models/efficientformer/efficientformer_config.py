from zeromodels.base import BaseConfig


class EfficientFormerConfig(BaseConfig):
    r"""Configuration for [`EfficientFormerModel`] / [`EfficientFormerImageClassify`].

    EfficientFormer is a hybrid conv/transformer network: convolutional MetaBlocks in
    the early stages and attention-based (`num_vit`) blocks in the last stage. One
    `zm_config.json` (declaring the canonical [`EfficientFormerImageClassify`]) sits on
    each variant's repo, and both the backbone and classifier load from it. Fields
    mirror the model constructor and serialize flat.

    Args:
        depths (`tuple`, *optional*, defaults to `(3, 2, 6, 4)`):
            Number of blocks per stage.
        embed_dim (`tuple`, *optional*, defaults to `(48, 96, 224, 448)`):
            Channel width per stage.
        num_vit (`int`, *optional*, defaults to 1):
            Number of attention blocks at the end of the last stage.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.efficientformer import (
    ...     EfficientFormerConfig,
    ...     EfficientFormerImageClassify,
    ... )

    >>> configuration = EfficientFormerConfig()
    >>> model = EfficientFormerImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "efficientformer"

    depths: tuple = (3, 2, 6, 4)
    embed_dim: tuple = (48, 96, 224, 448)
    num_vit: int = 1
    image_size: int = 224
    num_classes: int = 1000
