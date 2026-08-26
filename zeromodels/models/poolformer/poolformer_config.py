from zeromodels.base import BaseConfig


class PoolFormerConfig(BaseConfig):
    r"""Configuration for [`PoolFormerModel`] / [`PoolFormerImageClassify`].

    PoolFormer instantiates the MetaFormer template with a parameter-free average-
    pooling token mixer, showing the general architecture, not attention, drives much
    of the performance. One `kf_config.json` (declaring the canonical
    [`PoolFormerImageClassify`]) sits on each variant's repo, and both the backbone and
    classifier load from it. Fields mirror the model constructor and serialize flat.

    Args:
        embed_dim (`tuple`, *optional*, defaults to `(64, 128, 320, 512)`):
            Channel width per stage.
        depths (`tuple`, *optional*, defaults to `(2, 2, 6, 2)`):
            Number of blocks per stage.
        init_scale (`float`, *optional*, defaults to 1e-5):
            Initial value for the per-channel LayerScale.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.poolformer import (
    ...     PoolFormerConfig,
    ...     PoolFormerImageClassify,
    ... )

    >>> configuration = PoolFormerConfig()
    >>> model = PoolFormerImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "poolformer"

    embed_dim: tuple = (64, 128, 320, 512)
    depths: tuple = (2, 2, 6, 2)
    init_scale: float = 1e-5
    image_size: int = 224
    num_classes: int = 1000
