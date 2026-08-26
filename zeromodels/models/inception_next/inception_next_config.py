from zeromodels.base import BaseConfig


class InceptionNextConfig(BaseConfig):
    r"""Configuration for [`InceptionNextModel`] / [`InceptionNextImageClassify`].

    InceptionNeXt reworks the ConvNeXt block with an Inception-style depthwise mixer
    (split into small square, wide band, and identity branches) for higher throughput.
    One `kf_config.json` (declaring the canonical [`InceptionNextImageClassify`]) sits
    on each variant's repo, and both the backbone and classifier load from it. Fields
    mirror the model constructor and serialize flat.

    Args:
        depths (`tuple`, *optional*, defaults to `(2, 2, 6, 2)`):
            Number of blocks per stage.
        num_filters (`tuple`, *optional*, defaults to `(40, 80, 160, 320)`):
            Channel width per stage.
        mlp_ratios (`tuple`, *optional*, defaults to `(4, 4, 4, 3)`):
            MLP expansion ratio per stage.
        band_kernel_size (`int`, *optional*, defaults to 9):
            Kernel length of the wide 1xk / kx1 band branches.
        branch_ratio (`float`, *optional*, defaults to 0.25):
            Fraction of channels routed to each Inception mixer branch.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.inception_next import (
    ...     InceptionNextConfig,
    ...     InceptionNextImageClassify,
    ... )

    >>> configuration = InceptionNextConfig()
    >>> model = InceptionNextImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "inception_next"

    depths: tuple = (2, 2, 6, 2)
    num_filters: tuple = (40, 80, 160, 320)
    mlp_ratios: tuple = (4, 4, 4, 3)
    band_kernel_size: int = 9
    branch_ratio: float = 0.25
    image_size: int = 224
    num_classes: int = 1000
