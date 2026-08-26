from zeromodels.base import BaseConfig


class DenseNetConfig(BaseConfig):
    r"""Configuration for [`DenseNetModel`] / [`DenseNetImageClassify`].

    DenseNet connects each layer to every other layer within a dense block, so feature
    maps are concatenated (not summed), giving strong feature reuse at low parameter
    count. One `kf_config.json` (declaring the canonical [`DenseNetImageClassify`]) sits
    on each variant's repo, and both the backbone and classifier load from it. Fields
    mirror the model constructor and serialize flat.

    Args:
        depths (`tuple`, *optional*, defaults to `(6, 12, 24, 16)`):
            Number of dense layers per block.
        growth_rate (`int`, *optional*, defaults to 32):
            Number of feature maps each dense layer adds.
        initial_filter (`int`, *optional*, defaults to 64):
            Channel count of the stem convolution.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.densenet import DenseNetConfig, DenseNetImageClassify

    >>> configuration = DenseNetConfig()
    >>> model = DenseNetImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "densenet"

    depths: tuple = (6, 12, 24, 16)
    growth_rate: int = 32
    initial_filter: int = 64
    image_size: int = 224
    num_classes: int = 1000
