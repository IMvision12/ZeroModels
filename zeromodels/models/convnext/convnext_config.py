from zeromodels.base import BaseConfig


class ConvNeXtConfig(BaseConfig):
    r"""Configuration for [`ConvNeXtModel`] / [`ConvNeXtImageClassify`].

    ConvNeXt is a pure-convolutional network that modernizes a ResNet with a
    patchify stem, depthwise 7x7 convolutions, inverted bottlenecks, LayerNorm, and
    per-channel layer scaling. One `kf_config.json` (declaring the canonical
    [`ConvNeXtImageClassify`]) sits on each variant's repo, and both the backbone and
    classifier load from it. Fields mirror the model constructor and serialize flat.

    Args:
        depths (`tuple`, *optional*, defaults to `(3, 3, 9, 3)`):
            Number of ConvNeXt blocks per stage.
        projection_dim (`tuple`, *optional*, defaults to `(96, 192, 384, 768)`):
            Channel width per stage.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.convnext import ConvNeXtConfig, ConvNeXtImageClassify

    >>> configuration = ConvNeXtConfig()
    >>> model = ConvNeXtImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "convnext"

    depths: tuple = (3, 3, 9, 3)
    projection_dim: tuple = (96, 192, 384, 768)
    image_size: int = 224
    num_classes: int = 1000
