from zeromodels.base import BaseConfig


class ResNetConfig(BaseConfig):
    r"""Configuration for [`ResNetModel`] / [`ResNetImageClassify`].

    A ResNet stacks four stages of residual bottleneck blocks. The defaults
    describe an 18-block-style layout; the hosted variants override `depths` /
    `filters` (resnet50 / resnet101 / resnet152). One `kf_config.json` (declaring
    the canonical [`ResNetImageClassify`]) sits on each variant's repo, and both the
    backbone and the classifier load from it. Fields mirror the model constructor
    and serialize flat.

    Args:
        depths (`tuple`, *optional*, defaults to `(2, 2, 2, 2)`):
            Number of residual blocks per stage.
        filters (`tuple`, *optional*, defaults to `(64, 128, 256, 512)`):
            Base filter counts per stage (the output width is `filters[i] *
            expansion`).
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (used by
            [`ResNetImageClassify`]; the backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.resnet import ResNetConfig, ResNetImageClassify

    >>> configuration = ResNetConfig()
    >>> model = ResNetImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "resnet"

    depths: tuple = (2, 2, 2, 2)
    filters: tuple = (64, 128, 256, 512)
    num_classes: int = 1000
