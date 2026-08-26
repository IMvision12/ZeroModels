from zeromodels.base import BaseConfig


class ResNeXtConfig(BaseConfig):
    r"""Configuration for [`ResNeXtModel`] / [`ResNeXtImageClassify`].

    ResNeXt replaces ResNet's bottleneck with grouped ("cardinality") convolutions.
    The hosted variants override `depths` / `width_factor` (32x4d / 32x8d / 32x16d /
    32x32d). One `zm_config.json` (declaring the canonical
    [`ResNeXtImageClassify`]) sits on each variant's repo, and both the backbone and
    classifier load from it. Fields mirror the model constructor and serialize flat.

    Args:
        depths (`tuple`, *optional*, defaults to `(3, 4, 6, 3)`):
            Number of residual blocks per stage.
        filters (`tuple`, *optional*, defaults to `(64, 128, 256, 512)`):
            Base filter counts per stage.
        groups (`int`, *optional*, defaults to 32):
            Number of groups (cardinality) in the grouped convolutions.
        width_factor (`int`, *optional*, defaults to 2):
            Bottleneck width multiplier (32x`4`d = width_factor 2, 32x8d = 4, ...).
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.resnext import ResNeXtConfig, ResNeXtImageClassify

    >>> configuration = ResNeXtConfig()
    >>> model = ResNeXtImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "resnext"

    depths: tuple = (3, 4, 6, 3)
    filters: tuple = (64, 128, 256, 512)
    groups: int = 32
    width_factor: int = 2
    num_classes: int = 1000
