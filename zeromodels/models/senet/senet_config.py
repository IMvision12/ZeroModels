from zeromodels.base import BaseConfig


class SENetConfig(BaseConfig):
    r"""Configuration for [`SENetModel`] / [`SENetImageClassify`].

    SENet augments a ResNet (`bottleneck`) or ResNeXt (`resnext_block`, selected by
    `block_fn_name`) trunk with Squeeze-and-Excitation channel attention. One
    `zm_config.json` (declaring the canonical [`SENetImageClassify`]) sits on each
    variant's repo, and both the backbone and classifier load from it. Fields mirror
    the model constructor and serialize flat.

    Args:
        depths (`tuple`, *optional*, defaults to `(2, 2, 2, 2)`):
            Number of residual blocks per stage.
        filters (`tuple`, *optional*, defaults to `(64, 128, 256, 512)`):
            Base filter counts per stage.
        groups (`int`, *optional*, defaults to 32):
            Grouped-conv groups (used by the SE-ResNeXt variants).
        width_factor (`int`, *optional*, defaults to 2):
            Bottleneck width multiplier (SE-ResNeXt).
        senet (`bool`, *optional*, defaults to `True`):
            Whether to apply Squeeze-and-Excitation inside each block.
        block_fn_name (`str`, *optional*, defaults to `None`):
            Block builder: `None` -> bottleneck (SE-ResNet), `"resnext_block"` ->
            grouped (SE-ResNeXt).
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.senet import SENetConfig, SENetImageClassify

    >>> configuration = SENetConfig()
    >>> model = SENetImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "senet"

    depths: tuple = (2, 2, 2, 2)
    filters: tuple = (64, 128, 256, 512)
    groups: int = 32
    width_factor: int = 2
    senet: bool = True
    block_fn_name: str = None
    num_classes: int = 1000
