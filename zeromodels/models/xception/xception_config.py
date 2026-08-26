from zeromodels.base import BaseConfig


class XceptionConfig(BaseConfig):
    r"""Configuration for [`XceptionModel`] / [`XceptionImageClassify`].

    Xception replaces Inception modules with depthwise-separable convolutions and
    residual connections; the timm `41`/`65`/`71` presets differ in depth, with an
    optional pre-activation variant. One `kf_config.json` (declaring the canonical
    [`XceptionImageClassify`]) sits on each variant's repo, and both the backbone and
    classifier load from it. Fields mirror the model constructor and serialize flat.

    Args:
        config (`str`, *optional*, defaults to `"41"`):
            Depth preset, one of `"41"`, `"65"`, `"71"`.
        preact (`bool`, *optional*, defaults to `False`):
            Whether to use the pre-activation (aligned) Xception variant.
        bn_epsilon (`float`, *optional*, defaults to 1e-3):
            BatchNorm epsilon.
        image_size (`int`, *optional*, defaults to 299):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.xception import XceptionConfig, XceptionImageClassify

    >>> configuration = XceptionConfig()
    >>> model = XceptionImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "xception"

    config: str = "41"
    preact: bool = False
    bn_epsilon: float = 1e-3
    image_size: int = 299
    num_classes: int = 1000
