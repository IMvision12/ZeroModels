from zeromodels.base import BaseConfig


class SwinConfig(BaseConfig):
    r"""Configuration for [`SwinModel`] / [`SwinImageClassify`].

    The Swin Transformer computes self-attention inside shifted local windows and
    merges patches between stages to build a hierarchical feature pyramid. One
    `kf_config.json` (declaring the canonical [`SwinImageClassify`]) sits on each
    variant's repo, and both the backbone and classifier load from it. Fields mirror
    the model constructor and serialize flat.

    Args:
        window_size (`int`, *optional*, defaults to 7):
            Side length of the local attention window.
        embed_dim (`int`, *optional*, defaults to 96):
            Channel width of the first stage (doubles each stage).
        depths (`tuple`, *optional*, defaults to `(2, 2, 6, 2)`):
            Number of Swin blocks per stage.
        num_heads (`tuple`, *optional*, defaults to `(3, 6, 12, 24)`):
            Number of attention heads per stage.
        pretrain_size (`int`, *optional*, defaults to 224):
            Resolution the relative-position bias was trained at.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.swin import SwinConfig, SwinImageClassify

    >>> configuration = SwinConfig()
    >>> model = SwinImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "swin"

    window_size: int = 7
    embed_dim: int = 96
    depths: tuple = (2, 2, 6, 2)
    num_heads: tuple = (3, 6, 12, 24)
    pretrain_size: int = 224
    image_size: int = 224
    num_classes: int = 1000
