from zeromodels.base import BaseConfig


class PiTConfig(BaseConfig):
    r"""Configuration for [`PiTModel`] / [`PiTImageClassify`].

    PiT (Pooling-based Vision Transformer) introduces spatial pooling between three
    transformer stages so the token count shrinks while channels grow, ResNet-style.
    One `kf_config.json` (declaring the canonical [`PiTImageClassify`]) sits on each
    variant's repo, and both the backbone and classifier load from it. Fields mirror
    the model constructor and serialize flat.

    Args:
        patch_size (`int`, *optional*, defaults to 16):
            Side length of the initial conv patch embedding.
        stride (`int`, *optional*, defaults to 8):
            Stride of the patch embedding conv.
        embed_dim (`tuple`, *optional*, defaults to `(64, 128, 256)`):
            Channel width per stage.
        depth (`tuple`, *optional*, defaults to `(2, 6, 4)`):
            Number of transformer blocks per stage.
        heads (`tuple`, *optional*, defaults to `(2, 4, 8)`):
            Number of attention heads per stage.
        mlp_ratio (`int`, *optional*, defaults to 4):
            MLP hidden-dim expansion ratio.
        distilled (`bool`, *optional*, defaults to `False`):
            Whether the checkpoint has a distillation token and second head.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.pit import PiTConfig, PiTImageClassify

    >>> configuration = PiTConfig()
    >>> model = PiTImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "pit"

    patch_size: int = 16
    stride: int = 8
    embed_dim: tuple = (64, 128, 256)
    depth: tuple = (2, 6, 4)
    heads: tuple = (2, 4, 8)
    mlp_ratio: int = 4
    distilled: bool = False
    image_size: int = 224
    num_classes: int = 1000
