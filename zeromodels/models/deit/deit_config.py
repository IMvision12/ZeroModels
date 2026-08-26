from zeromodels.base import BaseConfig


class DeiTConfig(BaseConfig):
    r"""Configuration for [`DeiTModel`] / [`DeiTImageClassify`].

    DeiT is a ViT trained data-efficiently with an extra distillation token; the
    distilled variants (`use_distillation=True`) average a CLS head and a distillation
    head at inference. One `kf_config.json` (declaring the canonical
    [`DeiTImageClassify`]) sits on each variant's repo, and both the backbone and
    classifier load from it. Fields mirror the model constructor and serialize flat.

    Args:
        patch_size (`int`, *optional*, defaults to 16):
            Side length of each square image patch.
        embed_dim (`int`, *optional*, defaults to 192):
            Transformer hidden size.
        depth (`int`, *optional*, defaults to 12):
            Number of transformer encoder blocks.
        num_heads (`int`, *optional*, defaults to 3):
            Number of attention heads.
        use_distillation (`bool`, *optional*, defaults to `False`):
            Whether the checkpoint has a distillation token and second head.
        no_embed_class (`bool`, *optional*, defaults to `False`):
            Whether positional embeddings exclude the class/distillation tokens
            (the DeiT III recipe).
        layer_scale_init (`float`, *optional*, defaults to `None`):
            Initial per-channel LayerScale value; `None` disables LayerScale.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.deit import DeiTConfig, DeiTImageClassify

    >>> configuration = DeiTConfig()
    >>> model = DeiTImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "deit"

    patch_size: int = 16
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    use_distillation: bool = False
    no_embed_class: bool = False
    layer_scale_init: float = None
    image_size: int = 224
    num_classes: int = 1000
