from zeromodels.base import BaseConfig


class FlexiViTConfig(BaseConfig):
    r"""Configuration for [`FlexiViTModel`] / [`FlexiViTImageClassify`].

    FlexiViT is a ViT trained across a range of patch sizes so a single checkpoint can
    run at different patch resolutions; the released weights use `no_embed_class=True`.
    One `kf_config.json` (declaring the canonical [`FlexiViTImageClassify`]) sits on
    each variant's repo, and both the backbone and classifier load from it. Fields
    mirror the model constructor and serialize flat.

    Args:
        patch_size (`int`, *optional*, defaults to 16):
            Side length of each square image patch.
        embed_dim (`int`, *optional*, defaults to 384):
            Transformer hidden size.
        depth (`int`, *optional*, defaults to 12):
            Number of transformer encoder blocks.
        num_heads (`int`, *optional*, defaults to 6):
            Number of attention heads.
        no_embed_class (`bool`, *optional*, defaults to `True`):
            Whether positional embeddings exclude the class token (FlexiViT recipe).
        image_size (`int`, *optional*, defaults to 240):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.flexivit import FlexiViTConfig, FlexiViTImageClassify

    >>> configuration = FlexiViTConfig()
    >>> model = FlexiViTImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "flexivit"

    patch_size: int = 16
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    no_embed_class: bool = True
    image_size: int = 240
    num_classes: int = 1000
