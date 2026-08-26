from zeromodels.base import BaseConfig


class ViTConfig(BaseConfig):
    r"""Configuration for [`ViTModel`] / [`ViTImageClassify`].

    The Vision Transformer splits an image into patches, linearly embeds them with an
    added class token and positional embeddings, and applies a standard transformer
    encoder. One `zm_config.json` (declaring the canonical [`ViTImageClassify`]) sits
    on each variant's repo, and both the backbone and classifier load from it. Fields
    mirror the model constructor and serialize flat.

    Args:
        patch_size (`int`, *optional*, defaults to 16):
            Side length of each square image patch.
        embed_dim (`int`, *optional*, defaults to 768):
            Transformer hidden size.
        depth (`int`, *optional*, defaults to 12):
            Number of transformer encoder blocks.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            MLP hidden-dim expansion ratio.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether the qkv projection uses a bias.
        qk_norm (`bool`, *optional*, defaults to `False`):
            Whether to apply LayerNorm to queries and keys.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the weights were trained at.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classifier output classes (backbone ignores it).

    Examples:

    ```python
    >>> from zeromodels.models.vit import ViTConfig, ViTImageClassify

    >>> configuration = ViTConfig()
    >>> model = ViTImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "vit"

    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_norm: bool = False
    image_size: int = 224
    num_classes: int = 1000
