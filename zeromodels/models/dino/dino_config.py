from zeromodels.base import BaseConfig


class DinoViTConfig(BaseConfig):
    r"""Configuration for [`DinoViTModel`], a DINO Vision Transformer backbone.

    The defaults describe the DINO ViT-S/16 variant; other variants override the
    patch size and transformer dimensions. Fields mirror the model constructor and
    serialize flat to a repo's `zm_config.json`.

    Args:
        as_backbone (`bool`, *optional*, defaults to `False`):
            If `True`, output the list of per-block intermediate features (the last
            LayerNorm-normalized) for use as a backbone; if `False`, output only
            the final LayerNorm-normalized token sequence (CLS at index 0).
        patch_size (`int`, *optional*, defaults to 16):
            ViT patch size (8 or 16).
        embed_dim (`int`, *optional*, defaults to 384):
            Hidden dimension.
        depth (`int`, *optional*, defaults to 12):
            Number of transformer encoder layers.
        num_heads (`int`, *optional*, defaults to 6):
            Number of attention heads per layer.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            MLP expansion ratio.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the QKV projections.
        qk_norm (`bool`, *optional*, defaults to `False`):
            Whether to apply QK normalization.
        drop_rate (`float`, *optional*, defaults to 0.0):
            Dropout rate.
        attn_drop_rate (`float`, *optional*, defaults to 0.0):
            Attention dropout rate.
        include_normalization (`bool`, *optional*, defaults to `True`):
            Whether to prepend in-model image normalization (so raw images can be
            fed directly).
        normalization_mode (`str`, *optional*, defaults to `"imagenet"`):
            Normalization preset used when `include_normalization` is `True`.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the model is built for.

    Examples:

    ```python
    >>> from zeromodels.models.dino import DinoViTConfig, DinoViTModel

    >>> configuration = DinoViTConfig()
    >>> model = DinoViTModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "dino_vit"

    as_backbone: bool = False
    patch_size: int = 16
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_norm: bool = False
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    include_normalization: bool = True
    normalization_mode: str = "imagenet"
    image_size: int = 224


class DinoResNetConfig(BaseConfig):
    r"""Configuration for [`DinoResNetModel`], a DINO ResNet-50 backbone.

    The defaults describe the DINO ResNet-50 variant. Fields mirror the model
    constructor and serialize flat to a repo's `zm_config.json`.

    Args:
        as_backbone (`bool`, *optional*, defaults to `False`):
            If `True`, output the list of per-stage feature maps for use as a
            backbone; if `False`, output only the final-stage feature map.
        depths (`tuple`, *optional*, defaults to `(3, 4, 6, 3)`):
            Per-stage residual block counts.
        filters (`tuple`, *optional*, defaults to `(64, 128, 256, 512)`):
            Per-stage filter counts.
        include_normalization (`bool`, *optional*, defaults to `True`):
            Whether to prepend in-model image normalization.
        normalization_mode (`str`, *optional*, defaults to `"imagenet"`):
            Normalization preset used when `include_normalization` is `True`.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the model is built for.

    Examples:

    ```python
    >>> from zeromodels.models.dino import DinoResNetConfig, DinoResNetModel

    >>> configuration = DinoResNetConfig()
    >>> model = DinoResNetModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "dino_resnet"

    as_backbone: bool = False
    depths: tuple = (3, 4, 6, 3)
    filters: tuple = (64, 128, 256, 512)
    include_normalization: bool = True
    normalization_mode: str = "imagenet"
    image_size: int = 224
