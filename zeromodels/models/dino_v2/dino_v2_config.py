from zeromodels.base import BaseConfig


class DinoV2Config(BaseConfig):
    r"""Configuration for [`DinoV2Model`], a DINOv2 Vision Transformer backbone.

    The defaults describe the DINOv2 ViT-S/14 variant; other variants override the
    transformer dimensions. Fields mirror the model constructor and serialize flat
    to a repo's `zm_config.json`.

    Args:
        as_backbone (`bool`, *optional*, defaults to `False`):
            If `True`, output the list of per-block intermediate features (the last
            LayerNorm-normalized) for use as a backbone; if `False`, output only
            the final LayerNorm-normalized token sequence.
        patch_size (`int`, *optional*, defaults to 14):
            ViT patch size.
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
        layer_scale_init (`float`, *optional*, defaults to 1.0):
            LayerScale initialization value (DINOv2 uses per-layer scaling).
        include_normalization (`bool`, *optional*, defaults to `True`):
            Whether to prepend in-model image normalization.
        normalization_mode (`str`, *optional*, defaults to `"imagenet"`):
            Normalization preset used when `include_normalization` is `True`.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the model is built for.

    Examples:

    ```python
    >>> from zeromodels.models.dino_v2 import DinoV2Config, DinoV2Model

    >>> configuration = DinoV2Config()
    >>> model = DinoV2Model(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "dinov2"

    as_backbone: bool = False
    patch_size: int = 14
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_norm: bool = False
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    layer_scale_init: float = 1.0
    use_swiglu: bool = False
    include_normalization: bool = True
    normalization_mode: str = "imagenet"
    image_size: int = 224
