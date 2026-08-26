from zeromodels.base import BaseConfig


class DinoV3ViTConfig(BaseConfig):
    r"""Configuration for [`DinoV3ViTModel`], a DINOv3 Vision Transformer backbone.

    The defaults describe the DINOv3 ViT-B/16 variant; other variants override the
    transformer dimensions. DINOv3 ViTs use rotary position embeddings, register
    tokens, and (optionally) SwiGLU MLPs. Fields mirror the model constructor and
    serialize flat to a repo's `zm_config.json`.

    Args:
        as_backbone (`bool`, *optional*, defaults to `False`):
            If `True`, output the list of per-block intermediate features (the last
            LayerNorm-normalized) for use as a backbone; if `False`, output only
            the final LayerNorm-normalized token sequence.
        patch_size (`int`, *optional*, defaults to 16):
            ViT patch size.
        embed_dim (`int`, *optional*, defaults to 768):
            Hidden dimension.
        depth (`int`, *optional*, defaults to 12):
            Number of transformer encoder layers.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads per layer.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            MLP expansion ratio.
        use_swiglu (`bool`, *optional*, defaults to `False`):
            Whether the MLP is a SwiGLU (used by the larger variants).
        num_register_tokens (`int`, *optional*, defaults to 4):
            Number of learned register tokens prepended to the sequence.
        layer_scale_init (`float`, *optional*, defaults to 1.0):
            LayerScale initialization value.
        rope_theta (`float`, *optional*, defaults to 100.0):
            Rotary-position-embedding base frequency.
        query_bias (`bool`, *optional*, defaults to `True`):
            Whether the query projection uses a bias.
        key_bias (`bool`, *optional*, defaults to `False`):
            Whether the key projection uses a bias.
        value_bias (`bool`, *optional*, defaults to `True`):
            Whether the value projection uses a bias.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation used in the MLP.
        mlp_bias (`bool`, *optional*, defaults to `True`):
            Whether the MLP projections use a bias.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for the LayerNorm layers.
        include_normalization (`bool`, *optional*, defaults to `True`):
            Whether to prepend in-model image normalization.
        normalization_mode (`str`, *optional*, defaults to `"imagenet"`):
            Normalization preset used when `include_normalization` is `True`.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the model is built for.

    Examples:

    ```python
    >>> from zeromodels.models.dino_v3 import DinoV3ViTConfig, DinoV3ViTModel

    >>> configuration = DinoV3ViTConfig()
    >>> model = DinoV3ViTModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "dinov3_vit"

    as_backbone: bool = False
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    use_swiglu: bool = False
    num_register_tokens: int = 4
    layer_scale_init: float = 1.0
    rope_theta: float = 100.0
    query_bias: bool = True
    key_bias: bool = False
    value_bias: bool = True
    hidden_act: str = "gelu"
    mlp_bias: bool = True
    layer_norm_eps: float = 1e-5
    include_normalization: bool = True
    normalization_mode: str = "imagenet"
    image_size: int = 224


class DinoV3ConvNeXtConfig(BaseConfig):
    r"""Configuration for [`DinoV3ConvNeXtModel`], a DINOv3 ConvNeXt backbone.

    The defaults describe the DINOv3 ConvNeXt-Tiny variant; other variants override
    the per-stage depths and channel widths. Fields mirror the model constructor
    and serialize flat to a repo's `zm_config.json`.

    Args:
        as_backbone (`bool`, *optional*, defaults to `False`):
            If `True`, output the list of per-stage feature maps for use as a
            backbone; if `False`, output only the final-stage feature map.
        depths (`tuple`, *optional*, defaults to `(3, 3, 9, 3)`):
            Per-stage ConvNeXt block counts.
        projection_dim (`tuple`, *optional*, defaults to `(96, 192, 384, 768)`):
            Per-stage channel widths.
        include_normalization (`bool`, *optional*, defaults to `True`):
            Whether to prepend in-model image normalization.
        normalization_mode (`str`, *optional*, defaults to `"imagenet"`):
            Normalization preset used when `include_normalization` is `True`.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the model is built for.

    Examples:

    ```python
    >>> from zeromodels.models.dino_v3 import DinoV3ConvNeXtConfig, DinoV3ConvNeXtModel

    >>> configuration = DinoV3ConvNeXtConfig()
    >>> model = DinoV3ConvNeXtModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "dinov3_convnext"

    as_backbone: bool = False
    depths: tuple = (3, 3, 9, 3)
    projection_dim: tuple = (96, 192, 384, 768)
    include_normalization: bool = True
    normalization_mode: str = "imagenet"
    image_size: int = 224
