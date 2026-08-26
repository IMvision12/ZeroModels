"""DeepseekVLHybrid model configuration."""

from zeromodels.base import BaseConfig


class DeepseekVLHybridTextConfig(BaseConfig):
    r"""Configuration for the DeepseekVLHybrid text decoder (the `text_config` sub-config).

    Args:
        vocab_size (`int`, *optional*, defaults to 102400):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 4096):
            Text / residual-stream width.
        mlp_dim (`int`, *optional*, defaults to 11008):
            SwiGLU hidden width per text layer.
        num_layers (`int`, *optional*, defaults to 30):
            Number of text decoder blocks.
        num_heads (`int`, *optional*, defaults to 32):
            Query heads per text layer.
        num_kv_heads (`int`, *optional*, defaults to 32):
            Key/value heads per text layer.
        head_dim (`int`, *optional*, defaults to 128):
            Text per-head dim.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            Text RMSNorm epsilon.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Rotary base frequency.
        tie_embeddings (`bool`, *optional*, defaults to `False`):
            Whether [`DeepseekVLHybridConditionalGenerate`] ties the LM head to the token
            embeddings.

    Example:

    ```python
    >>> from zeromodels.models.deepseek_vl_hybrid import DeepseekVLHybridTextConfig

    >>> configuration = DeepseekVLHybridTextConfig()
    ```"""

    model_type = "deepseek_vl_hybrid_text"

    vocab_size: int = 102400
    embed_dim: int = 4096
    mlp_dim: int = 11008
    num_layers: int = 30
    num_heads: int = 32
    num_kv_heads: int = 32
    head_dim: int = 128
    norm_eps: float = 1e-06
    rope_theta: float = 10000.0
    tie_embeddings: bool = False


class DeepseekVLHybridVisionConfig(BaseConfig):
    r"""Configuration for the DeepseekVLHybrid vision tower (the `vision_config` sub-config).

    Args:
        embed_dim (`int`, *optional*, defaults to 1024):
            Low-res SigLIP vision tower hidden width.
        mlp_dim (`int`, *optional*, defaults to 4096):
            Low-res SigLIP vision tower MLP width.
        num_layers (`int`, *optional*, defaults to 24):
            Number of low-res SigLIP encoder blocks.
        num_heads (`int`, *optional*, defaults to 16):
            Low-res SigLIP attention heads.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            Low-res vision LayerNorm epsilon.
        image_size (`int`, *optional*, defaults to 384):
            Square low-res (SigLIP) input size in pixels.
        patch_size (`int`, *optional*, defaults to 16):
            Low-res vision patch size in pixels.

    Example:

    ```python
    >>> from zeromodels.models.deepseek_vl_hybrid import DeepseekVLHybridVisionConfig

    >>> configuration = DeepseekVLHybridVisionConfig()
    ```"""

    model_type = "deepseek_vl_hybrid_vision"

    embed_dim: int = 1024
    mlp_dim: int = 4096
    num_layers: int = 24
    num_heads: int = 16
    norm_eps: float = 1e-06
    image_size: int = 384
    patch_size: int = 16


class DeepseekVLHybridHighResConfig(BaseConfig):
    r"""Configuration for the DeepseekVLHybrid high-resolution vision encoder (the `high_res_config` sub-config).

    Args:
        embed_dim (`int`, *optional*, defaults to 768):
            High-res SAM/ViTDet tower hidden width.
        mlp_dim (`int`, *optional*, defaults to 3072):
            High-res SAM/ViTDet tower MLP width.
        num_layers (`int`, *optional*, defaults to 12):
            Number of high-res SAM/ViTDet encoder blocks.
        num_heads (`int`, *optional*, defaults to 12):
            High-res SAM/ViTDet attention heads.
        image_size (`int`, *optional*, defaults to 1024):
            Square high-res (SAM) input size in pixels.
        patch_size (`int`, *optional*, defaults to 16):
            High-res vision patch size in pixels.
        output_channels (`int`, *optional*, defaults to 256):
            SAM neck output channel count.
        window_size (`int`, *optional*, defaults to 14):
            SAM windowed-attention window size.
        global_attn_indexes (`tuple`, *optional*, defaults to `(2, 5, 8, 11)`):
            Indices of the SAM blocks that use global (non-windowed) attention.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            High-res vision LayerNorm epsilon.

    Example:

    ```python
    >>> from zeromodels.models.deepseek_vl_hybrid import DeepseekVLHybridHighResConfig

    >>> configuration = DeepseekVLHybridHighResConfig()
    ```"""

    model_type = "deepseek_vl_hybrid_high"

    embed_dim: int = 768
    mlp_dim: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    image_size: int = 1024
    patch_size: int = 16
    output_channels: int = 256
    window_size: int = 14
    global_attn_indexes: tuple = (2, 5, 8, 11)
    norm_eps: float = 1e-06


class DeepseekVLHybridConfig(BaseConfig):
    r"""Configuration for DeepseekVLHybrid: the composite holding each tower's sub-config.

    Args:
        text_config (`DeepseekVLHybridTextConfig` or `dict`, *optional*):
            Configuration of the DeepseekVLHybrid text decoder.
        vision_config (`DeepseekVLHybridVisionConfig` or `dict`, *optional*):
            Configuration of the DeepseekVLHybrid vision tower.
        high_res_config (`DeepseekVLHybridHighResConfig` or `dict`, *optional*):
            Configuration of the DeepseekVLHybrid high-resolution vision encoder.
        image_token_id (`int`, *optional*, defaults to 100015):
            The `<image_placeholder>` token id whose slots receive image features.

    Example:

    ```python
    >>> from zeromodels.models.deepseek_vl_hybrid import DeepseekVLHybridConfig, DeepseekVLHybridConditionalGenerate

    >>> configuration = DeepseekVLHybridConfig()
    >>> model = DeepseekVLHybridConditionalGenerate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "deepseek_vl_hybrid"

    sub_configs = {
        "text_config": DeepseekVLHybridTextConfig,
        "vision_config": DeepseekVLHybridVisionConfig,
        "high_res_config": DeepseekVLHybridHighResConfig,
    }
    sub_config_prefixes = {
        "text_config": "",
        "vision_config": "vision_",
        "high_res_config": "high_res_",
    }
    group_extras = {"vision_config": ("image_size", "patch_size")}

    text_config: DeepseekVLHybridTextConfig | dict | None = None
    vision_config: DeepseekVLHybridVisionConfig | dict | None = None
    high_res_config: DeepseekVLHybridHighResConfig | dict | None = None
    image_token_id: int = 100015
