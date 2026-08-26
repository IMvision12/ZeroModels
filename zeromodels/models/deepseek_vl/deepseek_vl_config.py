"""DeepseekVL model configuration."""

from zeromodels.base import BaseConfig


class DeepseekVLTextConfig(BaseConfig):
    r"""Configuration for the DeepseekVL text decoder (the `text_config` sub-config).

    Args:
        vocab_size (`int`, *optional*, defaults to 102400):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 2048):
            Text / residual-stream width.
        mlp_dim (`int`, *optional*, defaults to 5632):
            SwiGLU hidden width per text layer.
        num_layers (`int`, *optional*, defaults to 24):
            Number of text decoder blocks.
        num_heads (`int`, *optional*, defaults to 16):
            Query heads per text layer.
        num_kv_heads (`int`, *optional*, defaults to 16):
            Key/value heads per text layer.
        head_dim (`int`, *optional*, defaults to 128):
            Text per-head dim.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            Text RMSNorm epsilon.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Rotary base frequency.
        tie_embeddings (`bool`, *optional*, defaults to `False`):
            Whether [`DeepseekVLConditionalGenerate`] ties the LM head to the token embeddings.

    Example:

    ```python
    >>> from zeromodels.models.deepseek_vl import DeepseekVLTextConfig

    >>> configuration = DeepseekVLTextConfig()
    ```"""

    model_type = "deepseek_vl_text"

    vocab_size: int = 102400
    embed_dim: int = 2048
    mlp_dim: int = 5632
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 16
    head_dim: int = 128
    norm_eps: float = 1e-06
    rope_theta: float = 10000.0
    tie_embeddings: bool = False


class DeepseekVLVisionConfig(BaseConfig):
    r"""Configuration for the DeepseekVL vision tower (the `vision_config` sub-config).

    Args:
        embed_dim (`int`, *optional*, defaults to 1024):
            SigLIP vision tower hidden width.
        mlp_dim (`int`, *optional*, defaults to 4096):
            SigLIP vision tower MLP width.
        num_layers (`int`, *optional*, defaults to 24):
            Number of SigLIP encoder blocks.
        num_heads (`int`, *optional*, defaults to 16):
            SigLIP attention heads.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            Vision LayerNorm epsilon.
        image_size (`int`, *optional*, defaults to 384):
            Square vision input size in pixels.
        patch_size (`int`, *optional*, defaults to 16):
            Vision patch size in pixels.

    Example:

    ```python
    >>> from zeromodels.models.deepseek_vl import DeepseekVLVisionConfig

    >>> configuration = DeepseekVLVisionConfig()
    ```"""

    model_type = "deepseek_vl_vision"

    embed_dim: int = 1024
    mlp_dim: int = 4096
    num_layers: int = 24
    num_heads: int = 16
    norm_eps: float = 1e-06
    image_size: int = 384
    patch_size: int = 16


class DeepseekVLConfig(BaseConfig):
    r"""Configuration for DeepseekVL: the composite holding each tower's sub-config.

    Args:
        text_config (`DeepseekVLTextConfig` or `dict`, *optional*):
            Configuration of the DeepseekVL text decoder.
        vision_config (`DeepseekVLVisionConfig` or `dict`, *optional*):
            Configuration of the DeepseekVL vision tower.
        image_token_id (`int`, *optional*, defaults to 100015):
            The `<image_placeholder>` token id whose slots receive image features.

    Example:

    ```python
    >>> from zeromodels.models.deepseek_vl import DeepseekVLConfig, DeepseekVLConditionalGenerate

    >>> configuration = DeepseekVLConfig()
    >>> model = DeepseekVLConditionalGenerate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "deepseek_vl"

    sub_configs = {
        "text_config": DeepseekVLTextConfig,
        "vision_config": DeepseekVLVisionConfig,
    }
    sub_config_prefixes = {"text_config": "", "vision_config": "vision_"}
    group_extras = {"vision_config": ("image_size", "patch_size")}

    text_config: DeepseekVLTextConfig | dict | None = None
    vision_config: DeepseekVLVisionConfig | dict | None = None
    image_token_id: int = 100015
