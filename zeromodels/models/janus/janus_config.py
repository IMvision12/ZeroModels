"""Janus model configuration."""

from zeromodels.base import BaseConfig


class JanusTextConfig(BaseConfig):
    r"""Configuration for the Janus text decoder (the `text_config` sub-config).

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
            Whether [`JanusConditionalGenerate`] ties the LM head to the token embeddings.

    Example:

    ```python
    >>> from zeromodels.models.janus import JanusTextConfig

    >>> configuration = JanusTextConfig()
    ```"""

    model_type = "janus_text"

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


class JanusVisionConfig(BaseConfig):
    r"""Configuration for the Janus vision tower (the `vision_config` sub-config).

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
    >>> from zeromodels.models.janus import JanusVisionConfig

    >>> configuration = JanusVisionConfig()
    ```"""

    model_type = "janus_vision"

    embed_dim: int = 1024
    mlp_dim: int = 4096
    num_layers: int = 24
    num_heads: int = 16
    norm_eps: float = 1e-06
    image_size: int = 384
    patch_size: int = 16


class JanusConfig(BaseConfig):
    r"""Configuration for Janus: the composite holding each tower's sub-config.

    Args:
        text_config (`JanusTextConfig` or `dict`, *optional*):
            Configuration of the Janus text decoder.
        vision_config (`JanusVisionConfig` or `dict`, *optional*):
            Configuration of the Janus vision tower.
        image_token_id (`int`, *optional*, defaults to 100581):
            The `<image_placeholder>` token id whose slots receive image features.

    Example:

    ```python
    >>> from zeromodels.models.janus import JanusConfig, JanusConditionalGenerate

    >>> configuration = JanusConfig()
    >>> model = JanusConditionalGenerate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "janus"

    sub_configs = {"text_config": JanusTextConfig, "vision_config": JanusVisionConfig}
    sub_config_prefixes = {"text_config": "", "vision_config": "vision_"}
    group_extras = {"vision_config": ("image_size", "patch_size")}

    text_config: JanusTextConfig | dict | None = None
    vision_config: JanusVisionConfig | dict | None = None
    image_token_id: int = 100581
