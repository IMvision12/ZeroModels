"""LocateAnything model configuration."""

from zeromodels.base import BaseConfig


class LocateAnythingTextConfig(BaseConfig):
    r"""Configuration for the LocateAnything text decoder (the `text_config` sub-config).

    Args:
        vocab_size (`int`, *optional*, defaults to 152681):
            Text tokenizer vocabulary size.
        embed_dim (`int`, *optional*, defaults to 2048):
            Hidden size of the Qwen2 decoder.
        mlp_dim (`int`, *optional*, defaults to 11008):
            Feed-forward dimension of the decoder.
        num_layers (`int`, *optional*, defaults to 36):
            Number of decoder layers.
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads.
        num_kv_heads (`int`, *optional*, defaults to 2):
            Number of key/value heads (grouped-query attention).
        head_dim (`int`, *optional*, defaults to 128):
            Per-head dimension.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            RMSNorm epsilon.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            Decoder rotary-embedding base frequency.
        tie_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the LM head is tied to the token embedding.
        merge_kernel (`tuple`, *optional*, defaults to `(2, 2)`):
            Spatial merge kernel applied to vision tokens before projection.
        block_size (`int`, *optional*, defaults to 6):
            Parallel Box Decoding block size.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            Maximum sequence length the rotary cache is built for.

    Example:

    ```python
    >>> from zeromodels.models.locateanything import LocateAnythingTextConfig

    >>> configuration = LocateAnythingTextConfig()
    ```"""

    model_type = "locateanything_text"

    vocab_size: int = 152681
    embed_dim: int = 2048
    mlp_dim: int = 11008
    num_layers: int = 36
    num_heads: int = 16
    num_kv_heads: int = 2
    head_dim: int = 128
    norm_eps: float = 1e-06
    rope_theta: float = 1000000.0
    tie_embeddings: bool = True
    merge_kernel: tuple = (2, 2)
    block_size: int = 6
    max_position_embeddings: int = 32768


class LocateAnythingVisionConfig(BaseConfig):
    r"""Configuration for the LocateAnything vision tower (the `vision_config` sub-config).

    Args:
        embed_dim (`int`, *optional*, defaults to 1152):
            Hidden size of the MoonViT vision encoder.
        depth (`int`, *optional*, defaults to 27):
            Number of vision encoder layers.
        num_heads (`int`, *optional*, defaults to 16):
            Number of vision attention heads.
        mlp_dim (`int`, *optional*, defaults to 4304):
            Feed-forward dimension of the vision encoder.
        patch_size (`int`, *optional*, defaults to 14):
            Vision patch size.
        init_pos_h (`int`, *optional*, defaults to 64):
            Height of the pretrained position-embedding grid (interpolated).
        init_pos_w (`int`, *optional*, defaults to 64):
            Width of the pretrained position-embedding grid (interpolated).
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Vision rotary-embedding base frequency.

    Example:

    ```python
    >>> from zeromodels.models.locateanything import LocateAnythingVisionConfig

    >>> configuration = LocateAnythingVisionConfig()
    ```"""

    model_type = "locateanything_vision"

    embed_dim: int = 1152
    depth: int = 27
    num_heads: int = 16
    mlp_dim: int = 4304
    patch_size: int = 14
    init_pos_h: int = 64
    init_pos_w: int = 64
    rope_theta: float = 10000.0


class LocateAnythingConfig(BaseConfig):
    r"""Configuration for LocateAnything: the composite holding each tower's sub-config.

    Args:
        text_config (`LocateAnythingTextConfig` or `dict`, *optional*):
            Configuration of the LocateAnything text decoder.
        vision_config (`LocateAnythingVisionConfig` or `dict`, *optional*):
            Configuration of the LocateAnything vision tower.
        image_token_index (`int`, *optional*, defaults to 151665):
            Token id whose positions are replaced by vision features.

    Example:

    ```python
    >>> from zeromodels.models.locateanything import LocateAnythingConfig, LocateAnythingConditionalGenerate

    >>> configuration = LocateAnythingConfig()
    >>> model = LocateAnythingConditionalGenerate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "locateanything"

    sub_configs = {
        "text_config": LocateAnythingTextConfig,
        "vision_config": LocateAnythingVisionConfig,
    }
    sub_config_prefixes = {"text_config": "", "vision_config": "vision_"}

    text_config: LocateAnythingTextConfig | dict | None = None
    vision_config: LocateAnythingVisionConfig | dict | None = None
    image_token_index: int = 151665
