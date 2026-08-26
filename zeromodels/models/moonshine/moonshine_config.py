"""Moonshine model configuration."""

from zeromodels.base import BaseConfig


class MoonshineTextConfig(BaseConfig):
    r"""Configuration for the Moonshine text decoder (the `text_config` sub-config).

    Args:
        num_layers (`int`, *optional*, defaults to 6):
            Number of decoder transformer blocks.
        attention_heads (`int`, *optional*, defaults to 8):
            Decoder self-attention / cross-attention head count.
        num_kv_heads (`int`, *optional*, defaults to `None`):
            Decoder key/value head count; when `None`, equals the attention heads.
        ffn_dim (`int`, *optional*, defaults to 1152):
            Decoder MLP hidden dimension.
        activation (`str`, *optional*, defaults to `"silu"`):
            Decoder MLP activation (gated SiLU).
        vocab_size (`int`, *optional*, defaults to 32768):
            Token vocabulary size.
        max_position_embeddings (`int`, *optional*, defaults to 194):
            Maximum sequence length the rotary cache is built for.
        partial_rotary_factor (`float`, *optional*, defaults to 0.9):
            Fraction of head dimensions that receive rotary embedding (0.62 Base).
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Rotary-embedding base frequency.

    Example:

    ```python
    >>> from zeromodels.models.moonshine import MoonshineTextConfig

    >>> configuration = MoonshineTextConfig()
    ```"""

    model_type = "moonshine_text"

    num_layers: int = 6
    attention_heads: int = 8
    num_kv_heads: int = None
    ffn_dim: int = 1152
    activation: str = "silu"
    vocab_size: int = 32768
    max_position_embeddings: int = 194
    partial_rotary_factor: float = 0.9
    rope_theta: float = 10000.0


class MoonshineAudioConfig(BaseConfig):
    r"""Configuration for the Moonshine audio encoder (the `audio_config` sub-config).

    Args:
        num_layers (`int`, *optional*, defaults to 6):
            Number of encoder transformer blocks.
        attention_heads (`int`, *optional*, defaults to 8):
            Encoder self-attention head count.
        num_kv_heads (`int`, *optional*, defaults to `None`):
            Encoder key/value head count; when `None`, equals the attention heads.
        ffn_dim (`int`, *optional*, defaults to 1152):
            Encoder MLP hidden dimension.
        activation (`str`, *optional*, defaults to `"gelu"`):
            Encoder MLP activation.

    Example:

    ```python
    >>> from zeromodels.models.moonshine import MoonshineAudioConfig

    >>> configuration = MoonshineAudioConfig()
    ```"""

    model_type = "moonshine_audio"

    num_layers: int = 6
    attention_heads: int = 8
    num_kv_heads: int = None
    ffn_dim: int = 1152
    activation: str = "gelu"


class MoonshineConfig(BaseConfig):
    r"""Configuration for Moonshine: the composite holding each tower's sub-config.

    Args:
        text_config (`MoonshineTextConfig` or `dict`, *optional*):
            Configuration of the Moonshine text decoder.
        audio_config (`MoonshineAudioConfig` or `dict`, *optional*):
            Configuration of the Moonshine audio encoder.
        hidden_dim (`int`, *optional*, defaults to 288):
            Hidden / embedding dimension.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for every LayerNorm.

    Example:

    ```python
    >>> from zeromodels.models.moonshine import MoonshineConfig, MoonshineModel

    >>> configuration = MoonshineConfig()
    >>> model = MoonshineModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "moonshine"

    sub_configs = {
        "text_config": MoonshineTextConfig,
        "audio_config": MoonshineAudioConfig,
    }
    sub_config_prefixes = {"text_config": "decoder_", "audio_config": "encoder_"}
    group_extras = {
        "text_config": (
            "vocab_size",
            "max_position_embeddings",
            "partial_rotary_factor",
            "rope_theta",
        )
    }

    text_config: MoonshineTextConfig | dict | None = None
    audio_config: MoonshineAudioConfig | dict | None = None
    hidden_dim: int = 288
    layer_norm_eps: float = 1e-05
