"""Whisper model configuration."""

from zeromodels.base import BaseConfig


class WhisperTextConfig(BaseConfig):
    r"""Configuration for the Whisper text decoder (the `text_config` sub-config).

    Args:
        num_layers (`int`, *optional*, defaults to 4):
            Number of decoder transformer blocks.
        attention_heads (`int`, *optional*, defaults to 6):
            Decoder self-attention / cross-attention head count.
        ffn_dim (`int`, *optional*, defaults to 1536):
            Decoder MLP hidden dimension.
        vocab_size (`int`, *optional*, defaults to 51865):
            Token vocabulary size (51866 for the v3 variants).
        max_target_positions (`int`, *optional*, defaults to 448):
            Maximum decoded length.

    Example:

    ```python
    >>> from zeromodels.models.whisper import WhisperTextConfig

    >>> configuration = WhisperTextConfig()
    ```"""

    model_type = "whisper_text"

    num_layers: int = 4
    attention_heads: int = 6
    ffn_dim: int = 1536
    vocab_size: int = 51865
    max_target_positions: int = 448


class WhisperAudioConfig(BaseConfig):
    r"""Configuration for the Whisper audio encoder (the `audio_config` sub-config).

    Args:
        num_layers (`int`, *optional*, defaults to 4):
            Number of encoder transformer blocks.
        attention_heads (`int`, *optional*, defaults to 6):
            Encoder self-attention head count.
        ffn_dim (`int`, *optional*, defaults to 1536):
            Encoder MLP hidden dimension.
        num_mel_bins (`int`, *optional*, defaults to 80):
            Mel-bin count of the input log-mel spectrogram (128 for large-v3).
        max_source_positions (`int`, *optional*, defaults to 1500):
            Maximum encoder position.

    Example:

    ```python
    >>> from zeromodels.models.whisper import WhisperAudioConfig

    >>> configuration = WhisperAudioConfig()
    ```"""

    model_type = "whisper_audio"

    num_layers: int = 4
    attention_heads: int = 6
    ffn_dim: int = 1536
    num_mel_bins: int = 80
    max_source_positions: int = 1500


class WhisperConfig(BaseConfig):
    r"""Configuration for Whisper: the composite holding each tower's sub-config.

    Args:
        text_config (`WhisperTextConfig` or `dict`, *optional*):
            Configuration of the Whisper text decoder.
        audio_config (`WhisperAudioConfig` or `dict`, *optional*):
            Configuration of the Whisper audio encoder.
        hidden_dim (`int`, *optional*, defaults to 384):
            Hidden / embedding dimension.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            MLP activation (`"gelu"` exact, matches OpenAI).
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for every LayerNorm.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Whether to scale the decoder token embedding by `sqrt(hidden_dim)`.

    Example:

    ```python
    >>> from zeromodels.models.whisper import WhisperConfig, WhisperModel

    >>> configuration = WhisperConfig()
    >>> model = WhisperModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "whisper"

    sub_configs = {"text_config": WhisperTextConfig, "audio_config": WhisperAudioConfig}
    sub_config_prefixes = {"text_config": "decoder_", "audio_config": "encoder_"}
    group_extras = {
        "text_config": ("vocab_size", "max_target_positions"),
        "audio_config": ("num_mel_bins", "max_source_positions"),
    }

    text_config: WhisperTextConfig | dict | None = None
    audio_config: WhisperAudioConfig | dict | None = None
    hidden_dim: int = 384
    activation_function: str = "gelu"
    layer_norm_eps: float = 1e-05
    scale_embedding: bool = False
