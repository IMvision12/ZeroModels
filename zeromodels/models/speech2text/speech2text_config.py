"""Speech2Text model configuration."""

from zeromodels.base import BaseConfig


class Speech2TextTextConfig(BaseConfig):
    r"""Configuration for the Speech2Text text decoder (the `text_config` sub-config).

    Args:
        num_layers (`int`, *optional*, defaults to 6):
            Number of decoder transformer blocks.
        attention_heads (`int`, *optional*, defaults to 4):
            Decoder self-attention / cross-attention head count.
        ffn_dim (`int`, *optional*, defaults to 2048):
            Decoder MLP hidden dimension.
        vocab_size (`int`, *optional*, defaults to 10000):
            SentencePiece token vocabulary size.
        max_target_positions (`int`, *optional*, defaults to 1024):
            Maximum decoded length.
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id (used by the decoder position embedding).

    Example:

    ```python
    >>> from zeromodels.models.speech2text import Speech2TextTextConfig

    >>> configuration = Speech2TextTextConfig()
    ```"""

    model_type = "speech_to_text_text"

    num_layers: int = 6
    attention_heads: int = 4
    ffn_dim: int = 2048
    vocab_size: int = 10000
    max_target_positions: int = 1024
    pad_token_id: int = 1


class Speech2TextAudioConfig(BaseConfig):
    r"""Configuration for the Speech2Text audio encoder (the `audio_config` sub-config).

    Args:
        num_layers (`int`, *optional*, defaults to 12):
            Number of encoder transformer blocks.
        attention_heads (`int`, *optional*, defaults to 4):
            Encoder self-attention head count.
        ffn_dim (`int`, *optional*, defaults to 2048):
            Encoder MLP hidden dimension.
        num_mel_bins (`int`, *optional*, defaults to 80):
            Mel-filterbank channel count of the input features.
        max_source_positions (`int`, *optional*, defaults to 6000):
            Maximum encoder position.
        conv_channels (`int`, *optional*, defaults to 1024):
            Channel count of the Conv1d subsampler.
        conv_kernel_sizes (`tuple`, *optional*, defaults to `(5, 5)`):
            Kernel sizes of the Conv1d subsampler layers.
        num_conv_layers (`int`, *optional*, defaults to 2):
            Number of Conv1d subsampler layers.

    Example:

    ```python
    >>> from zeromodels.models.speech2text import Speech2TextAudioConfig

    >>> configuration = Speech2TextAudioConfig()
    ```"""

    model_type = "speech_to_text_audio"

    num_layers: int = 12
    attention_heads: int = 4
    ffn_dim: int = 2048
    num_mel_bins: int = 80
    max_source_positions: int = 6000
    conv_channels: int = 1024
    conv_kernel_sizes: tuple = (5, 5)
    num_conv_layers: int = 2


class Speech2TextConfig(BaseConfig):
    r"""Configuration for Speech2Text: the composite holding each tower's sub-config.

    Args:
        text_config (`Speech2TextTextConfig` or `dict`, *optional*):
            Configuration of the Speech2Text text decoder.
        audio_config (`Speech2TextAudioConfig` or `dict`, *optional*):
            Configuration of the Speech2Text audio encoder.
        hidden_dim (`int`, *optional*, defaults to 256):
            Hidden / embedding dimension.
        scale_embedding (`bool`, *optional*, defaults to `True`):
            Whether to scale the token embedding by `sqrt(hidden_dim)`.
        activation_function (`str`, *optional*, defaults to `"relu"`):
            MLP activation.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for every LayerNorm.

    Example:

    ```python
    >>> from zeromodels.models.speech2text import Speech2TextConfig, Speech2TextModel

    >>> configuration = Speech2TextConfig()
    >>> model = Speech2TextModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "speech_to_text"

    sub_configs = {
        "text_config": Speech2TextTextConfig,
        "audio_config": Speech2TextAudioConfig,
    }
    sub_config_prefixes = {"text_config": "decoder_", "audio_config": "encoder_"}
    group_extras = {
        "text_config": ("vocab_size", "max_target_positions", "pad_token_id"),
        "audio_config": (
            "num_mel_bins",
            "max_source_positions",
            "conv_channels",
            "conv_kernel_sizes",
            "num_conv_layers",
        ),
    }

    text_config: Speech2TextTextConfig | dict | None = None
    audio_config: Speech2TextAudioConfig | dict | None = None
    hidden_dim: int = 256
    scale_embedding: bool = True
    activation_function: str = "relu"
    layer_norm_eps: float = 1e-05
