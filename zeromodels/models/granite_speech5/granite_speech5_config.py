from zeromodels.base import BaseConfig


class GraniteSpeech5Config(BaseConfig):
    r"""Configuration for GraniteSpeech5: the conformer CTC ASR encoder.

    Granite Speech 5.0 is a self-conditioned CTC conformer (block-wise attention
    with Shaw's relative positions, two time-subsampling blocks) with a CTC head
    tied to the encoder's mid-layer self-conditioning projection. All fields are
    flat (the model has a single tower), matching the HF ``granite_speech5_ctc``
    checkpoint's encoder config plus the CTC vocab.

    Args:
        vocab_size (`int`, *optional*, defaults to 16384):
            CTC output vocabulary size (also the encoder's self-conditioning width).
        hidden_size (`int`, *optional*, defaults to 1024):
            Conformer hidden width.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Feed-forward hidden width per conformer block.
        num_hidden_layers (`int`, *optional*, defaults to 16):
            Number of conformer blocks.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Attention heads per block.
        head_dim (`int`, *optional*, defaults to 128):
            Per-head dimension.
        num_mel_bins (`int`, *optional*, defaults to 80):
            Mel bins; the stacked input feature width is ``num_mel_bins * 4``.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Feed-forward activation.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            Maximum relative-position span; the table holds ``2 * this + 1`` rows.
        context_size (`int`, *optional*, defaults to 128):
            Block-wise attention context window, in frames.
        conv_kernel_size (`int`, *optional*, defaults to 7):
            Depthwise-convolution kernel size.
        conv_expansion_factor (`int`, *optional*, defaults to 2):
            Convolution-module channel expansion factor.
        subsample_layers (`tuple`, *optional*, defaults to `(0, 1)`):
            Block indices that subsample time by 2.
        attention_bias (`bool`, *optional*, defaults to `True`):
            Whether the feed-forward Denses carry a bias.
        pad_token_id (`int`, *optional*, defaults to 0):
            CTC blank / padding id.

    Example:

    ```python
    >>> from zeromodels.models.granite_speech5 import GraniteSpeech5Config, GraniteSpeech5CTC

    >>> configuration = GraniteSpeech5Config()
    >>> model = GraniteSpeech5CTC(configuration)
    ```"""

    model_type = "granite_speech5_ctc"

    vocab_size: int = 16384
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 16
    num_attention_heads: int = 8
    head_dim: int = 128
    num_mel_bins: int = 80
    hidden_act: str = "silu"
    max_position_embeddings: int = 512
    context_size: int = 128
    conv_kernel_size: int = 7
    conv_expansion_factor: int = 2
    subsample_layers: tuple = (0, 1)
    attention_bias: bool = True
    pad_token_id: int = 0
