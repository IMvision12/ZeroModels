"""BART model configuration."""

from zeromodels.base import BaseConfig


class BartConfig(BaseConfig):
    r"""Configuration for BART (a denoising seq2seq transformer, Lewis et al. 2019).

    A standard post-norm encoder-decoder: a bidirectional encoder and an
    autoregressive decoder that cross-attends to it, with learned positional
    embeddings (offset by 2), a ``layernorm_embedding`` after the token +
    position embeddings, GELU feed-forwards, and token embeddings shared across
    the encoder, the decoder, and the LM head. Defaults are ``facebook/bart-large``.

    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Token vocabulary size (byte-level BPE, shared with RoBERTa).
        hidden_dim (`int`, *optional*, defaults to 1024):
            Hidden / embedding dimension (``d_model``).
        encoder_num_layers (`int`, *optional*, defaults to 12):
            Number of encoder transformer blocks.
        decoder_num_layers (`int`, *optional*, defaults to 12):
            Number of decoder transformer blocks.
        encoder_attention_heads (`int`, *optional*, defaults to 16):
            Encoder self-attention head count.
        decoder_attention_heads (`int`, *optional*, defaults to 16):
            Decoder self-/cross-attention head count.
        encoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Encoder MLP hidden dimension.
        decoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Decoder MLP hidden dimension.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            Maximum sequence length (learned position table size, plus the +2 offset).
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            Feed-forward activation.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Whether to scale the token embedding by ``sqrt(hidden_dim)``.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for every LayerNorm.
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            Dropout in the sequence-classification head (inference no-op).
        num_labels (`int`, *optional*, defaults to 3):
            Class count for :class:`BartSequenceClassify`.
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            Beginning-of-sequence token id (``<s>``).
        eos_token_id (`int`, *optional*, defaults to 2):
            End-of-sequence token id (``</s>``); also pooled by the classification head.
        decoder_start_token_id (`int`, *optional*, defaults to 2):
            First decoder token; BART starts generation from ``</s>``.

    Example:

    ```python
    >>> from zeromodels.models.bart import BartConfig, BartModel

    >>> configuration = BartConfig()
    >>> model = BartModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "bart"

    vocab_size: int = 50265
    hidden_dim: int = 1024
    encoder_num_layers: int = 12
    decoder_num_layers: int = 12
    encoder_attention_heads: int = 16
    decoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    decoder_ffn_dim: int = 4096
    max_position_embeddings: int = 1024
    activation_function: str = "gelu"
    scale_embedding: bool = False
    layer_norm_eps: float = 1e-05
    classifier_dropout: float = 0.0
    num_labels: int = 3
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    decoder_start_token_id: int = 2
