from zeromodels.base import BaseConfig


class T5Config(BaseConfig):
    r"""Configuration for the original T5 encoder-decoder ([`T5Model`]) and its heads.

    T5 is a text-to-text encoder-decoder transformer with relative position bias (no
    absolute or rotary positions), T5-style RMSNorm (scale only, no bias), pre-LayerNorm
    residual blocks, and bias-free projections. One `zm_config.json` (declaring the
    canonical [`T5Model`]) sits on each variant's repo; the backbone, generative head,
    encoder, and task-head classes all load from it. Fields mirror the model constructor
    and serialize flat.

    Args:
        vocab_size (`int`, *optional*, defaults to 32128):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 768):
            Model dimension (`d_model`).
        key_value_dim (`int`, *optional*, defaults to 64):
            Per-head query/key/value dimension (`d_kv`). The attention inner dimension is
            `num_heads * key_value_dim`, which may differ from `embed_dim`.
        mlp_dim (`int`, *optional*, defaults to 3072):
            Feed-forward intermediate size (`d_ff`).
        num_layers (`int`, *optional*, defaults to 12):
            Number of encoder layers.
        num_decoder_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            Number of buckets for the relative position bias.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            Maximum distance mapped by the relative position bias.
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            Feed-forward activation (`dense_act_fn`).
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            RMSNorm epsilon.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the LM head is tied to the shared token embedding. Original T5 ties
            them and scales the decoder output by `embed_dim ** -0.5` before the head.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id (also the decoder start token).
        eos_token_id (`int`, *optional*, defaults to 1):
            End-of-sequence token id.
        decoder_start_token_id (`int`, *optional*, defaults to 0):
            Token id fed as the first decoder input (T5 uses the pad token).

    Examples:

    ```python
    >>> from zeromodels.models.t5 import T5Config, T5Model

    >>> configuration = T5Config()
    >>> model = T5Model(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "t5"

    vocab_size: int = 32128
    embed_dim: int = 768
    key_value_dim: int = 64
    mlp_dim: int = 3072
    num_layers: int = 12
    num_decoder_layers: int = 12
    num_heads: int = 12
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    hidden_act: str = "relu"
    layer_norm_eps: float = 1e-6
    dropout: float = 0.1
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 1
    decoder_start_token_id: int = 0
