from zeromodels.base import BaseConfig


class DebertaV2Config(BaseConfig):
    r"""Configuration for the DeBERTa-v2 encoder ([`DebertaV2Model`]) and its heads.

    DeBERTa-v2 refines DeBERTa with a larger vocabulary, bucketed relative positions, a
    normalized relative embedding, and an optional convolutional token layer. One
    `zm_config.json` (declaring the canonical [`DebertaV2Model`]) sits on each variant's
    repo. Fields mirror the model constructor and serialize flat.

    Args:
        vocab_size (`int`, *optional*, defaults to 128100):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 1536):
            Hidden size.
        num_layers (`int`, *optional*, defaults to 24):
            Number of transformer encoder layers.
        num_heads (`int`, *optional*, defaults to 24):
            Number of attention heads.
        mlp_dim (`int`, *optional*, defaults to 6144):
            Feed-forward intermediate size.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            Maximum sequence length supported by the positional embeddings.
        max_relative_positions (`int`, *optional*, defaults to 512):
            Clamp range for the relative-position encoding.
        position_buckets (`int`, *optional*, defaults to 256):
            Number of log-spaced buckets for relative positions.
        pos_att_type (`tuple`, *optional*, defaults to `("p2c", "c2p")`):
            Which disentangled-attention terms to use.
        norm_rel_ebd (`bool`, *optional*, defaults to `True`):
            Whether to LayerNorm the relative position embeddings.
        conv_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size of the input convolution (0 disables it).
        conv_act (`str`, *optional*, defaults to `"gelu"`):
            Activation of the input convolution.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation used in the feed-forward blocks.
        layer_norm_eps (`float`, *optional*, defaults to 1e-7):
            LayerNorm epsilon.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.

    Examples:

    ```python
    >>> from zeromodels.models.deberta_v2 import DebertaV2Config, DebertaV2Model

    >>> configuration = DebertaV2Config()
    >>> model = DebertaV2Model(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "deberta_v2"

    vocab_size: int = 128100
    embed_dim: int = 1536
    num_layers: int = 24
    num_heads: int = 24
    mlp_dim: int = 6144
    max_position_embeddings: int = 512
    max_relative_positions: int = 512
    position_buckets: int = 256
    pos_att_type: tuple = ("p2c", "c2p")
    norm_rel_ebd: bool = True
    conv_kernel_size: int = 3
    conv_act: str = "gelu"
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-7
    pad_token_id: int = 0
