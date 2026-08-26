from zeromodels.base import BaseConfig


class DebertaConfig(BaseConfig):
    r"""Configuration for the DeBERTa encoder ([`DebertaModel`]) and its task heads.

    DeBERTa adds disentangled content/position attention and an enhanced mask decoder
    to a BERT-style encoder. One `kf_config.json` (declaring the canonical
    [`DebertaModel`]) sits on each variant's repo. Fields mirror the model constructor
    and serialize flat.

    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 768):
            Hidden size.
        num_layers (`int`, *optional*, defaults to 12):
            Number of transformer encoder layers.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads.
        mlp_dim (`int`, *optional*, defaults to 3072):
            Feed-forward intermediate size.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            Maximum sequence length supported by the positional embeddings.
        max_relative_positions (`int`, *optional*, defaults to 512):
            Clamp range for the relative-position encoding.
        pos_att_type (`tuple`, *optional*, defaults to `("c2p", "p2c")`):
            Which disentangled-attention terms to use (content-to-position and/or
            position-to-content).
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation used in the feed-forward blocks.
        layer_norm_eps (`float`, *optional*, defaults to 1e-7):
            LayerNorm epsilon.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.

    Examples:

    ```python
    >>> from zeromodels.models.deberta import DebertaConfig, DebertaModel

    >>> configuration = DebertaConfig()
    >>> model = DebertaModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "deberta"

    vocab_size: int = 50265
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    max_position_embeddings: int = 512
    max_relative_positions: int = 512
    pos_att_type: tuple = ("c2p", "p2c")
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-7
    pad_token_id: int = 0
