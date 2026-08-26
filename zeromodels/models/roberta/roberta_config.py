from zeromodels.base import BaseConfig


class RobertaConfig(BaseConfig):
    r"""Configuration for the RoBERTa encoder ([`RobertaModel`]) and its task heads.

    RoBERTa is a BERT-style encoder trained with a robust, longer masked-LM recipe (no
    next-sentence objective, byte-level BPE, a single token type). One `kf_config.json`
    (declaring the canonical [`RobertaModel`]) sits on each variant's repo; the encoder,
    masked-LM, and task-head classes all load from it. Fields mirror the model
    constructor and serialize flat.

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
        max_position_embeddings (`int`, *optional*, defaults to 514):
            Maximum sequence length supported by the positional embeddings.
        type_vocab_size (`int`, *optional*, defaults to 1):
            Number of token-type (segment) embeddings.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation used in the feed-forward blocks.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            LayerNorm epsilon.
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id (positions are offset past it).

    Examples:

    ```python
    >>> from zeromodels.models.roberta import RobertaConfig, RobertaModel

    >>> configuration = RobertaConfig()
    >>> model = RobertaModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "roberta"

    vocab_size: int = 50265
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    max_position_embeddings: int = 514
    type_vocab_size: int = 1
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    pad_token_id: int = 1
