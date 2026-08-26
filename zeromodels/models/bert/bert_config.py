from zeromodels.base import BaseConfig


class BertConfig(BaseConfig):
    r"""Configuration for the BERT encoder ([`BertModel`]) and its task heads.

    BERT is a bidirectional transformer encoder pre-trained with masked-language and
    next-sentence objectives. One `zm_config.json` (declaring the canonical
    [`BertModel`]) sits on each variant's repo; the encoder, masked-LM, and task-head
    classes all load from it. Fields mirror the model constructor and serialize flat.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
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
        type_vocab_size (`int`, *optional*, defaults to 2):
            Number of token-type (segment) embeddings.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation used in the feed-forward blocks.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            LayerNorm epsilon.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.

    Examples:

    ```python
    >>> from zeromodels.models.bert import BertConfig, BertModel

    >>> configuration = BertConfig()
    >>> model = BertModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "bert"

    vocab_size: int = 30522
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
