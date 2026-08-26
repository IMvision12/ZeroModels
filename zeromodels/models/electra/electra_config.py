from zeromodels.base import BaseConfig


class ElectraConfig(BaseConfig):
    r"""Configuration for the ELECTRA encoder ([`ElectraModel`]) and its task heads.

    ELECTRA is a BERT-style bidirectional transformer encoder pre-trained as a
    replaced-token discriminator (with a smaller generator producing the corrupted
    tokens). It embeds tokens at a separate `embedding_size` and projects up to
    `embed_dim` (the hidden size) when the two differ, and has **no pooler**. The
    discriminator repo (declaring [`ElectraModel`]) serves the encoder + downstream
    heads; the generator repo (declaring [`ElectraMaskedLM`]) serves the masked-LM.
    Fields mirror the model constructor and serialize flat.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Token vocabulary size.
        embedding_size (`int`, *optional*, defaults to 128):
            Token-embedding dimension. Projected to `embed_dim` when they differ.
        embed_dim (`int`, *optional*, defaults to 256):
            Hidden size of the transformer encoder.
        num_layers (`int`, *optional*, defaults to 12):
            Number of transformer encoder layers.
        num_heads (`int`, *optional*, defaults to 4):
            Number of attention heads.
        mlp_dim (`int`, *optional*, defaults to 1024):
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
    >>> from zeromodels.models.electra import ElectraConfig, ElectraModel

    >>> configuration = ElectraConfig()
    >>> model = ElectraModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "electra"

    vocab_size: int = 30522
    embedding_size: int = 128
    embed_dim: int = 256
    num_layers: int = 12
    num_heads: int = 4
    mlp_dim: int = 1024
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
