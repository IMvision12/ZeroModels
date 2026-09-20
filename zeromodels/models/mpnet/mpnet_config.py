from zeromodels.base import BaseConfig


class MPNetConfig(BaseConfig):
    r"""Configuration for the MPNet encoder ([`MPNetModel`]) and its task heads.

    MPNet is a BERT-style bidirectional encoder pre-trained with masked **and**
    permuted language modelling. It differs from BERT in three ways that matter to the
    implementation: there are no token-type (segment) embeddings, position ids are
    offset past the padding id as in RoBERTa, and every attention layer adds a shared
    **relative position bias** gathered from `relative_attention_num_buckets` buckets.
    One `zm_config.json` (declaring the canonical [`MPNetModel`]) sits on each variant's
    repo; the encoder, masked-LM, and task-head classes all load from it. Fields mirror
    the model constructor and serialize flat.

    Args:
        vocab_size (`int`, *optional*, defaults to 30527):
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
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            Number of buckets the relative position bias is gathered from. The bias is
            computed once and shared by every attention layer.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation used in the feed-forward blocks.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            LayerNorm epsilon.
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id (positions are offset past it).

    Examples:

    ```python
    >>> from zeromodels.models.mpnet import MPNetConfig, MPNetModel

    >>> configuration = MPNetConfig()
    >>> model = MPNetModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "mpnet"

    vocab_size: int = 30527
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    max_position_embeddings: int = 512
    relative_attention_num_buckets: int = 32
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 1
