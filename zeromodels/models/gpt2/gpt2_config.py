from zeromodels.base import BaseConfig


class GPT2Config(BaseConfig):
    r"""Configuration for the GPT-2 backbone ([`GPT2Model`]) and its generative
    head ([`GPT2TextGenerate`]).

    GPT-2 is a decoder-only transformer with learned token and absolute-position
    embeddings, pre-LayerNorm blocks, and a final LayerNorm. One `kf_config.json`
    (declaring the canonical [`GPT2Model`]) sits on each variant's repo; both the
    backbone and the generative head load from it. Fields mirror the model
    constructor and serialize flat.

    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 768):
            Model / residual-stream width.
        mlp_dim (`int`, *optional*, defaults to 3072):
            Feed-forward hidden width per block.
        num_layers (`int`, *optional*, defaults to 12):
            Number of decoder blocks.
        num_heads (`int`, *optional*, defaults to 12):
            Attention heads per block.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            Size of the learned position table.
        norm_eps (`float`, *optional*, defaults to 1e-5):
            LayerNorm epsilon.
        tie_embeddings (`bool`, *optional*, defaults to `True`):
            Whether [`GPT2TextGenerate`] ties the LM head to the token embeddings.

    Examples:

    ```python
    >>> from zeromodels.models.gpt2 import GPT2Config, GPT2Model

    >>> configuration = GPT2Config()
    >>> model = GPT2Model(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "gpt2"

    vocab_size: int = 50257
    embed_dim: int = 768
    mlp_dim: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    max_position_embeddings: int = 1024
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
