from zeromodels.base import BaseConfig


class Qwen2Config(BaseConfig):
    r"""Configuration for Qwen2: [`Qwen2Model`] and [`Qwen2TextGenerate`].

    Qwen2 is Alibaba's decoder-only transformer: grouped-query attention with a
    bias on the q/k/v projections, SwiGLU MLPs, RMSNorm, and 1D rotary positions.
    One `kf_config.json` sits on each variant's repo, and fields mirror the model
    constructor and serialize flat.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 896):
            Model (hidden) width.
        mlp_dim (`int`, *optional*, defaults to 4864):
            SwiGLU hidden width per layer.
        num_layers (`int`, *optional*, defaults to 24):
            Number of decoder blocks.
        num_heads (`int`, *optional*, defaults to 14):
            Number of query attention heads.
        num_kv_heads (`int`, *optional*, defaults to 2):
            Number of key/value heads (grouped-query attention).
        head_dim (`int`, *optional*):
            Per-head dimension; defaults to `embed_dim // num_heads`.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            RMSNorm epsilon.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            Rotary base frequency.
        tie_embeddings (`bool`, *optional*, defaults to `True`):
            Whether [`Qwen2TextGenerate`] ties the LM head to the token embedding.

    Examples:

    ```python
    >>> from zeromodels.models.qwen2 import Qwen2Config, Qwen2TextGenerate

    >>> configuration = Qwen2Config(embed_dim=3584, num_layers=28, num_kv_heads=4)
    >>> model = Qwen2TextGenerate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "qwen2"

    vocab_size: int = 151936
    embed_dim: int = 896
    mlp_dim: int = 4864
    num_layers: int = 24
    num_heads: int = 14
    num_kv_heads: int = 2
    head_dim: int | None = None
    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_embeddings: bool = True
