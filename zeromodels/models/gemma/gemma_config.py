from zeromodels.base import BaseConfig


class GemmaConfig(BaseConfig):
    r"""Configuration for Gemma: [`GemmaModel`] and [`GemmaTextGenerate`].

    Gemma is Google's decoder-only transformer: `(1 + w)` RMSNorm, GeGLU
    (tanh-approximate gelu) MLPs, scaled token embeddings, a `head_dim` (256)
    decoupled from `embed_dim // num_heads` (the 2B is multi-query, one K/V head),
    rotary positions, and a tied LM head. One `zm_config.json` sits on each
    variant's repo, and fields mirror the model constructor and serialize flat.

    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 2048):
            Model (hidden) width.
        mlp_dim (`int`, *optional*, defaults to 16384):
            GeGLU hidden width per layer.
        num_layers (`int`, *optional*, defaults to 18):
            Number of decoder blocks.
        num_heads (`int`, *optional*, defaults to 8):
            Number of query attention heads.
        num_kv_heads (`int`, *optional*, defaults to 1):
            Number of key/value heads (1 = multi-query, as in the 2B).
        head_dim (`int`, *optional*, defaults to 256):
            Per-head dimension (decoupled from `embed_dim // num_heads`).
        norm_eps (`float`, *optional*, defaults to 1e-6):
            RMSNorm epsilon.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Rotary base frequency.
        tie_embeddings (`bool`, *optional*, defaults to `True`):
            Whether [`GemmaTextGenerate`] ties the LM head to the token embedding.

    Examples:

    ```python
    >>> from zeromodels.models.gemma import GemmaConfig, GemmaTextGenerate

    >>> configuration = GemmaConfig(embed_dim=3072, num_layers=28, num_heads=16)
    >>> model = GemmaTextGenerate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "gemma"

    vocab_size: int = 256000
    embed_dim: int = 2048
    mlp_dim: int = 16384
    num_layers: int = 18
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
