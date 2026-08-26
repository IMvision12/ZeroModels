from zeromodels.base import BaseConfig


class Gemma2Config(BaseConfig):
    r"""Configuration for Gemma 2: [`Gemma2Model`] and [`Gemma2TextGenerate`].

    Gemma 2 is Google's decoder-only transformer, extending Gemma with alternating
    sliding-window / full causal attention, attention- and final-logit tanh
    soft-capping, a `query_pre_attn_scalar` query scaling, grouped-query attention,
    and pre/post-norm blocks. One `kf_config.json` sits on each variant's repo, and
    fields mirror the model constructor and serialize flat.

    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 2304):
            Model (hidden) width.
        mlp_dim (`int`, *optional*, defaults to 9216):
            GeGLU hidden width per layer.
        num_layers (`int`, *optional*, defaults to 26):
            Number of decoder blocks.
        num_heads (`int`, *optional*, defaults to 8):
            Number of query attention heads.
        num_kv_heads (`int`, *optional*, defaults to 4):
            Number of key/value heads (grouped-query attention).
        head_dim (`int`, *optional*, defaults to 256):
            Per-head dimension.
        query_pre_attn_scalar (`float`, *optional*, defaults to 256.0):
            Query scaling applied before attention (the 27B uses 144.0).
        attn_logit_softcapping (`float`, *optional*, defaults to 50.0):
            Tanh soft-cap on the attention logits.
        final_logit_softcapping (`float`, *optional*, defaults to 30.0):
            Tanh soft-cap on the output logits.
        sliding_window (`int`, *optional*, defaults to 4096):
            Window size on the alternating sliding-attention layers.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            RMSNorm epsilon.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Rotary base frequency.
        tie_embeddings (`bool`, *optional*, defaults to `True`):
            Whether [`Gemma2TextGenerate`] ties the LM head to the token embedding.

    Examples:

    ```python
    >>> from zeromodels.models.gemma2 import Gemma2Config, Gemma2TextGenerate

    >>> configuration = Gemma2Config(embed_dim=3584, num_layers=42, num_kv_heads=8)
    >>> model = Gemma2TextGenerate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "gemma2"

    vocab_size: int = 256000
    embed_dim: int = 2304
    mlp_dim: int = 9216
    num_layers: int = 26
    num_heads: int = 8
    num_kv_heads: int = 4
    head_dim: int = 256
    query_pre_attn_scalar: float = 256.0
    attn_logit_softcapping: float = 50.0
    final_logit_softcapping: float = 30.0
    sliding_window: int = 4096
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
