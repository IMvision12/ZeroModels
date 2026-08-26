from zeromodels.base import BaseConfig


class Qwen3Config(BaseConfig):
    r"""Configuration for Qwen3 (dense): [`Qwen3Model`] and [`Qwen3TextGenerate`].

    A Qwen3 decoder: grouped-query attention with per-head QK-norm and bias-free
    QKV projections, SwiGLU MLP, RMSNorm, and 1D rotary positions.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 1024):
            Model / residual-stream width.
        mlp_dim (`int`, *optional*, defaults to 3072):
            SwiGLU hidden width per layer.
        num_layers (`int`, *optional*, defaults to 28):
            Number of decoder blocks.
        num_heads (`int`, *optional*, defaults to 16):
            Query heads per layer.
        num_kv_heads (`int`, *optional*, defaults to 8):
            Key/value heads per layer (GQA).
        head_dim (`int`, *optional*, defaults to 128):
            Per-head dim.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            RMSNorm epsilon (shared by the per-head QK-norms).
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            Rotary base frequency.
        tie_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the LM head is tied to the token embedding."""

    model_type = "qwen3"

    vocab_size: int = 151936
    embed_dim: int = 1024
    mlp_dim: int = 3072
    num_layers: int = 28
    num_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_embeddings: bool = True
