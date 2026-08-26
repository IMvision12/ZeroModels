from zeromodels.base import BaseConfig


class Qwen2MoeConfig(BaseConfig):
    r"""Configuration for Qwen2-MoE: [`Qwen2MoeModel`] and [`Qwen2MoeTextGenerate`].

    Qwen2-MoE keeps the Qwen2 attention (GQA with q/k/v bias, RMSNorm, rotary
    positions) and replaces the dense MLP on the sparse layers with a softmax
    router over `num_experts` fused-einsum experts (top-`num_experts_per_tok`)
    plus an always-on shared expert. `Qwen1.5-MoE-A2.7B` and `Qwen2-57B-A14B` use
    this class. One `zm_config.json` sits on each variant's repo, and fields
    mirror the model constructor and serialize flat.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 2048):
            Model (hidden) width.
        num_layers (`int`, *optional*, defaults to 24):
            Number of decoder blocks.
        num_heads (`int`, *optional*, defaults to 16):
            Number of query attention heads.
        num_kv_heads (`int`, *optional*, defaults to 16):
            Number of key/value heads (grouped-query attention).
        head_dim (`int`, *optional*):
            Per-head dimension; defaults to `embed_dim // num_heads`.
        mlp_dim (`int`, *optional*, defaults to 5632):
            SwiGLU hidden width on the dense (non-MoE) layers.
        num_experts (`int`, *optional*, defaults to 60):
            Number of routed experts per MoE layer.
        num_experts_per_tok (`int`, *optional*, defaults to 4):
            Experts activated per token (top-k).
        moe_mlp_dim (`int`, *optional*, defaults to 1408):
            SwiGLU hidden width of each routed expert.
        shared_mlp_dim (`int`, *optional*, defaults to 5632):
            SwiGLU hidden width of the always-on shared expert.
        norm_topk_prob (`bool`, *optional*, defaults to `False`):
            Whether to renormalize the top-k routing weights to sum to 1.
        decoder_sparse_step (`int`, *optional*, defaults to 1):
            Every `step`-th layer is a MoE layer.
        mlp_only_layers (`tuple`, *optional*, defaults to `()`):
            Layer indices forced to a dense MLP instead of MoE.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            Rotary base frequency.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            RMSNorm epsilon.
        tie_embeddings (`bool`, *optional*, defaults to `False`):
            Whether [`Qwen2MoeTextGenerate`] ties the LM head to the token embedding.

    Examples:

    ```python
    >>> from zeromodels.models.qwen2_moe import Qwen2MoeConfig, Qwen2MoeTextGenerate

    >>> configuration = Qwen2MoeConfig(embed_dim=3584, num_layers=28, num_experts=64)
    >>> model = Qwen2MoeTextGenerate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "qwen2_moe"

    vocab_size: int = 151936
    embed_dim: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 16
    head_dim: int | None = None
    mlp_dim: int = 5632
    num_experts: int = 60
    num_experts_per_tok: int = 4
    moe_mlp_dim: int = 1408
    shared_mlp_dim: int = 5632
    norm_topk_prob: bool = False
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple = ()
    rope_theta: float = 1000000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = False
