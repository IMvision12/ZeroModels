from zeromodels.base import BaseConfig


class Qwen3MoeConfig(BaseConfig):
    r"""Configuration for Qwen3-MoE: [`Qwen3MoeModel`] and [`Qwen3MoeTextGenerate`].

    A Qwen3 decoder whose MLP is a sparse Mixture-of-Experts: a softmax router picks
    ``num_experts_per_tok`` of ``num_experts`` fused SwiGLU experts (a dense MLP is
    used instead on any layer in ``mlp_only_layers``, or off the
    ``decoder_sparse_step`` cadence). Attention is GQA with per-head QK-norm.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 2048):
            Model / residual-stream width.
        num_layers (`int`, *optional*, defaults to 48):
            Number of decoder blocks.
        num_heads (`int`, *optional*, defaults to 32):
            Query heads per layer.
        num_kv_heads (`int`, *optional*, defaults to 4):
            Key/value heads per layer (GQA).
        head_dim (`int`, *optional*, defaults to 128):
            Per-head dim.
        mlp_dim (`int`, *optional*, defaults to 6144):
            Dense-MLP hidden width (used on non-MoE layers).
        num_experts (`int`, *optional*, defaults to 128):
            Number of routed experts.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Experts activated per token.
        moe_mlp_dim (`int`, *optional*, defaults to 768):
            Hidden width of each expert.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Renormalize the top-k router probabilities.
        decoder_sparse_step (`int`, *optional*, defaults to 1):
            Use MoE every Nth layer (1 = every layer).
        mlp_only_layers (`tuple`, *optional*, defaults to `()`):
            Layer indices that use a dense MLP instead of MoE.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            Rotary base frequency.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            RMSNorm epsilon.
        tie_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the LM head is tied to the token embedding."""

    model_type = "qwen3_moe"

    vocab_size: int = 151936
    embed_dim: int = 2048
    num_layers: int = 48
    num_heads: int = 32
    num_kv_heads: int = 4
    head_dim: int = 128
    mlp_dim: int = 6144
    num_experts: int = 128
    num_experts_per_tok: int = 8
    moe_mlp_dim: int = 768
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple = ()
    rope_theta: float = 1000000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = False
