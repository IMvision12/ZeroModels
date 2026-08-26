from zeromodels.base import BaseConfig


class Glm4MoeConfig(BaseConfig):
    """Configuration for GLM-4.5/4.6/4.7 (MoE): [`Glm4MoeModel`] and [`Glm4MoeTextGenerate`].

    A flat MoE decoder: partial *NeoX* rotary, optional per-head QK norm, biased q/k/v,
    a DeepSeek-V3-style grouped-topk sigmoid router (with a learned
    ``e_score_correction_bias``) over a fused-einsum expert bank plus a shared expert;
    the first ``first_k_dense`` layers are plain dense SwiGLU.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Text / residual-stream width.
        num_layers: Number of decoder blocks.
        num_heads: Query heads per layer.
        num_kv_heads: Key/value heads per layer (GQA).
        head_dim: Per-head dim.
        mlp_dim: Dense-layer SwiGLU width (``intermediate_size``).
        moe_mlp_dim: Per-expert width (``moe_intermediate_size``).
        num_experts: Routed experts.
        num_experts_per_tok: Experts selected per token.
        n_shared_experts: Shared (always-on) experts.
        n_group / topk_group: Group-limited routing (1/1 = plain top-k).
        norm_topk_prob: Renormalize the selected router weights.
        routed_scaling_factor: Scale applied to the routed-expert output.
        first_k_dense: Leading dense (non-MoE) layers.
        partial_rotary_factor: Fraction of each head that receives rotary.
        use_qk_norm: Whether attention RMS-normalizes per-head q/k (4.6/4.7: True).
        norm_eps: RMSNorm epsilon.
        rope_theta: Rotary base frequency.
        attention_bias: Whether q/k/v carry a bias.
        tie_embeddings: Whether the LM head ties to the token embedding.
    """

    model_type = "glm4_moe"

    vocab_size: int = 151552
    embed_dim: int = 4096
    num_layers: int = 46
    num_heads: int = 96
    num_kv_heads: int = 8
    head_dim: int = 128
    mlp_dim: int = 10944
    moe_mlp_dim: int = 1408
    num_experts: int = 128
    num_experts_per_tok: int = 8
    n_shared_experts: int = 1
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    first_k_dense: int = 1
    partial_rotary_factor: float = 0.5
    use_qk_norm: bool = False
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    attention_bias: bool = False
    tie_embeddings: bool = False
