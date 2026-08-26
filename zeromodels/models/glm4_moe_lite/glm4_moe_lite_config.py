from zeromodels.base import BaseConfig


class Glm4MoeLiteConfig(BaseConfig):
    """Configuration for GLM-4.7-Flash: [`Glm4MoeLiteModel`] and [`Glm4MoeLiteTextGenerate`].

    ``glm4_moe_lite`` is the DeepSeek-V3 decoder (Multi-head Latent Attention +
    aux-loss-free DeepSeekMoE) with GLM weight names -- i.e. GLM-5 (`glm5_moe`)
    without the DSA sparse-attention indexer, which is definitionally
    ``deepseek_v3``. The first ``first_k_dense`` layers are dense; the rest route
    over a fused expert bank with a shared expert. Rope is interleaved DeepSeek
    rope over the ``qk_rope_head_dim`` slice.

    Args:
        vocab_size / embed_dim / num_layers / num_heads: Model geometry.
        mlp_dim: Dense-layer SwiGLU width (``intermediate_size``).
        moe_mlp_dim: Per-expert width (``moe_intermediate_size``).
        num_experts / num_experts_per_tok / n_shared_experts: MoE shape.
        n_group / topk_group / norm_topk_prob / routed_scaling_factor: Routing.
        first_k_dense: Leading dense layers.
        q_lora_rank: Query bottleneck.
        kv_lora_rank / qk_nope_head_dim / qk_rope_head_dim / v_head_dim: MLA.
        rope_theta: Base frequency.
        rope_scaling: The HF ``rope_scaling`` dict (yarn) or None.
        norm_eps: RMSNorm epsilon (block / model norms; MLA bottleneck norms are 1e-6).
        max_position_embeddings: Used by the yarn attention-factor default.
        tie_embeddings: Whether the LM head ties to the token embedding.
    """

    model_type = "glm4_moe_lite"

    vocab_size: int = 154880
    embed_dim: int = 2048
    num_layers: int = 47
    num_heads: int = 20
    mlp_dim: int = 10240
    moe_mlp_dim: int = 1536
    num_experts: int = 64
    num_experts_per_tok: int = 4
    n_shared_experts: int = 1
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.8
    first_k_dense: int = 1
    q_lora_rank: int | None = 768
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    rope_theta: float = 1000000.0
    rope_scaling: dict | None = None
    norm_eps: float = 1e-5
    max_position_embeddings: int = 202752
    tie_embeddings: bool = False
