from zeromodels.base import BaseConfig


class GlmConfig(BaseConfig):
    """Configuration for GLM-4: [`GlmModel`] and [`GlmTextGenerate`].

    A flat text-decoder config: pre-norm grouped-query attention with partial
    *interleaved* rotary embeddings, biased q/k/v, and a fused-SwiGLU MLP.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Text / residual-stream width.
        num_layers: Number of decoder blocks.
        num_heads: Query heads per layer.
        num_kv_heads: Key/value heads per layer (GQA).
        head_dim: Per-head dim.
        mlp_dim: SwiGLU hidden width (``intermediate_size``).
        partial_rotary_factor: Fraction of each head that receives rotary.
        norm_eps: RMSNorm epsilon.
        rope_theta: Rotary base frequency.
        attention_bias: Whether q/k/v carry a bias.
        tie_embeddings: Whether the LM head ties to the token embedding.
    """

    model_type = "glm"

    vocab_size: int = 151552
    embed_dim: int = 4096
    num_layers: int = 40
    num_heads: int = 32
    num_kv_heads: int = 2
    head_dim: int = 128
    mlp_dim: int = 13696
    partial_rotary_factor: float = 0.5
    norm_eps: float = 0.00000015625
    rope_theta: float = 10000.0
    attention_bias: bool = True
    tie_embeddings: bool = False
