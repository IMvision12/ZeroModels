from zeromodels.base import BaseConfig


class GptOssConfig(BaseConfig):
    r"""Configuration for GPT-OSS: [`GptOssModel`] and [`GptOssTextGenerate`].

    GPT-OSS is OpenAI's open-weight mixture-of-experts decoder: grouped-query
    attention with learned per-head attention sinks, alternating sliding-window /
    full causal attention, YaRN-scaled rotary positions, and a top-k-routed MoE
    feed-forward whose experts ship in MXFP4 (4-bit). One `zm_config.json` sits on
    each variant's repo, and fields mirror the model constructor and serialize flat.

    Args:
        vocab_size (`int`, *optional*, defaults to 201088):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 2880):
            Model (hidden) width.
        mlp_dim (`int`, *optional*, defaults to 2880):
            Per-expert hidden width.
        num_layers (`int`, *optional*, defaults to 24):
            Number of decoder blocks.
        num_heads (`int`, *optional*, defaults to 64):
            Number of query attention heads.
        num_kv_heads (`int`, *optional*, defaults to 8):
            Number of key/value heads (grouped-query attention).
        head_dim (`int`, *optional*, defaults to 64):
            Per-head dimension.
        num_experts (`int`, *optional*, defaults to 32):
            Number of MoE experts.
        num_experts_per_tok (`int`, *optional*, defaults to 4):
            Experts routed per token (top-k).
        sliding_window (`int`, *optional*, defaults to 128):
            Window size of the sliding-attention (even) layers.
        norm_eps (`float`, *optional*, defaults to 1e-5):
            RMSNorm epsilon.
        rope_theta (`float`, *optional*, defaults to 150000.0):
            Rotary base frequency.
        rope_factor (`float`, *optional*, defaults to 32.0):
            YaRN scaling factor.
        rope_beta_fast (`float`, *optional*, defaults to 32.0):
            YaRN beta_fast.
        rope_beta_slow (`float`, *optional*, defaults to 1.0):
            YaRN beta_slow.
        rope_truncate (`bool`, *optional*, defaults to `False`):
            YaRN correction-range truncation.
        rope_original_max_pos (`int`, *optional*, defaults to 4096):
            YaRN original context length.
        attention_bias (`bool`, *optional*, defaults to `True`):
            Whether q/k/v/o projections carry a bias.
        tie_embeddings (`bool`, *optional*, defaults to `False`):
            Whether [`GptOssTextGenerate`] ties the LM head to the embeddings.
    MXFP4-packed checkpoints (the official 20b / 120b) carry a
    ``quantization_config`` block (``{"quant_method": "mxfp4"}``); the model itself is
    quantization-agnostic, and a ``ZmQuantizer`` swaps in the packed expert bank at
    load time.

    Examples:

    ```python
    >>> from zeromodels.models.gpt_oss import GptOssConfig, GptOssTextGenerate

    >>> configuration = GptOssConfig(num_layers=24, num_experts=32)
    >>> model = GptOssTextGenerate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "gpt_oss"

    vocab_size: int = 201088
    embed_dim: int = 2880
    mlp_dim: int = 2880
    num_layers: int = 24
    num_heads: int = 64
    num_kv_heads: int = 8
    head_dim: int = 64
    num_experts: int = 32
    num_experts_per_tok: int = 4
    sliding_window: int = 128
    norm_eps: float = 1e-5
    rope_theta: float = 150000.0
    rope_factor: float = 32.0
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_truncate: bool = False
    rope_original_max_pos: int = 4096
    attention_bias: bool = True
    tie_embeddings: bool = False
