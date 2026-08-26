from zeromodels.base import BaseConfig


class ModernBertConfig(BaseConfig):
    r"""Configuration for the ModernBERT encoder ([`ModernBertModel`]) and its task heads.

    ModernBERT is a modernized bidirectional transformer encoder: rotary position
    embeddings, attention that alternates between a global (full) layer and local
    sliding-window layers, GeGLU feed-forwards, and pre-LayerNorm residuals, with
    no absolute-position or token-type embeddings. One `zm_config.json` (declaring
    the canonical [`ModernBertModel`]) sits on each variant's repo; the encoder,
    masked-LM, and task-head classes all load from it. Fields mirror the model
    constructor and serialize flat.

    Args:
        vocab_size (`int`, *optional*, defaults to 50368):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 768):
            Hidden size.
        num_layers (`int`, *optional*, defaults to 22):
            Number of transformer encoder layers.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads.
        mlp_dim (`int`, *optional*, defaults to 1152):
            GeGLU feed-forward hidden dimension.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            Maximum sequence length supported by the model.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation used in the GeGLU feed-forward blocks.
        norm_eps (`float`, *optional*, defaults to 1e-5):
            LayerNorm epsilon.
        local_attention (`int`, *optional*, defaults to 128):
            Total sliding-window size of the local-attention layers.
        global_attn_every_n_layers (`int`, *optional*, defaults to 3):
            Period of the global (full-attention) layers.
        global_rope_theta (`float`, *optional*, defaults to 160000.0):
            RoPE base for the global-attention layers.
        local_rope_theta (`float`, *optional*, defaults to 10000.0):
            RoPE base for the local-attention layers.
        pad_token_id (`int`, *optional*, defaults to 50283):
            Padding token id.

    Examples:

    ```python
    >>> from zeromodels.models.modernbert import ModernBertConfig, ModernBertModel

    >>> configuration = ModernBertConfig()
    >>> model = ModernBertModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "modernbert"

    vocab_size: int = 50368
    embed_dim: int = 768
    num_layers: int = 22
    num_heads: int = 12
    mlp_dim: int = 1152
    max_position_embeddings: int = 8192
    hidden_act: str = "gelu"
    norm_eps: float = 1e-5
    local_attention: int = 128
    global_attn_every_n_layers: int = 3
    global_rope_theta: float = 160000.0
    local_rope_theta: float = 10000.0
    pad_token_id: int = 50283
