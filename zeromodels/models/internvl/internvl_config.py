from zeromodels.base import BaseConfig


class InternVLTextConfig(BaseConfig):
    """Text-decoder config for InternVL (the ``text_config`` sub-config).

    Backbone-parametric: ``text_backbone`` selects the decoder recipe -- ``qwen2``
    (biased QKV, no QK norm, dense SwiGLU), ``qwen3`` (bias-free QKV + per-head QK
    norm, dense SwiGLU), or ``qwen3_moe`` (qwen3 attention + a sparse Qwen3-MoE
    feed-forward). The MoE fields are read only for ``qwen3_moe``.

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Text / residual-stream width.
        mlp_dim: Dense SwiGLU hidden width.
        num_layers: Number of decoder blocks.
        num_heads: Query heads per layer.
        num_kv_heads: Key/value heads per layer (GQA).
        head_dim: Per-head dim (defaults to ``embed_dim // num_heads``).
        text_backbone: ``qwen2`` / ``qwen3`` / ``qwen3_moe``.
        norm_eps: RMSNorm epsilon (shared by the QK norms).
        rope_theta: Rotary base frequency.
        tie_embeddings: Whether the LM head ties to the token embedding.
        num_experts: Routed experts (``qwen3_moe`` only; 0 = dense).
        num_experts_per_tok: Experts selected per token.
        moe_mlp_dim: Per-expert hidden width (``moe_intermediate_size``).
        norm_topk_prob: Renormalize the selected router weights to sum to one.
        decoder_sparse_step: Every ``step``-th layer is sparse.
        mlp_only_layers: Layer indices pinned dense.
    """

    model_type = "internvl_text"

    vocab_size: int = 151674
    embed_dim: int = 896
    mlp_dim: int = 4864
    num_layers: int = 24
    num_heads: int = 14
    num_kv_heads: int = 2
    head_dim: int | None = None
    text_backbone: str = "qwen2"
    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_embeddings: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_mlp_dim: int = 0
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple = ()


class InternVLVisionConfig(BaseConfig):
    """InternViT vision-tower config for InternVL (the ``vision_config`` sub-config).

    The 1B-14B checkpoints use the 300M ViT (``layer_norm``, biased attention, no
    QK norm); the 38B/78B use the 6B ViT (``rms_norm``, bias-free, full-width QK
    RMS-norm).

    Args:
        embed_dim: Vision hidden width.
        mlp_dim: Vision MLP width.
        num_layers: Vision encoder blocks.
        num_heads: Vision attention heads.
        image_size: Tile side length in pixels.
        patch_size: Patch side length in pixels.
        attention_bias: Whether vision q/k/v carry a bias (300M: True).
        qk_norm: Whether vision attention full-width RMS-norms q/k (6B: True).
        norm_type: ``layer_norm`` (300M) or ``rms_norm`` (6B).
        norm_eps: Vision norm epsilon.
        layer_scale_init: Initial vision layer-scale value.
    """

    model_type = "internvl_vision"

    embed_dim: int = 1024
    mlp_dim: int = 4096
    num_layers: int = 24
    num_heads: int = 16
    image_size: int = 448
    patch_size: int = 14
    attention_bias: bool = True
    qk_norm: bool = False
    norm_type: str = "layer_norm"
    norm_eps: float = 1e-6
    layer_scale_init: float = 0.1


class InternVLConfig(BaseConfig):
    """Configuration for InternVL: [`InternVLModel`] and [`InternVLConditionalGenerate`].

    A composite config: the pluggable text decoder lives in an
    [`InternVLTextConfig`] (``text_config``) and the InternViT tower in an
    [`InternVLVisionConfig`] (``vision_config``); ``downsample_ratio`` /
    ``image_token_id`` are the top-level image glue. The flat model constructor is
    fed by flattening the sub-configs (vision fields gain the ``vision_`` prefix,
    except ``image_size`` / ``patch_size``). One class loads InternVL 2.5 / 3 / 3.5
    (dense + MoE); the text tower is selected by ``text_config.text_backbone``.

    Args:
        text_config: Text-decoder config (defaults to an ``InternVLTextConfig``).
        vision_config: InternViT config (defaults to an ``InternVLVisionConfig``).
        downsample_ratio: Pixel-shuffle scale factor.
        image_token_id: ``<IMG_CONTEXT>`` placeholder id.
    """

    model_type = "internvl"

    sub_configs = {
        "text_config": InternVLTextConfig,
        "vision_config": InternVLVisionConfig,
    }
    sub_config_prefixes = {"text_config": "", "vision_config": "vision_"}
    group_extras = {"vision_config": ("image_size", "patch_size")}

    text_config: InternVLTextConfig | dict | None = None
    vision_config: InternVLVisionConfig | dict | None = None
    downsample_ratio: float = 0.5
    image_token_id: int = 151667
