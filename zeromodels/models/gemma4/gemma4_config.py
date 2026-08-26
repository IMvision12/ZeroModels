from zeromodels.base import BaseConfig


class Gemma4TextConfig(BaseConfig):
    r"""Text-decoder config for Gemma 4 (the ``text_config`` sub-config).

    Gemma 4's decoder geometry: sliding layers (``head_dim`` 256, full default rope
    at ``rope_local_theta``) alternate with global layers (``global_head_dim`` 512,
    ``num_global_kv_heads``, optional ``k_eq_v``, proportional partial rope at
    ``rope_theta``) on the schedule in ``layer_types`` (5:1 on E2B/E4B, 6-pattern
    elsewhere). GeGLU feed-forwards, an optional parallel MoE branch (26B-A4B), a
    learned per-layer scalar, and the E-variant extras: Per-Layer Embeddings
    (``hidden_size_per_layer_input``), tail KV-sharing (``num_kv_shared_layers``),
    and a double-wide MLP on the shared layers.

    Args:
        vocab_size (`int`, *optional*, defaults to 262144):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 3840):
            Text / residual-stream width.
        mlp_dim (`int`, *optional*, defaults to 15360):
            Dense GeGLU hidden width per layer.
        num_layers (`int`, *optional*, defaults to 48):
            Number of decoder blocks.
        num_heads (`int`, *optional*, defaults to 16):
            Query heads per layer.
        num_kv_heads (`int`, *optional*, defaults to 8):
            K/V heads on sliding layers.
        num_global_kv_heads (`int`, *optional*, defaults to 1):
            K/V heads on global layers.
        head_dim (`int`, *optional*, defaults to 256):
            Sliding-layer per-head dim.
        global_head_dim (`int`, *optional*, defaults to 512):
            Global-layer per-head dim.
        k_eq_v (`bool`, *optional*, defaults to `True`):
            Global layers reuse the key projection as the value.
        enable_moe (`bool`, *optional*, defaults to `False`):
            Whether layers carry the parallel expert branch.
        num_experts / num_experts_per_tok / moe_mlp_dim (`int`, *optional*):
            MoE parameters (26B-A4B).
        sliding_window (`int`, *optional*, defaults to 1024):
            Window of the sliding layers.
        sliding_window_pattern (`int`, *optional*, defaults to 6):
            Every ``pattern``-th layer is global when ``layer_types`` is unset.
        layer_types (`list`, *optional*):
            Explicit per-layer ``"sliding_attention"`` / ``"full_attention"``
            schedule (overrides ``sliding_window_pattern``).
        partial_rotary_factor (`float`, *optional*, defaults to 0.25):
            Fraction of the global head that is rotated.
        final_logit_softcapping (`float`, *optional*, defaults to 30.0):
            LM-head tanh softcap.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            RMSNorm epsilon.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            Global-layer rotary base.
        rope_local_theta (`float`, *optional*, defaults to 10000.0):
            Sliding-layer rotary base.
        tie_embeddings (`bool`, *optional*, defaults to `True`):
            Whether [`Gemma4ConditionalGenerate`] ties the LM head to the token embedding.
        hidden_size_per_layer_input (`int`, *optional*, defaults to 0):
            Per-Layer Embedding width (0 disables PLE; 256 on E2B/E4B).
        vocab_size_per_layer_input (`int`, *optional*, defaults to 262144):
            PLE auxiliary-embedding vocabulary size.
        num_kv_shared_layers (`int`, *optional*, defaults to 0):
            Tail layers that reuse an earlier layer's K/V (E-variants).
        use_double_wide_mlp (`bool`, *optional*, defaults to `False`):
            Double-width MLP on the KV-shared layers."""

    model_type = "gemma4_text"

    vocab_size: int = 262144
    embed_dim: int = 3840
    mlp_dim: int = 15360
    num_layers: int = 48
    num_heads: int = 16
    num_kv_heads: int = 8
    num_global_kv_heads: int = 1
    head_dim: int = 256
    global_head_dim: int = 512
    k_eq_v: bool = True
    enable_moe: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_mlp_dim: int = 0
    sliding_window: int = 1024
    sliding_window_pattern: int = 6
    layer_types: list | None = None
    partial_rotary_factor: float = 0.25
    final_logit_softcapping: float | None = 30.0
    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_local_theta: float = 10000.0
    tie_embeddings: bool = True
    hidden_size_per_layer_input: int = 0
    vocab_size_per_layer_input: int = 262144
    num_kv_shared_layers: int = 0
    use_double_wide_mlp: bool = False


class Gemma4VisionConfig(BaseConfig):
    r"""NaViT vision-tower config for Gemma 4 (the ``vision_config`` sub-config).

    Args:
        hidden_size (`int`, *optional*, defaults to 1152):
            Vision tower width.
        num_layers (`int`, *optional*, defaults to 27):
            Vision encoder blocks.
        num_heads (`int`, *optional*, defaults to 16):
            Vision attention heads.
        num_kv_heads (`int`, *optional*, defaults to 16):
            Vision K/V heads.
        head_dim (`int`, *optional*, defaults to 72):
            Vision per-head dim.
        intermediate_size (`int`, *optional*, defaults to 4304):
            Vision MLP width.
        patch_size (`int`, *optional*, defaults to 16):
            Teacher patch side in pixels.
        position_embedding_size (`int`, *optional*, defaults to 10240):
            Learned position-embedding table length.
        pooling_kernel_size (`int`, *optional*, defaults to 3):
            Spatial pooling kernel after the encoder.
        rope_theta (`float`, *optional*, defaults to 100.0):
            Vision rotary base.
        eps (`float`, *optional*, defaults to 1e-6):
            Vision RMSNorm epsilon.
        standardize (`bool`, *optional*, defaults to `True`):
            Per-image standardization.
        use_clipped_linears (`bool`, *optional*, defaults to `False`):
            Clipped-linear projections."""

    model_type = "gemma4_vision"

    hidden_size: int = 1152
    num_layers: int = 27
    num_heads: int = 16
    num_kv_heads: int = 16
    head_dim: int = 72
    intermediate_size: int = 4304
    patch_size: int = 16
    position_embedding_size: int = 10240
    pooling_kernel_size: int = 3
    rope_theta: float = 100.0
    eps: float = 1e-6
    standardize: bool = True
    use_clipped_linears: bool = False


class Gemma4AudioConfig(BaseConfig):
    r"""USM audio-tower config for Gemma 4 (the ``audio_config`` sub-config).

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Audio conformer width.
        num_layers (`int`, *optional*, defaults to 12):
            Conformer blocks.
        num_heads (`int`, *optional*, defaults to 8):
            Conformer attention heads.
        conv_channels (`tuple`, *optional*, defaults to `(128, 32)`):
            Sub-sampling conv channels.
        conv_kernel_size (`int`, *optional*, defaults to 5):
            Conformer conv kernel size.
        chunk_size (`int`, *optional*, defaults to 12):
            Local attention chunk size.
        context_left / context_right (`int`, *optional*, defaults to 13 / 0):
            Local attention context window.
        logit_cap (`float`, *optional*, defaults to 50.0):
            Attention logit tanh soft-cap.
        invalid_logits (`float`, *optional*, defaults to -1e9):
            Masked-position fill value.
        residual_weight (`float`, *optional*, defaults to 0.5):
            Conformer residual scaling.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            Audio RMSNorm epsilon.
        output_proj_dims (`int`, *optional*, defaults to 1536):
            Output projection width.
        use_clipped_linears (`bool`, *optional*, defaults to `True`):
            Clipped-linear projections."""

    model_type = "gemma4_audio"

    hidden_size: int = 1024
    num_layers: int = 12
    num_heads: int = 8
    conv_channels: tuple = (128, 32)
    conv_kernel_size: int = 5
    chunk_size: int = 12
    context_left: int = 13
    context_right: int = 0
    logit_cap: float = 50.0
    invalid_logits: float = -1e9
    residual_weight: float = 0.5
    norm_eps: float = 1e-6
    output_proj_dims: int = 1536
    use_clipped_linears: bool = True


class Gemma4Config(BaseConfig):
    r"""Configuration for Gemma 4: [`Gemma4Model`], [`Gemma4MultimodalModel`], and
    [`Gemma4ConditionalGenerate`].

    A composite config: the text decoder lives in a [`Gemma4TextConfig`]
    (``text_config``), the optional NaViT tower in a [`Gemma4VisionConfig`]
    (``vision_config``), and the optional USM tower in a [`Gemma4AudioConfig`]
    (``audio_config``); the ``*_token_id`` glue is top-level. An absent optional
    tower is ``None`` (dropped on serialize by BaseConfig). Unlike gemma3's flat
    constructor, [`Gemma4MultimodalModel`] takes the sub-configs **nested**, so
    ``constructor_kwargs`` emits the nested form and passes ``None`` for an absent
    tower (which is how the model skips building it).

    Args:
        text_config (`Gemma4TextConfig | dict`, *optional*):
            Text-decoder config (defaults to a `Gemma4TextConfig`).
        vision_config (`Gemma4VisionConfig | dict`, *optional*):
            NaViT tower config, or `None` for no vision tower.
        audio_config (`Gemma4AudioConfig | dict`, *optional*):
            USM tower config, or `None` for no audio tower.
        image_token_id (`int`, *optional*, defaults to 258880):
            Image soft-token placeholder id.
        video_token_id (`int`, *optional*, defaults to 258884):
            Video soft-token placeholder id.
        audio_token_id (`int`, *optional*, defaults to 258881):
            Audio soft-token placeholder id.
        pad_token_id (`int`, *optional*, defaults to 0):
            Pad id used to embed multimodal slots before scatter.
        use_bidirectional_vision (`bool`, *optional*, defaults to `True`):
            Blockwise bidirectional vision masking on sliding layers.

    Examples:

    ```python
    >>> from zeromodels.models.gemma4 import Gemma4Config, Gemma4ConditionalGenerate

    >>> configuration = Gemma4Config(
    ...     text_config={"embed_dim": 2560, "num_layers": 34},
    ...     vision_config={"num_layers": 27},
    ... )
    >>> model = Gemma4ConditionalGenerate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "gemma4"

    sub_configs = {
        "text_config": Gemma4TextConfig,
        "vision_config": Gemma4VisionConfig,
        "audio_config": Gemma4AudioConfig,
    }
    optional_sub_configs = ("vision_config", "audio_config")

    text_config: Gemma4TextConfig | dict | None = None
    vision_config: Gemma4VisionConfig | dict | None = None
    audio_config: Gemma4AudioConfig | dict | None = None
    image_token_id: int = 258880
    video_token_id: int = 258884
    audio_token_id: int = 258881
    pad_token_id: int = 0
    use_bidirectional_vision: bool = True

    def constructor_kwargs(self):
        kw = {
            name: getattr(self, name)
            for name in self.field_names()
            if name not in self.sub_configs
        }
        for key in self.sub_configs:
            obj = getattr(self, key)
            kw[key] = obj.constructor_kwargs() if obj is not None else None
        return kw
