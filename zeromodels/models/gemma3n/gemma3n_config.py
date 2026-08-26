from zeromodels.base import BaseConfig


class Gemma3nTextConfig(BaseConfig):
    r"""Text-decoder config for Gemma 3n (the ``text_config`` sub-config).

    Gemma 3n's on-device decoder recipe layers several novel pieces on the Gemma
    shape: **AltUp** (Alternating Updates, ``altup_num_inputs`` parallel hidden
    streams with a learned predict / correct step), **LAuReL** (a low-rank Learned
    Augmented Residual, ``laurel_rank``), **Per-Layer Embeddings** (an auxiliary
    embedding table + projection fed into every block), **MatFormer** (a per-layer
    ``mlp_dim`` when a list is given), **activation sparsity** (a Gaussian top-k gate
    on the first layers, ``activation_sparsity_pattern``), tail **KV-sharing**
    (``num_kv_shared_layers``), and a 5:1 sliding/global schedule with dual rotary
    bases. Attention carries per-head q/k/v norms (the value norm is scaleless).

    Args:
        vocab_size (`int`, *optional*, defaults to 262400):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 2048):
            Text / residual-stream width.
        mlp_dim (`int | list`, *optional*, defaults to 16384):
            GeGLU inner width; a per-layer list selects MatFormer sizes.
        num_layers (`int`, *optional*, defaults to 35):
            Number of decoder blocks.
        num_heads (`int`, *optional*, defaults to 8):
            Query heads per layer.
        num_kv_heads (`int`, *optional*, defaults to 2):
            Key/value heads per layer.
        head_dim (`int`, *optional*, defaults to 256):
            Per-head dimension.
        sliding_window (`int`, *optional*, defaults to 512):
            Window of the sliding-attention layers.
        sliding_window_pattern (`int`, *optional*, defaults to 5):
            Every ``pattern``-th layer is global when ``layer_types`` is unset.
        layer_types (`list`, *optional*):
            Explicit per-layer ``"sliding_attention"`` / ``"full_attention"`` schedule.
        final_logit_softcapping (`float`, *optional*, defaults to 30.0):
            Tanh soft-cap on the output logits.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            RMSNorm epsilon.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            Global-layer rotary base.
        rope_local_theta (`float`, *optional*, defaults to 10000.0):
            Sliding-layer rotary base.
        hidden_activation (`str`, *optional*, defaults to ``"gelu_pytorch_tanh"``):
            GeGLU activation.
        tie_embeddings (`bool`, *optional*, defaults to `True`):
            Whether [`Gemma3nTextGenerate`] ties the LM head to the token embedding.
        vocab_size_per_layer_input (`int`, *optional*, defaults to 262144):
            Per-Layer-Embedding auxiliary vocabulary size.
        hidden_size_per_layer_input (`int`, *optional*, defaults to 256):
            Per-Layer-Embedding width.
        altup_num_inputs (`int`, *optional*, defaults to 4):
            Number of parallel AltUp hidden streams.
        altup_active_idx (`int`, *optional*, defaults to 0):
            Stream index AltUp reads/writes as the active prediction.
        altup_coef_clip (`float`, *optional*, defaults to 120.0):
            Amplitude cap on AltUp coefficients (training-time only).
        altup_correct_scale (`bool`, *optional*, defaults to `True`):
            Scale the corrected active-stream output by a learned vector.
        num_kv_shared_layers (`int`, *optional*, defaults to 15):
            Tail layers that reuse an earlier layer's K/V (per attention type).
        laurel_rank (`int`, *optional*, defaults to 64):
            Inner rank of the LAuReL low-rank residual.
        activation_sparsity_pattern (`list`, *optional*):
            Per-layer Gaussian top-k sparsity factor (default: 0.95 on the first 10
            layers, 0.0 elsewhere)."""

    model_type = "gemma3n_text"

    vocab_size: int = 262400
    embed_dim: int = 2048
    mlp_dim: int | list = 16384
    num_layers: int = 35
    num_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = 256
    sliding_window: int = 512
    sliding_window_pattern: int = 5
    layer_types: list | None = None
    final_logit_softcapping: float | None = 30.0
    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_local_theta: float = 10000.0
    hidden_activation: str = "gelu_pytorch_tanh"
    tie_embeddings: bool = True
    vocab_size_per_layer_input: int = 262144
    hidden_size_per_layer_input: int = 256
    altup_num_inputs: int = 4
    altup_active_idx: int = 0
    altup_coef_clip: float = 120.0
    altup_correct_scale: bool = True
    num_kv_shared_layers: int = 15
    laurel_rank: int = 64
    activation_sparsity_pattern: list | None = None


class Gemma3nAudioConfig(BaseConfig):
    r"""USM audio-tower config for Gemma 3n (the ``audio_config`` sub-config).

    A Universal Speech Model conformer stack: a SubSampleConvProjection (two
    stride-2 convs with cumulative-over-time group norm) feeds ``conf_num_hidden_layers``
    conformer blocks (feed-forward, chunked local self-attention with relative
    position bias, light depthwise conv, feed-forward), all activation-clamped by
    ``gradient_clipping``.

    Args:
        vocab_size (`int`, *optional*, defaults to 128):
            Auxiliary vocabulary for the multimodal embedder soft tokens.
        vocab_offset (`int`, *optional*, defaults to 262272):
            Offset subtracted from audio soft-token ids into the 0-indexed table.
        input_feat_size (`int`, *optional*, defaults to 128):
            Mel feature dimension of the raw audio features.
        hidden_size (`int`, *optional*, defaults to 1536):
            Conformer width.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            RMSNorm epsilon.
        gradient_clipping (`float`, *optional*, defaults to 1e10):
            Symmetric activation clamp applied throughout the conformer.
        conf_attention_chunk_size (`int`, *optional*, defaults to 12):
            Local-attention block size.
        conf_attention_context_left / conf_attention_context_right (`int`, *optional*, defaults to 13 / 0):
            Left / right local-attention context.
        conf_attention_logit_cap (`float`, *optional*, defaults to 50.0):
            Attention logit tanh soft-cap.
        conf_num_attention_heads (`int`, *optional*, defaults to 8):
            Conformer attention heads.
        conf_num_hidden_layers (`int`, *optional*, defaults to 12):
            Conformer blocks.
        conf_conv_kernel_size (`int`, *optional*, defaults to 5):
            Light-conv kernel size.
        conf_reduction_factor (`int`, *optional*, defaults to 4):
            Temporal reduction between the conformer output and the soft tokens.
        conf_residual_weight (`float`, *optional*, defaults to 0.5):
            Feed-forward residual scaling.
        sscp_conv_channel_size (`tuple`, *optional*, defaults to `(128, 32)`):
            Sub-sampling conv channels.
        sscp_conv_group_norm_eps (`float`, *optional*, defaults to 1e-3):
            Cumulative group-norm epsilon.
        sscp_conv_kernel_size / sscp_conv_stride_size (`tuple`, *optional*):
            Sub-sampling conv (time, freq) kernels / strides."""

    model_type = "gemma3n_audio"

    vocab_size: int = 128
    vocab_offset: int = 262272
    input_feat_size: int = 128
    hidden_size: int = 1536
    rms_norm_eps: float = 1e-6
    gradient_clipping: float = 1e10
    conf_attention_chunk_size: int = 12
    conf_attention_context_left: int = 13
    conf_attention_context_right: int = 0
    conf_attention_logit_cap: float = 50.0
    conf_num_attention_heads: int = 8
    conf_num_hidden_layers: int = 12
    conf_conv_kernel_size: int = 5
    conf_reduction_factor: int = 4
    conf_residual_weight: float = 0.5
    sscp_conv_channel_size: tuple = (128, 32)
    sscp_conv_group_norm_eps: float = 1e-3
    sscp_conv_kernel_size: tuple = ((3, 3), (3, 3))
    sscp_conv_stride_size: tuple = ((2, 2), (2, 2))


class Gemma3nVisionConfig(BaseConfig):
    r"""MobileNet-V5 vision-tower config for Gemma 3n (the ``vision_config`` sub-config).

    The tower is a timm ``mobilenetv5_300m_enc`` encoder (a MobileNet-V5 stem +
    universal-inverted-residual / mobile-attention blocks + a multi-scale fusion
    adapter). The per-image feature map is flattened to
    ``vision_soft_tokens_per_image`` soft tokens by [`Gemma3nMultimodalEmbedder`].

    Args:
        architecture (`str`, *optional*, defaults to ``"mobilenetv5_300m_enc"``):
            timm encoder name (informational; the Keras tower is built directly).
        hidden_size (`int`, *optional*, defaults to 2048):
            MSFA output width fed to the multimodal embedder.
        vocab_size (`int`, *optional*, defaults to 128):
            Auxiliary vocabulary for the multimodal embedder soft tokens.
        vocab_offset (`int`, *optional*, defaults to 262144):
            Offset subtracted from vision soft-token ids into the 0-indexed table.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Embedder RMSNorm epsilon.
        do_pooling (`bool`, *optional*, defaults to `False`):
            Whether the timm encoder pools; Gemma 3n keeps the full feature map."""

    model_type = "gemma3n_vision"

    architecture: str = "mobilenetv5_300m_enc"
    hidden_size: int = 2048
    vocab_size: int = 128
    vocab_offset: int = 262144
    rms_norm_eps: float = 1e-6
    do_pooling: bool = False


class Gemma3nConfig(BaseConfig):
    r"""Configuration for Gemma 3n: [`Gemma3nModel`] and
    [`Gemma3nConditionalGenerate`].

    A composite config: the text decoder lives in a [`Gemma3nTextConfig`]
    (``text_config``), the MobileNet-V5 tower in a [`Gemma3nVisionConfig`]
    (``vision_config``), and the USM tower in a [`Gemma3nAudioConfig`]
    (``audio_config``); the soft-token counts and ``*_token_id`` glue are
    top-level. An absent optional tower is ``None`` (dropped on serialize by
    BaseConfig); [`Gemma3nModel`] takes the sub-configs **nested**, so
    ``constructor_kwargs`` emits the nested form and passes ``None`` for an absent
    tower (which is how the model skips building it).

    Args:
        text_config (`Gemma3nTextConfig | dict`, *optional*):
            Text-decoder config (defaults to a `Gemma3nTextConfig`).
        vision_config (`Gemma3nVisionConfig | dict`, *optional*):
            MobileNet-V5 tower config, or `None` for no vision tower.
        audio_config (`Gemma3nAudioConfig | dict`, *optional*):
            USM tower config, or `None` for no audio tower.
        audio_soft_tokens_per_image (`int`, *optional*, defaults to 188):
            Audio soft tokens produced per audio clip.
        vision_soft_tokens_per_image (`int`, *optional*, defaults to 256):
            Vision soft tokens produced per image.
        boi_token_id / eoi_token_id (`int`, *optional*, defaults to 255999 / 262144):
            Begin/end-of-image marker ids.
        image_token_id (`int`, *optional*, defaults to 262145):
            Image soft-token placeholder id (scatter target).
        boa_token_id / eoa_token_id (`int`, *optional*, defaults to 256000 / 262272):
            Begin/end-of-audio marker ids.
        audio_token_id (`int`, *optional*, defaults to 262273):
            Audio soft-token placeholder id (scatter target).
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the LM head is tied to the text token embedding.

    Examples:

    ```python
    >>> from zeromodels.models.gemma3n import Gemma3nConfig, Gemma3nConditionalGenerate

    >>> configuration = Gemma3nConfig(
    ...     text_config={"num_layers": 30},
    ...     audio_config=None,
    ... )
    >>> model = Gemma3nConditionalGenerate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "gemma3n"

    sub_configs = {
        "text_config": Gemma3nTextConfig,
        "vision_config": Gemma3nVisionConfig,
        "audio_config": Gemma3nAudioConfig,
    }
    optional_sub_configs = ()

    text_config: Gemma3nTextConfig | dict | None = None
    vision_config: Gemma3nVisionConfig | dict | None = None
    audio_config: Gemma3nAudioConfig | dict | None = None
    audio_soft_tokens_per_image: int = 188
    vision_soft_tokens_per_image: int = 256
    boi_token_id: int = 255999
    eoi_token_id: int = 262144
    image_token_id: int = 262145
    boa_token_id: int = 256000
    eoa_token_id: int = 262272
    audio_token_id: int = 262273
    tie_word_embeddings: bool = True

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
