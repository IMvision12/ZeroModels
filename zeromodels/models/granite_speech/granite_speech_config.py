"""GraniteSpeech model configuration."""

from zeromodels.base import BaseConfig


class GraniteSpeechTextConfig(BaseConfig):
    r"""Configuration for the GraniteSpeech text decoder (the `text_config` sub-config).

    The Granite decoder recipe: GQA attention, a SwiGLU MLP, and the four Granite
    scalar multipliers (`embedding_multiplier`, `residual_multiplier`,
    `attention_multiplier`, `logits_scaling`).

    Args:
        vocab_size (`int`, *optional*, defaults to 49160):
            Token vocabulary size.
        embed_dim (`int`, *optional*, defaults to 2048):
            Text / residual-stream width.
        mlp_dim (`int`, *optional*, defaults to 8192):
            SwiGLU hidden width per text layer.
        num_layers (`int`, *optional*, defaults to 40):
            Number of text decoder blocks.
        num_heads (`int`, *optional*, defaults to 32):
            Query heads per text layer.
        num_kv_heads (`int`, *optional*, defaults to 8):
            Key/value heads per text layer.
        norm_eps (`float`, *optional*, defaults to 1e-5):
            Text RMSNorm epsilon.
        rope_theta (`float`, *optional*, defaults to 10000000.0):
            Rotary base frequency.
        embedding_multiplier (`float`, *optional*, defaults to 12.0):
            Granite scalar multiplying the token embeddings.
        residual_multiplier (`float`, *optional*, defaults to 0.22):
            Granite scalar multiplying each residual-branch output.
        attention_multiplier (`float`, *optional*, defaults to 0.015625):
            Granite attention-logit scale (replaces the `1/sqrt(head_dim)` default).
        logits_scaling (`float`, *optional*, defaults to 8.0):
            Granite divisor applied to the final logits.
        tie_embeddings (`bool`, *optional*, defaults to `True`):
            Whether [`GraniteSpeechConditionalGenerate`] ties the LM head to the token
            embeddings.
        eos_token_id (`int`, *optional*, defaults to 0):
            End-of-sequence token id.

    Example:

    ```python
    >>> from zeromodels.models.granite_speech import GraniteSpeechTextConfig

    >>> configuration = GraniteSpeechTextConfig()
    ```"""

    model_type = "granite_speech_text"

    vocab_size: int = 49160
    embed_dim: int = 2048
    mlp_dim: int = 8192
    num_layers: int = 40
    num_heads: int = 32
    num_kv_heads: int = 8
    norm_eps: float = 1e-05
    rope_theta: float = 10000000.0
    embedding_multiplier: float = 12.0
    residual_multiplier: float = 0.22
    attention_multiplier: float = 0.015625
    logits_scaling: float = 8.0
    tie_embeddings: bool = True
    eos_token_id: int = 0


class GraniteSpeechAudioConfig(BaseConfig):
    r"""Configuration for the GraniteSpeech audio encoder (the `audio_config` sub-config).

    A conformer CTC encoder (block-local attention + a depthwise-convolution module
    per block) that turns log-mel features into the audio embeddings handed to the
    Q-Former projector.

    Args:
        input_dim (`int`, *optional*, defaults to 160):
            Input log-mel feature dimension.
        num_layers (`int`, *optional*, defaults to 16):
            Number of conformer encoder blocks.
        hidden_dim (`int`, *optional*, defaults to 1024):
            Conformer hidden width.
        feedforward_mult (`int`, *optional*, defaults to 4):
            Feed-forward expansion multiplier per conformer block.
        num_heads (`int`, *optional*, defaults to 8):
            Conformer attention heads.
        dim_head (`int`, *optional*, defaults to 128):
            Per-head dimension.
        output_dim (`int`, *optional*, defaults to 256):
            Encoder output width fed to the projector.
        context_size (`int`, *optional*, defaults to 200):
            Block-attention context window, in frames.
        max_pos_emb (`int`, *optional*, defaults to 512):
            Maximum relative-position embedding span.
        conv_kernel_size (`int`, *optional*, defaults to 15):
            Depthwise-convolution kernel size.
        conv_expansion_factor (`int`, *optional*, defaults to 2):
            Convolution-module channel expansion factor.

    Example:

    ```python
    >>> from zeromodels.models.granite_speech import GraniteSpeechAudioConfig

    >>> configuration = GraniteSpeechAudioConfig()
    ```"""

    model_type = "granite_speech_audio"

    input_dim: int = 160
    num_layers: int = 16
    hidden_dim: int = 1024
    feedforward_mult: int = 4
    num_heads: int = 8
    dim_head: int = 128
    output_dim: int = 256
    context_size: int = 200
    max_pos_emb: int = 512
    conv_kernel_size: int = 15
    conv_expansion_factor: int = 2


class GraniteSpeechConfig(BaseConfig):
    r"""Configuration for GraniteSpeech: the composite holding each tower's sub-config.

    The conformer `audio_config` feeds a BLIP-2 Q-Former projector whose output is
    fused into the Granite `text_config` decoder at the `audio_token_id` placeholder
    positions; the remaining fields configure that projector, the fusion, and the
    optional LoRA adapter over the fused positions.

    Args:
        text_config (`GraniteSpeechTextConfig` or `dict`, *optional*):
            Configuration of the GraniteSpeech text decoder.
        audio_config (`GraniteSpeechAudioConfig` or `dict`, *optional*):
            Configuration of the GraniteSpeech audio encoder.
        audio_token_id (`int`, *optional*, defaults to 49159):
            Placeholder token id whose positions receive the projected audio features.
        downsample_rate (`int`, *optional*, defaults to 5):
            Temporal downsample factor applied to encoder features before fusion.
        window_size (`int`, *optional*, defaults to 15):
            Number of encoder frames grouped per projector query window.
        has_lora_adapter (`bool`, *optional*, defaults to `True`):
            Whether the text decoder carries a LoRA adapter over the fused audio
            positions.
        lora_rank (`int`, *optional*, defaults to 64):
            LoRA adapter rank.
        lora_alpha (`int`, *optional*, defaults to 32):
            LoRA adapter scaling alpha.
        projector_dim (`int`, *optional*, defaults to 1024):
            Q-Former projector hidden width.
        projector_num_layers (`int`, *optional*, defaults to 2):
            Number of Q-Former projector blocks.
        projector_num_heads (`int`, *optional*, defaults to 16):
            Q-Former projector attention heads.
        projector_intermediate_size (`int`, *optional*, defaults to 4096):
            Q-Former projector feed-forward width.
        projector_cross_attention_frequency (`int`, *optional*, defaults to 1):
            Insert a cross-attention layer every `frequency` projector blocks.
        projector_layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            Q-Former projector LayerNorm epsilon.
        cat_hidden_layers (`tuple`, *optional*):
            Intermediate encoder layers concatenated with the final output before the
            projector (Plus only; `None` uses the final layer alone).

    Example:

    ```python
    >>> from zeromodels.models.granite_speech import GraniteSpeechConfig, GraniteSpeechConditionalGenerate

    >>> configuration = GraniteSpeechConfig()
    >>> model = GraniteSpeechConditionalGenerate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "granite_speech"

    sub_configs = {
        "text_config": GraniteSpeechTextConfig,
        "audio_config": GraniteSpeechAudioConfig,
    }
    sub_config_prefixes = {"text_config": "", "audio_config": "encoder_"}

    text_config: GraniteSpeechTextConfig | dict | None = None
    audio_config: GraniteSpeechAudioConfig | dict | None = None
    audio_token_id: int = 49159
    downsample_rate: int = 5
    window_size: int = 15
    has_lora_adapter: bool = True
    lora_rank: int = 64
    lora_alpha: int = 32
    projector_dim: int = 1024
    projector_num_layers: int = 2
    projector_num_heads: int = 16
    projector_intermediate_size: int = 4096
    projector_cross_attention_frequency: int = 1
    projector_layer_norm_eps: float = 1e-12
    cat_hidden_layers: tuple = None
