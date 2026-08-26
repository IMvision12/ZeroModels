from zeromodels.base import BaseConfig
from zeromodels.models.gemma4.gemma4_config import Gemma4Config, Gemma4TextConfig


class Gemma4UnifiedVisionConfig(BaseConfig):
    r"""Encoder-free vision-embedder config for Gemma 4 unified (``vision_config``).

    No transformer tower: images are raw merged pixel patches
    (``model_patch_size = patch_size * pooling_kernel_size``, 48px) projected by a
    Dense + factorized 2D position embedding of length ``mm_posemb_size``, then the
    shared soft-token projector into ``output_proj_dims`` (text width).

    Args:
        patch_size (`int`, *optional*, defaults to 16):
            Teacher patch side in pixels (before merging).
        pooling_kernel_size (`int`, *optional*, defaults to 3):
            Merge kernel side (``model_patch_size = patch_size * pooling_kernel_size``).
        mm_embed_dim (`int`, *optional*, defaults to 3840):
            Patch Dense projection width and position-table width.
        mm_posemb_size (`int`, *optional*, defaults to 1120):
            Length of the factorized 2D position table.
        output_proj_dims (`int`, *optional*, defaults to 3840):
            Soft-token projector output width (text hidden size).
        eps (`float`, *optional*, defaults to 1e-6):
            RMSNorm epsilon of the soft-token projector."""

    model_type = "gemma4_unified_vision"

    patch_size: int = 16
    pooling_kernel_size: int = 3
    mm_embed_dim: int = 3840
    mm_posemb_size: int = 1120
    output_proj_dims: int = 3840
    eps: float = 1e-6


class Gemma4UnifiedAudioConfig(BaseConfig):
    r"""Encoder-free audio-embedder config for Gemma 4 unified (``audio_config``).

    No conformer: each soft token is a raw 40ms waveform frame of
    ``audio_embed_dim`` (640) samples projected straight into ``output_proj_dims``.

    Args:
        audio_embed_dim (`int`, *optional*, defaults to 640):
            Raw samples per audio soft token (== frame length).
        output_proj_dims (`int`, *optional*, defaults to 640):
            Soft-token projector input width (== ``audio_embed_dim``).
        eps (`float`, *optional*, defaults to 1e-6):
            RMSNorm epsilon of the soft-token projector."""

    model_type = "gemma4_unified_audio"

    audio_embed_dim: int = 640
    output_proj_dims: int = 640
    eps: float = 1e-6


class Gemma4UnifiedConfig(Gemma4Config):
    r"""Configuration for Gemma 4 unified: [`Gemma4UnifiedModel`] /
    [`Gemma4UnifiedConditionalGenerate`].

    The composite for the encoder-free unified checkpoints (google/gemma-4-12B):
    the text decoder reuses [`Gemma4TextConfig`] (``text_config``), the vision and
    audio embedders are [`Gemma4UnifiedVisionConfig`] / [`Gemma4UnifiedAudioConfig`]
    (``vision_config`` / ``audio_config``, optional None when a tower is absent),
    and the ``*_token_id`` glue is top-level. Shares
    [`Gemma4Config`]'s nested-constructor / optional-tower handling.

    Args:
        text_config (`Gemma4TextConfig | dict`, *optional*):
            Text-decoder config (defaults to a `Gemma4TextConfig`).
        vision_config (`Gemma4UnifiedVisionConfig | dict`, *optional*):
            Encoder-free vision-embedder config, or `None`.
        audio_config (`Gemma4UnifiedAudioConfig | dict`, *optional*):
            Encoder-free audio-embedder config, or `None`.
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
    >>> from zeromodels.models.gemma4_unified import (
    ...     Gemma4UnifiedConfig, Gemma4UnifiedConditionalGenerate
    ... )

    >>> configuration = Gemma4UnifiedConfig(
    ...     text_config={"embed_dim": 3840, "num_layers": 48},
    ...     vision_config={"mm_embed_dim": 3840},
    ...     audio_config={"audio_embed_dim": 640},
    ... )
    >>> model = Gemma4UnifiedConditionalGenerate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "gemma4_unified"

    sub_configs = {
        "text_config": Gemma4TextConfig,
        "vision_config": Gemma4UnifiedVisionConfig,
        "audio_config": Gemma4UnifiedAudioConfig,
    }
    optional_sub_configs = ("vision_config", "audio_config")
