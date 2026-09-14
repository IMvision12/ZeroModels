from zeromodels.models.stable_diffusion_3.stable_diffusion_3_config import (
    StableDiffusion3Config,
    StableDiffusion3TransformerConfig,
)


class StableDiffusion3_5TransformerConfig(StableDiffusion3TransformerConfig):
    r"""The Stable Diffusion 3.5 MMDiT: a [`StableDiffusion3TransformerConfig`] with
    the SD 3.5 large layout.

    Same fields as [`StableDiffusion3TransformerConfig`]; the defaults are SD 3.5
    large (8B parameters): 38 blocks of 38 heads (2432 wide) with RMS-normalized
    queries and keys. SD 3.5 medium (2.5B, MMDiT-X) is 24 blocks of 24 heads with
    ``dual_attention_layers`` 0 to 12 and a 384-patch position grid; each repo's
    ``zm_config.json`` carries its own values."""

    num_layers: int = 38
    num_attention_heads: int = 38
    caption_projection_dim: int = 2432
    qk_norm: str | None = "rms_norm"


class StableDiffusion3_5Config(StableDiffusion3Config):
    r"""Configuration for [`StableDiffusion3_5Model`], the hosted SD 3.5 weights container.

    Same shape as [`StableDiffusion3Config`] (nested ``transformer_config`` /
    ``vae_config`` / ``text_config`` / ``text_config_2`` plus the scheduler, the
    token ids and the T5 length; flat constructor) with the
    [`StableDiffusion3_5TransformerConfig`] denoiser. The three SD 3.5 checkpoints
    (large, large-turbo, medium) share the VAE and the text encoders with SD 3.

    Examples:

    ```python
    >>> from zeromodels.models.stable_diffusion_3_5 import StableDiffusion3_5Config, StableDiffusion3_5Model

    >>> config = StableDiffusion3_5Config()
    >>> model = StableDiffusion3_5Model(config)  # random weights, SD 3.5 large shapes
    >>> model.config.transformer_config.qk_norm
    'rms_norm'
    ```"""

    model_type = "stable_diffusion_3_5"

    sub_configs = {
        "transformer_config": StableDiffusion3_5TransformerConfig,
        "vae_config": StableDiffusion3Config.sub_configs["vae_config"],
        "text_config": StableDiffusion3Config.sub_configs["text_config"],
        "text_config_2": StableDiffusion3Config.sub_configs["text_config_2"],
    }

    transformer_config: StableDiffusion3_5TransformerConfig | dict | None = None
