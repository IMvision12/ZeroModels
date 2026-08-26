"""SigLIP model configuration."""

from zeromodels.base import BaseConfig


class SigLIPTextConfig(BaseConfig):
    r"""Configuration for the SigLIP text decoder (the `text_config` sub-config).

    Args:
        hidden_dim (`int`, *optional*, defaults to 768):
            Hidden size of the text encoder.
        num_layers (`int`, *optional*, defaults to 12):
            Depth of the text encoder.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads in the text encoder.
        mlp_dim (`int`, *optional*, defaults to 3072):
            Feed-forward dimension of the text encoder.
        vocab_size (`int`, *optional*, defaults to 32000):
            Text tokenizer vocabulary size.
        max_seq_len (`int`, *optional*, defaults to 64):
            Maximum text sequence length.

    Example:

    ```python
    >>> from zeromodels.models.siglip import SigLIPTextConfig

    >>> configuration = SigLIPTextConfig()
    ```"""

    model_type = "siglip_text"

    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    vocab_size: int = 32000
    max_seq_len: int = 64


class SigLIPVisionConfig(BaseConfig):
    r"""Configuration for the SigLIP vision tower (the `vision_config` sub-config).

    Args:
        hidden_dim (`int`, *optional*, defaults to 768):
            Hidden size of the vision encoder.
        num_layers (`int`, *optional*, defaults to 12):
            Depth of the vision encoder.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads in the vision encoder.
        mlp_dim (`int`, *optional*, defaults to 3072):
            Feed-forward dimension of the vision encoder.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the vision tower is built for.
        patch_size (`int`, *optional*, defaults to 16):
            Patch size of the vision encoder.

    Example:

    ```python
    >>> from zeromodels.models.siglip import SigLIPVisionConfig

    >>> configuration = SigLIPVisionConfig()
    ```"""

    model_type = "siglip_vision"

    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    image_size: int = 224
    patch_size: int = 16


class SigLIPConfig(BaseConfig):
    r"""Configuration for SigLIP: the composite holding each tower's sub-config.

    Args:
        text_config (`SigLIPTextConfig` or `dict`, *optional*):
            Configuration of the SigLIP text encoder.
        vision_config (`SigLIPVisionConfig` or `dict`, *optional*):
            Configuration of the SigLIP vision tower.
        embed_dim (`int`, *optional*, defaults to 768):
            Shared image-text projection dimension.

    Example:

    ```python
    >>> from zeromodels.models.siglip import SigLIPConfig, SigLIPImageClassify

    >>> configuration = SigLIPConfig()
    >>> model = SigLIPImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "siglip"

    sub_configs = {"text_config": SigLIPTextConfig, "vision_config": SigLIPVisionConfig}
    sub_config_prefixes = {"text_config": "text_", "vision_config": "vision_"}
    group_extras = {
        "text_config": ("vocab_size", "max_seq_len"),
        "vision_config": ("image_size", "patch_size"),
    }

    text_config: SigLIPTextConfig | dict | None = None
    vision_config: SigLIPVisionConfig | dict | None = None
    embed_dim: int = 768
