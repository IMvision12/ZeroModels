"""MetaClip2 model configuration."""

from zeromodels.base import BaseConfig


class MetaClip2TextConfig(BaseConfig):
    r"""Configuration for the MetaClip2 text decoder (the `text_config` sub-config).

    Args:
        hidden_dim (`int`, *optional*, defaults to 512):
            Hidden size of the text encoder.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the text encoder.
        num_layers (`int`, *optional*, defaults to 12):
            Depth of the text encoder.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            MLP expansion ratio in the text blocks.
        vocab_size (`int`, *optional*, defaults to 901629):
            Text tokenizer vocabulary size (the multilingual XLM-R vocab; the mt5
            variants use 250000).
        max_seq_len (`int`, *optional*, defaults to 77):
            Maximum text sequence length.

    Example:

    ```python
    >>> from zeromodels.models.metaclip2 import MetaClip2TextConfig

    >>> configuration = MetaClip2TextConfig()
    ```"""

    model_type = "metaclip_2_text"

    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 12
    mlp_ratio: float = 4.0
    vocab_size: int = 901629
    max_seq_len: int = 77


class MetaClip2VisionConfig(BaseConfig):
    r"""Configuration for the MetaClip2 vision tower (the `vision_config` sub-config).

    Args:
        num_layers (`int`, *optional*, defaults to 12):
            Depth of the ViT vision encoder.
        hidden_dim (`int`, *optional*, defaults to 768):
            Hidden size of the vision encoder.
        patch_size (`int`, *optional*, defaults to 32):
            Patch size of the vision encoder.
        num_heads (`int`, *optional*, defaults to `None`):
            Number of attention heads in the vision encoder; when `None`, derived
            as `vision_hidden_dim // 64`.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            MLP expansion ratio in the vision blocks.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the vision tower is built for.

    Example:

    ```python
    >>> from zeromodels.models.metaclip2 import MetaClip2VisionConfig

    >>> configuration = MetaClip2VisionConfig()
    ```"""

    model_type = "metaclip_2_vision"

    num_layers: int = 12
    hidden_dim: int = 768
    patch_size: int = 32
    num_heads: int = None
    mlp_ratio: float = 4.0
    image_size: int = 224


class MetaClip2Config(BaseConfig):
    r"""Configuration for MetaClip2: the composite holding each tower's sub-config.

    Args:
        text_config (`MetaClip2TextConfig` or `dict`, *optional*):
            Configuration of the MetaClip2 text encoder.
        vision_config (`MetaClip2VisionConfig` or `dict`, *optional*):
            Configuration of the MetaClip2 vision tower.
        embed_dim (`int`, *optional*, defaults to 512):
            Shared image-text projection dimension.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            MLP activation (`"quick_gelu"` for the huge-quickgelu variant).
        eos_token_id (`int`, *optional*, defaults to 2):
            End-of-sequence token id used to pool the text features (the mt5
            variants use 1).

    Example:

    ```python
    >>> from zeromodels.models.metaclip2 import MetaClip2Config, MetaClip2ImageClassify

    >>> configuration = MetaClip2Config()
    >>> model = MetaClip2ImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "metaclip_2"

    sub_configs = {
        "text_config": MetaClip2TextConfig,
        "vision_config": MetaClip2VisionConfig,
    }
    sub_config_prefixes = {"text_config": "text_", "vision_config": "vision_"}
    group_extras = {
        "text_config": ("vocab_size", "max_seq_len"),
        "vision_config": ("image_size",),
    }

    text_config: MetaClip2TextConfig | dict | None = None
    vision_config: MetaClip2VisionConfig | dict | None = None
    embed_dim: int = 512
    hidden_act: str = "gelu"
    eos_token_id: int = 2
