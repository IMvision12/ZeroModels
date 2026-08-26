"""CLIP model configuration."""

from zeromodels.base import BaseConfig


class CLIPTextConfig(BaseConfig):
    r"""Configuration for the CLIP text decoder (the `text_config` sub-config).

    Args:
        hidden_dim (`int`, *optional*, defaults to 512):
            Hidden size of the text encoder.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the text encoder.
        num_layers (`int`, *optional*, defaults to 12):
            Depth of the text encoder.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            MLP expansion ratio in the text blocks.
        vocab_size (`int`, *optional*, defaults to 49408):
            Text tokenizer vocabulary size.
        max_seq_len (`int`, *optional*, defaults to 77):
            Maximum text sequence length.

    Example:

    ```python
    >>> from zeromodels.models.clip import CLIPTextConfig

    >>> configuration = CLIPTextConfig()
    ```"""

    model_type = "clip_text"

    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 12
    mlp_ratio: float = 4.0
    vocab_size: int = 49408
    max_seq_len: int = 77


class CLIPVisionConfig(BaseConfig):
    r"""Configuration for the CLIP vision tower (the `vision_config` sub-config).

    Args:
        num_layers (`int`, *optional*, defaults to 12):
            Depth of the ViT vision encoder.
        hidden_dim (`int`, *optional*, defaults to 768):
            Hidden size of the vision encoder.
        patch_size (`int`, *optional*, defaults to 32):
            Patch size of the vision encoder.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            MLP expansion ratio in the vision blocks.
        image_size (`int`, *optional*, defaults to 224):
            Square input resolution the vision tower is built for.

    Example:

    ```python
    >>> from zeromodels.models.clip import CLIPVisionConfig

    >>> configuration = CLIPVisionConfig()
    ```"""

    model_type = "clip_vision"

    num_layers: int = 12
    hidden_dim: int = 768
    patch_size: int = 32
    mlp_ratio: float = 4.0
    image_size: int = 224


class CLIPConfig(BaseConfig):
    r"""Configuration for CLIP: the composite holding each tower's sub-config.

    Args:
        text_config (`CLIPTextConfig` or `dict`, *optional*):
            Configuration of the CLIP text encoder.
        vision_config (`CLIPVisionConfig` or `dict`, *optional*):
            Configuration of the CLIP vision tower.
        embed_dim (`int`, *optional*, defaults to 512):
            Shared image-text projection dimension (`projection_dim`).
        hidden_act (`str`, *optional*, defaults to `"quick_gelu"`):
            MLP activation: `"quick_gelu"` for canonical OpenAI CLIP, `"gelu"` for
            the larger LAION / open_clip variants.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for every LayerNorm.

    Example:

    ```python
    >>> from zeromodels.models.clip import CLIPConfig, CLIPImageClassify

    >>> configuration = CLIPConfig()
    >>> model = CLIPImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "clip"

    sub_configs = {"text_config": CLIPTextConfig, "vision_config": CLIPVisionConfig}
    sub_config_prefixes = {"text_config": "text_", "vision_config": "vision_"}
    group_extras = {
        "text_config": ("vocab_size", "max_seq_len"),
        "vision_config": ("image_size",),
    }

    text_config: CLIPTextConfig | dict | None = None
    vision_config: CLIPVisionConfig | dict | None = None
    embed_dim: int = 512
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-05
