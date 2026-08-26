"""OwlViT model configuration."""

from zeromodels.base import BaseConfig


class OwlViTVisionConfig(BaseConfig):
    r"""Configuration for the OwlViT vision tower (the `vision_config` sub-config).

    Args:
        image_size (`int`, *optional*, defaults to 768):
            Input image resolution of the vision tower.
        patch_size (`int`, *optional*, defaults to 32):
            Patch size of the vision tower.
        hidden_dim (`int`, *optional*, defaults to 768):
            Hidden dimension of the vision tower.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Feed-forward dimension of the vision tower.
        num_layers (`int`, *optional*, defaults to 12):
            Number of transformer layers in the vision tower.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads in the vision tower.

    Example:

    ```python
    >>> from zeromodels.models.owlvit import OwlViTVisionConfig

    >>> configuration = OwlViTVisionConfig()
    ```"""

    model_type = "owlvit_vision"

    image_size: int = 768
    patch_size: int = 32
    hidden_dim: int = 768
    intermediate_size: int = 3072
    num_layers: int = 12
    num_heads: int = 12


class OwlViTTextConfig(BaseConfig):
    r"""Configuration for the OwlViT text decoder (the `text_config` sub-config).

    Args:
        hidden_dim (`int`, *optional*, defaults to 512):
            Hidden dimension of the text tower.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Feed-forward dimension of the text tower.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the text tower.
        num_layers (`int`, *optional*, defaults to 12):
            Number of transformer layers in the text tower.
        max_position_embeddings (`int`, *optional*, defaults to 16):
            Maximum text sequence length (per prompt) the text tower handles.
        vocab_size (`int`, *optional*, defaults to 49408):
            Vocabulary size of the CLIP text tokenizer.

    Example:

    ```python
    >>> from zeromodels.models.owlvit import OwlViTTextConfig

    >>> configuration = OwlViTTextConfig()
    ```"""

    model_type = "owlvit_text"

    hidden_dim: int = 512
    intermediate_size: int = 2048
    num_heads: int = 8
    num_layers: int = 12
    max_position_embeddings: int = 16
    vocab_size: int = 49408


class OwlViTConfig(BaseConfig):
    r"""Configuration for OwlViT: the composite holding each tower's sub-config.

    Args:
        vision_config (`OwlViTVisionConfig` or `dict`, *optional*):
            Configuration of the OwlViT vision tower.
        text_config (`OwlViTTextConfig` or `dict`, *optional*):
            Configuration of the OwlViT text encoder.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimension of the shared vision-text projection space.
        image_size (`int`, *optional*, defaults to `None`):
            Square input resolution to build for; `None` uses `vision_image_size`.

    Example:

    ```python
    >>> from zeromodels.models.owlvit import OwlViTConfig, OwlViTDetect

    >>> configuration = OwlViTConfig()
    >>> model = OwlViTDetect(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "owlvit"

    sub_configs = {"vision_config": OwlViTVisionConfig, "text_config": OwlViTTextConfig}
    sub_config_prefixes = {"vision_config": "vision_", "text_config": "text_"}

    vision_config: OwlViTVisionConfig | dict | None = None
    text_config: OwlViTTextConfig | dict | None = None
    projection_dim: int = 512
    image_size: int = None
