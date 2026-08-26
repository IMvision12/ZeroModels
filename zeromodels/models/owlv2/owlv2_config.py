"""Owlv2 model configuration."""

from zeromodels.base import BaseConfig


class Owlv2VisionConfig(BaseConfig):
    r"""Configuration for the Owlv2 vision tower (the `vision_config` sub-config).

    Args:
        image_size (`int`, *optional*, defaults to 960):
            Input image resolution of the vision tower.
        patch_size (`int`, *optional*, defaults to 16):
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
    >>> from zeromodels.models.owlv2 import Owlv2VisionConfig

    >>> configuration = Owlv2VisionConfig()
    ```"""

    model_type = "owlv2_vision"

    image_size: int = 960
    patch_size: int = 16
    hidden_dim: int = 768
    intermediate_size: int = 3072
    num_layers: int = 12
    num_heads: int = 12


class Owlv2TextConfig(BaseConfig):
    r"""Configuration for the Owlv2 text decoder (the `text_config` sub-config).

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
    >>> from zeromodels.models.owlv2 import Owlv2TextConfig

    >>> configuration = Owlv2TextConfig()
    ```"""

    model_type = "owlv2_text"

    hidden_dim: int = 512
    intermediate_size: int = 2048
    num_heads: int = 8
    num_layers: int = 12
    max_position_embeddings: int = 16
    vocab_size: int = 49408


class Owlv2Config(BaseConfig):
    r"""Configuration for Owlv2: the composite holding each tower's sub-config.

    Args:
        vision_config (`Owlv2VisionConfig` or `dict`, *optional*):
            Configuration of the Owlv2 vision tower.
        text_config (`Owlv2TextConfig` or `dict`, *optional*):
            Configuration of the Owlv2 text encoder.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimension of the shared vision-text projection space.
        image_size (`int`, *optional*, defaults to `None`):
            Square input resolution to build for; `None` uses `vision_image_size`.

    Example:

    ```python
    >>> from zeromodels.models.owlv2 import Owlv2Config, Owlv2Detect

    >>> configuration = Owlv2Config()
    >>> model = Owlv2Detect(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "owlv2"

    sub_configs = {"vision_config": Owlv2VisionConfig, "text_config": Owlv2TextConfig}
    sub_config_prefixes = {"vision_config": "vision_", "text_config": "text_"}

    vision_config: Owlv2VisionConfig | dict | None = None
    text_config: Owlv2TextConfig | dict | None = None
    projection_dim: int = 512
    image_size: int = None
