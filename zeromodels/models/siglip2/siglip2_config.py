"""SigLIP 2 model configuration."""

from zeromodels.models.siglip.siglip_config import (
    SigLIPConfig,
    SigLIPTextConfig,
    SigLIPVisionConfig,
)


class Siglip2TextConfig(SigLIPTextConfig):
    r"""Configuration for the SigLIP 2 text tower (the `text_config` sub-config): the
    SigLIP text encoder with the Gemma-style vocabulary.

    Example:

    ```python
    >>> from zeromodels.models.siglip2 import Siglip2TextConfig

    >>> configuration = Siglip2TextConfig()
    ```"""

    model_type = "siglip2_text"

    vocab_size: int = 256000


class Siglip2Config(SigLIPConfig):
    r"""Configuration for SigLIP 2: reuses the SigLIP architecture (the shared
    [`SigLIPVisionConfig`]); the text tower differs only in `vocab_size` (256000).

    Example:

    ```python
    >>> from zeromodels.models.siglip2 import Siglip2Config, SigLIP2ImageClassify

    >>> configuration = Siglip2Config()
    >>> model = SigLIP2ImageClassify(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "siglip2"

    sub_configs = {
        "text_config": Siglip2TextConfig,
        "vision_config": SigLIPVisionConfig,
    }

    text_config: Siglip2TextConfig | dict | None = None
    vision_config: SigLIPVisionConfig | dict | None = None
