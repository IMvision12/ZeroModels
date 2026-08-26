"""GraniteSpeechPlus model configuration."""

from zeromodels.models.granite_speech.granite_speech_config import (
    GraniteSpeechConfig,
)


class GraniteSpeechPlusConfig(GraniteSpeechConfig):
    r"""Configuration for GraniteSpeechPlus: the same composite structure as
    [`GraniteSpeechConfig`] (conformer `audio_config` + Granite `text_config`); the
    released variant sets its dimensions and `cat_hidden_layers` via the recipe.

    Example:

    ```python
    >>> from zeromodels.models.granite_speech_plus import (
    ...     GraniteSpeechPlusConfig,
    ...     GraniteSpeechPlusConditionalGenerate,
    ... )

    >>> configuration = GraniteSpeechPlusConfig()
    >>> model = GraniteSpeechPlusConditionalGenerate(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "granite_speech_plus"
