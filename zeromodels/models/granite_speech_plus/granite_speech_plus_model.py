import keras

from zeromodels.models.granite_speech.granite_speech_model import (
    GraniteSpeechConditionalGenerate,
    GraniteSpeechModel,
)

from .granite_speech_plus_config import GraniteSpeechPlusConfig

# GraniteSpeechPlus has its own weights repos (separate from GraniteSpeech). The
# backbone + generate head both load from the variant's repo, whose kf_config.json
# declares the canonical GraniteSpeechPlusConditionalGenerate.
GRANITE_SPEECH_PLUS_HUB_SIBLINGS = frozenset(
    {"GraniteSpeechPlusModel", "GraniteSpeechPlusConditionalGenerate"}
)


@keras.saving.register_keras_serializable(package="zeromodels")
class GraniteSpeechPlusModel(GraniteSpeechModel):
    """GraniteSpeechPlus backbone: :class:`GraniteSpeechModel` whose conformer CTC
    encoder concatenates a subset of intermediate layer outputs
    (``cat_hidden_layers``) with its final output before the projector (so the
    projector's ``encoder_hidden_size`` becomes
    ``encoder_hidden_dim * (len(cat_hidden_layers) + 1)``). All layers, fusion, the
    LoRA adapter and the weight transfer are reused from ``granite_speech``; this
    variant only points at the Plus config + release weights."""

    HF_MODEL_TYPE = "granite_speech_plus"
    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = GraniteSpeechPlusConfig
    HUB_REPO_SIBLINGS = GRANITE_SPEECH_PLUS_HUB_SIBLINGS


@keras.saving.register_keras_serializable(package="zeromodels")
class GraniteSpeechPlusConditionalGenerate(GraniteSpeechConditionalGenerate):
    """GraniteSpeechPlus with an LM head + fast ``.generate()`` (audio+text -> text)
    the Plus variant of :class:`GraniteSpeechConditionalGenerate`."""

    HF_MODEL_TYPE = "granite_speech_plus"
    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = GraniteSpeechPlusConfig
    HUB_REPO_SIBLINGS = GRANITE_SPEECH_PLUS_HUB_SIBLINGS
