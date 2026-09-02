"""Offline consistency checks for the AutoZM* loader tables.

The AutoZM* classes resolve a repo's model_type through the committed, hand-maintained
tables in ``zeromodels/auto/auto_mapping_names.py`` (transformers' ``MODEL_MAPPING_NAMES``
pattern). Those tables can drift from the code -- a renamed class leaves a dangling entry, a
new model is never added -- so this guards them offline (no network / no ``from_weights``,
which the integration suite never exercises). Registry-driven, not per-model, so it stays a
single file rather than a parametrized matrix entry.
"""

import inspect

import pytest

import zeromodels as zm
from zeromodels.auto import auto_factory as A
from zeromodels.auto import auto_mapping_names as names

# Task taxonomy + model iteration for the code<->tables consistency checks. These live in
# the test (which verifies the hand-maintained tables against the code), not in the factory
# (which is purely table-driven and does no model introspection).
_TASK_NORMALIZE = {
    "ForObjectDetection": "Detect",
    "DptDepthEstimation": "DepthEstimation",
    "DptSemanticSegment": "SemanticSegment",
    "DptDensePredict": "DensePredict",
    "UnifiedTextGenerate": "TextGenerate",
}
_TASK_SUFFIXES = sorted(
    {
        "ConditionalGenerate",
        "UnifiedTextGenerate",
        "TextGenerate",
        "ZeroShotClassify",
        "ImageClassify",
        "SequenceClassify",
        "TokenClassify",
        "AudioClassify",
        "NextSentencePredict",
        "SemanticSegment",
        "InstanceSegment",
        "PanopticSegment",
        "UniversalSegment",
        "DptDepthEstimation",
        "DptSemanticSegment",
        "DptDensePredict",
        "DepthEstimation",
        "DensePredict",
        "MaskedLM",
        "MultipleChoice",
        "QnA",
        "CTC",
        "Detect",
        "Segment",
        "Generate",
        "ForObjectDetection",
        "ImageEmbed",
        "TextEmbed",
        "TextModel",
        "VisionModel",
        "EncoderModel",
        "Model",
    },
    key=len,
    reverse=True,
)


def _task_of(class_name):
    for suffix in _TASK_SUFFIXES:
        if class_name.endswith(suffix):
            return _TASK_NORMALIZE.get(suffix, suffix)
    return None


def _hf_types(cls):
    hf = getattr(cls, "HF_MODEL_TYPE", None)
    if hf is None:
        return ()
    return (hf,) if isinstance(hf, str) else tuple(hf)


def _iter_model_classes():
    import zeromodels.models as models
    from zeromodels.base.base_model import BaseModel

    for family in sorted(n for n in dir(models) if not n.startswith("_")):
        package = getattr(models, family)
        for name in getattr(package, "__all__", []):
            obj = getattr(package, name, None)
            if inspect.isclass(obj) and issubclass(obj, BaseModel):
                yield name, obj


def test_top_level_and_task_autos_exported():
    for name in (
        "AutoZModel",
        "AutoZMConfig",
        "AutoZMTokenizer",
        "AutoZMProcessor",
        "AutoZMImageProcessor",
    ):
        assert hasattr(zm, name), f"zeromodels.{name} missing"
    for name in (
        "AutoZModel",
        "AutoZMDetect",
        "AutoZMImageClassify",
        "AutoZMTextGenerate",
        "AutoZMConditionalGenerate",
        "AutoZMSemanticSegment",
    ):
        assert isinstance(getattr(zm.auto, name), type), (
            f"zeromodels.auto.{name} missing"
        )


def test_no_transformers_name_collision():
    for name in (
        "AutoModel",
        "AutoConfig",
        "AutoTokenizer",
        "AutoProcessor",
        "AutoImageProcessor",
    ):
        assert not hasattr(zm, name), (
            f"zeromodels.{name} would collide with transformers"
        )


@pytest.mark.parametrize(
    "auto_name,model_type,expected_cls",
    [
        ("AutoZModel", "bert", "BertModel"),
        ("AutoZModel", "clip", "CLIPModel"),
        ("AutoZMDetect", "detr", "DETRDetect"),
        ("AutoZMImageClassify", "resnet", "ResNetImageClassify"),
        # Versioned families are disambiguated by the distinct config model_type.
        ("AutoZModel", "deberta_v2", "DebertaV2Model"),
        ("AutoZModel", "deberta_v3", "DebertaV3Model"),
    ],
)
def test_key_resolutions(auto_name, model_type, expected_cls):
    auto = getattr(zm.auto, auto_name)
    resolved = auto.mapping().get(model_type)
    assert resolved is not None, f"{auto_name} has no entry for {model_type!r}"
    assert resolved.__name__ == expected_cls


@pytest.mark.parametrize(
    "auto_name,model_type,expected_cls",
    [
        # Llama v1/v2 and DepthAnything v1/v2 genuinely share one config model_type;
        # the table's committed entry is the newer class (overridable via register()).
        ("AutoZModel", "llama", "Llama2Model"),
        ("AutoZMTextGenerate", "llama", "Llama2TextGenerate"),
        ("AutoZModel", "depth_anything", "DepthAnythingV2Model"),
    ],
)
def test_collision_defaults(auto_name, model_type, expected_cls):
    assert getattr(zm.auto, auto_name).mapping()[model_type].__name__ == expected_cls


def test_hf_alias_resolves_and_ambiguous_raises():
    assert zm.AutoZModel._resolve("xlm-roberta", "hf").__name__ == "XLMRobertaModel"
    with pytest.raises(ValueError, match="ambiguous"):
        zm.AutoZModel._resolve("deberta-v2", "hf")


def test_preprocessor_and_config_registries():
    assert zm.AutoZMConfig.for_model_type("detr").__name__ == "DetrConfig"
    assert zm.AutoZMTokenizer.mapping()["bert"].__name__ == "BertTokenizer"
    assert zm.AutoZMImageProcessor.mapping()["detr"].__name__ == "DETRImageProcessor"
    assert zm.AutoZMProcessor.mapping()["clip"].__name__ == "CLIPProcessor"


def test_read_model_type_rejects_bare_variant():
    with pytest.raises(ValueError, match="repo id"):
        A.read_model_type("resnet50_a1_in1k")


def test_register_override():
    class _Dummy:
        pass

    zm.AutoZModel.register("zzz_fake_type", _Dummy)
    try:
        assert zm.AutoZModel.mapping()["zzz_fake_type"] is _Dummy
    finally:
        A._MODEL_OVERRIDES.get("Model", {}).pop("zzz_fake_type", None)


# ---- table integrity (validity + completeness vs the code) ----


def test_every_table_value_resolves_and_matches_its_task():
    """No dangling / mis-filed entries: each value is a real class ending in its task."""
    for task, table in names.MODEL_TASK_MAPPING_NAMES.items():
        for model_type, class_name in table.items():
            A._resolve_class(class_name)  # raises if the class was renamed/removed
            assert _task_of(class_name) == task, (
                f"{class_name} is in the {task!r} table but its task is "
                f"{_task_of(class_name)!r}"
            )
    for flat in (
        names.CONFIG_MAPPING_NAMES,
        names.TOKENIZER_MAPPING_NAMES,
        names.PROCESSOR_MAPPING_NAMES,
        names.IMAGE_PROCESSOR_MAPPING_NAMES,
    ):
        for class_name in flat.values():
            A._resolve_class(class_name)


_COVERAGE_EXEMPT = {
    # Collision losers: a family sharing one config model_type -- the table holds the newer
    # sibling, the older is loadable via its own class.
    "LlamaModel",
    "LlamaTextGenerate",
    "DepthAnythingV1Model",
    "DepthAnythingV1DepthEstimation",
    # Redundant transformers-named alias whose model_type ("grounding-dino") already maps to
    # the zeromodels-convention sibling GroundingDinoDetect.
    "GroundingDinoForObjectDetection",
}


def _has_model_type(cls):
    if _hf_types(cls):
        return True
    config_cls = getattr(cls, "config_class", None)
    return bool(config_cls is not None and getattr(config_cls, "model_type", None))


def test_every_autodetectable_model_class_appears_in_a_table():
    """Every model class that carries a model_type is reachable through some AutoZM table
    (catches a new model added to the code but not to the hand-maintained tables). Bare
    backbones with no model_type are skipped: a repo has nothing to autodetect on, and they
    load via a task sibling or the concrete class."""
    mapped = set()
    for table in names.MODEL_TASK_MAPPING_NAMES.values():
        mapped.update(table.values())
    missing = sorted(
        name
        for name, cls in _iter_model_classes()
        if _has_model_type(cls) and name not in mapped and name not in _COVERAGE_EXEMPT
    )
    assert not missing, (
        "model class(es) with a model_type absent from "
        "zeromodels/auto/auto_mapping_names.py; add a row to the matching task table: "
        f"{missing}"
    )
