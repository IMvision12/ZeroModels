"""Auto loaders: resolve a repo to the right class by its ``model_type``.

The transformers ``AutoModel`` idea, zeromodels flavored. Every hosted repo records a
``model_type`` -- a zeromodels repo in its ``zm_config.json`` (the typed config's
``model_type``), a raw transformers/timm repo in its ``config.json``. Each Auto class holds
a ``model_type -> class`` mapping for one task, reads the ``model_type`` off the repo, and
dispatches to that class's own ``from_weights``. So you load without naming the concrete
class:

    from zeromodels import AutoZModel, AutoZMTokenizer
    from zeromodels.auto import AutoZMDetect

    backbone  = AutoZModel.from_weights("zeromodels/bert-base-uncased")   # -> BertModel
    detector  = AutoZMDetect.from_weights("zeromodels/detr-resnet-50")    # -> DETRDetect
    detector  = AutoZMDetect.from_weights("hf:facebook/detr-resnet-50")   # -> DETRDetect
    tokenizer = AutoZMTokenizer.from_weights("zeromodels/bert-base-uncased")

The classes are named ``AutoZM*`` (``AutoZModel`` for the backbone) so they never collide
with the transformers ``AutoModel`` / ``AutoTokenizer`` / ... when both libraries are
imported. Resolution is by ``model_type`` only (never by reading the stored
``model_class``), so the same mapping serves the zeromodels-repo and ``hf:`` paths.

The mappings are the **committed, hand-maintained tables** in
:mod:`zeromodels.auto.auto_mapping_names` (transformers' ``MODEL_MAPPING_NAMES`` pattern):
``model_type -> "ClassName"`` per task, resolved to the class lazily via a name index. Edit
a table entry (or add a row for a new model) to change what an ``AutoZM*`` loads; an offline
coverage test flags a model class that is missing from the tables. Runtime
``AutoZM*.register(model_type, cls)`` overrides an entry for the session.
"""

import inspect
import json
import os

from zeromodels.auto import auto_mapping_names as _names

_HF_PREFIX = "hf:"


def read_model_type(identifier):
    """Return ``(model_type, source)`` for a repo identifier -- the autodetect core.

    ``source`` is ``"hf"`` for an ``hf:org/repo`` id (read from ``config.json``) or
    ``"zm"`` for a zeromodels ``org/repo`` id (read from ``zm_config.json``). A bare
    variant string (no ``/``) has no remote config to read, so it is rejected with a
    message pointing at the concrete class.
    """
    if identifier.startswith(_HF_PREFIX):
        from huggingface_hub import hf_hub_download

        repo = identifier[len(_HF_PREFIX) :]
        if "/" not in repo:
            raise ValueError(
                f"'{identifier}': the 'hf:' prefix expects a Hugging Face repo id of the "
                f"form 'hf:org/name' (e.g. 'hf:openai/clip-vit-base-patch16'), but got "
                f"{repo!r} with no '/'."
            )
        token = os.environ.get("HF_TOKEN")
        with open(
            hf_hub_download(repo, "config.json", token=token), encoding="utf-8"
        ) as f:
            hf_config = json.load(f)
        model_type = hf_config.get("model_type")
        if not model_type:
            raise ValueError(
                f"'{identifier}': config.json has no 'model_type', so Auto* cannot "
                f"pick a class. Load with the concrete class's from_weights('{identifier}')."
            )
        return model_type, "hf"

    if "/" in identifier:
        from zeromodels.conversion.zm_config import load_zm_config

        spec = load_zm_config(identifier.rstrip("/"))
        model_type = spec.get("model_type") if spec else None
        if not model_type:
            raise ValueError(
                f"'{identifier}': no zm_config.json with a 'model_type' was found "
                f"(only zeromodels repos and 'hf:org/repo' ids are auto-detectable). "
                f"Load with the concrete class's from_weights('{identifier}')."
            )
        return model_type, "zm"

    raise ValueError(
        f"Auto*.from_weights needs a repo id ('org/repo' or 'hf:org/repo'), got the "
        f"bare variant {identifier!r}. Auto detection reads model_type from the repo's "
        f"config; a variant has no repo. Use the concrete class, e.g. "
        f"SomeModel.from_weights({identifier!r})."
    )


_CLASS_INDEX = None


def _class_index():
    global _CLASS_INDEX
    if _CLASS_INDEX is None:
        import zeromodels.models as models

        index = {}
        for family in sorted(n for n in dir(models) if not n.startswith("_")):
            package = getattr(models, family)
            for name in getattr(package, "__all__", []):
                obj = getattr(package, name, None)
                if inspect.isclass(obj):
                    index[name] = obj
        _CLASS_INDEX = index
    return _CLASS_INDEX


def _resolve_class(class_name):
    cls = _class_index().get(class_name)
    if cls is None:
        raise ValueError(
            f"auto_mapping_names references unknown class {class_name!r}; the class was "
            f"renamed or removed. Fix the entry in zeromodels/auto/auto_mapping_names.py."
        )
    return cls


_MODEL_OVERRIDES = {}  # task -> {model_type: class}
_PREPROC_OVERRIDES = {"tokenizer": {}, "processor": {}, "image_processor": {}}
_CONFIG_OVERRIDES = {}  # model_type -> config_class


class _BaseAuto:
    """Shared ``from_weights`` / ``from_config`` dispatch for one task's mapping table."""

    task = None

    @classmethod
    def _name_table(cls):
        return _names.MODEL_TASK_MAPPING_NAMES.get(cls.task, {})

    @classmethod
    def mapping(cls):
        """The live ``{model_type: class}`` dict for this task (inspectable)."""
        table = {mt: _resolve_class(n) for mt, n in cls._name_table().items()}
        table.update(_MODEL_OVERRIDES.get(cls.task, {}))
        return table

    @classmethod
    def _resolve(cls, model_type, source="zm"):
        override = _MODEL_OVERRIDES.get(cls.task, {})
        if model_type in override:
            return override[model_type]
        name = cls._name_table().get(model_type)
        if name is not None:
            return _resolve_class(name)
        candidates = _names.AMBIGUOUS_HF_TYPES.get(cls.task, {}).get(model_type)
        if candidates:
            raise ValueError(
                f"{cls.__name__}: model_type {model_type!r} is ambiguous (loads to any "
                f"of {candidates}); resolve it by loading the concrete class directly, "
                f"e.g. {candidates[0]}.from_weights(...)."
            )
        raise ValueError(
            f"{cls.__name__} has no class registered for model_type {model_type!r}. "
            f"Supported model_type values for this task: {sorted(cls._name_table())}."
        )

    @classmethod
    def from_weights(cls, identifier, **kwargs):
        """Detect the repo's ``model_type`` and load the matching class's weights."""
        model_type, source = read_model_type(identifier)
        return cls._resolve(model_type, source).from_weights(identifier, **kwargs)

    @classmethod
    def from_config(cls, config, **kwargs):
        """Build (random-init) the class matching ``config.model_type``."""
        model_type = (
            config.get("model_type") if isinstance(config, dict) else config.model_type
        )
        if not model_type:
            raise ValueError(f"{cls.__name__}.from_config: config has no model_type.")
        return cls._resolve(model_type, "zm")(config, **kwargs)

    @classmethod
    def register(cls, model_type, model_class):
        """Override / add a ``model_type -> class`` entry for this task (this session)."""
        _MODEL_OVERRIDES.setdefault(cls.task, {})[model_type] = model_class


class _BasePreprocessorAuto:
    """Shared dispatch for tokenizer / processor / image-processor autos."""

    task = None  # "tokenizer" | "processor" | "image_processor"

    @classmethod
    def _name_table(cls):
        return {
            "tokenizer": _names.TOKENIZER_MAPPING_NAMES,
            "processor": _names.PROCESSOR_MAPPING_NAMES,
            "image_processor": _names.IMAGE_PROCESSOR_MAPPING_NAMES,
        }[cls.task]

    @classmethod
    def mapping(cls):
        table = {mt: _resolve_class(n) for mt, n in cls._name_table().items()}
        table.update(_PREPROC_OVERRIDES[cls.task])
        return table

    @classmethod
    def _resolve(cls, model_type):
        if model_type in _PREPROC_OVERRIDES[cls.task]:
            return _PREPROC_OVERRIDES[cls.task][model_type]
        name = cls._name_table().get(model_type)
        if name is None:
            raise ValueError(
                f"{cls.__name__} has no {cls.task} registered for model_type "
                f"{model_type!r}. Supported: {sorted(cls._name_table())}."
            )
        return _resolve_class(name)

    @classmethod
    def from_weights(cls, identifier, **kwargs):
        model_type, source = read_model_type(identifier)
        return cls._resolve(model_type).from_weights(identifier, **kwargs)

    @classmethod
    def register(cls, model_type, preprocessor_class):
        _PREPROC_OVERRIDES[cls.task][model_type] = preprocessor_class


class AutoZMConfig:
    """Return the typed config class (or a built config) for a repo's ``model_type``."""

    @classmethod
    def _name_table(cls):
        return _names.CONFIG_MAPPING_NAMES

    @classmethod
    def mapping(cls):
        table = {mt: _resolve_class(n) for mt, n in cls._name_table().items()}
        table.update(_CONFIG_OVERRIDES)
        return table

    @classmethod
    def for_model_type(cls, model_type):
        if model_type in _CONFIG_OVERRIDES:
            return _CONFIG_OVERRIDES[model_type]
        name = cls._name_table().get(model_type)
        if name is None:
            raise ValueError(
                f"{cls.__name__} has no config for model_type {model_type!r}. "
                f"Supported: {sorted(cls._name_table())}."
            )
        return _resolve_class(name)

    @classmethod
    def from_weights(cls, identifier, **kwargs):
        """Build the typed config instance from a zeromodels repo's ``zm_config.json``."""
        from zeromodels.conversion.zm_config import ZM_METADATA_KEYS, load_zm_config

        if identifier.startswith(_HF_PREFIX) or "/" not in identifier:
            raise ValueError(
                f"{cls.__name__}.from_weights loads a zeromodels 'org/repo' (its typed "
                "zm_config.json); for an hf: repo build the model instead."
            )
        spec = load_zm_config(identifier.rstrip("/"))
        model_type = spec.get("model_type") if spec else None
        if not model_type:
            raise ValueError(f"'{identifier}': no zm_config.json with a model_type.")
        config_cls = cls.for_model_type(model_type)
        fields = {k: v for k, v in spec.items() if k not in ZM_METADATA_KEYS}
        fields.update(kwargs)
        return config_cls.from_dict(fields)

    @classmethod
    def register(cls, model_type, config_class):
        _CONFIG_OVERRIDES[model_type] = config_class


def _task_auto_name(task):
    return "AutoZModel" if task == "Model" else f"AutoZM{task}"


def _make_task_autos():
    """Create one ``AutoZM<Task>`` class per task table, populate globals + __all__."""
    created = {}
    for task in sorted(_names.MODEL_TASK_MAPPING_NAMES):
        cls_name = _task_auto_name(task)
        auto_cls = type(
            cls_name,
            (_BaseAuto,),
            {
                "task": task,
                "__doc__": (
                    f"Auto loader for the '{task}' task: resolves a repo's model_type to "
                    f"the matching *{task} class and loads it. See :class:`_BaseAuto`."
                ),
                "__module__": __name__,
            },
        )
        globals()[cls_name] = auto_cls
        created[cls_name] = auto_cls
    return created


AutoZMTokenizer = type(
    "AutoZMTokenizer",
    (_BasePreprocessorAuto,),
    {
        "task": "tokenizer",
        "__module__": __name__,
        "__doc__": "Auto loader: model_type -> the family's tokenizer class.",
    },
)
AutoZMProcessor = type(
    "AutoZMProcessor",
    (_BasePreprocessorAuto,),
    {
        "task": "processor",
        "__module__": __name__,
        "__doc__": "Auto loader: model_type -> the family's processor class.",
    },
)
AutoZMImageProcessor = type(
    "AutoZMImageProcessor",
    (_BasePreprocessorAuto,),
    {
        "task": "image_processor",
        "__module__": __name__,
        "__doc__": "Auto loader: model_type -> the family's image-processor class.",
    },
)

_TASK_AUTOS = _make_task_autos()


def all_mappings():
    """``{auto_class_name: {model_type: class}}`` for every task auto (inspection)."""
    return {name: auto.mapping() for name, auto in sorted(_TASK_AUTOS.items())}


__all__ = [
    "read_model_type",
    "all_mappings",
    "AutoZMConfig",
    "AutoZMTokenizer",
    "AutoZMProcessor",
    "AutoZMImageProcessor",
    *sorted(_TASK_AUTOS),
]
