import inspect

import keras

from zeromodels.base.base_config import BaseConfig
from zeromodels.base.base_mixin import WeightLoadingMixin

_KerasModelMeta = type(keras.Model)


def _keras_init_kwargs():
    """Constructor kwargs the keras base legitimately accepts.

    Everything a model forwards to ``super().__init__`` that is not one of these
    (``inputs`` / ``outputs`` / ``name`` / ``dtype`` / ...) is an unrecognized
    argument. Introspected from keras so it tracks the installed version.
    """
    allowed = {"inputs", "outputs", "name", "trainable", "dtype", "autocast"}
    for base in (keras.Model, keras.layers.Layer):
        try:
            params = inspect.signature(base.__init__).parameters
        except (TypeError, ValueError):
            continue
        for pname, p in params.items():
            if pname == "self" or p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            allowed.add(pname)
    return frozenset(allowed)


_KERAS_INIT_KWARGS = _keras_init_kwargs()


class _ConfigModelMeta(_KerasModelMeta):
    """Metaclass giving config-carrying models the transformers ``Model(config)`` API.

    Transforms a :class:`BaseConfig` passed as the first positional into the
    matching constructor kwargs *before* ``__init__`` runs (so the real
    ``__init__`` signature stays intact for Keras's own argspec introspection),
    with explicit kwargs still winning, and records ``self.config`` for both the
    config-object and the plain-kwargs construction paths. A no-op for models
    without a ``config_class``.
    """

    def __call__(cls, *args, **kwargs):
        config_obj = None
        overridden = False
        if args and isinstance(args[0], BaseConfig):
            config_obj = args[0]
            overridden = bool(kwargs)
            merged = dict(config_obj.constructor_kwargs())
            merged.update(kwargs)
            args, kwargs = (), merged
        obj = super().__call__(*args, **kwargs)
        config_cls = getattr(cls, "config_class", None)
        if config_cls is not None:
            if config_obj is not None and not overridden:
                obj.config = config_obj  # keep identity: model.config is config
            else:
                obj.config = config_cls.from_dict(kwargs)
        return obj


def strip_functional_graph_keys(config):
    """Drop keras functional-graph keys from a ``get_config`` dict.

    A model whose ``__init__`` takes ``*args`` / ``**kwargs`` trips keras's
    ``functional_like_constructor`` check, so ``super().get_config()`` returns the
    functional graph (``layers`` / ``input_layers`` / ``output_layers``) instead of the
    plain ``name`` / ``dtype`` config. Classes that rebuild via ``cls(**config)`` from
    their own fields must drop these keys so they never reach ``__init__``.
    """
    for key in ("input_layers", "output_layers", "layers"):
        config.pop(key, None)
    return config


def hf_num_classes(hf_config):
    """Derive the class count from a ``config.json`` dict.

    A serialized ``config.json`` typically stores ``id2label`` rather than a
    direct count, so this helper derives it from whichever of ``num_labels`` /
    ``id2label`` / ``label2id`` is present.
    """
    if "num_labels" in hf_config:
        return hf_config["num_labels"]
    id2label = hf_config.get("id2label")
    if id2label:
        return len(id2label)
    label2id = hf_config.get("label2id")
    if label2id:
        return len(label2id)
    raise KeyError(
        "Could not determine num_labels from HF config.json: "
        "neither 'num_labels' nor 'id2label' / 'label2id' is present."
    )


class BaseModel(WeightLoadingMixin, keras.Model, metaclass=_ConfigModelMeta):
    """Canonical zeromodels model base.

    Models build themselves functionally with
    ``super().__init__(inputs=..., outputs=...)``. This is the peer of
    :class:`BaseConfig` / :class:`BaseTokenizer` / :class:`BaseProcessor` and the
    single ``keras.Model`` base every model builds on (formerly
    ``FunctionalBaseModel``). Generative families keep their imperative KV-cache
    decode on the task side (a ``BaseGeneration`` mixin) over this functional
    backbone.
    """

    def __init__(self, *args, **kwargs):
        unexpected = sorted(k for k in kwargs if k not in _KERAS_INIT_KWARGS)
        if unexpected:
            raise TypeError(
                f"{type(self).__name__}() got unexpected keyword argument(s): "
                f"{', '.join(unexpected)}"
            )
        super().__init__(*args, **kwargs)

    def get_config(self):
        """Config for keras serialization, carrying any applied quantization.

        A no-op for unquantized models. When a :class:`ZmQuantizer` has run, the
        recipe is stamped into the config so a plain ``.keras`` save/reload
        rebuilds itself quantized (see :meth:`from_config`). Subclasses that
        override ``get_config`` call ``super().get_config()``, so they inherit
        this.
        """
        config = super().get_config()
        qc = getattr(self, "_quantization_config", None)
        if qc is not None:
            config["quantization_config"] = (
                qc.to_dict() if hasattr(qc, "to_dict") else dict(qc)
            )
        return config

    @classmethod
    def from_config(cls, config):
        """Rebuild from config, re-applying the quantizer when the block is present."""
        config = dict(config)
        quantization_config = config.pop("quantization_config", None)
        model = super().from_config(config)
        if quantization_config:
            from zeromodels.quantization import get_zm_quantizer

            get_zm_quantizer(quantization_config).preprocess_model(model)
        return model
