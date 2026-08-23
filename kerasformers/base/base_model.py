import keras

from kerasformers.base.base_config import BaseConfig
from kerasformers.base.base_mixin import WeightLoadingMixin, inference_scope

_KerasModelMeta = type(keras.Model)


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
    """Canonical kerasformers model base.

    Models build themselves functionally with
    ``super().__init__(inputs=..., outputs=...)``. This is the peer of
    :class:`BaseConfig` / :class:`BaseTokenizer` / :class:`BaseProcessor` and the
    single base every model migrates onto (formerly ``FunctionalBaseModel``).
    Generative families that still need an imperative KV-cache decode temporarily
    remain on :class:`SubclassedBaseModel` until they move to a functional
    backbone + task-side cached decode.
    """


class SubclassedBaseModel(WeightLoadingMixin, keras.Model, metaclass=_ConfigModelMeta):
    """Base for *imperative / subclassed* kerasformers models (Qwen LLMs & VLMs).

    Deliberately a **separate** ``keras.Model`` subclass from :class:`BaseModel`,
    not a subclass of it. When a functional model is built, Keras runs
    ``inject_functional_model_class`` and rewrites the functional base's
    ``__bases__`` from ``keras.Model`` to ``Functional`` (functional models rely
    on this for their subsequent builds). If subclassed models shared that base,
    ``Functional`` would leak into their MRO too, making Keras treat them as
    functional and fail with ``Functional.__init__() missing ... 'inputs' and
    'outputs'`` on construction, or ``'<Model>' object has no attribute
    '_inputs'`` on call. A separate base keeps subclassed models unaffected.
    """

    def get_config(self):
        """Config for keras serialization, carrying any applied quantization.

        A no-op for unquantized models. When a :class:`KfQuantizer` has run (via
        ``from_weights`` or a prior load), ``model._quantization_config`` is stamped
        into the config so a plain ``.keras`` save/reload rebuilds itself quantized
        (see :meth:`from_config`), the way Keras carries a quantized dtype policy.
        """
        config = super().get_config()
        qc = getattr(self, "_quantization_config", None)
        if qc is not None:
            # ``_quantization_config`` is a QuantizationConfig (weight-only subsystem)
            # or a plain {"quant_method": ...} dict (native, e.g. mxfp4). Normalize.
            config["quantization_config"] = (
                qc.to_dict() if hasattr(qc, "to_dict") else dict(qc)
            )
        return config

    @classmethod
    def from_config(cls, config):
        """Rebuild from config, re-applying the quantizer when the block is present.

        Builds the plain (quantization-agnostic) model, then runs the matching
        :class:`KfQuantizer` to swap in the packed layers before weights load.
        """
        config = dict(config)
        quantization_config = config.pop("quantization_config", None)
        model = super().from_config(config)
        if quantization_config:
            from kerasformers.quantization import get_kf_quantizer

            get_kf_quantizer(quantization_config).preprocess_model(model)
        return model

    def build_for_transfer(self):
        """Build every sublayer with a dummy forward, ready for a weight stream.

        Subclassed models build lazily on first call, so nothing has weights
        until they run once. The converted-weight cache reloads a model from a
        serialized *config* (an unbuilt skeleton) and then streams cached tensors
        onto ``self.weights``, which requires the weights to exist first. This
        runs the minimal forward that materializes them: a length-4 ``input_ids``
        batch, the text-LLM signature. Non-text subclassed models (VLMs, ASR)
        override with a signature-matching dummy input.

        Runs under :func:`inference_scope` so the build forward is graph-free on
        torch (an MXFP4 build would otherwise retain each layer's dequantized
        experts and OOM a large checkpoint).
        """
        with inference_scope():
            self({"input_ids": keras.ops.zeros((1, 4), dtype="int32")})
