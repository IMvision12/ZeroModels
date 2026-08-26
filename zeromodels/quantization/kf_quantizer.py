"""Model-level quantizers, the zeromodels analog of transformers' ``HfQuantizer``.

A :class:`KfQuantizer` reads a repo's ``quantization_config`` and prepares a
*quantization-agnostic* model for a quantized checkpoint: it swaps the relevant
float modules for their packed equivalents **before** the weights load, mirroring
``HfQuantizer._process_model_before_weight_loading``. The model classes therefore
carry no quantization flags; the quantizer is the single thing that produces packed
layers, on load.

Dispatch is a plain :func:`get_kf_quantizer` (keyed on ``quant_method``); there is no
``Auto`` registry yet. The methods are ``mxfp4`` (GPT-OSS native packed experts) and the
generic weight-only ``int8`` / ``int4`` / ``fp8``.
"""

from zeromodels.quantization.quantize import _named_children, _swap, _walk_layers
from zeromodels.quantization.quantized_layers import GptOssMXFP4Experts


class KfQuantizer:
    """Base class: prepare a float model to receive a natively-quantized checkpoint.

    Subclasses implement :meth:`_process_model_before_weight_loading` (the module
    swap). ``preprocess_model`` runs it and stamps ``model._quantization_config`` so
    the model serializes and reloads itself quantized (see the base model's
    ``get_config`` / ``from_config``).
    """

    quant_method = None
    requires_parameters_quantization = False

    def __init__(self, quantization_config):
        self.quantization_config = dict(quantization_config)

    def validate_environment(self, **kwargs):
        """Check backend / hardware support. Default: no requirements."""

    def update_load_dtype(self, load_dtype):
        """Adjust the build dtype for this scheme. Default: leave it unchanged."""
        return load_dtype

    def preprocess_model(self, model, **kwargs):
        model = self._process_model_before_weight_loading(model, **kwargs)
        # Stamp the model so it serializes + reloads itself quantized. Some paths
        # (quantize_model) already set a QuantizationConfig; keep it if so.
        if getattr(model, "_quantization_config", None) is None:
            model._quantization_config = dict(self.quantization_config)
        return model

    def postprocess_model(self, model, **kwargs):
        return self._process_model_after_weight_loading(model, **kwargs)

    def _process_model_before_weight_loading(self, model, **kwargs):
        raise NotImplementedError

    def _process_model_after_weight_loading(self, model, **kwargs):
        return model


class Mxfp4KfQuantizer(KfQuantizer):
    """Swap GPT-OSS float expert banks for the packed ``GptOssMXFP4Experts``.

    Mirrors transformers' ``Mxfp4HfQuantizer``: the model builds plain
    ``GptOssExperts``; this replaces each with the MXFP4 bank (uint8 nibble blocks +
    e8m0 scales) before the packed checkpoint loads. The float bank is detected by
    class name (``"GptOssExperts"``) to avoid a circular import back into the model.
    """

    quant_method = "mxfp4"

    def _process_model_before_weight_loading(self, model, **kwargs):
        swaps = []
        for parent in _walk_layers(model):
            for name, child in _named_children(parent).items():
                if name.startswith("_"):
                    continue
                if type(child).__name__ == "GptOssExperts":
                    swaps.append((parent, name, child))
        for parent, name, child in swaps:
            packed = GptOssMXFP4Experts(
                child.num_experts,
                child.embed_dim,
                child.mlp_dim,
                num_experts_per_tok=getattr(parent, "num_experts_per_tok", 4),
                name=name,
            )
            _swap(parent, name, child, packed)
        return model


class WeightOnlyKfQuantizer(KfQuantizer):
    """int8 / int4 / fp8 weight-only quantization, applied before the weights load.

    Swaps every eligible ``Dense`` / ``Embedding`` for its int / fp8 quantized layer
    so a repo whose ``quantization_config`` names one of these schemes builds integer
    storage the packed checkpoint loads into. Delegates to ``quantize_model``, which
    **clones** a functional model (its graph can't be swapped in place) and returns
    the new model, so callers must use the returned value. Generic across models
    (unlike the GPT-OSS-specific mxfp4 expert swap).
    """

    def _process_model_before_weight_loading(self, model, **kwargs):
        from zeromodels.quantization.quantize import quantize_model

        return quantize_model(model, self.quantization_config["quant_method"])


# quant_method -> KfQuantizer. mxfp4 is the GPT-OSS native packed-expert swap; the
# int/fp8 schemes are the generic weight-only path. (No Auto registry yet.)
_KF_QUANTIZERS = {
    "mxfp4": Mxfp4KfQuantizer,
    "int8": WeightOnlyKfQuantizer,
    "int4": WeightOnlyKfQuantizer,
    "fp8": WeightOnlyKfQuantizer,
}


def get_kf_quantizer(quantization_config):
    """Return a :class:`KfQuantizer` for a ``quantization_config`` block, or None.

    ``quantization_config`` is a ``{"quant_method": ...}`` dict (from a repo's
    kf_config.json or an upstream config.json). Returns None when it is empty.
    """
    if not quantization_config:
        return None
    method = quantization_config.get("quant_method")
    quantizer_cls = _KF_QUANTIZERS.get(method)
    if quantizer_cls is None:
        raise ValueError(
            f"no KfQuantizer registered for quant_method={method!r} "
            f"(known: {sorted(_KF_QUANTIZERS)})"
        )
    return quantizer_cls(quantization_config)
