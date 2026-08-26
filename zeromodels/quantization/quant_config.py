import inspect

import keras


@keras.saving.register_keras_serializable(package="zeromodels")
class QuantizationConfig:
    """Declarative recipe for quantizing a model.

    Controls which layers are quantized and at what precision, so production
    setups can keep sensitive layers in float and mix precisions per layer.

    Args:
        mode: Default scheme for quantized layers, ``"int8"``, ``"int4"``,
            ``"fp8"`` or ``"mxfp4"``.
        group_size: int4 block size along the input dim (ignored by
            int8/fp8/mxfp4; MXFP4 uses fixed 32-value blocks).
        skip_modules: Tuple of name substrings; any layer whose path contains one
            is left in float. Defaults to ``("lm_head",)`` (the output head is
            accuracy-sensitive).
        quantize_embeddings: If ``False``, ``Embedding`` layers stay float.
        overrides: ``{name_substring: mode}``, per-layer precision overrides
            (checked before ``mode``), e.g. ``{"decoder_layer_0": "int8"}`` to
            keep the first block at int8 while the rest go int4.
    """

    def __init__(
        self,
        mode="int8",
        group_size=32,
        skip_modules=("lm_head",),
        quantize_embeddings=True,
        overrides=None,
    ):
        if mode not in ("int8", "int4", "fp8", "mxfp4"):
            raise ValueError(
                f"mode must be 'int8', 'int4', 'fp8' or 'mxfp4', got {mode!r}"
            )
        self.mode = mode
        self.group_size = group_size
        self.skip_modules = tuple(skip_modules)
        self.quantize_embeddings = quantize_embeddings
        self.overrides = dict(overrides or {})

    def mode_for(self, path):
        """Resolve the scheme for a layer ``path`` (``None`` -> keep float)."""
        for pattern in self.skip_modules:
            if pattern in path:
                return None
        for pattern, mode in self.overrides.items():
            if pattern in path:
                return mode
        return self.mode

    def get_config(self):
        return {
            "mode": self.mode,
            "group_size": self.group_size,
            "skip_modules": list(self.skip_modules),
            "quantize_embeddings": self.quantize_embeddings,
            "overrides": self.overrides,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def quant_method(self):
        return self.mode

    def to_dict(self):
        return {"quant_method": self.quant_method, **self.get_config()}

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        method = data.pop("quant_method", None)
        params = set(inspect.signature(cls.__init__).parameters) - {"self"}
        if method is not None and "mode" in params and "mode" not in data:
            data["mode"] = method
        return cls(**{k: v for k, v in data.items() if k in params})

    def __repr__(self):
        return (
            f"{type(self).__name__}(mode={self.mode!r}, group_size={self.group_size}, "
            f"skip_modules={self.skip_modules}, "
            f"quantize_embeddings={self.quantize_embeddings}, "
            f"overrides={self.overrides})"
        )


SCHEMES = {
    "int8": QuantizationConfig("int8"),
    "int4": QuantizationConfig("int4", group_size=32),
    "int4-g64": QuantizationConfig("int4", group_size=64),
    "int4-g128": QuantizationConfig("int4", group_size=128),
    "fp8": QuantizationConfig("fp8"),
    "mxfp4": QuantizationConfig("mxfp4"),
}


def resolve_config(spec, group_size=32):
    """Coerce ``spec`` to a :class:`QuantizationConfig`.

    ``spec`` may be a config instance, a bare mode (``"int8"`` / ``"int4"`` /
    ``"fp8"`` / ``"mxfp4"``), or a named scheme (``"int4-g128"``, ...).
    """
    if isinstance(spec, QuantizationConfig):
        return spec
    if isinstance(spec, str):
        if spec in SCHEMES:
            return SCHEMES[spec]
        if spec in ("int8", "int4", "fp8", "mxfp4"):
            return QuantizationConfig(spec, group_size=group_size)
        raise ValueError(
            f"Unknown quantization spec {spec!r}. Use a QuantizationConfig, a mode "
            f"('int8'/'int4'/'fp8'/'mxfp4'), or a scheme {sorted(SCHEMES)}."
        )
    raise TypeError(
        f"quantization spec must be a str or QuantizationConfig, got {type(spec)}"
    )


@keras.saving.register_keras_serializable(package="zeromodels")
class Int8Config(QuantizationConfig):
    """Per-channel symmetric int8 config (transformers-style per-method config)."""

    def __init__(
        self, skip_modules=("lm_head",), quantize_embeddings=True, overrides=None
    ):
        super().__init__(
            mode="int8",
            skip_modules=skip_modules,
            quantize_embeddings=quantize_embeddings,
            overrides=overrides,
        )

    def get_config(self):
        config = super().get_config()
        del config["mode"], config["group_size"]
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Int4Config(QuantizationConfig):
    """Block-wise int4 config (``group_size`` is the per-method knob)."""

    def __init__(
        self,
        group_size=32,
        skip_modules=("lm_head",),
        quantize_embeddings=True,
        overrides=None,
    ):
        super().__init__(
            mode="int4",
            group_size=group_size,
            skip_modules=skip_modules,
            quantize_embeddings=quantize_embeddings,
            overrides=overrides,
        )

    def get_config(self):
        config = super().get_config()
        del config["mode"]
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Fp8Config(QuantizationConfig):
    """float8-e4m3 config (torch / jax only)."""

    def __init__(
        self, skip_modules=("lm_head",), quantize_embeddings=True, overrides=None
    ):
        super().__init__(
            mode="fp8",
            skip_modules=skip_modules,
            quantize_embeddings=quantize_embeddings,
            overrides=overrides,
        )

    def get_config(self):
        config = super().get_config()
        del config["mode"], config["group_size"]
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Mxfp4Config(QuantizationConfig):
    """OCP MXFP4 (e2m1, fixed 32-value blocks) config.

    Applies the same 4-bit float format GPT-OSS ships to any model's Dense /
    EinsumDense / expert kernels (contracting dims must be multiples of 32).
    Embeddings stay int8, like int4.
    """

    def __init__(
        self, skip_modules=("lm_head",), quantize_embeddings=True, overrides=None
    ):
        super().__init__(
            mode="mxfp4",
            skip_modules=skip_modules,
            quantize_embeddings=quantize_embeddings,
            overrides=overrides,
        )

    def get_config(self):
        config = super().get_config()
        del config["mode"], config["group_size"]
        return config
