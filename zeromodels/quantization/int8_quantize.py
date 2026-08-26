import keras
from keras import ops

from zeromodels.base import BaseQuantizer, normalize_axes

INT8_MAX = 127


@keras.saving.register_keras_serializable(package="zeromodels")
class Int8Quantizer(BaseQuantizer):
    """Per-channel symmetric int8 weight-only quantizer (~4x smaller).

    Absmax symmetric quantization with one float32 scale per output channel
    (``scale = max|w| / 127`` over the contracting ``axis``, no zero point). This
    is vector-wise per-channel int8, *not* LLM.int8(), which additionally keeps
    activation-outlier columns in fp16; on very large models per-channel int8 can
    degrade from weight outliers (group-wise int4, or keeping a layer in float
    via the config, is the lever). Backend-agnostic (pure ``keras.ops``); ``axis``
    may be an int or a tuple (N-D ``EinsumDense`` kernels).
    """

    mode = "int8"

    def quantize(self, weight, axis=0):
        weight = ops.convert_to_tensor(weight)
        amax = ops.max(ops.abs(weight), axis=axis, keepdims=True)
        scale = ops.maximum(amax / INT8_MAX, keras.config.epsilon())
        q = ops.clip(ops.round(weight / scale), -INT8_MAX, INT8_MAX)
        return ops.cast(q, "int8"), ops.cast(scale, "float32")

    def dequantize(self, packed, scale, axis=0, dtype=None):
        dtype = dtype or scale.dtype
        return ops.cast(packed, dtype) * ops.cast(scale, dtype)

    def storage_spec(self, weight_shape, axis=0):
        axes = normalize_axes(axis, len(weight_shape))
        scale_shape = tuple(1 if i in axes else d for i, d in enumerate(weight_shape))
        return {
            "kernel": (tuple(weight_shape), "int8"),
            "scale": (scale_shape, "float32"),
        }
