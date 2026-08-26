import keras
from keras import ops

from zeromodels.base import BaseQuantizer, single_axis

INT4_MAX = 7


def effective_group_size(in_dim, group_size):
    """Largest block size <= ``group_size`` dividing ``in_dim`` (falls back to a
    whole-axis block when ``group_size`` does not divide ``in_dim``)."""
    if in_dim % group_size == 0:
        return group_size
    for g in range(group_size, 0, -1):
        if in_dim % g == 0:
            return g
    return 1


@keras.saving.register_keras_serializable(package="zeromodels")
class Int4Quantizer(BaseQuantizer):
    """Block-wise symmetric int4 weight-only quantizer (~8x smaller).

    The contracting ``axis`` is split into blocks of ``group_size``; each
    (block, ...) gets its own absmax scale (the bitsandbytes block idea) and
    values are packed two-per-byte. Works on any rank: the axis is moved to the
    end, quantized/packed there, then moved back, so the same code serves 2-D
    ``Dense`` kernels (``axis=0``), N-D ``EinsumDense`` kernels, and fused MoE
    expert banks (``axis=-1``). Backend-agnostic (pure ``keras.ops``).
    """

    mode = "int4"

    def __init__(self, group_size=32):
        self.group_size = group_size

    def quantize(self, weight, axis=0):
        weight = ops.convert_to_tensor(weight)
        axis = single_axis(axis, len(weight.shape))
        w = ops.moveaxis(weight, axis, -1)
        packed, scale = self._quant_last(w)
        return ops.moveaxis(packed, -1, axis), ops.moveaxis(scale, -1, axis)

    def dequantize(self, packed, scale, axis=0, dtype=None):
        dtype = dtype or scale.dtype
        axis = single_axis(axis, len(packed.shape))
        p = ops.moveaxis(packed, axis, -1)
        s = ops.moveaxis(scale, axis, -1)
        deq = self._dequant_last(p, s, dtype)
        return ops.moveaxis(deq, -1, axis)

    def storage_spec(self, weight_shape, axis=0):
        axis = single_axis(axis, len(weight_shape))
        k = weight_shape[axis]
        eff = effective_group_size(k, self.group_size)
        kernel_shape = tuple(
            k // 2 if i == axis else d for i, d in enumerate(weight_shape)
        )
        scale_shape = tuple(
            k // eff if i == axis else d for i, d in enumerate(weight_shape)
        )
        return {"kernel": (kernel_shape, "uint8"), "scale": (scale_shape, "float32")}

    def _quant_last(self, w):
        shape = [int(s) for s in w.shape]
        leading, k = shape[:-1], shape[-1]
        if k % 2 != 0:
            raise ValueError(
                f"int4 needs an even contracting dim to pack two-per-byte, got {k}."
            )
        eff = effective_group_size(k, self.group_size)
        n_groups = k // eff
        wg = ops.reshape(w, (*leading, n_groups, eff))
        amax = ops.max(ops.abs(wg), axis=-1, keepdims=True)
        scale = ops.maximum(amax / INT4_MAX, keras.config.epsilon())
        q = ops.clip(ops.round(wg / scale), -INT4_MAX, INT4_MAX)
        q = ops.reshape(q, (*leading, k))
        packed = self._pack_last(q, leading, k)
        return packed, ops.cast(ops.squeeze(scale, -1), "float32")

    def _dequant_last(self, packed, scale, dtype):
        shape = [int(s) for s in packed.shape]
        leading, half = shape[:-1], shape[-1]
        k = half * 2
        eff = effective_group_size(k, self.group_size)
        n_groups = k // eff
        q = self._unpack_last(packed, leading, k)
        qg = ops.reshape(ops.cast(q, dtype), (*leading, n_groups, eff))
        deq = qg * ops.cast(ops.expand_dims(scale, -1), dtype)
        return ops.reshape(deq, (*leading, k))

    @staticmethod
    def _pack_last(q, leading, k):
        qu = ops.mod(ops.cast(q, "int32"), 16)
        r = ops.reshape(qu, (*leading, k // 2, 2))
        low = ops.take(r, 0, axis=-1)
        high = ops.take(r, 1, axis=-1)
        return ops.cast(low + high * 16, "uint8")

    @staticmethod
    def _unpack_last(packed, leading, k):
        p = ops.cast(packed, "int32")
        low = ops.mod(p, 16)
        high = ops.floor_divide(p, 16)
        low = ops.where(low > 7, low - 16, low)
        high = ops.where(high > 7, high - 16, high)
        r = ops.stack([low, high], axis=-1)
        return ops.reshape(r, (*leading, k))

    def get_config(self):
        return {"group_size": self.group_size}
