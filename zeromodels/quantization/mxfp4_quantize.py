import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseQuantizer, single_axis

# OCP MXFP4 (E2M1) codebook: 4-bit index -> value. Shared with the GPT-OSS
# converter's FP4_VALUES; kept here so the runtime dequant needs no converter.
FP4_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


# Midpoints between consecutive FP4 magnitudes |FP4_VALUES| = [0,.5,1,1.5,2,3,4,6];
# a value rounds to bucket k = count of thresholds it meets or exceeds.
_FP4_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def quantize_to_mxfp4(w):
    """Quantize a float tensor to packed MXFP4 (``blocks`` + e8m0 ``scales``).

    Backend-agnostic (``keras.ops``) inverse of :func:`dequantize_mxfp4`, and the
    pure-Keras counterpart of transformers' ``quantize_to_mxfp4`` (which needs the
    GPU triton kernel). The last axis (a multiple of 32) is the packing axis: it is
    split into 32-value blocks, each given a shared e8m0 power-of-two scale
    (``e = floor(log2(amax)) - 2``, OCP MXFP4) and rounded to the nearest FP4
    (e2m1) codebook value, two 4-bit indices packed per byte (low nibble first).

    Returns ``(blocks uint8 (..., G, 16), scales uint8 (..., G))``.
    """
    w = ops.cast(w, "float32")
    *prefix, d = w.shape
    groups = d // 32
    x = ops.reshape(w, (*prefix, groups, 32))
    amax = ops.max(ops.abs(x), axis=-1)  # (..., G)
    # e8m0 shared exponent; amax == 0 -> e = 0 (block dequantizes to zero anyway).
    log2 = ops.log(ops.where(amax > 0, amax, ops.ones_like(amax))) / ops.log(2.0)
    # +1e-3 guards float32 log2 rounding at exact powers of two (lattice amax is
    # nibble * 2^k, frequently an exact power of two) so floor picks the right
    # e8m0 bucket; far below the 0.585 gap to a non-power-of-two magnitude.
    exponent = ops.where(amax > 0, ops.floor(log2 + 1e-3) - 2.0, ops.zeros_like(amax))
    exponent = ops.clip(exponent, -127.0, 127.0)
    scaled = x / ops.expand_dims(ops.power(2.0, exponent), axis=-1)  # (..., G, 32)
    thresholds = ops.convert_to_tensor(np.asarray(_FP4_THRESHOLDS, "float32"))
    magnitude = ops.sum(
        ops.cast(ops.expand_dims(ops.abs(scaled), -1) >= thresholds, "int32"), axis=-1
    )  # (..., G, 32) in 0..7
    sign = ops.cast(ops.logical_and(scaled < 0, magnitude > 0), "int32") * 8
    nibbles = ops.reshape(magnitude + sign, (*prefix, groups, 16, 2))
    blocks = ops.cast(nibbles[..., 0] + nibbles[..., 1] * 16, "uint8")
    scales = ops.cast(exponent + 127.0, "uint8")
    return blocks, scales


def dequantize_mxfp4(blocks, scales, dtype="float32"):
    """Dequantize packed MXFP4 (``blocks`` + e8m0 ``scales``) to a float tensor.

    Backend-agnostic (``keras.ops``) port of Hugging Face's
    ``convert_moe_packed_tensors``, run on the fly in the expert layer's
    ``call`` so the weight is never stored dequantized.

    ``blocks`` is uint8 of shape ``(..., G, 16)``: each byte holds two 4-bit
    codebook indices, low nibble first. ``scales`` is uint8 ``(..., G)``: one
    e8m0 (power-of-two) exponent per 32-value block, biased by 127. Returns
    ``(..., G * 32)`` with the two nibbles interleaved (low, high), i.e. the
    per-block layout before any axis transpose.
    """
    table = ops.cast(ops.convert_to_tensor(np.asarray(FP4_VALUES, "float32")), dtype)
    packed = ops.cast(blocks, "int32")
    low = ops.take(table, ops.bitwise_and(packed, 0x0F), axis=0)  # (..., G, 16)
    high = ops.take(table, ops.right_shift(packed, 4), axis=0)  # (..., G, 16)
    values = ops.stack([low, high], axis=-1)  # (..., G, 16, 2): [lo, hi] per byte
    *prefix, groups, _, _ = values.shape
    values = ops.reshape(values, (*prefix, groups, 32))  # interleaved lo, hi
    exponent = ops.cast(scales, "float32") - 127.0
    scale = ops.cast(ops.expand_dims(ops.power(2.0, exponent), axis=-1), dtype)
    out = values * scale
    return ops.reshape(out, (*prefix, groups * 32))


@keras.saving.register_keras_serializable(package="zeromodels")
class MXFP4Quantizer(BaseQuantizer):
    """OCP MXFP4 (e2m1) weight-only quantizer (~4x smaller than bf16).

    The contracting ``axis`` is split into fixed 32-value blocks; each block gets
    a shared e8m0 (power-of-two) ``scale`` and its values are rounded to the FP4
    (e2m1) codebook, two 4-bit indices packed per uint8 byte. This is the format
    OpenAI ships GPT-OSS in; as a general ``quantize_model`` scheme it applies to
    any ``Dense`` / ``EinsumDense`` / fused-expert kernel whose contracting
    dimension is a multiple of 32. Like int4 it packs a **single** axis: the axis
    is moved to the end, packed there, then moved back, so the same code serves
    2-D ``Dense`` kernels (``axis=0``), N-D ``EinsumDense`` kernels, and MoE
    expert banks (``axis=-1``). The ``scale`` is uint8 (e8m0), not float. Reuses
    :func:`quantize_to_mxfp4` / :func:`dequantize_mxfp4`; backend-agnostic.
    """

    mode = "mxfp4"
    BLOCK = 32

    def quantize(self, weight, axis=0):
        weight = ops.convert_to_tensor(weight)
        axis = single_axis(axis, len(weight.shape))
        self._check_axis(weight.shape, axis)
        w = ops.moveaxis(weight, axis, -1)  # (..., K)
        blocks, scales = quantize_to_mxfp4(w)  # (..., G, 16), (..., G)
        prefix = [int(s) for s in blocks.shape[:-2]]
        groups = int(blocks.shape[-2])
        packed = ops.reshape(blocks, (*prefix, groups * 16))  # (..., K // 2)
        return ops.moveaxis(packed, -1, axis), ops.moveaxis(scales, -1, axis)

    def dequantize(self, packed, scale, axis=0, dtype=None):
        dtype = dtype or "float32"
        axis = single_axis(axis, len(packed.shape))
        p = ops.moveaxis(packed, axis, -1)  # (..., K // 2)
        s = ops.moveaxis(scale, axis, -1)  # (..., G)
        prefix = [int(x) for x in p.shape[:-1]]
        groups = int(p.shape[-1]) // 16
        blocks = ops.reshape(p, (*prefix, groups, 16))
        deq = dequantize_mxfp4(blocks, s, dtype)  # (..., K)
        return ops.moveaxis(deq, -1, axis)

    def storage_spec(self, weight_shape, axis=0):
        axis = single_axis(axis, len(weight_shape))
        self._check_axis(weight_shape, axis)
        k = int(weight_shape[axis])
        kernel_shape = tuple(
            k // 2 if i == axis else d for i, d in enumerate(weight_shape)
        )
        scale_shape = tuple(
            k // self.BLOCK if i == axis else d for i, d in enumerate(weight_shape)
        )
        return {"kernel": (kernel_shape, "uint8"), "scale": (scale_shape, "uint8")}

    def _check_axis(self, weight_shape, axis):
        k = int(weight_shape[axis])
        if k % self.BLOCK:
            raise ValueError(
                f"MXFP4 packs fixed {self.BLOCK}-value blocks, so the contracting "
                f"axis (dimension {k}) must be a multiple of {self.BLOCK}. Exclude "
                f"this layer with a QuantizationConfig skip pattern, or use int4."
            )

    def get_config(self):
        return {}
