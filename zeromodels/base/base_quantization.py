def normalize_axes(axis, ndim):
    """Normalize an int or tuple ``axis`` to a sorted tuple of positive axes."""
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(sorted(a % ndim for a in axis))


def single_axis(axis, ndim):
    """Resolve ``axis`` to one positive axis; error if it names more than one.

    Used by packed schemes (int4) that can only contract along a single axis.
    """
    axes = normalize_axes(axis, ndim)
    if len(axes) != 1:
        raise ValueError(
            f"this quantizer contracts a single axis, got {axis} ({len(axes)} axes)."
        )
    return axes[0]


class BaseQuantizer:
    """Base class for zeromodels weight-only quantizers.

    A quantizer compresses a float weight along its **contracting axis**: the
    axis (or axes) summed over in the owning layer's matmul / einsum, and
    reconstructs it on the fly. Because ``axis`` is explicit, one quantizer
    serves 2-D ``Dense`` kernels (``axis=0``), N-D ``EinsumDense`` kernels (axis
    derived from the equation), per-row embeddings (``axis=1``), and fused MoE
    expert banks (``axis=-1``).

    Contract:

    - :meth:`quantize` ``(weight, axis) -> (packed, scale)``
    - :meth:`dequantize` ``(packed, scale, axis, dtype) -> float weight``, pass
      the activation/compute ``dtype`` so mixed-precision graphs don't upcast.
    - :meth:`storage_spec` ``(weight_shape, axis) -> {"kernel": (shape, dtype),
      "scale": (shape, dtype)}`` so a layer can pre-create the stored weights
      before loading without ever materializing the float kernel.

    Follows the keras ``get_config`` / ``from_config`` contract for
    serializability. Subclasses: ``Int8Quantizer`` / ``Int4Quantizer`` /
    ``Fp8Quantizer``.
    """

    mode = None

    def quantize(self, weight, axis=0):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement quantize()."
        )

    def dequantize(self, packed, scale, axis=0, dtype=None):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement dequantize()."
        )

    def storage_spec(self, weight_shape, axis=0):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement storage_spec()."
        )

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
