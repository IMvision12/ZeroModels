from keras import ops

# Rejected tokens are pushed here rather than to -inf: finite, so their softmax
# probability underflows cleanly to 0 (the inverse-CDF draw can never pick them)
# without the NaNs a true -inf would risk.
NEG_INF = -1e9


class Sampler:
    """Maps logits ``(batch, vocab)`` + per-step uniform ``noise`` to next ids.

    ``stochastic`` tells :class:`BaseGeneration` whether to pre-compute random noise for
    the whole decode (*outside* the compiled loop, via a single ``SeedGenerator``).
    Greedy ignores the noise; stochastic samplers turn it into a token with an
    inverse-CDF categorical draw (:func:`categorical`), which needs only **one uniform
    per row** -- so the pre-drawn noise is ``(steps, batch)``, not
    ``(steps, batch, vocab)`` -- and runs no RNG inside the fused ``while_loop``. (An
    explicit ``seed`` is reproducible per backend; ``keras.random`` is backend-specific,
    so the draw is not identical across TF / JAX / Torch, though greedy is.)

    ``filter_logits`` is the candidate-restriction step (the analog of a
    Hugging Face ``LogitsWarper``): it returns logits with the rejected tokens
    pushed to ``NEG_INF`` and the kept tokens untouched. The base implementation
    keeps everything (greedy); ``TopKSampler`` / ``TopPSampler`` override it. It is
    split out from ``sample`` so the kept set can be compared against the reference
    warpers without drawing a token.
    """

    stochastic = False

    def sample(self, logits, noise):
        raise NotImplementedError(f"{type(self).__name__} must implement sample().")

    def filter_logits(self, logits):
        return logits

    def get_config(self):
        return {}


def gumbel(noise):
    # uniform(0, 1) -> Gumbel(0, 1)
    u = ops.clip(noise, 1e-9, 1.0)
    return -ops.log(-ops.log(u))


def categorical(masked_logits, noise):
    """Inverse-CDF categorical draw over ``softmax(masked_logits)``.

    ``masked_logits`` is ``(batch, vocab)`` (rejected tokens at :data:`NEG_INF`);
    ``noise`` is one uniform per row, ``(batch,)``. Returns ``(batch,)`` int32 ids.
    Needs only a single uniform per row, so the decode engine can pre-draw
    ``(steps, batch)`` noise instead of ``(steps, batch, vocab)``.

    The CDF is renormalised to end at exactly 1.0 and the uniform is clipped to
    ``[1e-9, 1.0]``, so the draw can never land on a rejected token at the
    ``u -> 0`` / ``u -> 1`` edges (the CDF only steps up on kept tokens, so the first
    index whose CDF reaches ``u`` is always a kept token).
    """
    probs = ops.softmax(ops.cast(masked_logits, "float32"), axis=-1)
    cdf = ops.cumsum(probs, axis=-1)
    cdf = cdf / cdf[..., -1:]
    u = ops.clip(ops.cast(noise, "float32"), 1e-9, 1.0)[..., None]
    idx = ops.sum(ops.cast(cdf < u, "int32"), axis=-1)
    return ops.cast(ops.minimum(idx, masked_logits.shape[-1] - 1), "int32")


def validate_temperature(temperature):
    # A non-positive temperature turns logits into +/-inf (and 0/0 -> NaN), which
    # would silently corrupt the draw; reject it up front like HF's
    # TemperatureLogitsWarper. NaN fails ``> 0`` too.
    if not (float(temperature) > 0.0):
        raise ValueError(
            f"temperature must be a strictly positive float, got {temperature!r}. "
            "For greedy decoding use GreedySampler()."
        )
