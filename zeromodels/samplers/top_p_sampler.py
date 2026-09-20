import keras
from keras import ops

from zeromodels.samplers.sampler import (
    NEG_INF,
    Sampler,
    categorical,
    validate_temperature,
)


@keras.saving.register_keras_serializable(package="zeromodels")
class TopPSampler(Sampler):
    """Nucleus sampling: the smallest set of top tokens with cumulative prob >= ``p``.

    Matches Hugging Face's ``TopPLogitsWarper``: sort by probability, keep tokens
    while the prefix *before* them holds less than ``p`` of the mass (so the token
    that crosses ``p`` is kept too), push the rest to ``NEG_INF``, then draw with an
    inverse-CDF categorical draw on the pre-supplied per-row noise. At least
    ``min_tokens_to_keep`` top tokens are always kept.

    Args are validated up front, because an out-of-range value would not raise on its
    own -- it would silently decode with a different strategy than requested. ``p``
    must be in ``(0, 1]``, ``min_tokens_to_keep`` must be ``>= 1``, and
    ``temperature`` must be strictly positive.
    """

    stochastic = True

    def __init__(self, p=0.9, temperature=1.0, min_tokens_to_keep=1):
        validate_temperature(temperature)
        if not (0.0 < float(p) <= 1.0):
            raise ValueError(
                f"p must be in the range (0, 1], got {p!r}. Use p=1.0 to sample from "
                "the full distribution, or GreedySampler() for greedy decoding."
            )
        if int(min_tokens_to_keep) < 1:
            raise ValueError(
                f"min_tokens_to_keep must be >= 1, got {min_tokens_to_keep!r}."
            )
        self.p = float(p)
        self.temperature = float(temperature)
        self.min_tokens_to_keep = int(min_tokens_to_keep)

    def filter_logits(self, logits):
        order = ops.argsort(-logits, axis=-1)  # descending
        sorted_logits = ops.take_along_axis(logits, order, axis=-1)
        probs = ops.softmax(sorted_logits, axis=-1)
        cumulative = ops.cumsum(probs, axis=-1) - probs  # exclusive prefix
        keep_sorted = cumulative < self.p
        floor = ops.arange(logits.shape[-1]) < self.min_tokens_to_keep
        keep_sorted = ops.logical_or(keep_sorted, floor[None, :])
        inverse = ops.argsort(order, axis=-1)  # scatter back to vocab order
        keep = ops.take_along_axis(keep_sorted, inverse, axis=-1)
        return ops.where(keep, logits, ops.full_like(logits, NEG_INF))

    def sample(self, logits, noise):
        logits = ops.cast(logits, "float32") / self.temperature
        return categorical(self.filter_logits(logits), noise)

    def get_config(self):
        return {
            "p": self.p,
            "temperature": self.temperature,
            "min_tokens_to_keep": self.min_tokens_to_keep,
        }
