from keras import ops

from zeromodels.samplers.sampler import (
    NEG_INF,
    Sampler,
    categorical,
    validate_temperature,
)


class TopKSampler(Sampler):
    """Sample from the ``k`` highest-logit tokens (temperature-scaled).

    Matches Hugging Face's ``TopKLogitsWarper``: keep the ``k`` largest logits
    (``k`` clamped to ``[1, vocab_size]`` -- HF's ``min_tokens_to_keep=1``, so a
    non-positive ``k`` falls back to greedy rather than masking everything), push the
    rest to ``NEG_INF``, then draw with an inverse-CDF categorical draw on the
    pre-supplied per-row noise. ``temperature`` must be strictly positive.
    """

    stochastic = True

    def __init__(self, k=50, temperature=1.0):
        validate_temperature(temperature)
        self.k = int(k)
        self.temperature = float(temperature)

    def filter_logits(self, logits):
        # Keep at least 1 token: a non-positive k would otherwise mask the whole
        # vocab, and sampling over an all-NEG_INF row draws a token uniformly.
        k = min(max(self.k, 1), int(logits.shape[-1]))
        kth = ops.min(ops.top_k(logits, k=k)[0], axis=-1, keepdims=True)
        return ops.where(logits < kth, ops.full_like(logits, NEG_INF), logits)

    def sample(self, logits, noise):
        logits = ops.cast(logits, "float32") / self.temperature
        return categorical(self.filter_logits(logits), noise)

    def get_config(self):
        return {"k": self.k, "temperature": self.temperature}
