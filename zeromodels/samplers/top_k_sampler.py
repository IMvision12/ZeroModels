from keras import ops

from zeromodels.samplers.sampler import NEG_INF, Sampler, gumbel


class TopKSampler(Sampler):
    """Sample from the ``k`` highest-logit tokens (temperature-scaled).

    Matches Hugging Face's ``TopKLogitsWarper``: keep the ``k`` largest logits
    (``k`` clamped to the vocabulary size), push the rest to ``NEG_INF``, then draw
    with the Gumbel-max trick on the pre-supplied noise.
    """

    stochastic = True

    def __init__(self, k=50, temperature=1.0):
        self.k = int(k)
        self.temperature = float(temperature)

    def filter_logits(self, logits):
        k = min(self.k, int(logits.shape[-1]))
        kth = ops.min(ops.top_k(logits, k=k)[0], axis=-1, keepdims=True)
        return ops.where(logits < kth, ops.full_like(logits, NEG_INF), logits)

    def sample(self, logits, noise):
        logits = ops.cast(logits, "float32") / self.temperature
        masked = self.filter_logits(logits)
        return ops.cast(ops.argmax(masked + gumbel(noise), axis=-1), "int32")

    def get_config(self):
        return {"k": self.k, "temperature": self.temperature}
