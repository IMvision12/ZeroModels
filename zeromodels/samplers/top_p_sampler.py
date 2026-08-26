from keras import ops

from zeromodels.samplers.sampler import NEG_INF, Sampler, gumbel


class TopPSampler(Sampler):
    """Nucleus sampling: the smallest set of top tokens with cumulative prob >= ``p``.

    Matches Hugging Face's ``TopPLogitsWarper`` (with its default
    ``min_tokens_to_keep=1``): sort by probability, keep tokens while the prefix
    *before* them holds less than ``p`` of the mass, which always retains the
    top-1 and the token that crosses ``p``: push the rest to ``NEG_INF``, then
    draw with the Gumbel-max trick on the pre-supplied noise.
    """

    stochastic = True

    def __init__(self, p=0.9, temperature=1.0):
        self.p = float(p)
        self.temperature = float(temperature)

    def filter_logits(self, logits):
        order = ops.argsort(-logits, axis=-1)  # descending
        sorted_logits = ops.take_along_axis(logits, order, axis=-1)
        probs = ops.softmax(sorted_logits, axis=-1)
        cumulative = ops.cumsum(probs, axis=-1) - probs  # exclusive prefix
        keep_sorted = cumulative < self.p  # nucleus (sorted order)
        inverse = ops.argsort(order, axis=-1)  # scatter back to vocab order
        keep = ops.take_along_axis(keep_sorted, inverse, axis=-1)
        return ops.where(keep, logits, ops.full_like(logits, NEG_INF))

    def sample(self, logits, noise):
        logits = ops.cast(logits, "float32") / self.temperature
        masked = self.filter_logits(logits)
        return ops.cast(ops.argmax(masked + gumbel(noise), axis=-1), "int32")

    def get_config(self):
        return {"p": self.p, "temperature": self.temperature}
