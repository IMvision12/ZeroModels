from keras import ops

from zeromodels.samplers.sampler import Sampler


class GreedySampler(Sampler):
    """Deterministic argmax: the default decoding strategy."""

    stochastic = False

    def sample(self, logits, noise):
        return ops.cast(ops.argmax(logits, axis=-1), "int32")
