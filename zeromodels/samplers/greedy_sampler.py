import keras
from keras import ops

from zeromodels.samplers.sampler import Sampler


@keras.saving.register_keras_serializable(package="zeromodels")
class GreedySampler(Sampler):
    """Deterministic argmax: the default decoding strategy."""

    stochastic = False

    def sample(self, logits, noise):
        return ops.cast(ops.argmax(logits, axis=-1), "int32")
