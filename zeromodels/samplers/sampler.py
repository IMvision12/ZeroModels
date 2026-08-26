from keras import ops

# Logits driven this far below the kept set never win the argmax draw (and, unlike
# -inf, stay finite so ``masked + gumbel(noise)`` cannot produce NaNs).
NEG_INF = -1e9


class Sampler:
    """Maps logits ``(batch, vocab)`` + per-step uniform ``noise`` to next ids.

    ``stochastic`` tells :class:`BaseGeneration` whether to pre-compute random noise for
    the whole decode (cross-backend, *outside* the compiled loop, via a single
    ``SeedGenerator``). Greedy ignores the noise; stochastic samplers turn it into
    a draw with the Gumbel-max trick (``argmax(masked_logits + gumbel(noise))``),
    so no RNG runs inside the fused ``while_loop`` and the result is identical on
    TF / JAX / Torch.

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
