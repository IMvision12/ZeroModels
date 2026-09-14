import inspect

import numpy as np
from keras import ops
from keras import random as keras_random


def linspace_float32(start, end, num):
    """``torch.linspace(start, end, num, dtype=float32)`` reproduced in numpy.

    The reference schedules are built in float32 torch, whose linspace kernel
    walks ``start + i * step`` over the first half and ``end - (num - 1 - i) * step``
    over the second, all in float32; numpy's float64-then-cast linspace differs in
    the last ulp, which the 1000-step cumulative product then amplifies.
    """
    start, end = np.float32(start), np.float32(end)
    i = np.arange(num, dtype=np.float32)
    step = (end - start) / np.float32(num - 1)
    return np.where(
        i < num // 2, start + i * step, end - (np.float32(num - 1) - i) * step
    ).astype(np.float32)


def make_beta_schedule(
    num_train_timesteps, beta_start, beta_end, beta_schedule, trained_betas=None
):
    """Return the ``num_train_timesteps`` betas for a diffusion noise schedule.

    ``scaled_linear`` (the Stable Diffusion default) spaces the square roots
    linearly, ``linear`` spaces the betas themselves, and ``squaredcos_cap_v2``
    is the cosine schedule. ``trained_betas`` overrides everything when given.
    Computed in float32 exactly as the reference (diffusers) does, so the
    per-timestep ``alphas_cumprod`` values agree to the bit.
    """
    if trained_betas is not None:
        return np.asarray(trained_betas, dtype=np.float32)
    if beta_schedule == "linear":
        return linspace_float32(beta_start, beta_end, num_train_timesteps)
    if beta_schedule == "scaled_linear":
        return (
            linspace_float32(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        )
    if beta_schedule == "squaredcos_cap_v2":
        steps = num_train_timesteps + 1
        t = (
            np.linspace(0, num_train_timesteps, steps, dtype=np.float64)
            / num_train_timesteps
        )
        alpha_bar = np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return np.clip(betas, 0, 0.999).astype(np.float32)
    raise ValueError(f"Unknown beta_schedule {beta_schedule!r}.")


class BaseScheduler:
    """Base class for the diffusion noise schedulers (samplers).

    A scheduler is weightless: it holds the training noise schedule
    (``betas`` / ``alphas_cumprod``) and turns a model's noise prediction into the
    next, less-noisy sample inside the denoising loop. The schedule is set up in
    numpy once at construction; :meth:`step` runs on backend tensors via
    ``keras.ops`` so it stays on device inside the loop. Concrete schedulers
    (DDIM, PNDM, Euler, ...) implement :meth:`set_timesteps` and :meth:`step`.

    Subclasses are constructed the same way a diffusers scheduler is, so a repo's
    ``scheduler/scheduler_config.json`` loads through :meth:`from_config`.

    Args:
        num_train_timesteps: Diffusion steps the schedule was trained with.
        beta_start / beta_end: Endpoints of the beta schedule.
        beta_schedule: ``"scaled_linear"`` (SD default), ``"linear"``, or
            ``"squaredcos_cap_v2"``.
        trained_betas: Explicit beta array, overriding the schedule.
        prediction_type: ``"epsilon"`` (predict noise), ``"v_prediction"``, or
            ``"sample"``.
        steps_offset: Added to the inference timesteps (SD ships ``1``).
        set_alpha_to_one: Use ``1.0`` as the final ``alpha_cumprod`` (else the
            first training value).
    """

    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        trained_betas=None,
        prediction_type="epsilon",
        steps_offset=0,
        set_alpha_to_one=True,
        **kwargs,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.trained_betas = trained_betas
        self.prediction_type = prediction_type
        self.steps_offset = steps_offset
        self.set_alpha_to_one = set_alpha_to_one

        self.betas = make_beta_schedule(
            num_train_timesteps, beta_start, beta_end, beta_schedule, trained_betas
        )
        self.alphas = (np.float32(1.0) - self.betas).astype(np.float32)
        # torch's float32 cumprod accumulates in double and stores float32
        self.alphas_cumprod = np.cumprod(self.alphas.astype(np.float64)).astype(
            np.float32
        )
        self.final_alpha_cumprod = (
            1.0 if set_alpha_to_one else float(self.alphas_cumprod[0])
        )

        self.num_inference_steps = None
        self.timesteps = np.arange(num_train_timesteps)[::-1].copy()

    @property
    def init_noise_sigma(self):
        """Scale applied to the initial pure-noise latent (1.0 for DDPM-style)."""
        return 1.0

    def scale_model_input(self, sample, timestep=None):
        """Scale the latent before the model sees it (identity for DDPM-style)."""
        return sample

    def set_timesteps(self, num_inference_steps):
        raise NotImplementedError

    def set_begin_index(self, index):
        """Start the loop at ``self.timesteps[index]`` (image-to-image skips the first
        steps); a no-op for the samplers that index by timestep value."""

    def step(self, model_output, timestep, sample, **kwargs):
        raise NotImplementedError

    def add_noise(self, original_samples, noise, timesteps):
        """Forward diffusion: mix ``original_samples`` with ``noise`` at ``timesteps``."""
        alphas_cumprod = np.asarray(self.alphas_cumprod)
        idx = np.asarray(timesteps).reshape(-1).astype(int)
        sqrt_alpha = np.sqrt(alphas_cumprod[idx])
        sqrt_one_minus = np.sqrt(1.0 - alphas_cumprod[idx])
        while sqrt_alpha.ndim < len(ops.shape(original_samples)):
            sqrt_alpha = sqrt_alpha[..., None]
            sqrt_one_minus = sqrt_one_minus[..., None]
        sqrt_alpha = ops.convert_to_tensor(sqrt_alpha, dtype=original_samples.dtype)
        sqrt_one_minus = ops.convert_to_tensor(
            sqrt_one_minus, dtype=original_samples.dtype
        )
        return sqrt_alpha * original_samples + sqrt_one_minus * noise

    def pred_original_and_epsilon(self, model_output, alpha_prod_t, sample):
        """Resolve ``(pred_original_sample, pred_epsilon)`` for the prediction type."""
        beta_prod_t = 1.0 - alpha_prod_t
        if self.prediction_type == "epsilon":
            pred_original = (
                sample - beta_prod_t**0.5 * model_output
            ) / alpha_prod_t**0.5
            pred_epsilon = model_output
        elif self.prediction_type == "v_prediction":
            pred_original = alpha_prod_t**0.5 * sample - beta_prod_t**0.5 * model_output
            pred_epsilon = alpha_prod_t**0.5 * model_output + beta_prod_t**0.5 * sample
        elif self.prediction_type == "sample":
            pred_original = model_output
            pred_epsilon = (
                sample - alpha_prod_t**0.5 * pred_original
            ) / beta_prod_t**0.5
        else:
            raise ValueError(f"Unknown prediction_type {self.prediction_type!r}.")
        return pred_original, pred_epsilon

    def to_config(self):
        return {
            "_class_name": type(self).__name__,
            "num_train_timesteps": self.num_train_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "beta_schedule": self.beta_schedule,
            "trained_betas": self.trained_betas,
            "prediction_type": self.prediction_type,
            "steps_offset": self.steps_offset,
            "set_alpha_to_one": self.set_alpha_to_one,
        }

    @classmethod
    def from_config(cls, config):
        """Build from a ``scheduler_config.json`` dict, ignoring unknown keys."""
        # A subclass signature is ``(own args, **kwargs)``: the schedule arguments
        # it forwards to the base live in the base signature, so accept the union
        # over the class hierarchy.
        params = set()
        for klass in cls.__mro__:
            if klass is object:
                continue
            params.update(inspect.signature(klass.__init__).parameters)
        accepted = {k: v for k, v in config.items() if k in params and k != "kwargs"}
        return cls(**accepted)


class DDIMScheduler(BaseScheduler):
    """Denoising Diffusion Implicit Models sampler (deterministic, ``eta=0``)."""

    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .astype(np.int64)
        )
        timesteps = timesteps + self.steps_offset
        self.timesteps = timesteps
        return self.timesteps

    def step(self, model_output, timestep, sample, eta=0.0, **kwargs):
        timestep = int(timestep)
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = float(self.alphas_cumprod[timestep])
        alpha_prod_t_prev = (
            float(self.alphas_cumprod[prev_timestep])
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        pred_original, pred_epsilon = self.pred_original_and_epsilon(
            model_output, alpha_prod_t, sample
        )
        pred_sample_direction = (1.0 - alpha_prod_t_prev) ** 0.5 * pred_epsilon
        prev_sample = alpha_prod_t_prev**0.5 * pred_original + pred_sample_direction
        return prev_sample


class PNDMScheduler(BaseScheduler):
    """Pseudo Numerical Methods sampler (the Stable Diffusion default).

    Runs a Runge-Kutta warm-up over the first steps (unless ``skip_prk_steps``)
    then the linear multi-step (PLMS) update over a running buffer of the last
    four noise predictions.
    """

    def __init__(self, skip_prk_steps=True, **kwargs):
        super().__init__(**kwargs)
        self.skip_prk_steps = skip_prk_steps
        self.cur_model_output = 0
        self.counter = 0
        self.cur_sample = None
        self.ets = []

    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        base = (
            np.arange(0, num_inference_steps) * step_ratio
        ).round() + self.steps_offset
        if self.skip_prk_steps:
            # PLMS: the second timestep is visited twice (the counter == 1 pseudo
            # step re-uses the first sample with the averaged prediction), so the
            # schedule has num_inference_steps + 1 entries, as in diffusers.
            plms = np.concatenate([base[:-1], base[-2:-1], base[-1:]])[::-1]
            timesteps = plms.copy().astype(np.int64)
        else:
            raise NotImplementedError(
                "PNDMScheduler currently supports skip_prk_steps=True only "
                "(the Stable Diffusion configuration)."
            )
        self.timesteps = timesteps
        self.cur_model_output = 0
        self.counter = 0
        self.cur_sample = None
        self.ets = []
        return self.timesteps

    def step(self, model_output, timestep, sample, **kwargs):
        timestep = int(timestep)
        diff = self.num_train_timesteps // self.num_inference_steps
        prev_timestep = timestep - diff

        if self.counter != 1:
            if len(self.ets) > 3:
                self.ets = self.ets[-3:]
            self.ets.append(model_output)
        else:
            prev_timestep = timestep
            timestep = timestep + diff

        if len(self.ets) == 1 and self.counter == 0:
            self.cur_sample = sample
        elif len(self.ets) == 1 and self.counter == 1:
            model_output = (model_output + self.ets[-1]) / 2
            sample = self.cur_sample
            self.cur_sample = None
        elif len(self.ets) == 2:
            model_output = (3 * self.ets[-1] - self.ets[-2]) / 2
        elif len(self.ets) == 3:
            model_output = (
                23 * self.ets[-1] - 16 * self.ets[-2] + 5 * self.ets[-3]
            ) / 12
        else:
            model_output = (1 / 24) * (
                55 * self.ets[-1]
                - 59 * self.ets[-2]
                + 37 * self.ets[-3]
                - 9 * self.ets[-4]
            )

        prev_sample = self._prev_sample(sample, timestep, prev_timestep, model_output)
        self.counter += 1
        return prev_sample

    def _prev_sample(self, sample, timestep, prev_timestep, model_output):
        alpha_prod_t = float(self.alphas_cumprod[timestep])
        alpha_prod_t_prev = (
            float(self.alphas_cumprod[prev_timestep])
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev

        if self.prediction_type == "v_prediction":
            model_output = alpha_prod_t**0.5 * model_output + beta_prod_t**0.5 * sample

        sample_coeff = (alpha_prod_t_prev / alpha_prod_t) ** 0.5
        model_output_denom_coeff = (
            alpha_prod_t * beta_prod_t_prev**0.5
            + (alpha_prod_t * beta_prod_t * alpha_prod_t_prev) ** 0.5
        )
        prev_sample = (
            sample_coeff * sample
            - (alpha_prod_t_prev - alpha_prod_t)
            * model_output
            / model_output_denom_coeff
        )
        return prev_sample

    def to_config(self):
        config = super().to_config()
        config["skip_prk_steps"] = self.skip_prk_steps
        return config


class EulerDiscreteScheduler(BaseScheduler):
    """Euler sampler over the karras-style sigma parameterization.

    Args:
        timestep_spacing: How the inference timesteps are spread over the training
            ones: ``"linspace"`` (evenly from ``num_train_timesteps - 1`` to 0),
            ``"leading"`` (multiples of the step ratio from 0, shifted by
            ``steps_offset``; the SD 1.x / 2.x and SDXL repos) or ``"trailing"``
            (multiples counted back from ``num_train_timesteps``; SDXL-Turbo).
        interpolation_type: ``"linear"`` interpolates the training sigmas at the
            timesteps; ``"log_linear"`` spaces them evenly in log space.
    """

    def __init__(
        self, timestep_spacing="linspace", interpolation_type="linear", **kwargs
    ):
        super().__init__(**kwargs)
        self.timestep_spacing = timestep_spacing
        self.interpolation_type = interpolation_type
        ac = self.alphas_cumprod
        sigmas = ((np.float32(1.0) - ac) / ac) ** np.float32(0.5)
        self.train_sigmas = sigmas.astype(np.float32)
        self.sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        self.step_index = 0

    @property
    def init_noise_sigma(self):
        # the reference scales the initial noise by the largest sigma itself for
        # "linspace" / "trailing" spacing and by sqrt(sigma_max^2 + 1) for "leading"
        max_sigma = float(self.sigmas.max())
        if self.timestep_spacing in ("linspace", "trailing"):
            return max_sigma
        return (max_sigma**2 + 1) ** 0.5

    def scale_model_input(self, sample, timestep=None):
        sigma = self.sigmas[self.step_index]
        return sample / ((sigma**2 + 1) ** 0.5)

    def set_begin_index(self, index):
        self.step_index = int(index)

    def add_noise(self, original_samples, noise, timesteps):
        # the Euler sample space is unscaled: x_t = x_0 + sigma_t * noise, sigma_t
        # being the inference sigma of the timestep (set_timesteps first)
        timesteps = np.asarray(timesteps, dtype=np.float32).reshape(-1)
        index = [int(np.nonzero(self.timesteps == t)[0][0]) for t in timesteps]
        sigma = self.sigmas[index]
        while sigma.ndim < len(ops.shape(original_samples)):
            sigma = sigma[..., None]
        sigma = ops.convert_to_tensor(sigma, dtype=original_samples.dtype)
        return original_samples + noise * sigma

    def spaced_timesteps(self, num_inference_steps):
        n_train = self.num_train_timesteps
        if self.timestep_spacing == "linspace":
            timesteps = np.linspace(
                0, n_train - 1, num_inference_steps, dtype=np.float32
            )
            return timesteps[::-1].copy()
        if self.timestep_spacing == "leading":
            step_ratio = n_train // num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()
            return timesteps[::-1].copy().astype(np.float32) + self.steps_offset
        if self.timestep_spacing == "trailing":
            step_ratio = n_train / num_inference_steps
            timesteps = np.arange(n_train, 0, -step_ratio).round()
            return timesteps.astype(np.float32) - 1
        raise ValueError(
            f"Unknown timestep_spacing {self.timestep_spacing!r}; expected "
            "'linspace', 'leading' or 'trailing'."
        )

    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        timesteps = self.spaced_timesteps(num_inference_steps)
        sigmas = self.train_sigmas
        if self.interpolation_type == "linear":
            sigmas = np.interp(timesteps, np.arange(len(sigmas)), sigmas)
        elif self.interpolation_type == "log_linear":
            sigmas = np.exp(
                np.linspace(
                    np.log(sigmas[-1]), np.log(sigmas[0]), num_inference_steps + 1
                )
            )[:-1]
        else:
            raise ValueError(
                f"Unknown interpolation_type {self.interpolation_type!r}; expected "
                "'linear' or 'log_linear'."
            )
        self.sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.timesteps = timesteps
        self.step_index = 0
        return self.timesteps

    def pred_original_sample(self, model_output, sigma, sample):
        # float32 sigma arithmetic, so the coefficients match the reference's bits
        if self.prediction_type == "epsilon":
            return sample - sigma * model_output
        if self.prediction_type == "v_prediction":
            return model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (
                sample / (sigma**2 + 1)
            )
        if self.prediction_type == "sample":
            return model_output
        raise ValueError(f"Unknown prediction_type {self.prediction_type!r}.")

    def step(self, model_output, timestep, sample, **kwargs):
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        pred_original = self.pred_original_sample(model_output, sigma, sample)
        derivative = (sample - pred_original) / sigma
        prev_sample = sample + derivative * (sigma_next - sigma)
        self.step_index += 1
        return prev_sample

    def to_config(self):
        config = super().to_config()
        config["timestep_spacing"] = self.timestep_spacing
        config["interpolation_type"] = self.interpolation_type
        return config


class EulerAncestralDiscreteScheduler(EulerDiscreteScheduler):
    """Ancestral Euler sampler: injects fresh noise at each step (stochastic)."""

    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras_random.SeedGenerator(seed)

    def step(self, model_output, timestep, sample, **kwargs):
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        pred_original = self.pred_original_sample(model_output, sigma, sample)
        sigma_up = (
            (sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2) ** 0.5
            if sigma > 0
            else np.float32(0.0)
        )
        sigma_down = (sigma_next**2 - sigma_up**2) ** 0.5
        derivative = (sample - pred_original) / sigma
        prev_sample = sample + derivative * (sigma_down - sigma)
        noise = keras_random.normal(
            ops.shape(sample), dtype=sample.dtype, seed=self.seed_generator
        )
        prev_sample = prev_sample + noise * sigma_up
        self.step_index += 1
        return prev_sample


class FlowMatchEulerDiscreteScheduler(BaseScheduler):
    """Euler sampler for rectified-flow models (Stable Diffusion 3 / 3.5, FLUX).

    There is no beta schedule: the noise level is the flow time ``sigma`` in
    ``[0, 1]`` (``x_t = (1 - sigma) x_0 + sigma * noise``), the model predicts the
    velocity and a step is ``x + (sigma_next - sigma) * v``. The timesteps handed to
    the model are ``sigma * num_train_timesteps``.

    Args:
        num_train_timesteps: The flow time resolution (1000).
        shift: Timestep shift towards noisier levels, ``shift * s / (1 + (shift - 1) s)``
            (3.0 for SD3 / SD3.5).
    """

    def __init__(self, num_train_timesteps=1000, shift=1.0, **kwargs):
        # the base (beta) schedule is irrelevant here; keep the constructor
        # compatible with from_config (a repo's scheduler_config may carry extra keys)
        super().__init__(num_train_timesteps=num_train_timesteps)
        self.shift = shift
        timesteps = np.linspace(
            1, num_train_timesteps, num_train_timesteps, dtype=np.float32
        )[::-1].copy()
        sigmas = timesteps / np.float32(num_train_timesteps)
        sigmas = self.shift_sigmas(sigmas)
        self.sigma_min = float(sigmas[-1])
        self.sigma_max = float(sigmas[0])
        self.sigmas = sigmas
        self.timesteps = sigmas * np.float32(num_train_timesteps)
        self.step_index = 0

    def shift_sigmas(self, sigmas):
        shift = np.float32(self.shift)
        return (shift * sigmas / (1 + (shift - 1) * sigmas)).astype(np.float32)

    @property
    def init_noise_sigma(self):
        return 1.0

    def set_begin_index(self, index):
        self.step_index = int(index)

    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        n_train = self.num_train_timesteps
        timesteps = np.linspace(
            self.sigma_max * n_train, self.sigma_min * n_train, num_inference_steps
        )
        sigmas = timesteps / n_train
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        sigmas = sigmas.astype(np.float32)
        self.sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.timesteps = sigmas * np.float32(n_train)
        self.step_index = 0
        return self.timesteps

    def add_noise(self, original_samples, noise, timesteps):
        # the forward flow: x_t = (1 - sigma) x_0 + sigma * noise
        timesteps = np.asarray(timesteps, dtype=np.float32).reshape(-1)
        index = [int(np.nonzero(self.timesteps == t)[0][0]) for t in timesteps]
        sigma = self.sigmas[index]
        while sigma.ndim < len(ops.shape(original_samples)):
            sigma = sigma[..., None]
        sigma = ops.convert_to_tensor(sigma, dtype=original_samples.dtype)
        return sigma * noise + (1.0 - sigma) * original_samples

    def step(self, model_output, timestep, sample, **kwargs):
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        prev_sample = sample + (sigma_next - sigma) * model_output
        self.step_index += 1
        return prev_sample

    def to_config(self):
        return {
            "_class_name": type(self).__name__,
            "num_train_timesteps": self.num_train_timesteps,
            "shift": self.shift,
        }


SCHEDULER_REGISTRY = {
    "DDIMScheduler": DDIMScheduler,
    "PNDMScheduler": PNDMScheduler,
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
    "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
    "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
}


def get_scheduler(config):
    """Build the scheduler named by a ``scheduler_config.json`` dict's ``_class_name``.

    Falls back to :class:`DDIMScheduler` (drop-in compatible) for a scheduler this
    port does not implement yet.
    """
    class_name = config.get("_class_name", "DDIMScheduler")
    cls = SCHEDULER_REGISTRY.get(class_name, DDIMScheduler)
    return cls.from_config(config)
