import keras
import numpy as np
from keras import ops

from zeromodels.base.base_mixin import inference_scope


class BaseDiffusion:
    """Backend-agnostic latent-diffusion generation, the peer of :class:`BaseGeneration`
    (mirrors diffusers' ``DiffusionPipeline.__call__``).

    A mixin added to a diffusion container (the family's ``XModel``, which builds
    its denoiser / VAE / text-encoder towers internally) to give it ``generate``,
    exactly as ``T5ConditionalGenerate(T5Model, BaseSeq2SeqGeneration)`` adds an
    imperative decode loop to a functional backbone. It bundles the machinery every
    diffusion task shares: resolving the generation settings, batching the
    conditioning for classifier-free guidance, sampling the initial latent, the
    scheduler's denoising loop, decoding the latent and the uint8 postprocess. The
    loop runs under :func:`inference_scope` (no autograd on torch).

    Performance follows :class:`BaseGeneration`: the denoiser call (``predict_noise``
    plus the guidance combine, the whole cost of a step) is wrapped in a per-backend
    compiled function cached on the instance, ``jax.jit`` with stateless variable
    threading on JAX, ``tf.function(jit_compile=True)`` on TensorFlow, eager on
    Torch. Its shapes are the same at every step, so one compile serves the whole
    loop; the scheduler update stays eager (a few elementwise ops on the latent, and
    stateful in the PNDM case).

    A model plugs in five hooks:

    - ``encode_prompt(input_ids, attention_mask=None) -> (batch, seq, dim)`` -- the
      conditioning for a batch of token ids (the text tower).
    - ``unconditional_ids(batch) -> (batch, seq)`` -- token ids of the empty prompt,
      the unconditional branch of classifier-free guidance.
    - ``predict_noise(latents, timesteps, embeddings) -> (batch, h, w, c)`` -- one
      denoiser call (the UNet), ``timesteps`` being ``(batch,)`` floats.
    - ``decode_latents(latents) -> (batch, H, W, 3)`` -- the decoded image in
      ``[-1, 1]``, channels-last (the VAE decoder, latent scaling included).
    - ``latent_shape -> (height, width, channels)`` of one initial latent, and a
      ``scheduler`` attribute (a :class:`~zeromodels.base.base_scheduler.BaseScheduler`).

    Generation settings resolve like the LM mixins': an explicit ``generate``
    argument wins, then the instance's ``generate_args`` (a repo's ``zm_config.json``
    ``generate_args`` re-attached on load, merged over the model class's own), then
    the defaults here (50 steps, guidance 7.5). Loading is the model's own
    ``from_weights``; on-the-fly ``hf:`` conversion is not supported for diffusion
    models (the weights are converted once and hosted under ``zeromodels/``).
    """

    DEFAULT_NUM_INFERENCE_STEPS = 50
    DEFAULT_GUIDANCE_SCALE = 7.5

    def encode_prompt(self, input_ids, attention_mask=None):
        raise NotImplementedError(
            f"{type(self).__name__} must implement encode_prompt()."
        )

    def unconditional_ids(self, batch):
        raise NotImplementedError(
            f"{type(self).__name__} must implement unconditional_ids()."
        )

    def predict_noise(self, latents, timesteps, embeddings):
        raise NotImplementedError(
            f"{type(self).__name__} must implement predict_noise()."
        )

    def decode_latents(self, latents):
        raise NotImplementedError(
            f"{type(self).__name__} must implement decode_latents()."
        )

    @property
    def latent_shape(self):
        raise NotImplementedError(f"{type(self).__name__} must define latent_shape.")

    def generate(
        self,
        input_ids,
        attention_mask=None,
        negative_input_ids=None,
        num_inference_steps=None,
        guidance_scale=None,
        seed=None,
        latents=None,
    ):
        """Generate images from tokenized prompts.

        Args:
            input_ids: ``(batch, seq)`` token ids, i.e. ``**tokenizer(prompts)``.
            attention_mask: The tokenizer's mask, handed to ``encode_prompt``.
            negative_input_ids: ``(batch, seq)`` tokenized negative prompt for
                classifier-free guidance, or a single ``(1, seq)`` row shared by the
                whole batch; defaults to the empty prompt.
            num_inference_steps: Scheduler steps (``generate_args`` / 50).
            guidance_scale: Classifier-free guidance strength (``generate_args`` /
                7.5); ``<= 1`` disables it.
            seed: RNG seed for the initial latent (reproducible per backend).
            latents: Explicit initial latent ``(batch, *latent_shape)``, for results
                that are identical across backends.

        Returns:
            ``(batch, H, W, 3)`` uint8 numpy images.
        """
        num_inference_steps, guidance_scale, seed = self.resolve_generation_args(
            num_inference_steps, guidance_scale, seed
        )
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch = int(input_ids.shape[0])
        with inference_scope():
            embeddings = self.encode_prompt(input_ids, attention_mask)
            if guidance_scale > 1.0:
                uncond_ids = (
                    ops.cast(ops.convert_to_tensor(negative_input_ids), "int32")
                    if negative_input_ids is not None
                    else self.unconditional_ids(batch)
                )
                if int(uncond_ids.shape[0]) == 1 and batch > 1:
                    uncond_ids = ops.repeat(uncond_ids, batch, axis=0)  # one for all
                embeddings = ops.concatenate(
                    [self.encode_prompt(uncond_ids), embeddings], axis=0
                )
            latents = self.prepare_latents(batch, seed=seed, latents=latents)
            latents = self.denoise(
                latents, embeddings, num_inference_steps, guidance_scale
            )
            image = self.decode_latents(latents)
        return self.postprocess_image(image)

    def resolve_generation_args(self, num_inference_steps, guidance_scale, seed):
        defaults = getattr(self, "generate_args", None) or {}
        if num_inference_steps is None:
            num_inference_steps = defaults.get(
                "num_inference_steps", self.DEFAULT_NUM_INFERENCE_STEPS
            )
        if guidance_scale is None:
            guidance_scale = defaults.get("guidance_scale", self.DEFAULT_GUIDANCE_SCALE)
        return (
            int(num_inference_steps),
            float(guidance_scale),
            None if seed is None else int(seed),
        )

    def prepare_latents(self, batch, seed=None, latents=None, dtype="float32"):
        """Sample the initial pure-noise latent, scaled by the scheduler's
        ``init_noise_sigma``.

        Pass explicit ``latents`` for reproducibility across backends (each
        backend's RNG differs, so a shared image needs shared latents, not a shared
        seed).
        """
        if latents is None:
            latents = keras.random.normal(
                (batch, *self.latent_shape), seed=seed, dtype=dtype
            )
        else:
            latents = ops.cast(ops.convert_to_tensor(latents), dtype)
        return latents * self.scheduler.init_noise_sigma

    @staticmethod
    def classifier_free_guidance(noise_pred, guidance_scale):
        """Combine the batched ``[uncond, cond]`` predictions into a guided one."""
        uncond, cond = ops.split(noise_pred, 2, axis=0)
        return uncond + guidance_scale * (cond - uncond)

    def denoise(self, latents, embeddings, num_inference_steps, guidance_scale):
        """Run the scheduler's denoising loop over ``predict_noise``.

        ``embeddings`` is ``[uncond, cond]`` batched along axis 0 when
        ``guidance_scale > 1`` (classifier-free guidance), else just ``cond``.
        Returns the denoised latent.
        """
        do_cfg = guidance_scale > 1.0
        batch = int(ops.shape(latents)[0])
        model_batch = batch * 2 if do_cfg else batch
        step = self.cached_denoise_step(do_cfg)
        guidance = ops.convert_to_tensor(guidance_scale, dtype="float32")
        scheduler = self.scheduler
        scheduler.set_timesteps(num_inference_steps)
        for t in scheduler.timesteps:
            model_input = (
                ops.concatenate([latents, latents], axis=0) if do_cfg else latents
            )
            model_input = scheduler.scale_model_input(model_input, t)
            timesteps = ops.full((model_batch,), float(t))
            noise_pred = step(model_input, timesteps, embeddings, guidance)
            latents = scheduler.step(noise_pred, t, latents)
        return latents

    def make_denoise_step(self, do_cfg):
        """One denoiser call, ``predict_noise`` plus the guidance combine, as a
        per-backend compiled function (see the class docstring). ``do_cfg`` is
        static (it fixes the batch layout); ``guidance_scale`` is a runtime scalar.
        """

        def step(model_input, timesteps, embeddings, guidance_scale):
            noise_pred = self.predict_noise(model_input, timesteps, embeddings)
            if do_cfg:
                noise_pred = self.classifier_free_guidance(noise_pred, guidance_scale)
            return noise_pred

        backend = keras.backend.backend()
        if backend == "jax":
            import itertools

            import jax

            def compiled(model_input, timesteps, embeddings, guidance_scale, state):
                trainable, non_trainable = state
                mapping = itertools.chain(
                    zip(self.trainable_variables, trainable),
                    zip(self.non_trainable_variables, non_trainable),
                )
                with keras.StatelessScope(state_mapping=mapping):
                    return step(model_input, timesteps, embeddings, guidance_scale)

            compiled = jax.jit(compiled)

            def run(model_input, timesteps, embeddings, guidance_scale):
                state = (
                    [v.value for v in self.trainable_variables],
                    [v.value for v in self.non_trainable_variables],
                )
                return compiled(
                    model_input, timesteps, embeddings, guidance_scale, state
                )

            return run

        if backend == "tensorflow":
            import tensorflow as tf

            return tf.function(step, jit_compile=True)

        return step

    def cached_denoise_step(self, do_cfg):
        # kept in __dict__ (not a tracked keras attribute), like the LM mixins' caches
        fns = self.__dict__.get("_denoise_steps")
        if fns is None:
            fns = self.__dict__["_denoise_steps"] = {}
        fn = fns.get(do_cfg)
        if fn is None:
            fn = fns[do_cfg] = self.make_denoise_step(do_cfg)
        return fn

    @staticmethod
    def postprocess_image(image):
        """Decoded image in ``[-1, 1]`` to a ``(batch, H, W, 3)`` uint8 numpy image."""
        image = ops.clip(image / 2.0 + 0.5, 0.0, 1.0)
        image = ops.convert_to_numpy(image)
        return np.clip(np.round(image * 255.0), 0, 255).astype("uint8")
