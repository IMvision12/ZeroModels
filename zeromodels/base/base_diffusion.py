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

    - ``encode_prompt(input_ids, attention_mask=None, **conditioning)`` -- the
      conditioning for a batch of token ids (the text tower): a ``(batch, seq, dim)``
      tensor, or any nested structure of batch-major tensors when the denoiser
      takes more than one (SDXL's ``{"encoder_hidden_states", "text_embeds",
      "time_ids"}``). Extra ``generate`` keyword arguments arrive here (a model's
      micro-conditioning), see below.
    - ``unconditional_ids(batch) -> (batch, seq)`` -- token ids of the empty prompt,
      the unconditional branch of classifier-free guidance. A model whose
      unconditional branch is not an encoded prompt (SDXL zeroes it) overrides
      ``encode_negative_prompt`` instead.
    - ``predict_noise(latents, timesteps, embeddings) -> (batch, h, w, c)`` -- one
      denoiser call (the UNet), ``timesteps`` being ``(batch,)`` floats and
      ``embeddings`` whatever ``encode_prompt`` returned (batched ``[uncond, cond]``
      along axis 0 under guidance).
    - ``decode_latents(latents) -> (batch, H, W, 3)`` -- the decoded image in
      ``[-1, 1]``, channels-last (the VAE decoder, latent scaling included).
    - ``latent_shape -> (height, width, channels)`` of one initial latent, and a
      ``scheduler`` attribute (a :class:`~zeromodels.base.base_scheduler.BaseScheduler`).
    - ``encode_latents(image) -> (batch, h, w, c)`` -- optional, the VAE-encoded
      (scaled) latent of a ``[-1, 1]`` channels-last image, for image-to-image.

    ``generate`` is also the image-to-image entry point (diffusers'
    ``Img2ImgPipeline``): given an ``image`` (or a clean ``latents``) and a
    ``strength``, it noises the encoded image to the matching point of the schedule
    and denoises the remaining steps. ``denoising_end`` stops a run early and
    ``denoising_start`` resumes one from a partially denoised latent, the SDXL
    base + refiner "ensemble of experts" (``output_type="latent"`` hands the latent
    over).

    Keyword arguments ``generate`` does not know are conditioning and go to
    ``encode_prompt``; a ``negative_``-prefixed twin (``negative_original_size`` for
    ``original_size``) goes to the negative branch instead, which otherwise reuses
    the positive value, the way ``negative_input_ids`` pairs with ``input_ids``.

    Generation settings resolve like the LM mixins': an explicit ``generate``
    argument wins, then the instance's ``generate_args`` (a repo's ``zm_config.json``
    ``generate_args`` re-attached on load, merged over the model class's own), then
    the defaults here (50 steps, guidance 7.5). Loading is the model's own
    ``from_weights``; on-the-fly ``hf:`` conversion is not supported for diffusion
    models (the weights are converted once and hosted under ``zeromodels/``).
    """

    DEFAULT_NUM_INFERENCE_STEPS = 50
    DEFAULT_GUIDANCE_SCALE = 7.5
    DEFAULT_STRENGTH = 0.8

    def encode_prompt(self, input_ids, attention_mask=None):
        raise NotImplementedError(
            f"{type(self).__name__} must implement encode_prompt()."
        )

    def unconditional_ids(self, batch):
        raise NotImplementedError(
            f"{type(self).__name__} must implement unconditional_ids()."
        )

    def encode_negative_prompt(self, negative_input_ids, batch, **conditioning):
        """The unconditional branch of classifier-free guidance: the encoded negative
        prompt, or the encoded empty prompt (``unconditional_ids``) when none is
        given. A single ``(1, seq)`` negative row is shared by the whole batch.
        """
        if negative_input_ids is None:
            negative_input_ids = self.unconditional_ids(batch)
        negative_input_ids = ops.cast(
            ops.convert_to_tensor(negative_input_ids), "int32"
        )
        if int(negative_input_ids.shape[0]) == 1 and batch > 1:
            negative_input_ids = ops.repeat(negative_input_ids, batch, axis=0)
        return self.encode_prompt(negative_input_ids, **conditioning)

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

    def encode_latents(self, image):
        raise NotImplementedError(
            f"{type(self).__name__} must implement encode_latents() for image-to-image."
        )

    def generate(
        self,
        input_ids,
        attention_mask=None,
        negative_input_ids=None,
        num_inference_steps=None,
        guidance_scale=None,
        seed=None,
        latents=None,
        image=None,
        strength=None,
        denoising_start=None,
        denoising_end=None,
        output_type="image",
        **conditioning,
    ):
        """Generate images from tokenized prompts (text-to-image), or from an image
        or latent and a prompt (image-to-image).

        Args:
            input_ids: ``(batch, seq)`` token ids, i.e. ``**tokenizer(prompts)``.
            attention_mask: The tokenizer's mask, handed to ``encode_prompt``.
            negative_input_ids: ``(batch, seq)`` tokenized negative prompt for
                classifier-free guidance, or a single ``(1, seq)`` row shared by the
                whole batch; defaults to the empty prompt.
            num_inference_steps: Scheduler steps (``generate_args`` / 50).
            guidance_scale: Classifier-free guidance strength (``generate_args`` /
                7.5); ``<= 1`` disables it.
            seed: RNG seed for the initial latent / added noise (reproducible per
                backend).
            latents: Text-to-image: the explicit initial noise
                ``(batch, *latent_shape)``, for results identical across backends.
                Image-to-image (with ``strength`` or ``denoising_start``): the
                clean, scaled latent to start from instead of ``image``.
            image: ``(batch, H, W, 3)`` uint8 or ``[0, 1]`` float images to start
                from (image-to-image); encoded by the VAE, then noised to the
                ``strength`` point of the schedule.
            strength: Image-to-image: the fraction of the schedule to run,
                ``(0, 1]`` (``1.0`` ignores the image); the noise added matches.
                Defaults to ``generate_args["strength"]`` when an ``image`` or
                ``latents`` are given without ``denoising_start``, else 0.8.
            denoising_start: Resume from a partially denoised latent at this
                fraction of the schedule, adding no noise (the refiner's half of
                the SDXL ensemble: the base ran with ``denoising_end`` at the
                same value and ``output_type="latent"``).
            denoising_end: Stop after this fraction of the schedule (the base's
                half of the ensemble).
            output_type: ``"image"`` (uint8 images) or ``"latent"`` (the final
                latent, a backend tensor, e.g. for a refiner).
            **conditioning: Model-specific conditioning handed to ``encode_prompt``
                (SDXL's ``original_size`` / ``crops_coords_top_left`` /
                ``target_size``); ``negative_<name>`` variants apply to the negative
                branch only.

        Returns:
            ``(batch, H, W, 3)`` uint8 numpy images, or the latent.
        """
        num_inference_steps, guidance_scale, seed = self.resolve_generation_args(
            num_inference_steps, guidance_scale, seed
        )
        positive, negative = self.split_conditioning(conditioning)
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch = int(input_ids.shape[0])
        image_to_image = (
            image is not None or strength is not None or denoising_start is not None
        )
        if image_to_image and strength is None and denoising_start is None:
            strength = (getattr(self, "generate_args", None) or {}).get(
                "strength", self.DEFAULT_STRENGTH
            )
        with inference_scope():
            embeddings = self.encode_prompt(input_ids, attention_mask, **positive)
            if guidance_scale > 1.0:
                uncond = self.encode_negative_prompt(
                    negative_input_ids, batch, **negative
                )
                embeddings = keras.tree.map_structure(
                    lambda u, c: ops.concatenate([u, c], axis=0), uncond, embeddings
                )
            # the scheduler's init_noise_sigma depends on the inference timesteps
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
            if denoising_end is not None:
                timesteps = timesteps[: self.steps_before(denoising_end, timesteps)]
            if image_to_image:
                timesteps, t_start = self.image_to_image_timesteps(
                    timesteps, strength, denoising_start
                )
                latents = self.prepare_image_latents(
                    image,
                    latents,
                    timesteps[0] if denoising_start is None else None,
                    seed=seed,
                )
                self.scheduler.set_begin_index(t_start)
            else:
                latents = self.prepare_latents(batch, seed=seed, latents=latents)
            latents = self.denoise(
                latents,
                embeddings,
                num_inference_steps,
                guidance_scale,
                timesteps=timesteps,
            )
            if output_type == "latent":
                return latents
            image = self.decode_latents(latents)
        return self.postprocess_image(image)

    @staticmethod
    def steps_before(fraction, timesteps):
        """How many of ``timesteps`` lie in the first ``fraction`` of the schedule
        (timesteps at or above the ``1 - fraction`` cutoff, diffusers' rounding)."""
        cutoff = int(round(1000 - fraction * 1000))
        return int(np.sum(np.asarray(timesteps, dtype=np.float64) >= cutoff))

    def image_to_image_timesteps(self, timesteps, strength, denoising_start):
        """The tail of ``timesteps`` an image-to-image run covers and its start
        index: the last ``strength`` fraction, or everything after
        ``denoising_start``."""
        n = len(timesteps)
        if denoising_start is not None:
            t_start = self.steps_before(denoising_start, timesteps)
        else:
            t_start = max(n - min(int(n * strength), n), 0)
        return timesteps[t_start:], t_start

    def prepare_image_latents(self, image, latents, timestep, seed=None):
        """The starting latent of an image-to-image run: the encoded ``image`` (or
        the given clean ``latents``), noised to ``timestep`` unless ``timestep`` is
        ``None`` (resuming a partially denoised latent)."""
        if latents is None:
            if image is None:
                raise ValueError("Image-to-image needs an image or latents.")
            image = ops.convert_to_tensor(image)
            if not keras.backend.is_float_dtype(image.dtype):
                image = ops.cast(image, "float32") / 255.0
            latents = self.encode_latents(ops.cast(image, "float32") * 2.0 - 1.0)
        latents = ops.cast(ops.convert_to_tensor(latents), "float32")
        if timestep is None:
            return latents
        noise = keras.random.normal(ops.shape(latents), seed=seed, dtype="float32")
        return self.scheduler.add_noise(latents, noise, [timestep])

    @staticmethod
    def split_conditioning(conditioning):
        """Split ``generate``'s extra keyword arguments into the positive branch's
        and the negative branch's (``negative_<name>`` overrides ``<name>`` there;
        a ``None`` override keeps the positive value)."""
        positive = {
            k: v for k, v in conditioning.items() if not k.startswith("negative_")
        }
        negative = dict(positive)
        for key, value in conditioning.items():
            if key.startswith("negative_") and value is not None:
                negative[key[len("negative_") :]] = value
        return positive, negative

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

    def denoise(
        self, latents, embeddings, num_inference_steps, guidance_scale, timesteps=None
    ):
        """Run the scheduler's denoising loop over ``predict_noise``.

        ``embeddings`` is whatever ``encode_prompt`` returned, ``[uncond, cond]``
        batched along axis 0 when ``guidance_scale > 1`` (classifier-free
        guidance), else just ``cond``. Loops over the scheduler's
        ``num_inference_steps`` timesteps, or over the given ``timesteps`` (a
        subset of them, the scheduler already set up). Returns the denoised latent.
        """
        do_cfg = guidance_scale > 1.0
        batch = int(ops.shape(latents)[0])
        model_batch = batch * 2 if do_cfg else batch
        step = self.cached_denoise_step(do_cfg)
        guidance = ops.convert_to_tensor(guidance_scale, dtype="float32")
        scheduler = self.scheduler
        if timesteps is None:
            scheduler.set_timesteps(num_inference_steps)
            timesteps = scheduler.timesteps
        for t in timesteps:
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
