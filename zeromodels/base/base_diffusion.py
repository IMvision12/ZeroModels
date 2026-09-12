import keras
import numpy as np
from keras import ops


class BaseDiffusion:
    """Generation mixin for diffusion models, the peer of :class:`BaseGeneration`.

    A diffusion task is a functional model (the family's ``XModel``, which builds
    its UNet / VAE / text encoder towers internally) plus this mixin, exactly as
    ``T5ConditionalGenerate(T5Model, BaseSeq2SeqGeneration)`` adds an imperative
    decode loop to a functional backbone. The mixin holds the machinery every
    diffusion task shares: sampling the initial latent, running the scheduler's
    denoising loop with classifier-free guidance, and turning the VAE decoder's
    output into uint8 images. A task implements ``generate(**tokenizer_inputs)``
    on top (``StableDiffusionTextToImage.generate``). Loading is the model's own
    ``from_weights``; on-the-fly ``hf:`` conversion is not supported for diffusion
    models (the weights are converted once and hosted under ``zeromodels/``).
    """

    def prepare_latents(
        self,
        batch,
        channels,
        height,
        width,
        init_noise_sigma,
        seed=None,
        latents=None,
        dtype="float32",
    ):
        """Sample the initial pure-noise latent, scaled by ``init_noise_sigma``.

        Pass explicit ``latents`` for reproducibility across backends (each
        backend's RNG differs, so a shared image needs shared latents, not a shared
        seed).
        """
        shape = (batch, height, width, channels)
        if latents is None:
            latents = keras.random.normal(shape, seed=seed, dtype=dtype)
        else:
            latents = ops.cast(latents, dtype)
        return latents * init_noise_sigma

    @staticmethod
    def classifier_free_guidance(noise_pred, guidance_scale):
        """Combine the batched ``[uncond, cond]`` predictions into a guided one."""
        uncond, cond = ops.split(noise_pred, 2, axis=0)
        return uncond + guidance_scale * (cond - uncond)

    def denoise(
        self, unet, scheduler, latents, embeddings, num_inference_steps, guidance_scale
    ):
        """Run the scheduler's denoising loop over ``unet``.

        ``embeddings`` is ``[uncond, cond]`` batched along axis 0 when
        ``guidance_scale > 1`` (classifier-free guidance), else just ``cond``.
        Returns the denoised latent.
        """
        do_cfg = guidance_scale is not None and guidance_scale > 1.0
        batch = int(ops.shape(latents)[0])
        model_batch = batch * 2 if do_cfg else batch
        scheduler.set_timesteps(num_inference_steps)
        for t in scheduler.timesteps:
            model_input = (
                ops.concatenate([latents, latents], axis=0) if do_cfg else latents
            )
            model_input = scheduler.scale_model_input(model_input, t)
            timesteps = ops.full((model_batch,), float(t))
            noise_pred = unet(
                {
                    "sample": model_input,
                    "timestep": timesteps,
                    "encoder_hidden_states": embeddings,
                }
            )["sample"]
            if do_cfg:
                noise_pred = self.classifier_free_guidance(noise_pred, guidance_scale)
            latents = scheduler.step(noise_pred, t, latents)
        return latents

    @staticmethod
    def postprocess_image(image):
        """VAE decoder output in ``[-1, 1]`` to a ``(B, H, W, 3)`` uint8 numpy image."""
        image = ops.clip(image / 2.0 + 0.5, 0.0, 1.0)
        image = ops.convert_to_numpy(image)
        return np.clip(np.round(image * 255.0), 0, 255).astype("uint8")
