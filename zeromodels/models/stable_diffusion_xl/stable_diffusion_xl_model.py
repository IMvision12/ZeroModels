import keras
from keras import layers, ops

from zeromodels.base.base_scheduler import EulerDiscreteScheduler
from zeromodels.models.clip.clip_layers import CLIPTextModelEmbedding
from zeromodels.models.clip.clip_model import residual_attention_block
from zeromodels.models.stable_diffusion.stable_diffusion_model import (
    AutoencoderKL,
    StableDiffusionModel,
    StableDiffusionTextToImage,
    UNet2DConditionModel,
)

from .stable_diffusion_xl_config import (
    StableDiffusionXLConfig,
    StableDiffusionXLRefinerConfig,
)

# The container and the task build the identical graph, so both load the one
# hosted repo (whose zm_config.json names the container).
STABLE_DIFFUSION_XL_HUB_SIBLINGS = frozenset(
    {"StableDiffusionXLModel", "StableDiffusionXLTextToImage"}
)
STABLE_DIFFUSION_XL_REFINER_HUB_SIBLINGS = frozenset(
    {"StableDiffusionXLRefinerModel", "StableDiffusionXLRefinerImageToImage"}
)


def build_text_encoder(
    config, hidden_act, layer_norm_eps, projection_dim=None, name="text_encoder"
):
    """Functional CLIP text tower (the ``clip`` family's layers and leaf names)
    that also exposes the penultimate hidden state SDXL conditions on.

    Outputs ``penultimate_hidden_state`` (the input of the last transformer block,
    no final LayerNorm: diffusers' ``hidden_states[-2]``), the usual
    ``last_hidden_state`` / ``pooler_output`` (EOT position, argmax of the ids) and,
    when ``projection_dim`` is set, ``text_embeds``, the pooled state through the
    bias-free ``text_projection`` (the second SDXL tower). Built under a
    ``keras.name_scope(name)`` so the two towers' identically named layers keep
    distinct weight paths inside the container.
    """
    length = config.max_seq_len
    with keras.name_scope(name):
        # int32 inputs: under a float16 load policy a float input would round the
        # ids (49407 is not representable in float16)
        token_ids = layers.Input(shape=(length,), dtype="int32", name="token_ids")
        padding_mask = layers.Input(shape=(length,), dtype="int32", name="padding_mask")
        x = CLIPTextModelEmbedding(
            vocab_size=config.vocab_size,
            max_seq_len=length,
            embed_dim=config.hidden_dim,
            name="text_model_embedding",
        )(token_ids)

        causal_mask = ops.cast(ops.triu(ops.ones((length, length)), k=1), "float32")
        causal_mask = causal_mask * (-1e8)
        key_mask = ops.reshape(ops.cast(padding_mask, "float32"), (-1, 1, 1, length))
        key_mask = (1.0 - ops.repeat(key_mask, length, axis=2)) * (-1e8)

        penultimate = None
        for i in range(config.num_layers):
            if i == config.num_layers - 1:
                penultimate = x
            x = residual_attention_block(
                x,
                proj_dim=config.hidden_dim,
                num_heads=config.num_heads,
                layer_name_prefix="text_model_encoder",
                layer_idx=i,
                causal_attention_mask=causal_mask,
                attention_mask=key_mask,
                mlp_ratio=config.mlp_ratio,
                hidden_act=hidden_act,
                layer_norm_eps=layer_norm_eps,
            )
        last_hidden_state = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="text_model_layernorm"
        )(x)
        one_hot = ops.one_hot(
            ops.argmax(token_ids, axis=-1), length, dtype=last_hidden_state.dtype
        )
        pooler_output = ops.einsum("bi,bij->bj", one_hot, last_hidden_state)

        outputs = {
            "penultimate_hidden_state": penultimate,
            "last_hidden_state": last_hidden_state,
            "pooler_output": pooler_output,
        }
        if projection_dim is not None:
            # the Dense runs on a (B, 1, D) view, as CLIPTextEmbed does, so the
            # kernel maps to the checkpoint's text_projection.weight unchanged
            projected = layers.Dense(
                projection_dim, use_bias=False, name="text_projection"
            )(ops.expand_dims(pooler_output, axis=1))
            outputs["text_embeds"] = ops.squeeze(projected, axis=1)
    return keras.Model(
        inputs={"token_ids": token_ids, "padding_mask": padding_mask},
        outputs=outputs,
        name=name,
    )


@keras.saving.register_keras_serializable(package="zeromodels")
class StableDiffusionXLModel(StableDiffusionModel):
    """Stable Diffusion XL weights container: the UNet, the VAE and the two text
    encoders (CLIP ViT-L/14 and OpenCLIP ViT-bigG/14) as one functional
    ``keras.Model``.

    The SD 1.x container (:class:`StableDiffusionModel`) with the SDXL
    configuration and a fourth component: four disconnected sub-graphs,
    ``{"sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"}
    -> "noise_pred"`` (the UNet with its ``text_time`` micro-conditioning),
    ``{"image", "latent"} -> "moments" / "image"`` (the VAE, built in float32
    whatever the load dtype, ``force_upcast``) and ``{"token_ids", "token_ids_2",
    "padding_mask"} -> "prompt_embeds" / "pooled_prompt_embeds"`` (the two text
    towers: their penultimate hidden states concatenated to the UNet's 2048-d
    context, and the second tower's projected pooled state). The components are
    ``.unet`` / ``.vae`` / ``.text_encoder`` / ``.text_encoder_2``. Hosted as one
    ``zm_config.json`` + float16 weights (the checkpoints' native precision) per
    variant under ``zeromodels/stable-diffusion-xl-base-1.0`` and
    ``zeromodels/sdxl-turbo``; on-the-fly ``hf:`` conversion is not supported for
    diffusion models.

    Args:
        **kwargs: The flat :class:`StableDiffusionXLConfig` fields (``unet_`` /
            ``vae_`` / ``text_`` / ``text_2_`` prefixed), or a config positionally.
            Pass ``unet_sample_size`` / ``vae_sample_size`` to build for another
            resolution (the weights are resolution-independent).
    """

    config_class = StableDiffusionXLConfig
    HUB_REPO_SIBLINGS = STABLE_DIFFUSION_XL_HUB_SIBLINGS

    def __init__(self, name="StableDiffusionXLModel", **kwargs):
        super().__init__(name=name, **kwargs)

    def build_components(self, config, data_format, channels_axis):
        u, v, t, t2 = (
            config.unet_config,
            config.vae_config,
            config.text_config,
            config.text_config_2,
        )
        unet = UNet2DConditionModel(
            u, data_format=data_format, channels_axis=channels_axis
        )
        vae = AutoencoderKL(
            **v.constructor_kwargs(),
            data_format=data_format,
            channels_axis=channels_axis,
        )
        components = {"unet": unet, "vae": vae}
        if t is not None:  # the refiner has no CLIP ViT-L/14 tower
            components["text_encoder"] = build_text_encoder(
                t, config.hidden_act, config.layer_norm_eps, name="text_encoder"
            )
        components["text_encoder_2"] = build_text_encoder(
            t2,
            t2.hidden_act,
            config.layer_norm_eps,
            projection_dim=t2.projection_dim,
            name="text_encoder_2",
        )
        return components

    def build_graph(self, config, components, data_format):
        u, length = config.unet_config, config.text_config_2.max_seq_len
        unet, vae = components["unet"], components["vae"]
        text_encoder = components.get("text_encoder")
        text_encoder_2 = components["text_encoder_2"]
        inputs = self.unet_vae_inputs(config, components, data_format)
        pooled_dim = (
            u.projection_class_embeddings_input_dim
            - u.num_time_ids * u.addition_time_embed_dim
        )
        inputs["text_embeds"] = layers.Input(shape=(pooled_dim,), name="text_embeds")
        inputs["time_ids"] = layers.Input(shape=(u.num_time_ids,), name="time_ids")
        if text_encoder is not None:
            inputs["token_ids"] = layers.Input(
                shape=(length,), dtype="int32", name="token_ids"
            )
        inputs["token_ids_2"] = layers.Input(
            shape=(length,), dtype="int32", name="token_ids_2"
        )
        inputs["padding_mask"] = layers.Input(
            shape=(length,), dtype="int32", name="padding_mask"
        )

        noise_pred = unet(
            {
                key: inputs[key]
                for key in (
                    "sample",
                    "timestep",
                    "encoder_hidden_states",
                    "text_embeds",
                    "time_ids",
                )
            }
        )["sample"]
        vae_out = vae({"image": inputs["image"], "latent": inputs["latent"]})
        text_out_2 = text_encoder_2(
            {
                "token_ids": inputs["token_ids_2"],
                "padding_mask": inputs["padding_mask"],
            }
        )
        hidden_states = [text_out_2["penultimate_hidden_state"]]
        if text_encoder is not None:
            text_out = text_encoder(
                {
                    "token_ids": inputs["token_ids"],
                    "padding_mask": inputs["padding_mask"],
                }
            )
            hidden_states.insert(0, text_out["penultimate_hidden_state"])
        prompt_embeds = (
            ops.concatenate(hidden_states, axis=-1)
            if len(hidden_states) > 1
            else hidden_states[0]
        )
        outputs = {
            "noise_pred": noise_pred,
            "moments": vae_out["moments"],
            "image": vae_out["sample"],
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": text_out_2["text_embeds"],
        }
        return inputs, outputs


@keras.saving.register_keras_serializable(package="zeromodels")
class StableDiffusionXLTextToImage(StableDiffusionTextToImage, StableDiffusionXLModel):
    """Text-to-image Stable Diffusion XL, pure Keras 3 and cross-backend (the
    diffusers ``StableDiffusionXLPipeline`` in this library's task-class form).

    :class:`StableDiffusionTextToImage`'s ``generate`` (``BaseDiffusion``) over the
    :class:`StableDiffusionXLModel` graph. The hooks follow the SDXL recipe: the
    prompt goes through both text towers (the second tower's ``!``-padded ids are
    derived from the first tokenizer's output), their penultimate hidden states
    are concatenated into the UNet context and the second tower's projected pooled
    state becomes the ``text_embeds`` micro-conditioning, next to the ``time_ids``
    built from ``original_size`` / ``crops_coords_top_left`` / ``target_size``.
    With no negative prompt the unconditional branch of classifier-free guidance
    is zero embeddings (``force_zeros_for_empty_prompt``), as in the reference.
    The sampler is the repo's ``scheduler_config`` (Euler with ``leading`` spacing
    for the base model, ancestral Euler with ``trailing`` spacing for SDXL-Turbo)::

        sd = StableDiffusionXLTextToImage.from_weights("zeromodels/stable-diffusion-xl-base-1.0")
        tokenizer = StableDiffusionXLTokenizer.from_weights("zeromodels/stable-diffusion-xl-base-1.0")
        image = sd.generate(**tokenizer("a photo of a cat"))  # (1, 1024, 1024, 3) uint8

    Args:
        scheduler: A :class:`BaseScheduler`; defaults to the config's
            ``scheduler_config``, else Euler with ``leading`` spacing.
        **kwargs: The flat :class:`StableDiffusionXLConfig` fields.
    """

    config_class = StableDiffusionXLConfig
    HUB_REPO_SIBLINGS = STABLE_DIFFUSION_XL_HUB_SIBLINGS
    # diffusers' StableDiffusionXLPipeline defaults; a repo's generate_args override
    generate_args = {"num_inference_steps": 50, "guidance_scale": 5.0}

    def __init__(self, scheduler=None, name="StableDiffusionXLTextToImage", **kwargs):
        super().__init__(scheduler=scheduler, name=name, **kwargs)

    def default_scheduler(self):
        return EulerDiscreteScheduler(timestep_spacing="leading", steps_offset=1)

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
        original_size=None,
        crops_coords_top_left=(0, 0),
        target_size=None,
        aesthetic_score=6.0,
        negative_original_size=None,
        negative_crops_coords_top_left=None,
        negative_target_size=None,
        negative_aesthetic_score=2.5,
    ):
        """Generate images from tokenized prompts, or refine an ``image`` / latent
        (see ``BaseDiffusion.generate`` for the shared arguments).

        The SDXL micro-conditioning arguments are ``(height, width)`` pairs in
        pixels: ``original_size`` and ``target_size`` default to the generated
        image's size, ``crops_coords_top_left`` to ``(0, 0)``; the ``negative_``
        variants condition the negative branch and default to the positive values.
        ``aesthetic_score`` / ``negative_aesthetic_score`` (6.0 / 2.5) replace the
        target size on the refiner (``requires_aesthetics_score``). For the SDXL
        ensemble, run the base with ``denoising_end=0.8, output_type="latent"`` and
        hand the latent to the refiner's ``generate(..., latents=...,
        denoising_start=0.8)``.
        """
        return super().generate(
            input_ids,
            attention_mask=attention_mask,
            negative_input_ids=negative_input_ids,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            latents=latents,
            image=image,
            strength=strength,
            denoising_start=denoising_start,
            denoising_end=denoising_end,
            output_type=output_type,
            original_size=original_size,
            crops_coords_top_left=crops_coords_top_left,
            target_size=target_size,
            aesthetic_score=aesthetic_score,
            negative_original_size=negative_original_size,
            negative_crops_coords_top_left=negative_crops_coords_top_left,
            negative_target_size=negative_target_size,
            negative_aesthetic_score=negative_aesthetic_score,
        )

    def image_size(self):
        size = self.vae.sample_size
        return tuple(size) if isinstance(size, (tuple, list)) else (size, size)

    def time_ids(
        self,
        batch,
        original_size=None,
        crops_coords_top_left=(0, 0),
        target_size=None,
        aesthetic_score=6.0,
    ):
        """The ``(batch, 6)`` float32 micro-conditioning row: original size, crop
        offset and target size, each ``(height, width)``; ``(batch, 5)`` with the
        aesthetic score in place of the target size when
        ``requires_aesthetics_score`` (the refiner)."""
        original_size = tuple(original_size or self.image_size())
        row = list(original_size) + list(crops_coords_top_left)
        if self.config.requires_aesthetics_score:
            row = row + [aesthetic_score]
        else:
            row = row + list(target_size or self.image_size())
        return ops.convert_to_tensor([row] * batch, dtype="float32")

    def unconditional_ids(self, batch):
        cfg = self.config
        length = cfg.text_config_2.max_seq_len
        row = [cfg.bos_token_id, cfg.eos_token_id] + [cfg.pad_token_id] * (length - 2)
        return ops.convert_to_tensor([row] * batch, dtype="int32")

    def prompt_mask(self, input_ids):
        # the tokens up to and including the first <|endoftext|>; what follows is
        # padding (the first tokenizer pads with <|endoftext|> itself)
        is_eos = ops.cast(ops.equal(input_ids, self.config.eos_token_id), "int32")
        return ops.equal(ops.cumsum(is_eos, axis=1) - is_eos, 0)

    def encode_prompt(
        self,
        input_ids,
        attention_mask=None,
        original_size=None,
        crops_coords_top_left=(0, 0),
        target_size=None,
        aesthetic_score=6.0,
    ):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        if attention_mask is None:
            real = self.prompt_mask(input_ids)
        else:
            real = ops.greater(
                ops.cast(ops.convert_to_tensor(attention_mask), "int32"), 0
            )
        # the second tokenizer pads with "!" (id 0) where the first pads with
        # <|endoftext|>; the towers attend the padding (no attention mask), as the
        # reference does
        input_ids_2 = ops.where(real, input_ids, self.config.pad_token_id_2)
        ones = ops.ones_like(input_ids)
        out_2 = self.text_encoder_2({"token_ids": input_ids_2, "padding_mask": ones})
        hidden = out_2["penultimate_hidden_state"]
        if self.config.text_config is not None:
            out = self.text_encoder({"token_ids": input_ids, "padding_mask": ones})
            hidden = ops.concatenate([out["penultimate_hidden_state"], hidden], -1)
        batch = int(input_ids.shape[0])
        return {
            "encoder_hidden_states": hidden,
            "text_embeds": out_2["text_embeds"],
            "time_ids": self.time_ids(
                batch,
                original_size,
                crops_coords_top_left,
                target_size,
                aesthetic_score,
            ),
        }

    def encode_negative_prompt(self, negative_input_ids, batch, **conditioning):
        if (
            negative_input_ids is not None
            or not self.config.force_zeros_for_empty_prompt
        ):
            return super().encode_negative_prompt(
                negative_input_ids, batch, **conditioning
            )
        # force_zeros_for_empty_prompt: zero embeddings stand in for the empty
        # prompt; the micro-conditioning keeps its (negative) values
        u, length = self.config.unet_config, self.config.text_config_2.max_seq_len
        pooled_dim = (
            u.projection_class_embeddings_input_dim
            - u.num_time_ids * u.addition_time_embed_dim
        )
        return {
            "encoder_hidden_states": ops.zeros(
                (batch, length, u.cross_attention_dim), dtype="float32"
            ),
            "text_embeds": ops.zeros((batch, pooled_dim), dtype="float32"),
            "time_ids": self.time_ids(batch, **conditioning),
        }

    def predict_noise(self, latents, timesteps, embeddings):
        return self.unet({"sample": latents, "timestep": timesteps, **embeddings})[
            "sample"
        ]


@keras.saving.register_keras_serializable(package="zeromodels")
class StableDiffusionXLRefinerModel(StableDiffusionXLModel):
    """Stable Diffusion XL refiner weights container: the refiner UNet, the VAE and
    the OpenCLIP ViT-bigG/14 text encoder as one functional ``keras.Model``.

    :class:`StableDiffusionXLModel` with the refiner configuration
    (:class:`StableDiffusionXLRefinerConfig`): three disconnected sub-graphs (no
    CLIP ViT-L/14 tower, so the UNet's context is the second tower's 1280-d
    penultimate state alone), the four-level refiner UNet and five
    micro-conditioning ids (size, crop, aesthetic score). Components:
    ``.unet`` / ``.vae`` / ``.text_encoder_2``. Hosted under
    ``zeromodels/stable-diffusion-xl-refiner-1.0`` (float16, the VAE float32).

    Args:
        **kwargs: The flat :class:`StableDiffusionXLRefinerConfig` fields, or a
            config positionally.
    """

    config_class = StableDiffusionXLRefinerConfig
    HUB_REPO_SIBLINGS = STABLE_DIFFUSION_XL_REFINER_HUB_SIBLINGS

    def __init__(self, name="StableDiffusionXLRefinerModel", **kwargs):
        super().__init__(name=name, **kwargs)


@keras.saving.register_keras_serializable(package="zeromodels")
class StableDiffusionXLRefinerImageToImage(
    StableDiffusionXLTextToImage, StableDiffusionXLRefinerModel
):
    """Image-to-image Stable Diffusion XL refiner, pure Keras 3 and cross-backend
    (diffusers' ``StableDiffusionXLImg2ImgPipeline`` with the refiner weights).

    :class:`StableDiffusionXLTextToImage`'s hooks over the
    :class:`StableDiffusionXLRefinerModel` graph, the prompt encoded by the
    OpenCLIP ViT-bigG/14 tower alone and the aesthetic score
    (``aesthetic_score`` 6.0 / ``negative_aesthetic_score`` 2.5) as the fifth
    micro-conditioning id. ``generate`` refines an ``image`` (noised to
    ``strength``, 0.3 by default) or, as the second half of the SDXL ensemble of
    experts, a latent the base model left partially denoised
    (``denoising_start`` at the base's ``denoising_end``)::

        base = StableDiffusionXLTextToImage.from_weights("zeromodels/stable-diffusion-xl-base-1.0")
        refiner = StableDiffusionXLRefinerImageToImage.from_weights("zeromodels/stable-diffusion-xl-refiner-1.0")
        tokenizer = StableDiffusionXLTokenizer.from_weights("zeromodels/stable-diffusion-xl-base-1.0")
        inputs = tokenizer("a photo of a cat")
        latent = base.generate(**inputs, denoising_end=0.8, output_type="latent")
        image = refiner.generate(**inputs, latents=latent, denoising_start=0.8)

    Args:
        scheduler: A :class:`BaseScheduler`; defaults to the config's
            ``scheduler_config``, else Euler with ``leading`` spacing.
        **kwargs: The flat :class:`StableDiffusionXLRefinerConfig` fields.
    """

    config_class = StableDiffusionXLRefinerConfig
    HUB_REPO_SIBLINGS = STABLE_DIFFUSION_XL_REFINER_HUB_SIBLINGS
    # diffusers' StableDiffusionXLImg2ImgPipeline defaults
    generate_args = {"num_inference_steps": 50, "guidance_scale": 5.0, "strength": 0.3}

    def __init__(
        self, scheduler=None, name="StableDiffusionXLRefinerImageToImage", **kwargs
    ):
        super().__init__(scheduler=scheduler, name=name, **kwargs)
