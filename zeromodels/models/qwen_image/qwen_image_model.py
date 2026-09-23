"""Qwen-Image models: transformer, container, and text-to-image task."""

from __future__ import annotations

import keras
import numpy as np
from keras import layers, ops

from zeromodels.base import BaseDiffusion, BaseModel
from zeromodels.base.base_mixin import inference_scope
from zeromodels.base.base_scheduler import (
    FlowMatchEulerDiscreteScheduler,
    get_scheduler,
)
from zeromodels.models.qwen2_5_vl.qwen2_5_vl_model import Qwen2_5VLModel
from zeromodels.models.qwen_image.qwen_image_config import (
    QwenImageConfig,
    QwenImageTextConfig,
    QwenImageTransformerConfig,
)
from zeromodels.models.qwen_image.qwen_image_layers import (
    QwenImageAdaLayerNormContinuous,
    QwenImageEmbedRope,
    QwenImageRMSNorm,
    QwenImageTimestepProjEmbeddings,
    QwenImageTransformerBlock,
)
from zeromodels.models.qwen_image.qwen_image_vae import AutoencoderKLQwenImage
from zeromodels.models.stable_diffusion.stable_diffusion_layers import safe_name

QWEN_IMAGE_HUB_SIBLINGS = frozenset({"QwenImageModel", "QwenImageTextToImage"})
PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


def calculate_shift(
    image_seq_len,
    base_seq_len=256,
    max_seq_len=4096,
    base_shift=0.5,
    max_shift=1.15,
):
    """Resolution-dependent flow-match shift (Diffusers ``calculate_shift``)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def pack_latents(latents, height, width):
    """Pack ``(B, H, W, C)`` latents into ``(B, H/2 * W/2, C*4)`` (Diffusers)."""
    batch = ops.shape(latents)[0]
    channels = ops.shape(latents)[-1]
    latents = ops.reshape(
        latents, (batch, height // 2, 2, width // 2, 2, channels)
    )
    latents = ops.transpose(latents, (0, 1, 3, 5, 2, 4))
    return ops.reshape(
        latents, (batch, (height // 2) * (width // 2), channels * 4)
    )


def unpack_latents(latents, height, width, channels):
    """Unpack ``(B, seq, C*4)`` to ``(B, H, W, C)``."""
    batch = ops.shape(latents)[0]
    latents = ops.reshape(
        latents, (batch, height // 2, width // 2, channels, 2, 2)
    )
    latents = ops.transpose(latents, (0, 1, 4, 2, 5, 3))
    return ops.reshape(latents, (batch, height, width, channels))


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImageTransformer2DModel(BaseModel):
    """Qwen-Image denoiser (Diffusers ``QwenImageTransformer2DModel``).

    Double-stream MMDiT over **packed** latents ``(B, seq, in_channels)`` and
    text features ``(B, text_seq, joint_attention_dim)``. Inputs dict keys:
    ``sample``, ``timestep``, ``encoder_hidden_states``, optional
    ``encoder_hidden_states_mask``. Output ``{"sample": packed velocity}``.

    Built for a fixed packed sequence length derived from ``sample_size``
    (latent side before 2×2 packing).
    """

    HF_MODEL_TYPE = None
    config_class = QwenImageTransformerConfig

    def __init__(
        self,
        patch_size=2,
        in_channels=64,
        out_channels=16,
        num_layers=60,
        attention_head_dim=128,
        num_attention_heads=24,
        joint_attention_dim=3584,
        axes_dims_rope=(16, 56, 56),
        guidance_embeds=False,
        sample_size=128,
        text_seq_len=512,
        name="QwenImageTransformer2DModel",
        **kwargs,
    ):
        del guidance_embeds  # base Qwen-Image is not guidance-distilled
        keras_kwargs = {k: kwargs.pop(k) for k in ("trainable", "dtype") if k in kwargs}
        axes_dims_rope = tuple(axes_dims_rope)
        inner_dim = num_attention_heads * attention_head_dim
        sample_h = (
            sample_size[0]
            if isinstance(sample_size, (tuple, list))
            else sample_size
        )
        pack_h = pack_w = sample_h // patch_size
        packed_seq = pack_h * pack_w

        pos_embed = QwenImageEmbedRope(
            theta=10000,
            axes_dim=axes_dims_rope,
            scale_rope=True,
            module_path="pos_embed",
        )
        time_text_embed = QwenImageTimestepProjEmbeddings(
            embedding_dim=inner_dim, module_path="time_text_embed"
        )
        txt_norm = QwenImageRMSNorm(eps=1e-6, module_path="txt_norm")
        img_in = layers.Dense(inner_dim, name=safe_name("img_in"))
        txt_in = layers.Dense(inner_dim, name=safe_name("txt_in"))
        blocks = [
            QwenImageTransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                module_path=f"transformer_blocks.{i}",
            )
            for i in range(num_layers)
        ]
        norm_out = QwenImageAdaLayerNormContinuous(
            inner_dim, num_chunks=2, module_path="norm_out"
        )
        proj_out = layers.Dense(
            patch_size * patch_size * out_channels,
            name=safe_name("proj_out"),
        )

        sample_in = layers.Input(shape=(packed_seq, in_channels), name="sample")
        timestep_in = layers.Input(shape=(), name="timestep")
        enc_in = layers.Input(
            shape=(text_seq_len, joint_attention_dim), name="encoder_hidden_states"
        )
        enc_mask_in = layers.Input(
            shape=(text_seq_len,), dtype="int32", name="encoder_hidden_states_mask"
        )

        hidden = img_in(sample_in)
        encoder = txt_in(txt_norm(enc_in))
        temb = time_text_embed(timestep_in)
        pos_embed.build(None)
        vid_freqs, txt_freqs = pos_embed.call(pack_h, pack_w, text_seq_len, frame=1)
        # Bake RoPE tables into the graph as constants (static T2I resolution).
        image_rotary_emb = (
            ops.convert_to_tensor(ops.convert_to_numpy(vid_freqs)),
            ops.convert_to_tensor(ops.convert_to_numpy(txt_freqs)),
        )
        mask = ops.cast(enc_mask_in, "bool")
        for block in blocks:
            encoder, hidden = block(
                hidden,
                encoder,
                temb,
                encoder_hidden_states_mask=mask,
                image_rotary_emb=image_rotary_emb,
            )
        hidden = norm_out([hidden, temb])
        output = proj_out(hidden)

        super().__init__(
            inputs={
                "sample": sample_in,
                "timestep": timestep_in,
                "encoder_hidden_states": enc_in,
                "encoder_hidden_states_mask": enc_mask_in,
            },
            outputs={"sample": output},
            name=name,
            **keras_kwargs,
        )

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.joint_attention_dim = joint_attention_dim
        self.axes_dims_rope = axes_dims_rope
        self.sample_size = sample_size
        self.text_seq_len = text_seq_len
        self.inner_dim = inner_dim
        self.pack_h = pack_h
        self.pack_w = pack_w
        self.packed_seq = packed_seq
        self.pos_embed = pos_embed
        self.time_text_embed = time_text_embed
        self.txt_norm = txt_norm
        self.img_in = img_in
        self.txt_in = txt_in
        self.transformer_blocks = blocks
        self.norm_out = norm_out
        self.proj_out = proj_out

    @classmethod
    def kwargs_from_diffusers_config(cls, cfg):
        return {
            "patch_size": cfg.get("patch_size", 2),
            "in_channels": cfg.get("in_channels", 64),
            "out_channels": cfg.get("out_channels", 16),
            "num_layers": cfg.get("num_layers", 60),
            "attention_head_dim": cfg.get("attention_head_dim", 128),
            "num_attention_heads": cfg.get("num_attention_heads", 24),
            "joint_attention_dim": cfg.get("joint_attention_dim", 3584),
            "axes_dims_rope": tuple(cfg.get("axes_dims_rope", (16, 56, 56))),
            "guidance_embeds": bool(cfg.get("guidance_embeds", False)),
        }


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImageTextEncoderModel(Qwen2_5VLModel):
    """Qwen-Image prompt encoder: the Qwen2.5-VL-7B text tower, no vision / LM head.

    The ``text_encoder`` component of :class:`QwenImageModel`.
    Inputs ``input_ids`` / ``attention_mask``; output ``last_hidden_state``.
    """

    HF_MODEL_TYPE = None
    config_class = QwenImageTextConfig

    def __init__(self, max_seq_len=1024, name="text_encoder", **kwargs):
        kwargs["build_vision"] = False
        super().__init__(name=name, **kwargs)
        self.max_seq_len = max_seq_len


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImageModel(BaseModel):
    """Qwen-Image weights container: transformer + VAE + Qwen2.5-VL text tower.

    One functional ``keras.Model`` with disconnected paths (Diffusers
    ``QwenImagePipeline`` components). Hosted as ``zeromodels/qwen-image``
    (sharded weights). On-the-fly ``hf:`` conversion is not supported.
    """

    config_class = QwenImageConfig
    HF_MODEL_TYPE = None
    HUB_REPO_SIBLINGS = QWEN_IMAGE_HUB_SIBLINGS

    def __init__(self, name="QwenImageModel", **kwargs):
        keras_kwargs = {k: kwargs.pop(k) for k in ("trainable", "dtype") if k in kwargs}
        config = self.config_class.from_dict(kwargs)

        components = self.build_components(config)
        inputs, outputs = self.build_graph(config, components)
        super().__init__(inputs=inputs, outputs=outputs, name=name, **keras_kwargs)

        for attr, component in components.items():
            setattr(self, attr, component)

    def build_components(self, config):
        d, v = config.transformer_config, config.vae_config
        transformer = QwenImageTransformer2DModel(
            patch_size=d.patch_size,
            in_channels=d.in_channels,
            out_channels=d.out_channels,
            num_layers=d.num_layers,
            attention_head_dim=d.attention_head_dim,
            num_attention_heads=d.num_attention_heads,
            joint_attention_dim=d.joint_attention_dim,
            axes_dims_rope=d.axes_dims_rope,
            guidance_embeds=d.guidance_embeds,
            sample_size=d.sample_size,
            text_seq_len=config.max_sequence_length,
        )
        vae = AutoencoderKLQwenImage(
            base_dim=v.base_dim,
            z_dim=v.z_dim,
            dim_mult=v.dim_mult,
            num_res_blocks=v.num_res_blocks,
            attn_scales=v.attn_scales,
            temperal_downsample=v.temperal_downsample,
            dropout=v.dropout,
            input_channels=v.input_channels,
            latents_mean=v.latents_mean,
            latents_std=v.latents_std,
            sample_size=v.sample_size,
        )
        text_encoder = QwenImageTextEncoderModel(config.text_config)
        return {"transformer": transformer, "vae": vae, "text_encoder": text_encoder}

    def build_graph(self, config, components):
        d, v, t = config.transformer_config, config.vae_config, config.text_config
        transformer, vae, text_encoder = (
            components["transformer"],
            components["vae"],
            components["text_encoder"],
        )
        text_seq = config.max_sequence_length
        img_h = img_w = (
            v.sample_size
            if not isinstance(v.sample_size, (tuple, list))
            else v.sample_size[0]
        )
        lat_h = img_h // vae.vae_scale_factor
        lat_w = img_w // vae.vae_scale_factor

        inputs = {
            "sample": layers.Input(
                shape=(transformer.packed_seq, d.in_channels), name="sample"
            ),
            "timestep": layers.Input(shape=(), name="timestep"),
            "encoder_hidden_states": layers.Input(
                shape=(text_seq, d.joint_attention_dim), name="encoder_hidden_states"
            ),
            "encoder_hidden_states_mask": layers.Input(
                shape=(text_seq,), dtype="int32", name="encoder_hidden_states_mask"
            ),
            "image": layers.Input(shape=(img_h, img_w, 3), name="image"),
            "latent": layers.Input(shape=(lat_h, lat_w, v.z_dim), name="latent"),
            "token_ids": layers.Input(
                shape=(t.max_seq_len,), dtype="int32", name="token_ids"
            ),
            "padding_mask": layers.Input(
                shape=(t.max_seq_len,), dtype="int32", name="padding_mask"
            ),
        }
        noise_pred = transformer(
            {
                "sample": inputs["sample"],
                "timestep": inputs["timestep"],
                "encoder_hidden_states": inputs["encoder_hidden_states"],
                "encoder_hidden_states_mask": inputs["encoder_hidden_states_mask"],
            }
        )["sample"]
        vae_out = vae({"image": inputs["image"], "latent": inputs["latent"]})
        text_out = text_encoder(
            {"input_ids": inputs["token_ids"], "attention_mask": inputs["padding_mask"]}
        )
        return inputs, {
            "noise_pred": noise_pred,
            "moments": vae_out["moments"],
            "image": vae_out["sample"],
            "prompt_embeds": text_out["last_hidden_state"],
        }

    def get_config(self):
        config = super().get_config()
        config.update(self.config.constructor_kwargs())
        return config

    def from_hf(self, *args, **kwargs):
        raise NotImplementedError(
            "On-the-fly hf: conversion is not supported for Qwen-Image; "
            "use convert_qwen_image_diffusers_to_keras.py and "
            "from_weights('zeromodels/qwen-image')."
        )


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImageTextToImage(QwenImageModel, BaseDiffusion):
    """Text-to-image Qwen-Image (Diffusers ``QwenImagePipeline``).

    ::

        model = QwenImageTextToImage.from_weights("zeromodels/qwen-image")
        tok = QwenImageTokenizer.from_weights("zeromodels/qwen-image")
        image = model.generate(**tok("a cat"), height=1024, width=1024)

    Uses true CFG (``guidance_scale`` / Diffusers ``true_cfg_scale``) with
    dual forward passes and prediction-norm renormalization. Packed latents +
    flow-match Euler with dynamic resolution shifting.
    """

    config_class = QwenImageConfig
    HUB_REPO_SIBLINGS = QWEN_IMAGE_HUB_SIBLINGS
    generate_args = {"num_inference_steps": 50, "guidance_scale": 4.0}
    DEFAULT_GUIDANCE_SCALE = 4.0

    def __init__(self, scheduler=None, name="QwenImageTextToImage", **kwargs):
        scheduler_config = kwargs.get("scheduler_config")
        super().__init__(name=name, **kwargs)
        if scheduler is None:
            scheduler = (
                get_scheduler(scheduler_config)
                if scheduler_config
                else self.default_scheduler()
            )
        self.scheduler = scheduler

    def default_scheduler(self):
        return FlowMatchEulerDiscreteScheduler(
            shift=1.0,
            use_dynamic_shifting=True,
            base_shift=0.5,
            max_shift=0.9,
            base_image_seq_len=256,
            max_image_seq_len=8192,
            shift_terminal=0.02,
            time_shift_type="exponential",
        )

    @property
    def vae_scale_factor(self):
        return self.vae.vae_scale_factor

    @property
    def latent_shape(self):
        # Packed shape used by prepare_latents after packing.
        h, w = self._latent_side()
        channels = self.vae.z_dim
        return ((h // 2) * (w // 2), channels * 4)

    def _latent_side(self, height=None, width=None):
        scale = self.vae_scale_factor * 2  # VAE 8× and pack 2×
        if height is None or width is None:
            side = self.config.default_sample_size * self.vae_scale_factor
            height = width = side
        # Match Diffusers: round down to multiple of vae_scale_factor*2, then *2 latent.
        h = 2 * (int(height) // scale)
        w = 2 * (int(width) // scale)
        return h, w

    def unconditional_ids(self, batch):
        # Empty / space negative prompt is encoded by the tokenizer template;
        # here build a minimal pad row the task replaces via encode_negative_prompt.
        length = self.config.text_config.max_seq_len
        row = [self.config.pad_token_id] * length
        return ops.convert_to_tensor([row] * batch, dtype="int32")

    def encode_prompt(self, input_ids, attention_mask=None, **conditioning):
        """Encode ChatML-templated token ids → truncated prompt embeds + mask.

        Expects ``input_ids`` already wrapped with the Diffusers prompt template
        (see :class:`QwenImageTokenizer`). Drops the template prefix
        (``prompt_template_encode_start_idx``) and pads/truncates to
        ``max_sequence_length``.
        """
        del conditioning
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        if attention_mask is None:
            attention_mask = ops.ones_like(input_ids)
        else:
            attention_mask = ops.cast(ops.convert_to_tensor(attention_mask), "int32")

        out = self.text_encoder(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        hidden = out["last_hidden_state"]
        drop = int(self.config.prompt_template_encode_start_idx)
        max_len = int(self.config.max_sequence_length)

        # Gather non-padding tokens per row, drop template prefix, then pad.
        # Implemented with numpy for clarity on host; convert back to tensors.
        hidden_np = ops.convert_to_numpy(hidden)
        mask_np = ops.convert_to_numpy(attention_mask)
        batch = hidden_np.shape[0]
        dim = hidden_np.shape[-1]
        embeds = np.zeros((batch, max_len, dim), dtype=hidden_np.dtype)
        out_mask = np.zeros((batch, max_len), dtype=np.int32)
        for i in range(batch):
            valid = hidden_np[i][mask_np[i].astype(bool)]
            valid = valid[drop:][:max_len]
            n = valid.shape[0]
            embeds[i, :n] = valid
            out_mask[i, :n] = 1
        return {
            "encoder_hidden_states": ops.convert_to_tensor(embeds),
            "encoder_hidden_states_mask": ops.convert_to_tensor(out_mask),
        }

    def predict_noise(self, latents, timesteps, embeddings):
        return self.transformer(
            {
                "sample": latents,
                "timestep": timesteps,
                "encoder_hidden_states": embeddings["encoder_hidden_states"],
                "encoder_hidden_states_mask": embeddings[
                    "encoder_hidden_states_mask"
                ],
            }
        )["sample"]

    def decode_latents(self, latents, height=None, width=None):
        h, w = self._latent_side(height, width)
        latents = unpack_latents(latents, h, w, self.vae.z_dim)
        # Diffusers: latents / (1/std) + mean  ==  latents * std + mean
        mean = ops.convert_to_tensor(self.vae.latents_mean, dtype="float32")
        std = ops.convert_to_tensor(self.vae.latents_std, dtype="float32")
        mean = ops.reshape(mean, (1, 1, 1, -1))
        std = ops.reshape(std, (1, 1, 1, -1))
        latents = latents * std + mean
        return self.vae.decode(latents)

    def prepare_latents(self, batch, seed=None, latents=None, dtype="float32"):
        h, w = self._latent_side(
            getattr(self, "_gen_height", None), getattr(self, "_gen_width", None)
        )
        channels = self.vae.z_dim
        if latents is None:
            # Spatial noise then pack (matches Diffusers prepare_latents).
            noise = keras.random.normal(
                (batch, h, w, channels), seed=seed, dtype=dtype
            )
            latents = pack_latents(noise, h, w)
        else:
            latents = ops.cast(ops.convert_to_tensor(latents), dtype)
        return latents * self.scheduler.init_noise_sigma

    def denoise(
        self, latents, embeddings, num_inference_steps, guidance_scale, timesteps=None
    ):
        """True CFG: separate cond/uncond forwards + norm renormalization."""
        do_cfg = guidance_scale > 1.0
        scheduler = self.scheduler
        if timesteps is None:
            scheduler.set_timesteps(num_inference_steps)
            timesteps = scheduler.timesteps
        # embeddings may be a pair (uncond, cond) under CFG from generate().
        if do_cfg and isinstance(embeddings, (tuple, list)):
            uncond_emb, cond_emb = embeddings
        else:
            cond_emb = embeddings
            uncond_emb = None

        for t in timesteps:
            t_batch = ops.full((ops.shape(latents)[0],), float(t) / 1000.0)
            noise_pred = self.predict_noise(latents, t_batch, cond_emb)
            if do_cfg and uncond_emb is not None:
                neg_pred = self.predict_noise(latents, t_batch, uncond_emb)
                comb = neg_pred + guidance_scale * (noise_pred - neg_pred)
                cond_norm = ops.sqrt(
                    ops.sum(ops.square(noise_pred), axis=-1, keepdims=True)
                )
                comb_norm = ops.sqrt(
                    ops.sum(ops.square(comb), axis=-1, keepdims=True)
                )
                noise_pred = comb * (cond_norm / (comb_norm + 1e-8))
            latents = scheduler.step(noise_pred, t, latents)
        return latents

    def generate(
        self,
        input_ids,
        attention_mask=None,
        negative_input_ids=None,
        negative_attention_mask=None,
        num_inference_steps=None,
        guidance_scale=None,
        seed=None,
        latents=None,
        height=None,
        width=None,
        output_type="image",
        **conditioning,
    ):
        num_inference_steps, guidance_scale, seed = self.resolve_generation_args(
            num_inference_steps, guidance_scale, seed
        )
        self._gen_height = height
        self._gen_width = width
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch = int(input_ids.shape[0])

        with inference_scope():
            embeddings = self.encode_prompt(input_ids, attention_mask, **conditioning)
            do_cfg = guidance_scale > 1.0 and (
                negative_input_ids is not None
                or conditioning.get("negative_prompt") is not None
            )
            # Diffusers enables true CFG when a negative prompt is provided.
            if guidance_scale > 1.0 and negative_input_ids is None:
                # Encode empty/space negative via unconditional_ids path.
                neg_ids = self.unconditional_ids(batch)
                neg_mask = ops.ones_like(neg_ids)
                # Prefer caller-supplied negative when present.
                uncond = self.encode_prompt(neg_ids, neg_mask)
                do_cfg = True
            elif negative_input_ids is not None:
                uncond = self.encode_prompt(
                    negative_input_ids, negative_attention_mask
                )
                do_cfg = guidance_scale > 1.0
            else:
                uncond = None
                do_cfg = False

            # Dynamic flow-match timesteps (mu from packed sequence length).
            h, w = self._latent_side(height, width)
            image_seq_len = (h // 2) * (w // 2)
            sched_cfg = getattr(self.scheduler, "config_dict", None) or {}
            mu = calculate_shift(
                image_seq_len,
                sched_cfg.get("base_image_seq_len", 256),
                sched_cfg.get("max_image_seq_len", 4096),
                sched_cfg.get("base_shift", 0.5),
                sched_cfg.get("max_shift", 0.9),
            )
            sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
            if hasattr(self.scheduler, "set_timesteps"):
                try:
                    self.scheduler.set_timesteps(
                        num_inference_steps, sigmas=sigmas, mu=mu
                    )
                except TypeError:
                    self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps

            latents = self.prepare_latents(batch, seed=seed, latents=latents)
            emb = (uncond, embeddings) if do_cfg else embeddings
            latents = self.denoise(
                latents,
                emb,
                num_inference_steps,
                guidance_scale if do_cfg else 1.0,
                timesteps=timesteps,
            )
            if output_type == "latent":
                return latents
            image = self.decode_latents(latents, height=height, width=width)
        return self.postprocess_image(image)
