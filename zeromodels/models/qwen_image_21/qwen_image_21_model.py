"""Qwen-Image-2.1 models: single-stream transformer, container, and T2I task."""

from __future__ import annotations

import keras
import numpy as np
from keras import layers, ops

from zeromodels.base import BaseDiffusion, BaseModel, CausalMask
from zeromodels.base.base_mixin import inference_scope
from zeromodels.base.base_scheduler import (
    FlowMatchEulerDiscreteScheduler,
    get_scheduler,
)
from zeromodels.models.qwen3_vl.qwen3_vl_model import Qwen3VLTextModel, qwen3_text_cos_sin
from zeromodels.models.qwen_image_21.qwen_image_21_config import (
    DEFAULT_LATENTS_MEAN,
    DEFAULT_LATENTS_STD,
    QwenImage21Config,
    QwenImage21TextConfig,
    QwenImage21TransformerConfig,
    QwenImage21VAEConfig,
)
from zeromodels.models.qwen_image_21.qwen_image_21_layers import (
    IMG_TOKENS_PER_SLOT,
    QwenImage21AdaLayerNormContinuous,
    QwenImage21TextProjection,
    QwenImage21TimestepProjEmbeddings,
    QwenImage21TransformerBlock,
    build_block_causal_additive_mask,
    build_qwenimage21_rope_angles,
    build_token_metadata,
)
from zeromodels.models.qwen_image_21.qwen_image_21_vae import (
    QwenImage21CausalConv,
    QwenImage21Decoder3d,
    QwenImage21Encoder3d,
)
from zeromodels.models.stable_diffusion.stable_diffusion_layers import safe_name

QWEN_IMAGE_21_HUB_SIBLINGS = frozenset(
    {"QwenImage21Model", "QwenImage21TextToImage"}
)


def pack_latents(latents, height, width):
    """Flatten ``(B, H, W, C)`` → ``(B, H*W, C)`` (2.1 consumes latents unpatched)."""
    batch = ops.shape(latents)[0]
    channels = ops.shape(latents)[-1]
    return ops.reshape(latents, (batch, height * width, channels))


def unpack_latents(latents, height, width, channels):
    """Unflatten ``(B, H*W, C)`` → ``(B, H, W, C)``."""
    batch = ops.shape(latents)[0]
    return ops.reshape(latents, (batch, height, width, channels))


def _t2i_image_pad_mask(text_seq_len, latent_h, latent_w):
    """Bool mask over the VLM+target-slot sequence before 2×2 expansion."""
    target_slots = (latent_h * latent_w) // IMG_TOKENS_PER_SLOT
    return np.concatenate(
        [
            np.zeros(text_seq_len, dtype=bool),
            np.ones(target_slots, dtype=bool),
        ]
    )


def _expand_image_pad_mask(img_mask):
    """Expand each VLM image slot to ``IMG_TOKENS_PER_SLOT`` latent tokens."""
    repeats = np.where(img_mask, IMG_TOKENS_PER_SLOT, 1)
    return np.repeat(img_mask, repeats)


@keras.saving.register_keras_serializable(package="zeromodels")
class AutoencoderKLQwenImage21(BaseModel):
    """Qwen-Image-2.1 VAE (Diffusers ``AutoencoderKLQwenImage21``), channels-last.

    64-channel latents, 16× spatial compression, residual encoder/decoder, RGBA
    (4-channel) IO. Graph built for ``sample_size``; weights are resolution-
    independent.
    """

    config_class = QwenImage21VAEConfig
    HF_MODEL_TYPE = None

    def __init__(
        self,
        base_dim=96,
        decoder_base_dim=144,
        z_dim=64,
        dim_mult=(1, 2, 4, 8, 8),
        num_res_blocks=2,
        attn_scales=(),
        temperal_downsample=(False, True, True, True),
        dropout=0.0,
        input_channels=4,
        out_channels=4,
        is_residual=True,
        scale_factor_spatial=16,
        latents_mean=DEFAULT_LATENTS_MEAN,
        latents_std=DEFAULT_LATENTS_STD,
        sample_size=1024,
        name="AutoencoderKLQwenImage21",
        **kwargs,
    ):
        dim_mult = tuple(dim_mult)
        temperal_downsample = tuple(temperal_downsample)
        attn_scales = tuple(attn_scales)
        latents_mean = tuple(latents_mean)
        latents_std = tuple(latents_std)
        temperal_upsample = tuple(reversed(temperal_downsample))
        decoder_base_dim = (
            int(decoder_base_dim) if decoder_base_dim is not None else int(base_dim)
        )

        h_img, w_img = (
            sample_size
            if isinstance(sample_size, (tuple, list))
            else (sample_size, sample_size)
        )
        spatial_compression_ratio = int(scale_factor_spatial)
        h_lat, w_lat = h_img // spatial_compression_ratio, w_img // spatial_compression_ratio

        encoder = QwenImage21Encoder3d(
            dim=base_dim,
            z_dim=z_dim * 2,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            dropout=dropout,
            input_channels=input_channels,
            is_residual=is_residual,
            module_path="encoder",
        )
        decoder = QwenImage21Decoder3d(
            dim=decoder_base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_upsample=temperal_upsample,
            dropout=dropout,
            out_channels=out_channels,
            is_residual=is_residual,
            module_path="decoder",
        )
        quant_conv = QwenImage21CausalConv(
            z_dim * 2, kernel_size=1, padding=0, module_path="quant_conv"
        )
        post_quant_conv = QwenImage21CausalConv(
            z_dim, kernel_size=1, padding=0, module_path="post_quant_conv"
        )

        image_in = layers.Input(shape=(h_img, w_img, input_channels), name="image")
        latent_in = layers.Input(shape=(h_lat, w_lat, z_dim), name="latent")

        image_5d = ops.expand_dims(image_in, axis=1)
        moments_5d = quant_conv(encoder(image_5d))
        moments = ops.squeeze(moments_5d, axis=1)

        latent_5d = ops.expand_dims(latent_in, axis=1)
        decoded_5d = decoder(post_quant_conv(latent_5d))
        decoded = ops.squeeze(decoded_5d, axis=1)
        decoded = ops.clip(decoded, -1.0, 1.0)

        super().__init__(
            inputs={"image": image_in, "latent": latent_in},
            outputs={"moments": moments, "sample": decoded},
            name=name,
            **kwargs,
        )

        self.base_dim = base_dim
        self.decoder_base_dim = decoder_base_dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_upsample
        self.dropout = dropout
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.is_residual = is_residual
        self.latents_mean = latents_mean
        self.latents_std = latents_std
        self.sample_size = sample_size
        self.spatial_compression_ratio = spatial_compression_ratio
        self.vae_scale_factor = spatial_compression_ratio
        self.encoder = encoder
        self.decoder = decoder
        self.quant_conv = quant_conv
        self.post_quant_conv = post_quant_conv

    def _to_ndhwc(self, x, is_latent=False):
        static_ndim = len(x.shape)
        if static_ndim == 4:
            return ops.expand_dims(x, axis=1)
        if static_ndim == 5:
            c1 = int(x.shape[1]) if x.shape[1] is not None else None
            c_last = int(x.shape[-1]) if x.shape[-1] is not None else None
            expect = self.z_dim if is_latent else self.input_channels
            if c1 == expect and c_last != expect:
                return ops.transpose(x, (0, 2, 3, 4, 1))
            return x
        raise ValueError(f"Expected 4D or 5D tensor, got shape {x.shape}")

    def encode(self, x, sample=False, seed=None):
        was_4d = len(x.shape) == 4
        x5 = self._to_ndhwc(x, is_latent=False)
        moments = self.quant_conv(self.encoder(x5))
        mean, logvar = ops.split(moments, 2, axis=-1)
        if sample:
            logvar = ops.clip(logvar, -30.0, 20.0)
            std = ops.exp(0.5 * logvar)
            noise = keras.random.normal(ops.shape(mean), dtype=mean.dtype, seed=seed)
            z = mean + std * noise
        else:
            z = mean
        if was_4d:
            z = ops.squeeze(z, axis=1)
        return z

    def decode(self, z):
        was_4d = len(z.shape) == 4
        z5 = self._to_ndhwc(z, is_latent=True)
        out = self.decoder(self.post_quant_conv(z5))
        out = ops.clip(out, -1.0, 1.0)
        if was_4d:
            out = ops.squeeze(out, axis=1)
        return out

    @classmethod
    def kwargs_from_diffusers_config(cls, cfg):
        return {
            "base_dim": cfg.get("base_dim", 96),
            "decoder_base_dim": cfg.get("decoder_base_dim", 144),
            "z_dim": cfg.get("z_dim", 64),
            "dim_mult": tuple(cfg.get("dim_mult", (1, 2, 4, 8, 8))),
            "num_res_blocks": cfg.get("num_res_blocks", 2),
            "attn_scales": tuple(cfg.get("attn_scales") or ()),
            "temperal_downsample": tuple(
                cfg.get("temperal_downsample", (False, True, True, True))
            ),
            "dropout": cfg.get("dropout", 0.0),
            "input_channels": cfg.get("in_channels", 4),
            "out_channels": cfg.get("out_channels", 4),
            "is_residual": bool(cfg.get("is_residual", True)),
            "scale_factor_spatial": cfg.get("scale_factor_spatial", 16),
            "latents_mean": tuple(cfg.get("latents_mean", DEFAULT_LATENTS_MEAN)),
            "latents_std": tuple(cfg.get("latents_std", DEFAULT_LATENTS_STD)),
        }


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21Transformer2DModel(BaseModel):
    """Qwen-Image-2.1 denoiser (Diffusers ``QwenImage21Transformer2DModel``).

    Single-stream block-causal DiT over unpatched latents and Qwen3-VL text
    features. Built for a fixed T2I layout (``sample_size`` latent side +
    ``text_seq_len``). Output ``{"sample": target velocity}``.
    """

    HF_MODEL_TYPE = None
    config_class = QwenImage21TransformerConfig

    def __init__(
        self,
        patch_size=1,
        in_channels=64,
        out_channels=64,
        num_layers=32,
        attention_head_dim=128,
        num_attention_heads=32,
        context_in_dim=4096,
        mlp_ratio=3,
        axes_dims_rope=(16, 56, 56),
        eps=1e-6,
        causal_condition=True,
        sample_size=64,
        text_seq_len=512,
        name="QwenImage21Transformer2DModel",
        **kwargs,
    ):
        keras_kwargs = {k: kwargs.pop(k) for k in ("trainable", "dtype") if k in kwargs}
        axes_dims_rope = tuple(axes_dims_rope)
        inner_dim = num_attention_heads * attention_head_dim
        sample_h = (
            sample_size[0] if isinstance(sample_size, (tuple, list)) else sample_size
        )
        sample_w = (
            sample_size[1] if isinstance(sample_size, (tuple, list)) else sample_size
        )
        assert patch_size == 1, "Qwen-Image-2.1 consumes latents unpatched"
        img_seq = sample_h * sample_w
        img_shapes = [(1, sample_h, sample_w)]

        # Prefill T2I joint layout (text + target image, no condition images).
        slot_mask = _t2i_image_pad_mask(text_seq_len, sample_h, sample_w)
        image_pad_mask = _expand_image_pad_mask(slot_mask)
        joint_seq = int(image_pad_mask.shape[0])
        image_ids, target_token_mask = build_token_metadata(image_pad_mask, img_shapes)
        rope_angles = build_qwenimage21_rope_angles(
            img_shapes, image_pad_mask, axes_dims_rope
        )
        attn_mask_np = build_block_causal_additive_mask(image_ids)

        time_text_embed = QwenImage21TimestepProjEmbeddings(
            embedding_dim=inner_dim, module_path="time_text_embed"
        )
        txt_in = QwenImage21TextProjection(
            context_in_dim, inner_dim, eps=eps, module_path="txt_in"
        )
        img_in = layers.Dense(
            inner_dim, use_bias=False, name=safe_name("img_in")
        )
        modulation = layers.Dense(
            4 * inner_dim, use_bias=False, name=safe_name("modulation.1")
        )
        blocks = [
            QwenImage21TransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                mlp_ratio=mlp_ratio,
                eps=eps,
                module_path=f"transformer_blocks.{i}",
            )
            for i in range(num_layers)
        ]
        norm_out = QwenImage21AdaLayerNormContinuous(
            inner_dim, inner_dim, eps=eps, module_path="norm_out"
        )
        proj_out = layers.Dense(
            patch_size * patch_size * (out_channels or in_channels),
            use_bias=False,
            name=safe_name("proj_out"),
        )

        sample_in = layers.Input(shape=(img_seq, in_channels), name="sample")
        timestep_in = layers.Input(shape=(), name="timestep")
        enc_in = layers.Input(
            shape=(text_seq_len, context_in_dim), name="encoder_hidden_states"
        )
        enc_mask_in = layers.Input(
            shape=(text_seq_len,), dtype="int32", name="encoder_hidden_states_mask"
        )

        hidden = img_in(sample_in)
        encoder = txt_in(enc_in)

        # Pure T2I layout: text tokens then target-image tokens (no condition images).
        # Equivalent to Diffusers' expand/scatter path when img_mask is text-False +
        # target-slot-True.
        joint = ops.concatenate([encoder, hidden], axis=1)

        rotary = ops.convert_to_tensor(rope_angles)
        attn_mask = ops.convert_to_tensor(attn_mask_np)
        target_mask_t = ops.convert_to_tensor(target_token_mask)
        text_pos = np.flatnonzero(~image_pad_mask).astype(np.int32)

        # Fold encoder padding into the block-causal key mask.
        from zeromodels.models.qwen_image_21.qwen_image_21_layers import MASK_NEG

        text_scatter = np.zeros((joint_seq, text_seq_len), dtype=np.float32)
        for j, pos in enumerate(text_pos):
            text_scatter[pos, j] = 1.0
        text_scatter_t = ops.convert_to_tensor(text_scatter)
        text_valid_on_joint = ops.einsum(
            "bt,jt->bj", ops.cast(enc_mask_in, "float32"), text_scatter_t
        )
        is_text = ops.convert_to_tensor((~image_pad_mask).astype(np.float32))
        key_ok = text_valid_on_joint * is_text[None, :] + (1.0 - is_text[None, :])
        attn_mask = attn_mask + (1.0 - key_ok)[:, None, None, :] * MASK_NEG

        if causal_condition:
            # Extra t=0 row: text tokens modulate from it; target uses the real step.
            t_all = ops.concatenate(
                [timestep_in, ops.zeros((1,), dtype=timestep_in.dtype)], axis=0
            )
            temb = time_text_embed(t_all)
            mod_mask = target_mask_t
        else:
            temb = time_text_embed(timestep_in)
            mod_mask = None

        mod = modulation(ops.silu(temb))
        for block in blocks:
            joint = block(
                joint,
                mod,
                rotary_emb=rotary,
                attention_mask=attn_mask,
                target_token_mask=mod_mask,
            )
        joint = norm_out([joint, temb, mod_mask])
        output_full = proj_out(joint)
        # Return only target-image tokens (Diffusers pipeline slices the same way).
        output = output_full[:, -img_seq:, :]
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
        self.out_channels = out_channels or in_channels
        self.num_layers = num_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.context_in_dim = context_in_dim
        self.mlp_ratio = mlp_ratio
        self.axes_dims_rope = axes_dims_rope
        self.eps = eps
        self.causal_condition = causal_condition
        self.sample_size = sample_size
        self.text_seq_len = text_seq_len
        self.inner_dim = inner_dim
        self.img_seq = img_seq
        self.joint_seq = joint_seq
        self.time_text_embed = time_text_embed
        self.txt_in = txt_in
        self.img_in = img_in
        self.modulation = modulation
        self.transformer_blocks = blocks
        self.norm_out = norm_out
        self.proj_out = proj_out

    @classmethod
    def kwargs_from_diffusers_config(cls, cfg):
        return {
            "patch_size": cfg.get("patch_size", 1),
            "in_channels": cfg.get("in_channels", 64),
            "out_channels": cfg.get("out_channels", 64),
            "num_layers": cfg.get("num_layers", 32),
            "attention_head_dim": cfg.get("attention_head_dim", 128),
            "num_attention_heads": cfg.get("num_attention_heads", 32),
            "context_in_dim": cfg.get("context_in_dim", 4096),
            "mlp_ratio": cfg.get("mlp_ratio", 3),
            "axes_dims_rope": tuple(cfg.get("axes_dims_rope", (16, 56, 56))),
            "eps": cfg.get("eps", 1e-6),
            "causal_condition": bool(cfg.get("causal_condition", True)),
        }


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21TextEncoderModel(BaseModel):
    """Qwen-Image-2.1 prompt encoder: Qwen3-VL text tower, no vision / LM head.

    Returns **pre-final-norm** hidden states (Diffusers hooks the last RMSNorm so
    the transformer sees the decoder output before it).
    """

    HF_MODEL_TYPE = None
    config_class = QwenImage21TextConfig
    output_logits = False

    def __init__(self, name="text_encoder", **kwargs):
        keras_kwargs = {k: kwargs.pop(k) for k in ("trainable", "dtype") if k in kwargs}
        if "max_seq_len" in kwargs and not any(
            k in kwargs for k in ("embed_dim", "num_layers", "vocab_size")
        ):
            # Allow ``QwenImage21TextEncoderModel(config)`` via from_dict path below.
            pass
        config = self.config_class.from_dict(kwargs) if kwargs else self.config_class()
        max_seq_len = int(getattr(config, "max_seq_len", 1024))

        language_model = Qwen3VLTextModel(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            mlp_dim=config.mlp_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            norm_eps=config.norm_eps,
            name="language_model",
        )
        causal_mask = CausalMask(name="causal_mask")

        input_ids = layers.Input(shape=(None,), dtype="int32", name="input_ids")
        attention_mask = layers.Input(
            shape=(None,), dtype="int32", name="attention_mask"
        )
        hidden = language_model.token_embedding(input_ids)
        pos1 = ops.cumsum(ops.ones_like(input_ids), axis=-1) - 1
        pos = ops.stack([pos1, pos1, pos1], axis=0)
        cos, sin = qwen3_text_cos_sin(
            pos, config.head_dim, config.rope_theta, tuple(config.mrope_section)
        )
        mask = causal_mask(input_ids, attention_mask)
        # Decoder layers only — skip final_norm (Diffusers forward hook).
        for layer in language_model.decoder_layers:
            hidden = layer(hidden, cos, sin, attention_mask=mask)

        super().__init__(
            inputs={"input_ids": input_ids, "attention_mask": attention_mask},
            outputs={"last_hidden_state": hidden},
            name=name,
            **keras_kwargs,
        )
        self.language_model = language_model
        self.causal_mask_layer = causal_mask
        self.max_seq_len = max_seq_len
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.mlp_dim = config.mlp_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.norm_eps = config.norm_eps
        self.rope_theta = config.rope_theta
        self.mrope_section = tuple(config.mrope_section)
        self.tie_embeddings = config.tie_embeddings


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21Model(BaseModel):
    """Qwen-Image-2.1 weights container: transformer + VAE + Qwen3-VL text tower."""

    config_class = QwenImage21Config
    HF_MODEL_TYPE = None
    HUB_REPO_SIBLINGS = QWEN_IMAGE_21_HUB_SIBLINGS

    def __init__(self, name="QwenImage21Model", **kwargs):
        keras_kwargs = {k: kwargs.pop(k) for k in ("trainable", "dtype") if k in kwargs}
        config = self.config_class.from_dict(kwargs)
        components = self.build_components(config)
        inputs, outputs = self.build_graph(config, components)
        super().__init__(inputs=inputs, outputs=outputs, name=name, **keras_kwargs)
        for attr, component in components.items():
            setattr(self, attr, component)

    def build_components(self, config):
        d, v = config.transformer_config, config.vae_config
        transformer = QwenImage21Transformer2DModel(
            patch_size=d.patch_size,
            in_channels=d.in_channels,
            out_channels=d.out_channels,
            num_layers=d.num_layers,
            attention_head_dim=d.attention_head_dim,
            num_attention_heads=d.num_attention_heads,
            context_in_dim=d.context_in_dim,
            mlp_ratio=d.mlp_ratio,
            axes_dims_rope=d.axes_dims_rope,
            eps=d.eps,
            causal_condition=d.causal_condition,
            sample_size=d.sample_size,
            text_seq_len=config.max_sequence_length,
        )
        vae = AutoencoderKLQwenImage21(
            base_dim=v.base_dim,
            decoder_base_dim=v.decoder_base_dim,
            z_dim=v.z_dim,
            dim_mult=v.dim_mult,
            num_res_blocks=v.num_res_blocks,
            attn_scales=v.attn_scales,
            temperal_downsample=v.temperal_downsample,
            dropout=v.dropout,
            input_channels=v.input_channels,
            out_channels=v.out_channels,
            is_residual=v.is_residual,
            scale_factor_spatial=v.scale_factor_spatial,
            latents_mean=v.latents_mean,
            latents_std=v.latents_std,
            sample_size=v.sample_size,
        )
        text_kw = (
            config.text_config.constructor_kwargs()
            if hasattr(config.text_config, "constructor_kwargs")
            else dict(config.text_config)
        )
        text_encoder = QwenImage21TextEncoderModel(**text_kw)
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
                shape=(transformer.img_seq, d.in_channels), name="sample"
            ),
            "timestep": layers.Input(shape=(), name="timestep"),
            "encoder_hidden_states": layers.Input(
                shape=(text_seq, d.context_in_dim), name="encoder_hidden_states"
            ),
            "encoder_hidden_states_mask": layers.Input(
                shape=(text_seq,), dtype="int32", name="encoder_hidden_states_mask"
            ),
            "image": layers.Input(
                shape=(img_h, img_w, v.input_channels), name="image"
            ),
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
            "On-the-fly hf: conversion is not supported for Qwen-Image-2.1; "
            "use convert_qwen_image_21_diffusers_to_keras.py and "
            "from_weights('zeromodels/qwen-image-2.1')."
        )


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21TextToImage(QwenImage21Model, BaseDiffusion):
    """Text-to-image Qwen-Image-2.1 (Diffusers ``QwenImage21Pipeline``).

    ::

        model = QwenImage21TextToImage.from_weights("zeromodels/qwen-image-2.1")
        tok = QwenImage21Tokenizer.from_weights("zeromodels/qwen-image-2.1")
        image = model.generate(**tok("a cat"), height=1024, width=1024)

    Defaults match the reference: 40 flow-match steps, ``guidance_scale=1.0``
    (no CFG). Pass ``guidance_scale>1`` with a negative prompt for true CFG.
    """

    config_class = QwenImage21Config
    HUB_REPO_SIBLINGS = QWEN_IMAGE_21_HUB_SIBLINGS
    generate_args = {"num_inference_steps": 40, "guidance_scale": 1.0}
    DEFAULT_GUIDANCE_SCALE = 1.0

    def __init__(self, scheduler=None, name="QwenImage21TextToImage", **kwargs):
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
        h, w = self._latent_side()
        return (h * w, self.vae.z_dim)

    def _latent_side(self, height=None, width=None):
        scale = self.vae_scale_factor * 2
        if height is None or width is None:
            side = self.config.default_sample_size * self.vae_scale_factor
            height = width = side
        h = 2 * (int(height) // scale)
        w = 2 * (int(width) // scale)
        return h, w

    def unconditional_ids(self, batch):
        length = self.config.text_config.max_seq_len
        row = [self.config.pad_token_id] * length
        return ops.convert_to_tensor([row] * batch, dtype="int32")

    def encode_prompt(self, input_ids, attention_mask=None, **conditioning):
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
                "encoder_hidden_states_mask": embeddings["encoder_hidden_states_mask"],
            }
        )["sample"]

    def decode_latents(self, latents, height=None, width=None):
        h, w = self._latent_side(height, width)
        latents = unpack_latents(latents, h, w, self.vae.z_dim)
        mean = ops.convert_to_tensor(self.vae.latents_mean, dtype="float32")
        std = ops.convert_to_tensor(self.vae.latents_std, dtype="float32")
        mean = ops.reshape(mean, (1, 1, 1, -1))
        std = ops.reshape(std, (1, 1, 1, -1))
        latents = latents * std + mean
        image = self.vae.decode(latents)
        # VAE emits RGBA; T2I postprocess uses RGB.
        if int(image.shape[-1]) == 4:
            image = image[..., :3]
        return image

    def prepare_latents(self, batch, seed=None, latents=None, dtype="float32"):
        h, w = self._latent_side(
            getattr(self, "_gen_height", None), getattr(self, "_gen_width", None)
        )
        channels = self.vae.z_dim
        if latents is None:
            noise = keras.random.normal((batch, h, w, channels), seed=seed, dtype=dtype)
            latents = pack_latents(noise, h, w)
        else:
            latents = ops.cast(ops.convert_to_tensor(latents), dtype)
        return latents * self.scheduler.init_noise_sigma

    def denoise(
        self, latents, embeddings, num_inference_steps, guidance_scale, timesteps=None
    ):
        do_cfg = guidance_scale > 1.0
        scheduler = self.scheduler
        if timesteps is None:
            scheduler.set_timesteps(num_inference_steps)
            timesteps = scheduler.timesteps
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
                noise_pred = neg_pred + guidance_scale * (noise_pred - neg_pred)
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
            if guidance_scale > 1.0 and negative_input_ids is None:
                # Diffusers only enables CFG when a negative prompt is provided.
                do_cfg = False
                uncond = None
            elif negative_input_ids is not None and guidance_scale > 1.0:
                uncond = self.encode_prompt(negative_input_ids, negative_attention_mask)
                do_cfg = True
            else:
                uncond = None
                do_cfg = False

            h, w = self._latent_side(height, width)
            image_seq_len = h * w
            sched_cfg = getattr(self.scheduler, "config_dict", None) or {}
            base_seq_len = sched_cfg.get("base_image_seq_len", 256)
            max_seq_len = sched_cfg.get("max_image_seq_len", 4096)
            base_shift = sched_cfg.get("base_shift", 0.5)
            max_shift = sched_cfg.get("max_shift", 0.9)
            m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
            mu = image_seq_len * m + (base_shift - m * base_seq_len)
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
