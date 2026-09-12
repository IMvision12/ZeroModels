import keras
from keras import layers, ops

from zeromodels.base import BaseModel
from zeromodels.base.base_diffusion import BaseDiffusion
from zeromodels.base.base_mixin import build_dtype_scope
from zeromodels.base.base_scheduler import PNDMScheduler, get_scheduler
from zeromodels.models.clip import CLIPTextModel

from .convert_stable_diffusion_diffusers_to_keras import (
    RenamedStateDict,
    transfer_component,
)
from .stable_diffusion_config import (
    AutoencoderKLConfig,
    StableDiffusionConfig,
    UNet2DConditionConfig,
)
from .stable_diffusion_layers import (
    GROUP_EPS,
    Downsample2D,
    ResnetBlock2D,
    Transformer2DModel,
    Upsample2D,
    VaeAttentionBlock,
    group_norm,
    safe_name,
    time_embedding_mlp,
    timestep_embedding,
)

# The container and the task build the identical graph, so both load the one
# hosted repo (whose zm_config.json names the container), like CLIP's heads.
STABLE_DIFFUSION_HUB_SIBLINGS = frozenset(
    {"StableDiffusionModel", "StableDiffusionTextToImage"}
)

CROSS_ATTN_DOWN = "CrossAttnDownBlock2D"
DOWN = "DownBlock2D"
CROSS_ATTN_UP = "CrossAttnUpBlock2D"
UP = "UpBlock2D"


@keras.saving.register_keras_serializable(package="zeromodels")
class UNet2DConditionModel(BaseModel):
    """Stable Diffusion denoising UNet (diffusers ``UNet2DConditionModel``).

    A channels-last, cross-backend port of the SD 1.x UNet: a symmetric
    down/mid/up convolutional UNet whose ResNet blocks are conditioned on the
    sinusoidal timestep embedding and whose ``CrossAttn`` blocks attend the image
    latents to the text ``encoder_hidden_states``. It predicts the noise (or ``v``)
    on a latent, and is called once per denoising step by the pipeline.

    Inputs are a dict ``{"sample", "timestep", "encoder_hidden_states"}`` (plus
    ``"text_embeds"`` and ``"time_ids"`` for the SDXL ``text_time`` conditioning);
    the output dict is ``{"sample": (B, H, W, out_channels)}``. Built for a fixed
    latent resolution (``sample_size``), since the Transformer2D reshapes need
    static spatial dims; the weights are resolution-independent, so rebuild the
    graph for another size and reload.

    Args mirror the diffusers config: ``in_channels`` / ``out_channels`` (4),
    ``block_out_channels`` ((320, 640, 1280, 1280)), ``layers_per_block`` (2),
    ``cross_attention_dim`` (768), ``num_attention_heads`` (8, or one value per
    level), ``norm_num_groups`` (32), ``use_linear_projection`` (a linear token
    projection instead of the 1x1 conv), ``transformer_layers_per_block`` (1, or
    one value per level: SDXL stacks (1, 2, 10) blocks per Transformer2D) and the
    ``down_block_types`` / ``up_block_types`` lists. ``addition_embed_type``
    ``"text_time"`` adds SDXL's micro-conditioning to the timestep embedding: the
    pooled text embedding and the ``num_time_ids`` size / crop values, each
    sinusoidally embedded to ``addition_time_embed_dim``, concatenated
    (``projection_class_embeddings_input_dim`` wide) and projected by the
    ``add_embedding`` MLP.
    """

    HF_MODEL_TYPE = None
    config_class = UNet2DConditionConfig

    def __init__(
        self,
        sample_size=64,
        in_channels=4,
        out_channels=4,
        down_block_types=(CROSS_ATTN_DOWN, CROSS_ATTN_DOWN, CROSS_ATTN_DOWN, DOWN),
        up_block_types=(UP, CROSS_ATTN_UP, CROSS_ATTN_UP, CROSS_ATTN_UP),
        block_out_channels=(320, 640, 1280, 1280),
        layers_per_block=2,
        cross_attention_dim=768,
        num_attention_heads=8,
        norm_num_groups=32,
        use_linear_projection=False,
        transformer_layers_per_block=1,
        addition_embed_type=None,
        addition_time_embed_dim=256,
        projection_class_embeddings_input_dim=None,
        num_time_ids=6,
        text_seq_len=77,
        data_format=None,
        channels_axis=None,
        name="UNet2DConditionModel",
        **kwargs,
    ):
        data_format = data_format or keras.config.image_data_format()
        if channels_axis is None:
            channels_axis = -1 if data_format == "channels_last" else 1
        levels = len(block_out_channels)
        # heads per level (diffusers' attention_head_dim list), mirrored on the way up
        if isinstance(num_attention_heads, (tuple, list)):
            heads_per_level = tuple(num_attention_heads)
        else:
            heads_per_level = (num_attention_heads,) * levels
        if isinstance(transformer_layers_per_block, (tuple, list)):
            depth_per_level = tuple(transformer_layers_per_block)
        else:
            depth_per_level = (transformer_layers_per_block,) * levels
        time_embed_dim = block_out_channels[0] * 4
        sample_h, sample_w = (
            sample_size
            if isinstance(sample_size, (tuple, list))
            else (sample_size, sample_size)
        )

        sample_in = layers.Input(
            shape=(in_channels, sample_h, sample_w)
            if data_format == "channels_first"
            else (sample_h, sample_w, in_channels),
            name="sample",
        )
        timestep_in = layers.Input(shape=(), name="timestep")
        context = layers.Input(
            shape=(text_seq_len, cross_attention_dim), name="encoder_hidden_states"
        )

        temb = timestep_embedding(timestep_in, block_out_channels[0])
        temb = time_embedding_mlp(temb, time_embed_dim, name="time_embedding")

        extra_inputs = {}
        if addition_embed_type == "text_time":
            # SDXL micro-conditioning: the pooled text embedding and the sinusoidal
            # embeddings of the size / crop ids, projected and added to temb
            pooled_dim = (
                projection_class_embeddings_input_dim
                - num_time_ids * addition_time_embed_dim
            )
            text_embeds_in = layers.Input(shape=(pooled_dim,), name="text_embeds")
            time_ids_in = layers.Input(shape=(num_time_ids,), name="time_ids")
            time_embeds = timestep_embedding(
                ops.reshape(time_ids_in, (-1,)), addition_time_embed_dim
            )
            time_embeds = ops.reshape(
                time_embeds, (-1, num_time_ids * addition_time_embed_dim)
            )
            add_embeds = ops.concatenate([text_embeds_in, time_embeds], axis=-1)
            temb = temb + time_embedding_mlp(
                add_embeds, time_embed_dim, name="add_embedding"
            )
            extra_inputs = {"text_embeds": text_embeds_in, "time_ids": time_ids_in}
        elif addition_embed_type is not None:
            raise ValueError(
                f"Unsupported addition_embed_type {addition_embed_type!r}; expected "
                "None or 'text_time'."
            )

        sample = layers.Conv2D(
            block_out_channels[0],
            3,
            padding="same",
            data_format=data_format,
            name="conv_in",
        )(sample_in)

        skips = [sample]
        for i, block_type in enumerate(down_block_types):
            out_ch = block_out_channels[i]
            for j in range(layers_per_block):
                sample = ResnetBlock2D(
                    out_ch,
                    module_path=f"down_blocks.{i}.resnets.{j}",
                    groups=norm_num_groups,
                    data_format=data_format,
                    channels_axis=channels_axis,
                )([sample, temb])
                if block_type == CROSS_ATTN_DOWN:
                    sample = Transformer2DModel(
                        out_ch,
                        heads_per_level[i],
                        module_path=f"down_blocks.{i}.attentions.{j}",
                        groups=norm_num_groups,
                        use_linear_projection=use_linear_projection,
                        num_layers=depth_per_level[i],
                        data_format=data_format,
                        channels_axis=channels_axis,
                    )([sample, context])
                skips.append(sample)
            if i != len(down_block_types) - 1:
                sample = Downsample2D(
                    out_ch,
                    module_path=f"down_blocks.{i}.downsamplers.0",
                    data_format=data_format,
                    channels_axis=channels_axis,
                )(sample)
                skips.append(sample)

        mid_ch = block_out_channels[-1]
        sample = ResnetBlock2D(
            mid_ch,
            module_path="mid_block.resnets.0",
            groups=norm_num_groups,
            data_format=data_format,
            channels_axis=channels_axis,
        )([sample, temb])
        sample = Transformer2DModel(
            mid_ch,
            heads_per_level[-1],
            module_path="mid_block.attentions.0",
            groups=norm_num_groups,
            use_linear_projection=use_linear_projection,
            num_layers=depth_per_level[-1],
            data_format=data_format,
            channels_axis=channels_axis,
        )([sample, context])
        sample = ResnetBlock2D(
            mid_ch,
            module_path="mid_block.resnets.1",
            groups=norm_num_groups,
            data_format=data_format,
            channels_axis=channels_axis,
        )([sample, temb])

        reversed_channels = list(reversed(block_out_channels))
        reversed_heads = list(reversed(heads_per_level))
        reversed_depths = list(reversed(depth_per_level))
        for i, block_type in enumerate(up_block_types):
            out_ch = reversed_channels[i]
            for j in range(layers_per_block + 1):
                sample = layers.Concatenate(axis=channels_axis)([sample, skips.pop()])
                sample = ResnetBlock2D(
                    out_ch,
                    module_path=f"up_blocks.{i}.resnets.{j}",
                    groups=norm_num_groups,
                    data_format=data_format,
                    channels_axis=channels_axis,
                )([sample, temb])
                if block_type == CROSS_ATTN_UP:
                    sample = Transformer2DModel(
                        out_ch,
                        reversed_heads[i],
                        module_path=f"up_blocks.{i}.attentions.{j}",
                        groups=norm_num_groups,
                        use_linear_projection=use_linear_projection,
                        num_layers=reversed_depths[i],
                        data_format=data_format,
                        channels_axis=channels_axis,
                    )([sample, context])
            if i != len(up_block_types) - 1:
                sample = Upsample2D(
                    out_ch,
                    module_path=f"up_blocks.{i}.upsamplers.0",
                    data_format=data_format,
                    channels_axis=channels_axis,
                )(sample)

        sample = group_norm(
            sample,
            name="conv_norm_out",
            channels_axis=channels_axis,
            groups=norm_num_groups,
        )
        sample = ops.silu(sample)
        sample = layers.Conv2D(
            out_channels,
            3,
            padding="same",
            data_format=data_format,
            name="conv_out",
        )(sample)

        super().__init__(
            inputs={
                "sample": sample_in,
                "timestep": timestep_in,
                "encoder_hidden_states": context,
                **extra_inputs,
            },
            outputs={"sample": sample},
            name=name,
            **kwargs,
        )

        self.data_format = data_format
        self.channels_axis = channels_axis
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_block_types = tuple(down_block_types)
        self.up_block_types = tuple(up_block_types)
        self.block_out_channels = tuple(block_out_channels)
        self.layers_per_block = layers_per_block
        self.cross_attention_dim = cross_attention_dim
        self.num_attention_heads = (
            tuple(num_attention_heads)
            if isinstance(num_attention_heads, (tuple, list))
            else num_attention_heads
        )
        self.norm_num_groups = norm_num_groups
        self.use_linear_projection = use_linear_projection
        self.transformer_layers_per_block = (
            tuple(transformer_layers_per_block)
            if isinstance(transformer_layers_per_block, (tuple, list))
            else transformer_layers_per_block
        )
        self.addition_embed_type = addition_embed_type
        self.addition_time_embed_dim = addition_time_embed_dim
        self.projection_class_embeddings_input_dim = (
            projection_class_embeddings_input_dim
        )
        self.num_time_ids = num_time_ids
        self.text_seq_len = text_seq_len

    @classmethod
    def from_diffusers_config(cls, config, **kwargs):
        """Build from a diffusers ``unet/config.json`` dict."""
        return cls(**cls.kwargs_from_diffusers_config(config), **kwargs)

    @staticmethod
    def kwargs_from_diffusers_config(config):
        """The constructor kwargs a diffusers ``unet/config.json`` dict describes."""
        # diffusers' "attention_head_dim" is the head COUNT (legacy name), a scalar
        # or one value per level
        heads = config.get("num_attention_heads") or config.get("attention_head_dim", 8)
        if isinstance(heads, (list, tuple)):
            heads = tuple(heads)
        depth = config.get("transformer_layers_per_block", 1)
        if isinstance(depth, (list, tuple)):
            depth = tuple(depth)
        kwargs = {
            "sample_size": config.get("sample_size", 64),
            "in_channels": config.get("in_channels", 4),
            "out_channels": config.get("out_channels", 4),
            "down_block_types": tuple(config["down_block_types"]),
            "up_block_types": tuple(config["up_block_types"]),
            "block_out_channels": tuple(config["block_out_channels"]),
            "layers_per_block": config.get("layers_per_block", 2),
            "cross_attention_dim": config.get("cross_attention_dim", 768),
            "num_attention_heads": heads,
            "norm_num_groups": config.get("norm_num_groups", 32),
            "use_linear_projection": config.get("use_linear_projection", False),
            "transformer_layers_per_block": depth,
        }
        if config.get("addition_embed_type") is not None:
            kwargs.update(
                addition_embed_type=config["addition_embed_type"],
                addition_time_embed_dim=config.get("addition_time_embed_dim", 256),
                projection_class_embeddings_input_dim=config[
                    "projection_class_embeddings_input_dim"
                ],
            )
        return kwargs

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        transfer_component(keras_model, state_dict)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sample_size": self.sample_size,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "down_block_types": self.down_block_types,
                "up_block_types": self.up_block_types,
                "block_out_channels": self.block_out_channels,
                "layers_per_block": self.layers_per_block,
                "cross_attention_dim": self.cross_attention_dim,
                "num_attention_heads": self.num_attention_heads,
                "norm_num_groups": self.norm_num_groups,
                "use_linear_projection": self.use_linear_projection,
                "transformer_layers_per_block": self.transformer_layers_per_block,
                "addition_embed_type": self.addition_embed_type,
                "addition_time_embed_dim": self.addition_time_embed_dim,
                "projection_class_embeddings_input_dim": (
                    self.projection_class_embeddings_input_dim
                ),
                "num_time_ids": self.num_time_ids,
                "text_seq_len": self.text_seq_len,
                "name": self.name,
            }
        )
        return config


def vae_resnet(channels, module_path, groups, data_format, channels_axis):
    """A VAE ``ResnetBlock2D``: no timestep conditioning, GroupNorm eps 1e-6."""
    return ResnetBlock2D(
        channels,
        module_path=module_path,
        groups=groups,
        eps=GROUP_EPS,
        time_embedding=False,
        data_format=data_format,
        channels_axis=channels_axis,
    )


def build_vae_encoder(
    image_size,
    in_channels,
    block_out_channels,
    layers_per_block,
    latent_channels,
    groups,
    data_format,
    channels_axis,
    name="vae_encoder",
):
    """Functional VAE encoder ``image -> 2*latent_channels moments``; its leaves are
    named with the ``encoder.*`` keys of a diffusers ``AutoencoderKL``."""
    h_img, w_img = (
        image_size
        if isinstance(image_size, (tuple, list))
        else (image_size, image_size)
    )
    img = layers.Input(
        shape=(in_channels, h_img, w_img)
        if data_format == "channels_first"
        else (h_img, w_img, in_channels),
        name="image",
    )
    x = layers.Conv2D(
        block_out_channels[0],
        3,
        padding="same",
        data_format=data_format,
        name=safe_name("encoder.conv_in"),
    )(img)
    for i, ch in enumerate(block_out_channels):
        for j in range(layers_per_block):
            x = vae_resnet(
                ch,
                f"encoder.down_blocks.{i}.resnets.{j}",
                groups,
                data_format,
                channels_axis,
            )(x)
        if i != len(block_out_channels) - 1:
            # the VAE pads bottom/right only before its stride-2 conv (padding=0)
            x = Downsample2D(
                ch,
                module_path=f"encoder.down_blocks.{i}.downsamplers.0",
                padding=0,
                data_format=data_format,
                channels_axis=channels_axis,
            )(x)
    mid = block_out_channels[-1]
    x = vae_resnet(
        mid, "encoder.mid_block.resnets.0", groups, data_format, channels_axis
    )(x)
    x = VaeAttentionBlock(
        mid,
        module_path="encoder.mid_block.attentions.0",
        groups=groups,
        data_format=data_format,
        channels_axis=channels_axis,
    )(x)
    x = vae_resnet(
        mid, "encoder.mid_block.resnets.1", groups, data_format, channels_axis
    )(x)
    x = group_norm(
        x,
        name="encoder.conv_norm_out",
        channels_axis=channels_axis,
        groups=groups,
        eps=GROUP_EPS,
    )
    x = ops.silu(x)
    x = layers.Conv2D(
        2 * latent_channels,
        3,
        padding="same",
        data_format=data_format,
        name=safe_name("encoder.conv_out"),
    )(x)
    return keras.Model(img, x, name=name)


def build_vae_decoder(
    latent_size,
    out_channels,
    block_out_channels,
    layers_per_block,
    latent_channels,
    groups,
    data_format,
    channels_axis,
    name="vae_decoder",
):
    """Functional VAE decoder ``latent -> image``; its leaves are named with the
    ``decoder.*`` keys of a diffusers ``AutoencoderKL``."""
    h_lat, w_lat = (
        latent_size
        if isinstance(latent_size, (tuple, list))
        else (latent_size, latent_size)
    )
    z = layers.Input(
        shape=(latent_channels, h_lat, w_lat)
        if data_format == "channels_first"
        else (h_lat, w_lat, latent_channels),
        name="latent",
    )
    mid = block_out_channels[-1]
    x = layers.Conv2D(
        mid,
        3,
        padding="same",
        data_format=data_format,
        name=safe_name("decoder.conv_in"),
    )(z)
    x = vae_resnet(
        mid, "decoder.mid_block.resnets.0", groups, data_format, channels_axis
    )(x)
    x = VaeAttentionBlock(
        mid,
        module_path="decoder.mid_block.attentions.0",
        groups=groups,
        data_format=data_format,
        channels_axis=channels_axis,
    )(x)
    x = vae_resnet(
        mid, "decoder.mid_block.resnets.1", groups, data_format, channels_axis
    )(x)
    reversed_channels = list(reversed(block_out_channels))
    for i, ch in enumerate(reversed_channels):
        for j in range(layers_per_block + 1):
            x = vae_resnet(
                ch,
                f"decoder.up_blocks.{i}.resnets.{j}",
                groups,
                data_format,
                channels_axis,
            )(x)
        if i != len(reversed_channels) - 1:
            x = Upsample2D(
                ch,
                module_path=f"decoder.up_blocks.{i}.upsamplers.0",
                data_format=data_format,
                channels_axis=channels_axis,
            )(x)
    x = group_norm(
        x,
        name="decoder.conv_norm_out",
        channels_axis=channels_axis,
        groups=groups,
        eps=GROUP_EPS,
    )
    x = ops.silu(x)
    x = layers.Conv2D(
        out_channels,
        3,
        padding="same",
        data_format=data_format,
        name=safe_name("decoder.conv_out"),
    )(x)
    return keras.Model(z, x, name=name)


@keras.saving.register_keras_serializable(package="zeromodels")
class AutoencoderKL(BaseModel):
    """Stable Diffusion VAE (diffusers ``AutoencoderKL``), channels-last.

    Maps an image to a low-resolution latent and back: the text-to-image task
    decodes the denoised latent to pixels with :meth:`decode`, and img2img encodes
    with :meth:`encode`. Downsamples by ``2 ** (len(block_out_channels) - 1)`` (8
    for Stable Diffusion), so a 512x512 image has a 64x64x4 latent.

    Encode and decode are separate entry points, so, like ``CLIPModel``'s two
    towers, this is one functional graph with two disconnected paths:
    ``{"image", "latent"}`` in, ``{"moments", "sample"}`` out
    (``image -> encoder -> quant_conv -> moments`` and
    ``latent -> post_quant_conv -> decoder -> sample``). :meth:`encode` and
    :meth:`decode` run one path on its own. Being a single functional model is what
    lets its weights ride inside the hosted :class:`StableDiffusionModel`'s one
    ``model.weights.h5``.

    Args mirror the diffusers config: ``in_channels`` / ``out_channels`` (3),
    ``latent_channels`` (4), ``block_out_channels`` ((128, 256, 512, 512)),
    ``layers_per_block`` (2), ``norm_num_groups`` (32), ``sample_size`` (the image
    resolution the encoder/decoder graphs are built for), ``scaling_factor``
    (0.18215) and ``force_upcast`` (build this VAE in float32 whatever dtype the
    rest of the model loads in: the SDXL VAE overflows in float16).
    """

    config_class = AutoencoderKLConfig
    HF_MODEL_TYPE = None

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        norm_num_groups=32,
        sample_size=512,
        scaling_factor=0.18215,
        force_upcast=False,
        data_format=None,
        channels_axis=None,
        name="AutoencoderKL",
        **kwargs,
    ):
        data_format = data_format or keras.config.image_data_format()
        if channels_axis is None:
            channels_axis = -1 if data_format == "channels_last" else 1
        h_img, w_img = (
            sample_size
            if isinstance(sample_size, (tuple, list))
            else (sample_size, sample_size)
        )
        factor = 2 ** (len(block_out_channels) - 1)
        h_lat, w_lat = h_img // factor, w_img // factor

        # force_upcast: the layers are created under a float32 policy whatever the
        # global (load) dtype is, as diffusers upcasts such a VAE around decode
        with build_dtype_scope("float32" if force_upcast else None):
            encoder = build_vae_encoder(
                (h_img, w_img),
                in_channels,
                block_out_channels,
                layers_per_block,
                latent_channels,
                norm_num_groups,
                data_format,
                channels_axis,
            )
            decoder = build_vae_decoder(
                (h_lat, w_lat),
                out_channels,
                block_out_channels,
                layers_per_block,
                latent_channels,
                norm_num_groups,
                data_format,
                channels_axis,
            )
            quant_conv = layers.Conv2D(
                2 * latent_channels,
                1,
                data_format=data_format,
                name="quant_conv",
            )
            post_quant_conv = layers.Conv2D(
                latent_channels,
                1,
                data_format=data_format,
                name="post_quant_conv",
            )

            image_in = layers.Input(
                shape=(in_channels, h_img, w_img)
                if data_format == "channels_first"
                else (h_img, w_img, in_channels),
                name="image",
            )
            latent_in = layers.Input(
                shape=(latent_channels, h_lat, w_lat)
                if data_format == "channels_first"
                else (h_lat, w_lat, latent_channels),
                name="latent",
            )
            moments = quant_conv(encoder(image_in))
            decoded = decoder(post_quant_conv(latent_in))

            super().__init__(
                inputs={"image": image_in, "latent": latent_in},
                outputs={"moments": moments, "sample": decoded},
                name=name,
                **kwargs,
            )

        self.data_format = data_format
        self.channels_axis = channels_axis
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.block_out_channels = tuple(block_out_channels)
        self.layers_per_block = layers_per_block
        self.norm_num_groups = norm_num_groups
        self.sample_size = sample_size
        self.scaling_factor = scaling_factor
        self.force_upcast = force_upcast
        self.vae_scale_factor = factor
        self.encoder = encoder
        self.decoder = decoder
        self.quant_conv = quant_conv
        self.post_quant_conv = post_quant_conv

    def encode(self, image, sample=False, seed=None):
        moments = self.quant_conv(self.encoder(image))
        mean, logvar = ops.split(moments, 2, axis=self.channels_axis)
        if not sample:
            return mean
        logvar = ops.clip(logvar, -30.0, 20.0)
        std = ops.exp(0.5 * logvar)
        noise = keras.random.normal(ops.shape(mean), dtype=mean.dtype, seed=seed)
        return mean + std * noise

    def decode(self, latent):
        return self.decoder(self.post_quant_conv(latent))

    def get_config(self):
        config = super().get_config()
        config.update(self.config.constructor_kwargs())
        return config

    @classmethod
    def from_diffusers_config(cls, config, sample_size=512, **kwargs):
        return cls(
            in_channels=config.get("in_channels", 3),
            out_channels=config.get("out_channels", 3),
            latent_channels=config.get("latent_channels", 4),
            block_out_channels=tuple(config["block_out_channels"]),
            layers_per_block=config.get("layers_per_block", 2),
            norm_num_groups=config.get("norm_num_groups", 32),
            sample_size=sample_size,
            scaling_factor=config.get("scaling_factor", 0.18215),
            force_upcast=config.get("force_upcast", False),
            **kwargs,
        )

    def transfer_from_hf(self, state_dict):
        # legacy query/key/value/proj_attn spelling of a raw checkpoint read
        transfer_component(self, RenamedStateDict(state_dict))


@keras.saving.register_keras_serializable(package="zeromodels")
class StableDiffusionModel(BaseModel):
    """Stable Diffusion 1.x weights container: the UNet, the VAE and the CLIP text
    encoder as one functional ``keras.Model``.

    This is the hosted unit: a ``zeromodels/<variant>`` repo carries one
    ``zm_config.json`` (a :class:`StableDiffusionConfig`) and one
    ``model.weights.h5``, and the standard ``from_weights("zeromodels/<variant>")``
    loader builds this container and loads every component's weights in one go. It
    is the ``XModel`` of the family; :class:`StableDiffusionTextToImage` is the task
    class, this same graph plus the denoising loop (``BaseDiffusion``) and a
    scheduler, the way ``T5ConditionalGenerate`` extends ``T5Model``.

    Like ``CLIPModel`` (text + vision towers in one graph), it is a single
    functional model whose graph is three disconnected paths, one per component:
    ``{"sample", "timestep", "encoder_hidden_states"} -> "noise_pred"`` (UNet),
    ``{"image", "latent"} -> "moments" / "image"`` (VAE) and
    ``{"token_ids", "padding_mask"} -> "text_embeds"`` (CLIP). The components are
    exposed as ``.unet`` / ``.vae`` / ``.text_encoder`` (they share this model's
    weights) and are what the task class runs step by step; calling the container
    end to end is rarely useful. On-the-fly ``hf:`` conversion is deliberately
    unsupported for diffusion models; the weights are converted once by
    ``convert_stable_diffusion_diffusers_to_keras.py`` and hosted.

    The constructor is flat, like every zeromodels model: the sub-config fields
    arrive prefixed ``unet_`` / ``vae_`` / ``text_`` (see
    :class:`StableDiffusionConfig`), or pass a config positionally; ``.config`` is
    always the typed config. Pass ``unet_sample_size`` / ``vae_sample_size`` to
    build for another resolution (the weights are resolution-independent).
    """

    config_class = StableDiffusionConfig
    HF_MODEL_TYPE = None
    HUB_REPO_SIBLINGS = STABLE_DIFFUSION_HUB_SIBLINGS

    def __init__(self, name="StableDiffusionModel", **kwargs):
        keras_kwargs = {k: kwargs.pop(k) for k in ("trainable", "dtype") if k in kwargs}
        config = self.config_class.from_dict(kwargs)  # regroup the flat kwargs

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

        components = self.build_components(config, data_format, channels_axis)
        inputs, outputs = self.build_graph(config, components, data_format)
        super().__init__(inputs=inputs, outputs=outputs, name=name, **keras_kwargs)

        self.data_format = data_format
        self.channels_axis = channels_axis
        for attr, component in components.items():
            setattr(self, attr, component)

    def build_components(self, config, data_format, channels_axis):
        """The trained components, ``{attribute name: model}``: the UNet, the VAE
        and the CLIP text tower (subclasses swap or add towers)."""
        u, v, t = config.unet_config, config.vae_config, config.text_config
        unet = UNet2DConditionModel(
            u, data_format=data_format, channels_axis=channels_axis
        )
        vae = AutoencoderKL(
            **v.constructor_kwargs(),
            data_format=data_format,
            channels_axis=channels_axis,
        )
        text_encoder = CLIPTextModel(
            max_seq_len=t.max_seq_len,
            vocab_size=t.vocab_size,
            text_hidden_dim=t.hidden_dim,
            text_num_heads=t.num_heads,
            text_num_layers=t.num_layers,
            text_mlp_ratio=t.mlp_ratio,
            hidden_act=config.hidden_act,
            layer_norm_eps=config.layer_norm_eps,
        )
        return {"unet": unet, "vae": vae, "text_encoder": text_encoder}

    def unet_vae_inputs(self, config, components, data_format):
        """The UNet's and the VAE's ``keras.Input`` dict (shared by every SD family)."""
        u, v = config.unet_config, config.vae_config
        vae = components["vae"]
        sample_h, sample_w = (
            u.sample_size
            if isinstance(u.sample_size, (tuple, list))
            else (u.sample_size, u.sample_size)
        )
        img_h, img_w = (
            v.sample_size
            if isinstance(v.sample_size, (tuple, list))
            else (v.sample_size, v.sample_size)
        )
        lat_h, lat_w = img_h // vae.vae_scale_factor, img_w // vae.vae_scale_factor
        inputs = {
            "sample": layers.Input(
                shape=(u.in_channels, sample_h, sample_w)
                if data_format == "channels_first"
                else (sample_h, sample_w, u.in_channels),
                name="sample",
            ),
            "timestep": layers.Input(shape=(), name="timestep"),
            "encoder_hidden_states": layers.Input(
                shape=(u.text_seq_len, u.cross_attention_dim),
                name="encoder_hidden_states",
            ),
            "image": layers.Input(
                shape=(v.in_channels, img_h, img_w)
                if data_format == "channels_first"
                else (img_h, img_w, v.in_channels),
                name="image",
            ),
            "latent": layers.Input(
                shape=(v.latent_channels, lat_h, lat_w)
                if data_format == "channels_first"
                else (lat_h, lat_w, v.latent_channels),
                name="latent",
            ),
        }
        return inputs

    def build_graph(self, config, components, data_format):
        """Wire the components into the container's ``(inputs, outputs)`` dicts:
        three disconnected paths, one per component."""
        t = config.text_config
        unet, vae, text_encoder = (
            components["unet"],
            components["vae"],
            components["text_encoder"],
        )
        inputs = self.unet_vae_inputs(config, components, data_format)
        inputs["token_ids"] = layers.Input(
            shape=(t.max_seq_len,), dtype="int32", name="token_ids"
        )
        inputs["padding_mask"] = layers.Input(
            shape=(t.max_seq_len,), dtype="int32", name="padding_mask"
        )

        noise_pred = unet(
            {
                "sample": inputs["sample"],
                "timestep": inputs["timestep"],
                "encoder_hidden_states": inputs["encoder_hidden_states"],
            }
        )["sample"]
        vae_out = vae({"image": inputs["image"], "latent": inputs["latent"]})
        text_embeds = text_encoder(
            {"token_ids": inputs["token_ids"], "padding_mask": inputs["padding_mask"]}
        )["last_hidden_state"]
        outputs = {
            "noise_pred": noise_pred,
            "moments": vae_out["moments"],
            "image": vae_out["sample"],
            "text_embeds": text_embeds,
        }
        return inputs, outputs

    @classmethod
    def from_hf(cls, repo, **kwargs):
        raise NotImplementedError(
            "On-the-fly `hf:` conversion is not supported for diffusion models. Load "
            "the preconverted weights from a zeromodels repo, e.g. "
            f"{cls.__name__}.from_weights('zeromodels/stable-diffusion-v1-5'), or run "
            "convert_stable_diffusion_diffusers_to_keras.py to produce them from "
            f"{repo!r}."
        )

    @classmethod
    def from_variant(cls, variant, /, quantization=None, **kwargs):
        # A bare name is the hosted repo's basename. `quantization` is applied by
        # from_weights after the load, exactly as on the org/repo branch.
        return cls.from_hub_repo(f"zeromodels/{variant}", **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(self.config.constructor_kwargs())
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class StableDiffusionTextToImage(StableDiffusionModel, BaseDiffusion):
    """Text-to-image Stable Diffusion 1.x, pure Keras 3 and cross-backend (the
    diffusers ``StableDiffusionPipeline`` in this library's task-class form).

    :class:`StableDiffusionModel` (the UNet, VAE and CLIP text towers, built
    internally like ``CLIPModel``) plus :class:`BaseDiffusion`'s ``generate``, the
    way ``T5ConditionalGenerate`` is ``T5Model`` plus a generation mixin: this class
    only supplies the hooks (``encode_prompt`` on the CLIP tower, the empty prompt's
    ``unconditional_ids``, ``predict_noise`` on the UNet, ``decode_latents`` on the
    VAE) and the scheduler. It is the same graph and weights as the container, so
    both classes load the same hosted repo.

    Inference is the library's usual two-step, tokenizer then ``generate``::

        sd = StableDiffusionTextToImage.from_weights("zeromodels/stable-diffusion-v1-5")
        tokenizer = StableDiffusionTokenizer.from_weights("zeromodels/stable-diffusion-v1-5")
        image = sd.generate(**tokenizer("a photo of a cat"))  # (1, 512, 512, 3) uint8

    The graphs are built for the repo's resolution (512px); pass
    ``unet_sample_size`` / ``vae_sample_size`` overrides to ``from_weights`` to
    build for another size (the weights are resolution-independent). On-the-fly
    ``hf:`` conversion is not supported for diffusion models.

    Args:
        scheduler: A :class:`BaseScheduler`. Defaults to the one the config's
            ``scheduler_config`` describes (the checkpoint's own sampler and noise
            schedule, carried in the repo's ``zm_config.json``), or PNDM with the SD
            1.x schedule when the config has none. Swap any time with
            ``sd.scheduler = EulerDiscreteScheduler()``.
        **kwargs: The flat :class:`StableDiffusionConfig` fields (see
            :class:`StableDiffusionModel`).
    """

    HUB_REPO_SIBLINGS = STABLE_DIFFUSION_HUB_SIBLINGS
    # Default generation settings (diffusers' StableDiffusionPipeline defaults);
    # a repo's zm_config.json generate_args override them on load.
    generate_args = {"num_inference_steps": 50, "guidance_scale": 7.5}

    def __init__(self, scheduler=None, name="StableDiffusionTextToImage", **kwargs):
        # read from the flat kwargs: the metaclass attaches self.config only after
        # __init__ returns
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
        """The family's sampler when the config carries no ``scheduler_config``."""
        return PNDMScheduler(steps_offset=1)

    @property
    def latent_shape(self):
        size = self.unet.sample_size
        height, width = size if isinstance(size, (tuple, list)) else (size, size)
        if self.data_format == "channels_first":
            return (self.unet.in_channels, height, width)
        return (height, width, self.unet.in_channels)

    def unconditional_ids(self, batch):
        cfg = self.config
        length = cfg.text_config.max_seq_len
        row = [cfg.bos_token_id, cfg.eos_token_id] + [cfg.pad_token_id] * (length - 2)
        return ops.convert_to_tensor([row] * batch, dtype="int32")

    def encode_prompt(self, input_ids, attention_mask=None):
        # SD attends the padding tokens (the reference passes no attention mask),
        # so the text tower gets an all-ones mask regardless of the tokenizer's
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        mask = ops.ones_like(input_ids)
        out = self.text_encoder({"token_ids": input_ids, "padding_mask": mask})
        return out["last_hidden_state"]

    def predict_noise(self, latents, timesteps, embeddings):
        return self.unet(
            {
                "sample": latents,
                "timestep": timesteps,
                "encoder_hidden_states": embeddings,
            }
        )["sample"]

    def decode_latents(self, latents):
        image = self.vae.decode(latents / self.vae.scaling_factor)
        if self.data_format == "channels_first":
            image = ops.transpose(image, (0, 2, 3, 1))  # generate() hands out HWC
        return image

    def encode_latents(self, image):
        # the posterior mean, scaled: the image-to-image starting point (the
        # reference samples the posterior; its noise is negligible next to the
        # noise added for the strength)
        if self.data_format == "channels_first":
            image = ops.transpose(image, (0, 3, 1, 2))  # generate() hands in HWC
        return self.vae.encode(image) * self.vae.scaling_factor
