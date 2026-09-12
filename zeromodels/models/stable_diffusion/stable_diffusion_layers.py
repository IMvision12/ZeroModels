import math

import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention

GROUP_EPS = 1e-6
GROUPS = 32


def safe_name(module_path):
    return module_path.replace(".", "__")


def timestep_embedding(
    timesteps, dim, flip_sin_to_cos=True, downscale_freq_shift=0, max_period=10000
):
    """Sinusoidal timestep embedding ``(B,) -> (B, dim)`` (diffusers ``Timesteps``).

    Matches ``get_timestep_embedding``: half the channels are cosines and half
    sines of geometrically spaced frequencies; ``flip_sin_to_cos`` puts cosines
    first (the Stable Diffusion convention).
    """
    half = dim // 2
    exponent = -math.log(max_period) * ops.arange(0, half, dtype="float32")
    exponent = exponent / (half - downscale_freq_shift)
    freqs = ops.exp(exponent)
    args = ops.cast(timesteps, "float32")[:, None] * freqs[None, :]
    if flip_sin_to_cos:
        emb = ops.concatenate([ops.cos(args), ops.sin(args)], axis=-1)
    else:
        emb = ops.concatenate([ops.sin(args), ops.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = ops.concatenate([emb, ops.zeros_like(emb[:, :1])], axis=-1)
    return emb


def time_embedding_mlp(temb, dim, name):
    """``Linear -> SiLU -> Linear`` timestep-embedding projection."""
    temb = layers.Dense(dim, name=safe_name(f"{name}.linear_1"))(temb)
    temb = ops.silu(temb)
    temb = layers.Dense(dim, name=safe_name(f"{name}.linear_2"))(temb)
    return temb


def group_norm(x, name, channels_axis, groups=GROUPS, eps=1e-5):
    return layers.GroupNormalization(
        groups=groups, axis=channels_axis, epsilon=eps, name=safe_name(name)
    )(x)


@keras.saving.register_keras_serializable(package="zeromodels")
class ResnetBlock2D(layers.Layer):
    """Diffusers ``ResnetBlock2D``: GroupNorm/SiLU/conv, additive time embedding,
    GroupNorm/SiLU/conv, plus a 1x1 shortcut when the channel count changes.

    Args:
        out_channels: Output channel count (the shortcut conv appears when it differs
            from the input's).
        module_path: The block's diffusers module path (``down_blocks.0.resnets.0``); the
            leaves are named ``<module_path>.norm1`` etc. for the converter.
        groups: GroupNorm groups.
        eps: GroupNorm epsilon (1e-5 in the UNet, 1e-6 in the VAE).
        time_embedding: Whether the block is conditioned on a timestep embedding
            (``call([x, temb])``); the VAE's blocks are not (``call(x)``).
        data_format: ``"channels_last"`` or ``"channels_first"``; defaults to
            ``keras.config.image_data_format()``.
        channels_axis: The channel axis of that layout (``-1`` or ``1``).
    """

    def __init__(
        self,
        out_channels,
        module_path,
        groups=GROUPS,
        eps=1e-5,
        time_embedding=True,
        data_format=None,
        channels_axis=None,
        **kwargs,
    ):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.module_path = module_path
        self.groups = groups
        self.eps = eps
        self.time_embedding = time_embedding
        self.data_format = data_format or keras.config.image_data_format()
        self.channels_axis = (
            channels_axis
            if channels_axis is not None
            else (-1 if self.data_format == "channels_last" else 1)
        )

        self.norm1 = layers.GroupNormalization(
            groups=groups,
            axis=self.channels_axis,
            epsilon=eps,
            name=safe_name(f"{module_path}.norm1"),
        )
        self.conv1 = layers.Conv2D(
            out_channels,
            3,
            padding="same",
            data_format=self.data_format,
            name=safe_name(f"{module_path}.conv1"),
        )
        self.time_emb_proj = (
            layers.Dense(out_channels, name=safe_name(f"{module_path}.time_emb_proj"))
            if time_embedding
            else None
        )
        self.norm2 = layers.GroupNormalization(
            groups=groups,
            axis=self.channels_axis,
            epsilon=eps,
            name=safe_name(f"{module_path}.norm2"),
        )
        self.conv2 = layers.Conv2D(
            out_channels,
            3,
            padding="same",
            data_format=self.data_format,
            name=safe_name(f"{module_path}.conv2"),
        )
        self.conv_shortcut = None

    def build(self, input_shape):
        x_shape = input_shape[0] if self.time_embedding else input_shape
        in_channels = int(x_shape[self.channels_axis])
        if self.data_format == "channels_first":
            mid_shape = (x_shape[0], self.out_channels, x_shape[2], x_shape[3])
        else:
            mid_shape = (x_shape[0], x_shape[1], x_shape[2], self.out_channels)
        self.norm1.build(x_shape)
        self.conv1.build(x_shape)
        if self.time_embedding:
            self.time_emb_proj.build(input_shape[1])
        self.norm2.build(mid_shape)
        self.conv2.build(mid_shape)
        if in_channels != self.out_channels:
            self.conv_shortcut = layers.Conv2D(
                self.out_channels,
                1,
                padding="valid",
                data_format=self.data_format,
                name=safe_name(f"{self.module_path}.conv_shortcut"),
            )
            self.conv_shortcut.build(x_shape)
        self.built = True

    def call(self, inputs):
        x, temb = inputs if self.time_embedding else (inputs, None)
        h = self.conv1(ops.silu(self.norm1(x)))
        if temb is not None:
            t = self.time_emb_proj(ops.silu(temb))
            if self.data_format == "channels_first":
                h = h + t[:, :, None, None]
            else:
                h = h + t[:, None, None, :]
        h = self.conv2(ops.silu(self.norm2(h)))
        residual = x if self.conv_shortcut is None else self.conv_shortcut(x)
        return residual + h

    def compute_output_shape(self, input_shape):
        x_shape = input_shape[0] if self.time_embedding else input_shape
        if self.data_format == "channels_first":
            return (x_shape[0], self.out_channels, x_shape[2], x_shape[3])
        return (x_shape[0], x_shape[1], x_shape[2], self.out_channels)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_channels": self.out_channels,
                "module_path": self.module_path,
                "groups": self.groups,
                "eps": self.eps,
                "time_embedding": self.time_embedding,
                "data_format": self.data_format,
                "channels_axis": self.channels_axis,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class CrossAttention(layers.Layer):
    """Diffusers ``Attention`` (``to_q`` / ``to_k`` / ``to_v`` / ``to_out.0``) over
    ``(B, N, C)`` tokens; self-attention when called without a context.

    Args:
        query_dim: Token width (also the inner attention width).
        heads: Attention heads.
        module_path: Diffusers module path of the attention (``...attn1``).
        qkv_bias: Whether the q/k/v projections carry a bias: the UNet's do not,
            the VAE's (diffusers' pre-refactor ``AttentionBlock``) do.
    """

    def __init__(self, query_dim, heads, module_path, qkv_bias=False, **kwargs):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.query_dim = query_dim
        self.heads = heads
        self.module_path = module_path
        self.qkv_bias = qkv_bias
        self.head_dim = query_dim // heads
        self.scale = self.head_dim**-0.5
        self.to_q = layers.Dense(
            query_dim, use_bias=qkv_bias, name=safe_name(f"{module_path}.to_q")
        )
        self.to_k = layers.Dense(
            query_dim, use_bias=qkv_bias, name=safe_name(f"{module_path}.to_k")
        )
        self.to_v = layers.Dense(
            query_dim, use_bias=qkv_bias, name=safe_name(f"{module_path}.to_v")
        )
        self.to_out = layers.Dense(query_dim, name=safe_name(f"{module_path}.to_out.0"))

    def build(self, input_shape, context_shape=None):
        self.to_q.build(input_shape)
        kv_shape = input_shape if context_shape is None else context_shape
        self.to_k.build(kv_shape)
        self.to_v.build(kv_shape)
        self.to_out.build(tuple(input_shape[:-1]) + (self.query_dim,))
        self.built = True

    def split_heads(self, t):
        t = ops.reshape(t, (-1, ops.shape(t)[1], self.heads, self.head_dim))
        return ops.transpose(t, (0, 2, 1, 3))

    def call(self, x, context=None):
        kv = x if context is None else context
        q = self.split_heads(self.to_q(x))
        k = self.split_heads(self.to_k(kv))
        v = self.split_heads(self.to_v(kv))
        out = fused_attention(q, k, v, self.scale)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (-1, ops.shape(x)[1], self.query_dim))
        return self.to_out(out)

    def compute_output_shape(self, input_shape, context_shape=None):
        return tuple(input_shape[:-1]) + (self.query_dim,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "query_dim": self.query_dim,
                "heads": self.heads,
                "module_path": self.module_path,
                "qkv_bias": self.qkv_bias,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GEGLUFeedForward(layers.Layer):
    """Diffusers ``FeedForward`` with GEGLU: ``proj`` to ``2 * inner`` gated by GELU,
    then back down (``net.0.proj`` / ``net.2``).

    Args:
        dim: Token width.
        module_path: Diffusers module path (``...ff``).
        mult: Inner width multiplier (4).
    """

    def __init__(self, dim, module_path, mult=4, **kwargs):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.dim = dim
        self.module_path = module_path
        self.mult = mult
        self.proj = layers.Dense(
            dim * mult * 2, name=safe_name(f"{module_path}.net.0.proj")
        )
        self.out = layers.Dense(dim, name=safe_name(f"{module_path}.net.2"))

    def build(self, input_shape):
        self.proj.build(input_shape)
        self.out.build(tuple(input_shape[:-1]) + (self.dim * self.mult,))
        self.built = True

    def call(self, x):
        hidden, gate = ops.split(self.proj(x), 2, axis=-1)
        return self.out(
            hidden * ops.gelu(gate, approximate=False)
        )  # exact erf GELU, as diffusers

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.dim,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"dim": self.dim, "module_path": self.module_path, "mult": self.mult}
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class BasicTransformerBlock(layers.Layer):
    """Self-attention, cross-attention, GEGLU feed-forward, each pre-normed with a
    residual (diffusers ``BasicTransformerBlock``).

    Args:
        dim: Token width.
        heads: Attention heads.
        module_path: Diffusers module path (``...transformer_blocks.0``).
    """

    def __init__(self, dim, heads, module_path, **kwargs):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.dim = dim
        self.heads = heads
        self.module_path = module_path
        self.norm1 = layers.LayerNormalization(
            epsilon=1e-5, name=safe_name(f"{module_path}.norm1")
        )
        self.attn1 = CrossAttention(dim, heads, module_path=f"{module_path}.attn1")
        self.norm2 = layers.LayerNormalization(
            epsilon=1e-5, name=safe_name(f"{module_path}.norm2")
        )
        self.attn2 = CrossAttention(dim, heads, module_path=f"{module_path}.attn2")
        self.norm3 = layers.LayerNormalization(
            epsilon=1e-5, name=safe_name(f"{module_path}.norm3")
        )
        self.ff = GEGLUFeedForward(dim, module_path=f"{module_path}.ff")

    def build(self, input_shape):
        x_shape, context_shape = input_shape
        self.norm1.build(x_shape)
        self.attn1.build(x_shape)
        self.norm2.build(x_shape)
        self.attn2.build(x_shape, context_shape)
        self.norm3.build(x_shape)
        self.ff.build(x_shape)
        self.built = True

    def call(self, inputs):
        x, context = inputs
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        return x + self.ff(self.norm3(x))

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[0])

    def get_config(self):
        config = super().get_config()
        config.update(
            {"dim": self.dim, "heads": self.heads, "module_path": self.module_path}
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Transformer2DModel(layers.Layer):
    """Diffusers ``Transformer2DModel``: GroupNorm, 1x1 in-proj, one
    ``BasicTransformerBlock`` over the flattened spatial tokens, 1x1 out-proj,
    residual. ``call([x, context])`` with ``x`` in the active image data format.

    Args:
        channels: Feature-map channels (token width).
        heads: Attention heads.
        module_path: Diffusers module path (``down_blocks.0.attentions.0``).
        groups: GroupNorm groups.
        data_format: ``"channels_last"`` or ``"channels_first"``; defaults to
            ``keras.config.image_data_format()``.
        channels_axis: The channel axis of that layout (``-1`` or ``1``).
    """

    def __init__(
        self,
        channels,
        heads,
        module_path,
        groups=GROUPS,
        data_format=None,
        channels_axis=None,
        **kwargs,
    ):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.channels = channels
        self.heads = heads
        self.module_path = module_path
        self.groups = groups
        self.data_format = data_format or keras.config.image_data_format()
        self.channels_axis = (
            channels_axis
            if channels_axis is not None
            else (-1 if self.data_format == "channels_last" else 1)
        )
        self.norm = layers.GroupNormalization(
            groups=groups,
            axis=self.channels_axis,
            epsilon=GROUP_EPS,
            name=safe_name(f"{module_path}.norm"),
        )
        self.proj_in = layers.Conv2D(
            channels,
            1,
            padding="valid",
            data_format=self.data_format,
            name=safe_name(f"{module_path}.proj_in"),
        )
        self.transformer_block = BasicTransformerBlock(
            channels, heads, module_path=f"{module_path}.transformer_blocks.0"
        )
        self.proj_out = layers.Conv2D(
            channels,
            1,
            padding="valid",
            data_format=self.data_format,
            name=safe_name(f"{module_path}.proj_out"),
        )

    def build(self, input_shape):
        x_shape, context_shape = input_shape
        if self.data_format == "channels_first":
            height, width = x_shape[2], x_shape[3]
            proj_shape = (x_shape[0], self.channels, height, width)
        else:
            height, width = x_shape[1], x_shape[2]
            proj_shape = (x_shape[0], height, width, self.channels)
        self.norm.build(x_shape)
        self.proj_in.build(x_shape)
        self.transformer_block.build(
            ((x_shape[0], height * width, self.channels), context_shape)
        )
        self.proj_out.build(proj_shape)
        self.built = True

    def call(self, inputs):
        x, context = inputs
        h = self.proj_in(self.norm(x))
        shape = ops.shape(h)
        if self.data_format == "channels_first":
            # (B, C, H, W) -> (B, H*W, C)
            height, width = shape[2], shape[3]
            h = ops.transpose(h, (0, 2, 3, 1))
        else:
            height, width = shape[1], shape[2]
        h = ops.reshape(h, (-1, height * width, self.channels))
        h = self.transformer_block([h, context])
        h = ops.reshape(h, (-1, height, width, self.channels))
        if self.data_format == "channels_first":
            h = ops.transpose(h, (0, 3, 1, 2))
        return x + self.proj_out(h)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[0])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "heads": self.heads,
                "module_path": self.module_path,
                "groups": self.groups,
                "data_format": self.data_format,
                "channels_axis": self.channels_axis,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Downsample2D(layers.Layer):
    """Strided 3x3 conv downsample (diffusers ``Downsample2D``).

    Args:
        channels: Output channels.
        module_path: Diffusers module path (``down_blocks.0.downsamplers.0``).
        padding: 1 for the UNet (symmetric), 0 for the VAE encoder, which pads only
            the bottom/right edge before its unpadded stride-2 conv.
        data_format: ``"channels_last"`` or ``"channels_first"``; defaults to
            ``keras.config.image_data_format()``.
        channels_axis: The channel axis of that layout (``-1`` or ``1``).
    """

    def __init__(
        self,
        channels,
        module_path,
        padding=1,
        data_format=None,
        channels_axis=None,
        **kwargs,
    ):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.channels = channels
        self.module_path = module_path
        self.padding = padding
        self.data_format = data_format or keras.config.image_data_format()
        self.channels_axis = (
            channels_axis
            if channels_axis is not None
            else (-1 if self.data_format == "channels_last" else 1)
        )
        self.conv = layers.Conv2D(
            channels,
            3,
            strides=2,
            padding="valid",
            data_format=self.data_format,
            name=safe_name(f"{module_path}.conv"),
        )

    def pad_amounts(self):
        return ((1, 1), (1, 1)) if self.padding == 1 else ((0, 1), (0, 1))

    def padded_shape(self, input_shape):
        (top, bottom), (left, right) = self.pad_amounts()
        if self.data_format == "channels_first":
            batch, channels, height, width = input_shape
            return (batch, channels, height + top + bottom, width + left + right)
        batch, height, width, channels = input_shape
        return (batch, height + top + bottom, width + left + right, channels)

    def build(self, input_shape):
        self.conv.build(self.padded_shape(input_shape))
        self.built = True

    def call(self, x):
        (top, bottom), (left, right) = self.pad_amounts()
        if self.data_format == "channels_first":
            x = ops.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)))
        else:
            x = ops.pad(x, ((0, 0), (top, bottom), (left, right), (0, 0)))
        return self.conv(x)

    def compute_output_shape(self, input_shape):
        padded = self.padded_shape(input_shape)
        if self.data_format == "channels_first":
            batch, _, height, width = padded
            return (batch, self.channels, (height - 3) // 2 + 1, (width - 3) // 2 + 1)
        batch, height, width, _ = padded
        return (batch, (height - 3) // 2 + 1, (width - 3) // 2 + 1, self.channels)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "module_path": self.module_path,
                "padding": self.padding,
                "data_format": self.data_format,
                "channels_axis": self.channels_axis,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Upsample2D(layers.Layer):
    """Nearest 2x upsample + 3x3 conv (diffusers ``Upsample2D``).

    Args:
        channels: Output channels.
        module_path: Diffusers module path (``up_blocks.0.upsamplers.0``).
        data_format: ``"channels_last"`` or ``"channels_first"``; defaults to
            ``keras.config.image_data_format()``.
        channels_axis: The channel axis of that layout (``-1`` or ``1``).
    """

    def __init__(
        self, channels, module_path, data_format=None, channels_axis=None, **kwargs
    ):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.channels = channels
        self.module_path = module_path
        self.data_format = data_format or keras.config.image_data_format()
        self.channels_axis = (
            channels_axis
            if channels_axis is not None
            else (-1 if self.data_format == "channels_last" else 1)
        )
        self.conv = layers.Conv2D(
            channels,
            3,
            padding="same",
            data_format=self.data_format,
            name=safe_name(f"{module_path}.conv"),
        )

    def upsampled_shape(self, input_shape, channels):
        if self.data_format == "channels_first":
            batch, _, height, width = input_shape
            return (batch, channels, height * 2, width * 2)
        batch, height, width, _ = input_shape
        return (batch, height * 2, width * 2, channels)

    def build(self, input_shape):
        channels = input_shape[self.channels_axis]
        self.conv.build(self.upsampled_shape(input_shape, channels))
        self.built = True

    def call(self, x):
        if self.data_format == "channels_first":
            x = ops.repeat(ops.repeat(x, 2, axis=2), 2, axis=3)
        else:
            x = ops.repeat(ops.repeat(x, 2, axis=1), 2, axis=2)
        return self.conv(x)

    def compute_output_shape(self, input_shape):
        return self.upsampled_shape(input_shape, self.channels)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "module_path": self.module_path,
                "data_format": self.data_format,
                "channels_axis": self.channels_axis,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class VaeAttentionBlock(layers.Layer):
    """Single-head spatial self-attention of the VAE mid block: GroupNorm, attention
    over the flattened pixels (biased q/k/v), residual.

    Args:
        channels: Feature-map channels.
        module_path: Diffusers module path (``mid_block.attentions.0``).
        groups: GroupNorm groups.
        data_format: ``"channels_last"`` or ``"channels_first"``; defaults to
            ``keras.config.image_data_format()``.
        channels_axis: The channel axis of that layout (``-1`` or ``1``).
    """

    def __init__(
        self,
        channels,
        module_path,
        groups=GROUPS,
        data_format=None,
        channels_axis=None,
        **kwargs,
    ):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.channels = channels
        self.module_path = module_path
        self.groups = groups
        self.data_format = data_format or keras.config.image_data_format()
        self.channels_axis = (
            channels_axis
            if channels_axis is not None
            else (-1 if self.data_format == "channels_last" else 1)
        )
        self.group_norm = layers.GroupNormalization(
            groups=groups,
            axis=self.channels_axis,
            epsilon=GROUP_EPS,
            name=safe_name(f"{module_path}.group_norm"),
        )
        self.attention = CrossAttention(
            channels,
            heads=1,
            module_path=module_path,
            qkv_bias=True,
            name=safe_name(f"{module_path}.attn"),
        )

    def build(self, input_shape):
        if self.data_format == "channels_first":
            height, width = input_shape[2], input_shape[3]
        else:
            height, width = input_shape[1], input_shape[2]
        self.group_norm.build(input_shape)
        self.attention.build((input_shape[0], height * width, self.channels))
        self.built = True

    def call(self, x):
        h = self.group_norm(x)
        shape = ops.shape(h)
        if self.data_format == "channels_first":
            # (B, C, H, W) -> (B, H*W, C)
            height, width = shape[2], shape[3]
            h = ops.transpose(h, (0, 2, 3, 1))
        else:
            height, width = shape[1], shape[2]
        h = self.attention(ops.reshape(h, (-1, height * width, self.channels)))
        h = ops.reshape(h, (-1, height, width, self.channels))
        if self.data_format == "channels_first":
            h = ops.transpose(h, (0, 3, 1, 2))
        return x + h

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "module_path": self.module_path,
                "groups": self.groups,
                "data_format": self.data_format,
                "channels_axis": self.channels_axis,
            }
        )
        return config
