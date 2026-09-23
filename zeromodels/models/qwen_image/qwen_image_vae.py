"""Keras 3 port of Diffusers ``AutoencoderKLQwenImage`` (Qwen-Image / Wan VAE).

Internal activations use channels-last 5D ``(B, T, H, W, C)`` (NDHWC) so
``Conv3D`` is portable across Keras backends. Diffusers stores the same stacks
as NCDHW ``Conv3d``; converters must transpose kernels:

* Conv3D: ``(O, I, T, H, W)`` -> ``(T, H, W, I, O)``
* Conv2D (resample / attention): ``(O, I, H, W)`` -> ``(H, W, I, O)``
* RMSNorm ``gamma``: squeeze Diffusers ``(C, 1, 1[, 1])`` -> ``(C,)``

v1 targets single-frame T2I (``T=1``): feat-cache video streaming is a no-op and
temporal ``time_conv`` paths inside ``Resample`` are skipped, matching Diffusers'
first-chunk behaviour when the cache is cold. Tiling / slicing are omitted.
Module paths mirror Diffusers for weight conversion.
"""

from __future__ import annotations

import math

import keras
from keras import layers, ops

from zeromodels.base import BaseModel
from zeromodels.base.base_attention import fused_attention
from zeromodels.base.base_config import BaseConfig
from zeromodels.models.stable_diffusion.stable_diffusion_layers import safe_name

# ---------------------------------------------------------------------------
# Defaults (Qwen/Qwen-Image VAE config.json)
# ---------------------------------------------------------------------------

DEFAULT_LATENTS_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
DEFAULT_LATENTS_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)


class QwenImageVAEConfig(BaseConfig):
    """Configuration for :class:`AutoencoderKLQwenImage`.

    Fields match Diffusers ``AutoencoderKLQwenImage``; ``sample_size`` is the
    ZeroModels graph-build resolution (weights are resolution-independent).
    """

    model_type = "autoencoder_kl_qwen_image"

    base_dim: int = 96
    z_dim: int = 16
    dim_mult: tuple = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: tuple = ()
    temperal_downsample: tuple = (False, True, True)
    dropout: float = 0.0
    input_channels: int = 3
    latents_mean: tuple = DEFAULT_LATENTS_MEAN
    latents_std: tuple = DEFAULT_LATENTS_STD
    sample_size: int = 1024


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _as_tuple3(value):
    if isinstance(value, int):
        return (value, value, value)
    value = tuple(value)
    if len(value) != 3:
        raise ValueError(f"Expected int or length-3 tuple, got {value!r}")
    return value


def ncdhw_to_ndhwc(x):
    """``(B, C, T, H, W)`` -> ``(B, T, H, W, C)``."""
    return ops.transpose(x, (0, 2, 3, 4, 1))


def ndhwc_to_ncdhw(x):
    """``(B, T, H, W, C)`` -> ``(B, C, T, H, W)``."""
    return ops.transpose(x, (0, 4, 1, 2, 3))


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImageRMSNorm(layers.Layer):
    """RMSNorm used by the Qwen-Image / Wan VAE (Diffusers ``QwenImageRMS_norm``)."""

    def __init__(self, dim, images=True, module_path=None, **kwargs):
        if module_path is not None:
            kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.images = bool(images)
        self.module_path = module_path
        self.scale = self.dim**0.5

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(self.dim,),
            initializer="ones",
            trainable=True,
            name="gamma",
        )
        self.built = True

    def call(self, x):
        # Diffusers: F.normalize(x, dim=channels) * sqrt(dim) * gamma, i.e. an L2
        # normalize (eps 1e-12 on the norm) over the last (channel) axis here.
        dtype = x.dtype
        x_f = ops.cast(x, "float32")
        norm = ops.sqrt(ops.sum(ops.square(x_f), axis=-1, keepdims=True))
        x_f = x_f / ops.maximum(norm, 1e-12)
        x_f = x_f * self.scale * ops.cast(self.gamma, "float32")
        return ops.cast(x_f, dtype)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "images": self.images,
                "module_path": self.module_path,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImageCausalConv3d(layers.Layer):
    """Causal 3D conv with asymmetric temporal pad (Diffusers ``QwenImageCausalConv3d``).

    Diffusers pads ``(W_l, W_r, H_l, H_r, T_l, T_r)`` with ``T_l = 2 * temporal_pad``
    and ``T_r = 0``. Keras uses NDHWC ``Conv3D`` + ``ops.pad``.

    Weight mapping from Diffusers ``Conv3d``: ``(O, I, T, H, W) -> (T, H, W, I, O)``.
    """

    def __init__(
        self,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        module_path=None,
        **kwargs,
    ):
        if module_path is not None:
            kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.out_channels = int(out_channels)
        self.kernel_size = _as_tuple3(kernel_size)
        self.stride = _as_tuple3(stride)
        pad = _as_tuple3(padding)
        self.padding_t, self.padding_h, self.padding_w = pad
        # Causal: double temporal left pad, zero right pad.
        self._pad_t_left = 2 * self.padding_t
        self._pad_t_right = 0
        self._pad_h = self.padding_h
        self._pad_w = self.padding_w
        self.module_path = module_path
        # Leaf Conv3D keeps the Diffusers module path name so conversion's
        # last-two-segment mapping yields ``encoder.conv_in.weight`` etc.
        # (path ``.../encoder__conv_in/encoder__conv_in/kernel``).
        leaf_name = safe_name(module_path) if module_path else "conv"
        self.conv = layers.Conv3D(
            self.out_channels,
            self.kernel_size,
            strides=self.stride,
            padding="valid",
            data_format="channels_last",
            name=leaf_name,
        )

    def _padded_shape(self, input_shape):
        b, t, h, w, c = input_shape
        t2 = (
            None
            if t is None
            else t + self._pad_t_left + self._pad_t_right
        )
        h2 = None if h is None else h + 2 * self._pad_h
        w2 = None if w is None else w + 2 * self._pad_w
        return (b, t2, h2, w2, c)

    def build(self, input_shape):
        self.conv.build(self._padded_shape(input_shape))
        self.built = True

    def call(self, x):
        # NDHWC pad: [[B], [T], [H], [W], [C]]
        x = ops.pad(
            x,
            (
                (0, 0),
                (self._pad_t_left, self._pad_t_right),
                (self._pad_h, self._pad_h),
                (self._pad_w, self._pad_w),
                (0, 0),
            ),
        )
        return self.conv(x)

    def compute_output_shape(self, input_shape):
        b, t, h, w, _ = self._padded_shape(input_shape)
        kt, kh, kw = self.kernel_size
        st, sh, sw = self.stride

        def _out(size, k, s):
            if size is None:
                return None
            return (size - k) // s + 1

        return (b, _out(t, kt, st), _out(h, kh, sh), _out(w, kw, sw), self.out_channels)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_channels": self.out_channels,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": (self.padding_t, self.padding_h, self.padding_w),
                "module_path": self.module_path,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImageResample(layers.Layer):
    """Spatial / spatio-temporal resample (Diffusers ``QwenImageResample``).

    For ``T=1`` image inference the temporal ``time_conv`` is created (weights)
    but not applied, matching Diffusers' cold feat-cache first chunk.
    """

    def __init__(self, dim, mode, module_path, apply_temporal=False, **kwargs):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.mode = mode
        self.module_path = module_path
        self.apply_temporal = bool(apply_temporal)

        self.spatial_conv = None
        self.time_conv = None
        self._downsample = mode in ("downsample2d", "downsample3d")
        self._upsample = mode in ("upsample2d", "upsample3d")

        if mode in ("upsample2d", "upsample3d"):
            self.spatial_conv = layers.Conv2D(
                dim // 2,
                3,
                padding="same",
                data_format="channels_last",
                name=safe_name(f"{module_path}.resample.1"),
            )
            if mode == "upsample3d":
                self.time_conv = QwenImageCausalConv3d(
                    dim * 2,
                    kernel_size=(3, 1, 1),
                    padding=(1, 0, 0),
                    module_path=f"{module_path}.time_conv",
                )
        elif mode in ("downsample2d", "downsample3d"):
            self.spatial_conv = layers.Conv2D(
                dim,
                3,
                strides=2,
                padding="valid",
                data_format="channels_last",
                name=safe_name(f"{module_path}.resample.1"),
            )
            if mode == "downsample3d":
                self.time_conv = QwenImageCausalConv3d(
                    dim,
                    kernel_size=(3, 1, 1),
                    stride=(2, 1, 1),
                    padding=(0, 0, 0),
                    module_path=f"{module_path}.time_conv",
                )
        elif mode != "none":
            raise ValueError(
                f"Unknown resample mode {mode!r}; expected none/upsample2d/"
                "upsample3d/downsample2d/downsample3d."
            )

    def build(self, input_shape):
        # input_shape: (B, T, H, W, C)
        b, t, h, w, c = input_shape
        if self.spatial_conv is not None:
            if self._downsample:
                # ZeroPad2d((0,1,0,1)) -> H+1, W+1
                h_p = None if h is None else h + 1
                w_p = None if w is None else w + 1
                self.spatial_conv.build((b, h_p, w_p, c))
            else:
                h_u = None if h is None else h * 2
                w_u = None if w is None else w * 2
                self.spatial_conv.build((b, h_u, w_u, c))
        if self.time_conv is not None:
            # Downsample3d time_conv is stride-2 with k_t=3 and no pad; T=1 cannot
            # run it. Build with a synthetic T so kernels exist for conversion; the
            # T=1 image path never calls time_conv (cold-cache Diffusers behaviour).
            b, t, h, w, c = input_shape
            t_build = t if isinstance(t, int) and t >= 4 else 4
            self.time_conv.build((b, t_build, h, w, c))
        self.built = True

    def call(self, x):
        # Optional temporal branch (video streaming); skipped for T=1 image path
        # (matches Diffusers' cold feat-cache first chunk, which skips time_conv).
        if (
            self.apply_temporal
            and self.time_conv is not None
            and self.mode == "upsample3d"
        ):
            b = ops.shape(x)[0]
            t = ops.shape(x)[1]
            h = ops.shape(x)[2]
            w = ops.shape(x)[3]
            c = self.dim
            x = self.time_conv(x)
            # (B, T, H, W, 2C) -> (B, T*2, H, W, C) interleaved like Diffusers
            x = ops.reshape(x, (b, t, h, w, 2, c))
            x = ops.transpose(x, (0, 1, 4, 2, 3, 5))
            x = ops.reshape(x, (b, t * 2, h, w, c))

        b = ops.shape(x)[0]
        t = ops.shape(x)[1]
        h = ops.shape(x)[2]
        w = ops.shape(x)[3]
        c = ops.shape(x)[4]
        # Merge batch and time for 2D ops: (B*T, H, W, C)
        x2 = ops.reshape(x, (b * t, h, w, c))

        if self._upsample:
            x2 = ops.image.resize(x2, (h * 2, w * 2), interpolation="nearest")
            x2 = self.spatial_conv(x2)
            out_c = self.dim // 2
            out_h, out_w = h * 2, w * 2
        elif self._downsample:
            x2 = ops.pad(x2, ((0, 0), (0, 1), (0, 1), (0, 0)))
            x2 = self.spatial_conv(x2)
            out_c = self.dim
            out_h = (h + 1 - 3) // 2 + 1
            out_w = (w + 1 - 3) // 2 + 1
        else:
            out_c = c
            out_h, out_w = h, w

        return ops.reshape(x2, (b, t, out_h, out_w, out_c))

    def compute_output_shape(self, input_shape):
        b, t, h, w, c = input_shape
        if self._upsample:
            return (
                b,
                t,
                None if h is None else h * 2,
                None if w is None else w * 2,
                self.dim // 2,
            )
        if self._downsample:
            return (
                b,
                t,
                None if h is None else (h + 1 - 3) // 2 + 1,
                None if w is None else (w + 1 - 3) // 2 + 1,
                self.dim,
            )
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "mode": self.mode,
                "module_path": self.module_path,
                "apply_temporal": self.apply_temporal,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImageResidualBlock(layers.Layer):
    """Residual block with causal 3D convs (Diffusers ``QwenImageResidualBlock``)."""

    def __init__(
        self,
        in_dim,
        out_dim,
        module_path,
        dropout=0.0,
        **kwargs,
    ):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.module_path = module_path
        self.dropout_rate = float(dropout)

        self.norm1 = QwenImageRMSNorm(
            in_dim, images=False, module_path=f"{module_path}.norm1"
        )
        self.conv1 = QwenImageCausalConv3d(
            out_dim, 3, padding=1, module_path=f"{module_path}.conv1"
        )
        self.norm2 = QwenImageRMSNorm(
            out_dim, images=False, module_path=f"{module_path}.norm2"
        )
        self.dropout = layers.Dropout(self.dropout_rate, name=safe_name(f"{module_path}.dropout"))
        self.conv2 = QwenImageCausalConv3d(
            out_dim, 3, padding=1, module_path=f"{module_path}.conv2"
        )
        self.conv_shortcut = None
        if in_dim != out_dim:
            self.conv_shortcut = QwenImageCausalConv3d(
                out_dim, 1, padding=0, module_path=f"{module_path}.conv_shortcut"
            )

    def build(self, input_shape):
        self.norm1.build(input_shape)
        self.conv1.build(input_shape)
        mid = list(input_shape)
        mid[-1] = self.out_dim
        mid = tuple(mid)
        self.norm2.build(mid)
        self.conv2.build(mid)
        if self.conv_shortcut is not None:
            self.conv_shortcut.build(input_shape)
        self.built = True

    def call(self, x, training=None):
        shortcut = x if self.conv_shortcut is None else self.conv_shortcut(x)
        h = self.norm1(x)
        h = ops.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = ops.silu(h)
        h = self.dropout(h, training=training)
        h = self.conv2(h)
        return h + shortcut

    def compute_output_shape(self, input_shape):
        out = list(input_shape)
        out[-1] = self.out_dim
        return tuple(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_dim": self.in_dim,
                "out_dim": self.out_dim,
                "module_path": self.module_path,
                "dropout": self.dropout_rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImageAttentionBlock(layers.Layer):
    """Single-head causal self-attention over spatial tokens (per time step)."""

    def __init__(self, dim, module_path, **kwargs):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.module_path = module_path
        self.norm = QwenImageRMSNorm(dim, images=True, module_path=f"{module_path}.norm")
        self.to_qkv = layers.Conv2D(
            dim * 3,
            1,
            data_format="channels_last",
            name=safe_name(f"{module_path}.to_qkv"),
        )
        self.proj = layers.Conv2D(
            dim,
            1,
            data_format="channels_last",
            name=safe_name(f"{module_path}.proj"),
        )

    def build(self, input_shape):
        # (B, T, H, W, C)
        b, t, h, w, c = input_shape
        self.norm.build((b, h, w, c))
        self.to_qkv.build((b, h, w, c))
        self.proj.build((b, h, w, c))
        self.built = True

    def call(self, x):
        identity = x
        b = ops.shape(x)[0]
        t = ops.shape(x)[1]
        h = ops.shape(x)[2]
        w = ops.shape(x)[3]
        c = self.dim
        x2 = ops.reshape(x, (b * t, h, w, c))
        x2 = self.norm(x2)
        qkv = self.to_qkv(x2)
        # (B*T, H, W, 3C) -> (B*T, 1, HW, 3C) -> split to q,k,v (B*T, 1, HW, C)
        qkv = ops.reshape(qkv, (b * t, 1, h * w, c * 3))
        q, k, v = ops.split(qkv, 3, axis=-1)
        scale = 1.0 / math.sqrt(c)
        attn = fused_attention(q, k, v, scale)
        attn = ops.reshape(attn, (b * t, h, w, c))
        attn = self.proj(attn)
        attn = ops.reshape(attn, (b, t, h, w, c))
        return attn + identity

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "module_path": self.module_path})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImageMidBlock(layers.Layer):
    """Mid block: resnet -> (attn -> resnet) * num_layers."""

    def __init__(self, dim, module_path, dropout=0.0, num_layers=1, **kwargs):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.module_path = module_path
        self.dropout_rate = float(dropout)
        self.num_layers = int(num_layers)

        self.resnets = [
            QwenImageResidualBlock(
                dim, dim, f"{module_path}.resnets.0", dropout=dropout
            )
        ]
        self.attentions = []
        for i in range(num_layers):
            self.attentions.append(
                QwenImageAttentionBlock(dim, f"{module_path}.attentions.{i}")
            )
            self.resnets.append(
                QwenImageResidualBlock(
                    dim, dim, f"{module_path}.resnets.{i + 1}", dropout=dropout
                )
            )

    def build(self, input_shape):
        for layer in self.resnets:
            layer.build(input_shape)
        for layer in self.attentions:
            layer.build(input_shape)
        self.built = True

    def call(self, x, training=None):
        x = self.resnets[0](x, training=training)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = resnet(x, training=training)
        return x

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "module_path": self.module_path,
                "dropout": self.dropout_rate,
                "num_layers": self.num_layers,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImageEncoder3d(layers.Layer):
    """3D VAE encoder (Diffusers ``QwenImageEncoder3d``)."""

    def __init__(
        self,
        dim=96,
        z_dim=32,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temperal_downsample=(False, True, True),
        dropout=0.0,
        input_channels=3,
        module_path="encoder",
        apply_temporal=False,
        **kwargs,
    ):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.z_dim = int(z_dim)
        self.dim_mult = tuple(dim_mult)
        self.num_res_blocks = int(num_res_blocks)
        self.attn_scales = tuple(attn_scales)
        self.temperal_downsample = tuple(temperal_downsample)
        self.dropout_rate = float(dropout)
        self.input_channels = int(input_channels)
        self.module_path = module_path
        self.apply_temporal = bool(apply_temporal)

        dims = [dim * u for u in [1] + list(self.dim_mult)]
        scale = 1.0

        self.conv_in = QwenImageCausalConv3d(
            dims[0], 3, padding=1, module_path=f"{module_path}.conv_in"
        )
        self.down_blocks = []
        idx = 0
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(self.num_res_blocks):
                self.down_blocks.append(
                    QwenImageResidualBlock(
                        in_dim,
                        out_dim,
                        f"{module_path}.down_blocks.{idx}",
                        dropout=dropout,
                    )
                )
                idx += 1
                if scale in self.attn_scales:
                    self.down_blocks.append(
                        QwenImageAttentionBlock(
                            out_dim, f"{module_path}.down_blocks.{idx}"
                        )
                    )
                    idx += 1
                in_dim = out_dim
            if i != len(self.dim_mult) - 1:
                mode = (
                    "downsample3d"
                    if self.temperal_downsample[i]
                    else "downsample2d"
                )
                self.down_blocks.append(
                    QwenImageResample(
                        out_dim,
                        mode,
                        f"{module_path}.down_blocks.{idx}",
                        apply_temporal=self.apply_temporal,
                    )
                )
                idx += 1
                scale /= 2.0

        self.mid_block = QwenImageMidBlock(
            dims[-1], f"{module_path}.mid_block", dropout=dropout, num_layers=1
        )
        self.norm_out = QwenImageRMSNorm(
            dims[-1], images=False, module_path=f"{module_path}.norm_out"
        )
        self.conv_out = QwenImageCausalConv3d(
            z_dim, 3, padding=1, module_path=f"{module_path}.conv_out"
        )
        self._out_dim = dims[-1]

    def build(self, input_shape):
        self.conv_in.build(input_shape)
        x_shape = list(input_shape)
        x_shape[-1] = self.dim
        x_shape = tuple(x_shape)
        # Walk blocks updating spatial dims approximately for build.
        shape = x_shape
        for layer in self.down_blocks:
            layer.build(shape)
            shape = layer.compute_output_shape(shape)
        self.mid_block.build(shape)
        self.norm_out.build(shape)
        self.conv_out.build(shape)
        self.built = True

    def call(self, x, training=None):
        x = self.conv_in(x)
        for layer in self.down_blocks:
            if isinstance(layer, QwenImageResidualBlock):
                x = layer(x, training=training)
            else:
                x = layer(x)
        x = self.mid_block(x, training=training)
        x = self.norm_out(x)
        x = ops.silu(x)
        x = self.conv_out(x)
        return x

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[-1] = self.dim
        shape = tuple(shape)
        for layer in self.down_blocks:
            shape = layer.compute_output_shape(shape)
        out = list(shape)
        out[-1] = self.z_dim
        # Spatial / temporal sizes: 3 spatial downsamples -> /8; T unchanged (T=1 path).
        b, t, h, w, _ = input_shape
        factor = 2 ** (len(self.dim_mult) - 1)
        return (
            b,
            t,
            None if h is None else h // factor,
            None if w is None else w // factor,
            self.z_dim,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "z_dim": self.z_dim,
                "dim_mult": self.dim_mult,
                "num_res_blocks": self.num_res_blocks,
                "attn_scales": self.attn_scales,
                "temperal_downsample": self.temperal_downsample,
                "dropout": self.dropout_rate,
                "input_channels": self.input_channels,
                "module_path": self.module_path,
                "apply_temporal": self.apply_temporal,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImageUpBlock(layers.Layer):
    """Decoder up block (Diffusers ``QwenImageUpBlock``)."""

    def __init__(
        self,
        in_dim,
        out_dim,
        num_res_blocks,
        module_path,
        dropout=0.0,
        upsample_mode=None,
        apply_temporal=False,
        **kwargs,
    ):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_res_blocks = int(num_res_blocks)
        self.module_path = module_path
        self.dropout_rate = float(dropout)
        self.upsample_mode = upsample_mode
        self.apply_temporal = bool(apply_temporal)

        self.resnets = []
        current = in_dim
        for i in range(num_res_blocks + 1):
            self.resnets.append(
                QwenImageResidualBlock(
                    current,
                    out_dim,
                    f"{module_path}.resnets.{i}",
                    dropout=dropout,
                )
            )
            current = out_dim

        self.upsampler = None
        if upsample_mode is not None:
            self.upsampler = QwenImageResample(
                out_dim,
                upsample_mode,
                f"{module_path}.upsamplers.0",
                apply_temporal=apply_temporal,
            )

    def build(self, input_shape):
        shape = input_shape
        for layer in self.resnets:
            layer.build(shape)
            shape = layer.compute_output_shape(shape)
        if self.upsampler is not None:
            self.upsampler.build(shape)
        self.built = True

    def call(self, x, training=None):
        for resnet in self.resnets:
            x = resnet(x, training=training)
        if self.upsampler is not None:
            x = self.upsampler(x)
        return x

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[-1] = self.out_dim
        shape = tuple(shape)
        if self.upsampler is not None:
            shape = self.upsampler.compute_output_shape(shape)
        return shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_dim": self.in_dim,
                "out_dim": self.out_dim,
                "num_res_blocks": self.num_res_blocks,
                "module_path": self.module_path,
                "dropout": self.dropout_rate,
                "upsample_mode": self.upsample_mode,
                "apply_temporal": self.apply_temporal,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImageDecoder3d(layers.Layer):
    """3D VAE decoder (Diffusers ``QwenImageDecoder3d``)."""

    def __init__(
        self,
        dim=96,
        z_dim=16,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temperal_upsample=(True, True, False),
        dropout=0.0,
        input_channels=3,
        module_path="decoder",
        apply_temporal=False,
        **kwargs,
    ):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.z_dim = int(z_dim)
        self.dim_mult = tuple(dim_mult)
        self.num_res_blocks = int(num_res_blocks)
        self.attn_scales = tuple(attn_scales)
        self.temperal_upsample = tuple(temperal_upsample)
        self.dropout_rate = float(dropout)
        self.input_channels = int(input_channels)
        self.module_path = module_path
        self.apply_temporal = bool(apply_temporal)

        dims = [dim * u for u in [self.dim_mult[-1]] + list(self.dim_mult[::-1])]

        self.conv_in = QwenImageCausalConv3d(
            dims[0], 3, padding=1, module_path=f"{module_path}.conv_in"
        )
        self.mid_block = QwenImageMidBlock(
            dims[0], f"{module_path}.mid_block", dropout=dropout, num_layers=1
        )
        self.up_blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0:
                in_dim = in_dim // 2
            upsample_mode = None
            if i != len(self.dim_mult) - 1:
                upsample_mode = (
                    "upsample3d" if self.temperal_upsample[i] else "upsample2d"
                )
            self.up_blocks.append(
                QwenImageUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    module_path=f"{module_path}.up_blocks.{i}",
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                    apply_temporal=apply_temporal,
                )
            )
        self.norm_out = QwenImageRMSNorm(
            dims[-1], images=False, module_path=f"{module_path}.norm_out"
        )
        self.conv_out = QwenImageCausalConv3d(
            input_channels, 3, padding=1, module_path=f"{module_path}.conv_out"
        )
        self._out_channels = dims[-1]

    def build(self, input_shape):
        self.conv_in.build(input_shape)
        shape = list(input_shape)
        shape[-1] = self.dim * self.dim_mult[-1]
        shape = tuple(shape)
        self.mid_block.build(shape)
        for block in self.up_blocks:
            block.build(shape)
            shape = block.compute_output_shape(shape)
        self.norm_out.build(shape)
        self.conv_out.build(shape)
        self.built = True

    def call(self, x, training=None):
        x = self.conv_in(x)
        x = self.mid_block(x, training=training)
        for block in self.up_blocks:
            x = block(x, training=training)
        x = self.norm_out(x)
        x = ops.silu(x)
        x = self.conv_out(x)
        return x

    def compute_output_shape(self, input_shape):
        b, t, h, w, _ = input_shape
        factor = 2 ** (len(self.dim_mult) - 1)
        return (
            b,
            t,
            None if h is None else h * factor,
            None if w is None else w * factor,
            self.input_channels,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "z_dim": self.z_dim,
                "dim_mult": self.dim_mult,
                "num_res_blocks": self.num_res_blocks,
                "attn_scales": self.attn_scales,
                "temperal_upsample": self.temperal_upsample,
                "dropout": self.dropout_rate,
                "input_channels": self.input_channels,
                "module_path": self.module_path,
                "apply_temporal": self.apply_temporal,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="zeromodels")
class AutoencoderKLQwenImage(BaseModel):
    """Qwen-Image VAE (Diffusers ``AutoencoderKLQwenImage``), channels-last NDHWC.

    Encode / decode a single image frame (``T=1``). The functional graph is built
    for ``sample_size`` but conv weights are resolution-independent. Public
    helpers accept channels-last HWC images or Diffusers NCDHW 5D tensors.

    Latent normalisation with ``latents_mean`` / ``latents_std`` is left to the
    pipeline (stored on the model for that purpose).
    """

    config_class = QwenImageVAEConfig
    HF_MODEL_TYPE = None

    def __init__(
        self,
        base_dim=96,
        z_dim=16,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temperal_downsample=(False, True, True),
        dropout=0.0,
        input_channels=3,
        latents_mean=DEFAULT_LATENTS_MEAN,
        latents_std=DEFAULT_LATENTS_STD,
        sample_size=1024,
        apply_temporal=False,
        name="AutoencoderKLQwenImage",
        **kwargs,
    ):
        dim_mult = tuple(dim_mult)
        temperal_downsample = tuple(temperal_downsample)
        attn_scales = tuple(attn_scales)
        latents_mean = tuple(latents_mean)
        latents_std = tuple(latents_std)
        temperal_upsample = tuple(reversed(temperal_downsample))

        h_img, w_img = (
            sample_size
            if isinstance(sample_size, (tuple, list))
            else (sample_size, sample_size)
        )
        spatial_compression_ratio = 2 ** len(temperal_downsample)
        h_lat, w_lat = h_img // spatial_compression_ratio, w_img // spatial_compression_ratio

        encoder = QwenImageEncoder3d(
            dim=base_dim,
            z_dim=z_dim * 2,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            dropout=dropout,
            input_channels=input_channels,
            module_path="encoder",
            apply_temporal=apply_temporal,
        )
        decoder = QwenImageDecoder3d(
            dim=base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_upsample=temperal_upsample,
            dropout=dropout,
            input_channels=input_channels,
            module_path="decoder",
            apply_temporal=apply_temporal,
        )
        quant_conv = QwenImageCausalConv3d(
            z_dim * 2, kernel_size=1, padding=0, module_path="quant_conv"
        )
        post_quant_conv = QwenImageCausalConv3d(
            z_dim, kernel_size=1, padding=0, module_path="post_quant_conv"
        )

        # Functional graph: HWC image + HWC-ish latent (T squeezed to channels-last 4D
        # with an explicit time axis of 1 in the 5D path).
        image_in = layers.Input(shape=(h_img, w_img, input_channels), name="image")
        latent_in = layers.Input(shape=(h_lat, w_lat, z_dim), name="latent")

        image_5d = ops.expand_dims(image_in, axis=1)  # (B, 1, H, W, C)
        moments_5d = quant_conv(encoder(image_5d))
        moments = ops.squeeze(moments_5d, axis=1)  # (B, h, w, 2*z)

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
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_upsample
        self.dropout = dropout
        self.input_channels = input_channels
        self.latents_mean = latents_mean
        self.latents_std = latents_std
        self.sample_size = sample_size
        self.apply_temporal = apply_temporal
        self.spatial_compression_ratio = spatial_compression_ratio
        self.vae_scale_factor = spatial_compression_ratio
        self.encoder = encoder
        self.decoder = decoder
        self.quant_conv = quant_conv
        self.post_quant_conv = post_quant_conv

    # -- public encode / decode ------------------------------------------------

    def _to_ndhwc(self, x, is_latent=False):
        """Normalize inputs to NDHWC ``(B, T, H, W, C)``."""
        static_ndim = len(x.shape)
        if static_ndim == 4:
            # (B, H, W, C) channels-last image or latent
            return ops.expand_dims(x, axis=1)
        if static_ndim == 5:
            # Detect NCDHW (Diffusers): channel axis at 1 matches input_channels / z_dim.
            c1 = int(x.shape[1]) if x.shape[1] is not None else None
            c_last = int(x.shape[-1]) if x.shape[-1] is not None else None
            expect = self.z_dim if is_latent else self.input_channels
            if c1 == expect and c_last != expect:
                return ncdhw_to_ndhwc(x)
            return x
        raise ValueError(f"Expected 4D or 5D tensor, got shape {x.shape}")

    def _maybe_squeeze_t(self, x, original_was_4d):
        if original_was_4d:
            return ops.squeeze(x, axis=1)
        return x

    def encode(self, x, sample=False, seed=None, return_ncdhw=False):
        """Encode image(s) to latents (mean, or reparameterized sample).

        Args:
            x: ``(B, H, W, 3)`` HWC, ``(B, 1, H, W, 3)`` NDHWC, or
                ``(B, 3, 1, H, W)`` Diffusers NCDHW.
            sample: If True, draw ``z ~ N(mean, std)``; else return mean.
            seed: RNG seed for sampling.
            return_ncdhw: If True, return Diffusers layout ``(B, C, T, H, W)``.
        """
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
        if return_ncdhw:
            return ndhwc_to_ncdhw(z)
        return self._maybe_squeeze_t(z, was_4d)

    def decode(self, z, return_ncdhw=False):
        """Decode latents to RGB in ``[-1, 1]``.

        Args:
            z: ``(B, h, w, z_dim)``, ``(B, 1, h, w, z_dim)`` NDHWC, or
                ``(B, z_dim, 1, h, w)`` NCDHW.
            return_ncdhw: If True, return Diffusers layout.
        """
        was_4d = len(z.shape) == 4
        z5 = self._to_ndhwc(z, is_latent=True)
        x = self.decoder(self.post_quant_conv(z5))
        x = ops.clip(x, -1.0, 1.0)
        if return_ncdhw:
            return ndhwc_to_ncdhw(x)
        return self._maybe_squeeze_t(x, was_4d)

    def get_config(self):
        config = super().get_config()
        config.update(self.config.constructor_kwargs())
        config["apply_temporal"] = self.apply_temporal
        return config

    @classmethod
    def from_diffusers_config(cls, config, sample_size=1024, **kwargs):
        return cls(
            base_dim=config.get("base_dim", 96),
            z_dim=config.get("z_dim", 16),
            dim_mult=tuple(config.get("dim_mult", (1, 2, 4, 4))),
            num_res_blocks=config.get("num_res_blocks", 2),
            attn_scales=tuple(config.get("attn_scales", ())),
            temperal_downsample=tuple(
                config.get("temperal_downsample", (False, True, True))
            ),
            dropout=config.get("dropout", 0.0),
            input_channels=config.get("input_channels", 3),
            latents_mean=tuple(config.get("latents_mean", DEFAULT_LATENTS_MEAN)),
            latents_std=tuple(config.get("latents_std", DEFAULT_LATENTS_STD)),
            sample_size=sample_size,
            **kwargs,
        )
