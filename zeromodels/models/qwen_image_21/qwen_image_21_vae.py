from __future__ import annotations

import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention
from zeromodels.models.qwen_image.qwen_image_vae import QwenImageRMSNorm
from zeromodels.models.stable_diffusion.stable_diffusion_layers import safe_name


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21CausalConv(layers.Layer):
    """Spatial Conv2d with NDHWC T-squeeze (Diffusers ``QwenImage21CausalConv3d``)."""

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
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            ks = tuple(kernel_size)
            self.kernel_size = (ks[-2], ks[-1]) if len(ks) == 3 else (ks[0], ks[1])
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            st = tuple(stride)
            self.stride = (st[-2], st[-1]) if len(st) == 3 else (st[0], st[1])
        if isinstance(padding, int):
            self.pad_h = self.pad_w = padding
        else:
            pad = tuple(padding)
            self.pad_h = pad[0] if len(pad) >= 1 else 0
            self.pad_w = pad[1] if len(pad) >= 2 else self.pad_h
        self.module_path = module_path
        self.conv = layers.Conv2D(
            self.out_channels,
            self.kernel_size,
            strides=self.stride,
            padding="valid",
            data_format="channels_last",
            name=safe_name(module_path) if module_path else "conv",
        )

    def build(self, input_shape):
        # (B, T, H, W, C) → build Conv2D on (B, H, W, C)
        b, _, h, w, c = input_shape
        h_p = None if h is None else h + 2 * self.pad_h
        w_p = None if w is None else w + 2 * self.pad_w
        self.conv.build((b, h_p, w_p, c))
        self.built = True

    def call(self, x):
        # x: (B, T, H, W, C) with T=1 for T2I
        b = ops.shape(x)[0]
        t = ops.shape(x)[1]
        h = ops.shape(x)[2]
        w = ops.shape(x)[3]
        c = ops.shape(x)[4]
        x2 = ops.reshape(x, (b * t, h, w, c))
        if self.pad_h or self.pad_w:
            x2 = ops.pad(
                x2,
                (
                    (0, 0),
                    (self.pad_h, self.pad_h),
                    (self.pad_w, self.pad_w),
                    (0, 0),
                ),
            )
        x2 = self.conv(x2)
        out_h = ops.shape(x2)[1]
        out_w = ops.shape(x2)[2]
        return ops.reshape(x2, (b, t, out_h, out_w, self.out_channels))

    def compute_output_shape(self, input_shape):
        b, t, h, w, _ = input_shape
        if h is None or w is None:
            return (b, t, None, None, self.out_channels)
        h_p = h + 2 * self.pad_h
        w_p = w + 2 * self.pad_w
        out_h = (h_p - self.kernel_size[0]) // self.stride[0] + 1
        out_w = (w_p - self.kernel_size[1]) // self.stride[1] + 1
        return (b, t, out_h, out_w, self.out_channels)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_channels": self.out_channels,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": (self.pad_h, self.pad_w),
                "module_path": self.module_path,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21AvgDown3D(layers.Layer):
    """Average downsample shortcut (Diffusers ``QwenImage21AvgDown3D``), NDHWC."""

    def __init__(self, in_channels, out_channels, factor_t, factor_s=1, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.factor_t = int(factor_t)
        self.factor_s = int(factor_s)
        self.factor = self.factor_t * self.factor_s * self.factor_s
        if self.in_channels * self.factor % self.out_channels != 0:
            raise ValueError(
                f"in_channels ({in_channels}) * factor ({self.factor}) must be "
                f"divisible by out_channels ({out_channels})."
            )
        self.group_size = self.in_channels * self.factor // self.out_channels

    def call(self, x):
        ft, fs = self.factor_t, self.factor_s
        pad_t = (ft - (ops.shape(x)[1] % ft)) % ft
        x = ops.pad(x, ((0, 0), (pad_t, 0), (0, 0), (0, 0), (0, 0)))
        b = ops.shape(x)[0]
        t = ops.shape(x)[1]
        h = ops.shape(x)[2]
        w = ops.shape(x)[3]
        c = self.in_channels
        x = ops.reshape(x, (b, t // ft, ft, h // fs, fs, w // fs, fs, c))
        x = ops.transpose(x, (0, 1, 3, 5, 7, 2, 4, 6))
        x = ops.reshape(x, (b, t // ft, h // fs, w // fs, c * self.factor))
        x = ops.reshape(
            x,
            (b, t // ft, h // fs, w // fs, self.out_channels, self.group_size),
        )
        return ops.mean(x, axis=-1)

    def compute_output_shape(self, input_shape):
        b, t, h, w, _ = input_shape
        ft, fs = self.factor_t, self.factor_s

        def down(size, factor):
            if size is None:
                return None
            pad = (factor - size % factor) % factor
            return (size + pad) // factor

        return (
            b,
            down(t, ft) if t is not None else None,
            down(h, fs),
            down(w, fs),
            self.out_channels,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "factor_t": self.factor_t,
                "factor_s": self.factor_s,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21DupUp3D(layers.Layer):
    """Duplicate upsample shortcut (Diffusers ``QwenImage21DupUp3D``), NDHWC."""

    def __init__(self, in_channels, out_channels, factor_t, factor_s=1, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.factor_t = int(factor_t)
        self.factor_s = int(factor_s)
        self.factor = self.factor_t * self.factor_s * self.factor_s
        assert self.out_channels * self.factor % self.in_channels == 0
        self.repeats = self.out_channels * self.factor // self.in_channels

    def call(self, x, first_chunk=False):
        x = ops.repeat(x, self.repeats, axis=-1)
        b = ops.shape(x)[0]
        t = ops.shape(x)[1]
        h = ops.shape(x)[2]
        w = ops.shape(x)[3]
        ft, fs = self.factor_t, self.factor_s
        x = ops.reshape(x, (b, t, h, w, self.out_channels, ft, fs, fs))
        x = ops.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))
        x = ops.reshape(x, (b, t * ft, h * fs, w * fs, self.out_channels))
        if first_chunk and ft > 1:
            x = x[:, ft - 1 :, :, :, :]
        return x

    def compute_output_shape(self, input_shape):
        b, t, h, w, _ = input_shape
        ft, fs = self.factor_t, self.factor_s
        return (
            b,
            None if t is None else t * ft,
            None if h is None else h * fs,
            None if w is None else w * fs,
            self.out_channels,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "factor_t": self.factor_t,
                "factor_s": self.factor_s,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21Resample(layers.Layer):
    """Spatial resample (Diffusers ``QwenImage21Resample``), T=1 / no temporal path."""

    def __init__(self, dim, mode, module_path, upsample_out_dim=None, **kwargs):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.mode = mode
        self.module_path = module_path
        self.upsample_out_dim = (
            int(upsample_out_dim) if upsample_out_dim is not None else self.dim // 2
        )
        self._downsample = mode in ("downsample2d", "downsample3d")
        self._upsample = mode in ("upsample2d", "upsample3d")
        self.spatial_conv = None
        if self._upsample:
            self.spatial_conv = layers.Conv2D(
                self.upsample_out_dim,
                3,
                padding="same",
                data_format="channels_last",
                name=safe_name(f"{module_path}.resample.1"),
            )
        elif self._downsample:
            self.spatial_conv = layers.Conv2D(
                self.dim,
                3,
                strides=2,
                padding="valid",
                data_format="channels_last",
                name=safe_name(f"{module_path}.resample.1"),
            )
        elif mode != "none":
            raise ValueError(f"Unknown resample mode {mode!r}")

    def build(self, input_shape):
        b, t, h, w, c = input_shape
        if self.spatial_conv is not None:
            if self._downsample:
                self.spatial_conv.build(
                    (b, None if h is None else h + 1, None if w is None else w + 1, c)
                )
            else:
                self.spatial_conv.build(
                    (b, None if h is None else h * 2, None if w is None else w * 2, c)
                )
        self.built = True

    def call(self, x):
        b = ops.shape(x)[0]
        t = ops.shape(x)[1]
        h = ops.shape(x)[2]
        w = ops.shape(x)[3]
        c = ops.shape(x)[4]
        x2 = ops.reshape(x, (b * t, h, w, c))
        if self._upsample:
            x2 = ops.image.resize(x2, (h * 2, w * 2), interpolation="nearest")
            x2 = self.spatial_conv(x2)
            out_c = self.upsample_out_dim
            out_h, out_w = h * 2, w * 2
        elif self._downsample:
            x2 = ops.pad(x2, ((0, 0), (0, 1), (0, 1), (0, 0)))
            x2 = self.spatial_conv(x2)
            out_c = self.dim
            out_h = (h + 1 - 3) // 2 + 1
            out_w = (w + 1 - 3) // 2 + 1
        else:
            out_c, out_h, out_w = c, h, w
        return ops.reshape(x2, (b, t, out_h, out_w, out_c))

    def compute_output_shape(self, input_shape):
        b, t, h, w, c = input_shape
        if self._upsample:
            return (
                b,
                t,
                None if h is None else h * 2,
                None if w is None else w * 2,
                self.upsample_out_dim,
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
                "upsample_out_dim": self.upsample_out_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21ResidualBlock(layers.Layer):
    """Residual block with spatial causal convs."""

    def __init__(self, in_dim, out_dim, module_path, dropout=0.0, **kwargs):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.module_path = module_path
        self.dropout_rate = float(dropout)
        self.norm1 = QwenImageRMSNorm(
            in_dim, images=False, module_path=f"{module_path}.norm1"
        )
        self.conv1 = QwenImage21CausalConv(
            out_dim, 3, padding=1, module_path=f"{module_path}.conv1"
        )
        self.norm2 = QwenImageRMSNorm(
            out_dim, images=False, module_path=f"{module_path}.norm2"
        )
        self.conv2 = QwenImage21CausalConv(
            out_dim, 3, padding=1, module_path=f"{module_path}.conv2"
        )
        self.conv_shortcut = None
        if in_dim != out_dim:
            self.conv_shortcut = QwenImage21CausalConv(
                out_dim, 1, padding=0, module_path=f"{module_path}.conv_shortcut"
            )
        self.dropout = layers.Dropout(dropout) if dropout > 0 else None

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
        h = self.conv_shortcut(x) if self.conv_shortcut is not None else x
        x = self.norm1(x)
        x = ops.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = ops.silu(x)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        x = self.conv2(x)
        return x + h

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[-1] = self.out_dim
        return tuple(shape)

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
class QwenImage21AttentionBlock(layers.Layer):
    """Spatial self-attention over H*W (Diffusers ``QwenImage21AttentionBlock``)."""

    def __init__(self, dim, module_path, **kwargs):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.module_path = module_path
        self.norm = QwenImageRMSNorm(
            dim, images=False, module_path=f"{module_path}.norm"
        )
        self.to_qkv = layers.Dense(
            dim * 3, use_bias=True, name=safe_name(f"{module_path}.to_qkv")
        )
        self.proj = layers.Dense(
            dim, use_bias=True, name=safe_name(f"{module_path}.proj")
        )

    def build(self, input_shape):
        self.norm.build(input_shape)
        self.to_qkv.build((*input_shape[:-1], self.dim))
        self.proj.build((*input_shape[:-1], self.dim))
        self.built = True

    def call(self, x):
        residual = x
        x = self.norm(x)
        b = ops.shape(x)[0]
        t = ops.shape(x)[1]
        h = ops.shape(x)[2]
        w = ops.shape(x)[3]
        c = self.dim
        x = ops.reshape(x, (b * t, h * w, c))
        qkv = self.to_qkv(x)
        q, k, v = ops.split(qkv, 3, axis=-1)
        q = ops.expand_dims(q, 1)
        k = ops.expand_dims(k, 1)
        v = ops.expand_dims(v, 1)
        out = fused_attention(q, k, v, self.dim**-0.5)
        out = ops.squeeze(out, 1)
        out = self.proj(out)
        out = ops.reshape(out, (b, t, h, w, c))
        return out + residual

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "module_path": self.module_path})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21MidBlock(layers.Layer):
    def __init__(self, dim, module_path, dropout=0.0, num_layers=1, **kwargs):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.module_path = module_path
        self.resnets = [
            QwenImage21ResidualBlock(
                dim, dim, f"{module_path}.resnets.0", dropout=dropout
            )
        ]
        self.attentions = []
        for i in range(num_layers):
            self.attentions.append(
                QwenImage21AttentionBlock(dim, f"{module_path}.attentions.{i}")
            )
            self.resnets.append(
                QwenImage21ResidualBlock(
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
        config.update({"dim": self.dim, "module_path": self.module_path})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21ResidualDownBlock(layers.Layer):
    def __init__(
        self,
        in_dim,
        out_dim,
        module_path,
        dropout=0.0,
        num_res_blocks=2,
        temperal_downsample=False,
        down_flag=False,
        **kwargs,
    ):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.module_path = module_path
        self.num_res_blocks = int(num_res_blocks)
        self.temperal_downsample = bool(temperal_downsample)
        self.down_flag = bool(down_flag)
        self.avg_shortcut = QwenImage21AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
            name=safe_name(f"{module_path}.avg_shortcut"),
        )
        self.resnets = []
        current = in_dim
        for i in range(num_res_blocks):
            self.resnets.append(
                QwenImage21ResidualBlock(
                    current, out_dim, f"{module_path}.resnets.{i}", dropout=dropout
                )
            )
            current = out_dim
        self.downsampler = None
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = QwenImage21Resample(
                out_dim, mode, f"{module_path}.downsampler"
            )

    def build(self, input_shape):
        self.avg_shortcut.build(input_shape)
        shape = input_shape
        for layer in self.resnets:
            layer.build(shape)
            shape = layer.compute_output_shape(shape)
        if self.downsampler is not None:
            self.downsampler.build(shape)
        self.built = True

    def call(self, x, training=None):
        shortcut = self.avg_shortcut(x)
        for resnet in self.resnets:
            x = resnet(x, training=training)
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x + shortcut

    def compute_output_shape(self, input_shape):
        shape = input_shape
        for layer in self.resnets:
            shape = layer.compute_output_shape(shape)
        if self.downsampler is not None:
            shape = self.downsampler.compute_output_shape(shape)
        return shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_dim": self.in_dim,
                "out_dim": self.out_dim,
                "module_path": self.module_path,
                "num_res_blocks": self.num_res_blocks,
                "temperal_downsample": self.temperal_downsample,
                "down_flag": self.down_flag,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21ResidualUpBlock(layers.Layer):
    def __init__(
        self,
        in_dim,
        out_dim,
        module_path,
        dropout=0.0,
        num_res_blocks=2,
        temperal_upsample=False,
        up_flag=False,
        **kwargs,
    ):
        kwargs.setdefault("name", safe_name(module_path))
        super().__init__(**kwargs)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.module_path = module_path
        self.num_res_blocks = int(num_res_blocks)
        self.temperal_upsample = bool(temperal_upsample)
        self.up_flag = bool(up_flag)
        self.avg_shortcut = None
        if up_flag:
            self.avg_shortcut = QwenImage21DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2,
                name=safe_name(f"{module_path}.avg_shortcut"),
            )
        self.resnets = []
        current = in_dim
        for i in range(num_res_blocks + 1):
            self.resnets.append(
                QwenImage21ResidualBlock(
                    current, out_dim, f"{module_path}.resnets.{i}", dropout=dropout
                )
            )
            current = out_dim
        self.upsampler = None
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler = QwenImage21Resample(
                out_dim, mode, f"{module_path}.upsampler", upsample_out_dim=out_dim
            )

    def build(self, input_shape):
        if self.avg_shortcut is not None:
            self.avg_shortcut.build(input_shape)
        shape = input_shape
        for layer in self.resnets:
            layer.build(shape)
            shape = layer.compute_output_shape(shape)
        if self.upsampler is not None:
            self.upsampler.build(shape)
        self.built = True

    def call(self, x, training=None, first_chunk=False):
        shortcut = None
        if self.avg_shortcut is not None:
            shortcut = self.avg_shortcut(x, first_chunk=first_chunk)
        for resnet in self.resnets:
            x = resnet(x, training=training)
        if self.upsampler is not None:
            x = self.upsampler(x)
        if shortcut is not None:
            x = x + shortcut
        return x

    def compute_output_shape(self, input_shape):
        shape = input_shape
        for layer in self.resnets:
            shape = layer.compute_output_shape(shape)
        if self.upsampler is not None:
            shape = self.upsampler.compute_output_shape(shape)
        return shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_dim": self.in_dim,
                "out_dim": self.out_dim,
                "module_path": self.module_path,
                "num_res_blocks": self.num_res_blocks,
                "temperal_upsample": self.temperal_upsample,
                "up_flag": self.up_flag,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21Encoder3d(layers.Layer):
    def __init__(
        self,
        dim=96,
        z_dim=128,
        dim_mult=(1, 2, 4, 8, 8),
        num_res_blocks=2,
        attn_scales=(),
        temperal_downsample=(False, True, True, True),
        dropout=0.0,
        input_channels=4,
        is_residual=True,
        module_path="encoder",
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
        self.is_residual = bool(is_residual)
        self.module_path = module_path

        dims = [dim * u for u in [1] + list(dim_mult)]
        self.conv_in = QwenImage21CausalConv(
            dims[0], 3, padding=1, module_path=f"{module_path}.conv_in"
        )
        self.down_blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            self.down_blocks.append(
                QwenImage21ResidualDownBlock(
                    in_dim,
                    out_dim,
                    f"{module_path}.down_blocks.{i}",
                    dropout=dropout,
                    num_res_blocks=num_res_blocks,
                    temperal_downsample=(
                        temperal_downsample[i] if i != len(dim_mult) - 1 else False
                    ),
                    down_flag=i != len(dim_mult) - 1,
                )
            )
        self.mid_block = QwenImage21MidBlock(
            dims[-1], f"{module_path}.mid_block", dropout=dropout, num_layers=1
        )
        self.norm_out = QwenImageRMSNorm(
            dims[-1], images=False, module_path=f"{module_path}.norm_out"
        )
        self.conv_out = QwenImage21CausalConv(
            z_dim, 3, padding=1, module_path=f"{module_path}.conv_out"
        )

    def build(self, input_shape):
        self.conv_in.build(input_shape)
        shape = self.conv_in.compute_output_shape(input_shape)
        for block in self.down_blocks:
            block.build(shape)
            shape = block.compute_output_shape(shape)
        self.mid_block.build(shape)
        self.norm_out.build(shape)
        self.conv_out.build(shape)
        self.built = True

    def call(self, x, training=None):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x, training=training)
        x = self.mid_block(x, training=training)
        x = self.norm_out(x)
        x = ops.silu(x)
        return self.conv_out(x)

    def compute_output_shape(self, input_shape):
        shape = self.conv_in.compute_output_shape(input_shape)
        for block in self.down_blocks:
            shape = block.compute_output_shape(shape)
        shape = list(shape)
        shape[-1] = self.z_dim
        return tuple(shape)

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
                "is_residual": self.is_residual,
                "module_path": self.module_path,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21Decoder3d(layers.Layer):
    def __init__(
        self,
        dim=144,
        z_dim=64,
        dim_mult=(1, 2, 4, 8, 8),
        num_res_blocks=2,
        attn_scales=(),
        temperal_upsample=(True, True, True, False),
        dropout=0.0,
        out_channels=4,
        is_residual=True,
        module_path="decoder",
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
        self.out_channels = int(out_channels)
        self.is_residual = bool(is_residual)
        self.module_path = module_path

        dims = [dim * u for u in [self.dim_mult[-1]] + list(self.dim_mult[::-1])]
        self.conv_in = QwenImage21CausalConv(
            dims[0], 3, padding=1, module_path=f"{module_path}.conv_in"
        )
        self.mid_block = QwenImage21MidBlock(
            dims[0], f"{module_path}.mid_block", dropout=dropout, num_layers=1
        )
        self.up_blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            up_flag = i != len(self.dim_mult) - 1
            self.up_blocks.append(
                QwenImage21ResidualUpBlock(
                    in_dim,
                    out_dim,
                    f"{module_path}.up_blocks.{i}",
                    dropout=dropout,
                    num_res_blocks=num_res_blocks,
                    temperal_upsample=self.temperal_upsample[i] if up_flag else False,
                    up_flag=up_flag,
                )
            )
        self.norm_out = QwenImageRMSNorm(
            dims[-1], images=False, module_path=f"{module_path}.norm_out"
        )
        self.conv_out = QwenImage21CausalConv(
            out_channels, 3, padding=1, module_path=f"{module_path}.conv_out"
        )

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
            x = block(x, training=training, first_chunk=True)
        x = self.norm_out(x)
        x = ops.silu(x)
        return self.conv_out(x)

    def compute_output_shape(self, input_shape):
        b, t, h, w, _ = input_shape
        scale = 2 ** (len(self.dim_mult) - 1)
        return (
            b,
            t,
            None if h is None else h * scale,
            None if w is None else w * scale,
            self.out_channels,
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
                "out_channels": self.out_channels,
                "is_residual": self.is_residual,
                "module_path": self.module_path,
            }
        )
        return config
