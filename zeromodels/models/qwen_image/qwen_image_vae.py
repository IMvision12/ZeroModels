import math

import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention
from zeromodels.models.stable_diffusion.stable_diffusion_layers import safe_name


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
        self.kernel_size = (
            (kernel_size, kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else tuple(kernel_size)
        )
        self.stride = (
            (stride, stride, stride) if isinstance(stride, int) else tuple(stride)
        )
        pad = (
            (padding, padding, padding) if isinstance(padding, int) else tuple(padding)
        )
        self.padding_t, self.padding_h, self.padding_w = pad
        self._pad_t_left = 2 * self.padding_t
        self._pad_t_right = 0
        self._pad_h = self.padding_h
        self._pad_w = self.padding_w
        self.module_path = module_path
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
        t2 = None if t is None else t + self._pad_t_left + self._pad_t_right
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
                h_p = None if h is None else h + 1
                w_p = None if w is None else w + 1
                self.spatial_conv.build((b, h_p, w_p, c))
            else:
                h_u = None if h is None else h * 2
                w_u = None if w is None else w * 2
                self.spatial_conv.build((b, h_u, w_u, c))
        if self.time_conv is not None:
            b, t, h, w, c = input_shape
            t_build = t if isinstance(t, int) and t >= 4 else 4
            self.time_conv.build((b, t_build, h, w, c))
        self.built = True

    def call(self, x):
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
            x = ops.reshape(x, (b, t, h, w, 2, c))
            x = ops.transpose(x, (0, 1, 4, 2, 3, 5))
            x = ops.reshape(x, (b, t * 2, h, w, c))

        b = ops.shape(x)[0]
        t = ops.shape(x)[1]
        h = ops.shape(x)[2]
        w = ops.shape(x)[3]
        c = ops.shape(x)[4]
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
        self.dropout = layers.Dropout(
            self.dropout_rate, name=safe_name(f"{module_path}.dropout")
        )
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
        self.norm = QwenImageRMSNorm(
            dim, images=True, module_path=f"{module_path}.norm"
        )
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
                mode = "downsample3d" if self.temperal_downsample[i] else "downsample2d"
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
