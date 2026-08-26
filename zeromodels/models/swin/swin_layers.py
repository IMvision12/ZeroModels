import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="zeromodels")
class SwinWindowPartition(layers.Layer):
    """Partitions input tensor into non-overlapping windows.

    Divides the input feature map into non-overlapping windows for efficient
    attention computation. Supports both standard and fused attention modes.

    Args:
        window_size: int, size of each window (height and width).
        fused: bool, if True, operates in fused attention mode. Default False.
        num_heads: int, number of attention heads. Required when fused=True.
        qkv_mult: int, multiplier for QKV transformations. Default 3.
        data_format: string, ``"channels_last"`` or ``"channels_first"``.
            Default ``"channels_last"``.
        **kwargs: Additional keyword arguments passed to the Layer class.
    """

    def __init__(
        self,
        window_size,
        fused=False,
        num_heads=None,
        qkv_mult=3,
        data_format="channels_last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.fused = fused
        self.num_heads = num_heads
        self.qkv_mult = qkv_mult
        self.data_format = data_format

        if self.fused and self.num_heads is None:
            raise ValueError("num_heads must be set when fused=True")

    def call(self, inputs, height=None, width=None):
        if len(inputs.shape) != 4:
            raise ValueError("Expecting inputs rank to be 4.")
        if height is None or width is None:
            raise ValueError("Height and width must be provided")

        windows_height = height // self.window_size
        windows_width = width // self.window_size
        cf = self.data_format == "channels_first"

        if not self.fused:
            channels = inputs.shape[1] if cf else inputs.shape[-1]
            if channels is None:
                raise ValueError("Channel dimension must be defined.")

            if cf:
                x = ops.reshape(
                    inputs,
                    [
                        -1,
                        channels,
                        windows_height,
                        self.window_size,
                        windows_width,
                        self.window_size,
                    ],
                )
                # (B, C, wh, ws, ww, ws) -> (B, wh, ww, ws, ws, C): channels must
                # end up last so the flatten groups (window_size**2, channels).
                x = ops.transpose(x, [0, 2, 4, 3, 5, 1])
                outputs = ops.reshape(x, [-1, self.window_size**2, channels])
            else:
                x = ops.reshape(
                    inputs,
                    [
                        -1,
                        windows_height,
                        self.window_size,
                        windows_width,
                        self.window_size,
                        channels,
                    ],
                )
                x = ops.transpose(x, [0, 1, 3, 2, 4, 5])
                outputs = ops.reshape(x, [-1, self.window_size**2, channels])
        else:
            full_channels = inputs.shape[1] if cf else inputs.shape[-1]
            if full_channels is None:
                raise ValueError("Channel dimension must be defined.")
            head_channels = full_channels // (self.qkv_mult * self.num_heads)

            if cf:
                x = ops.reshape(
                    inputs,
                    [
                        -1,
                        self.qkv_mult,
                        self.num_heads,
                        head_channels,
                        windows_height,
                        self.window_size,
                        windows_width,
                        self.window_size,
                    ],
                )
                x = ops.transpose(x, [1, 0, 4, 6, 2, 5, 7, 3])
                outputs = ops.reshape(
                    x,
                    [
                        self.qkv_mult,
                        -1,
                        self.num_heads,
                        self.window_size**2,
                        head_channels,
                    ],
                )
            else:
                x = ops.reshape(
                    inputs,
                    [
                        -1,
                        windows_height,
                        self.window_size,
                        windows_width,
                        self.window_size,
                        self.qkv_mult,
                        self.num_heads,
                        head_channels,
                    ],
                )
                x = ops.transpose(x, [5, 0, 1, 3, 6, 2, 4, 7])
                outputs = ops.reshape(
                    x,
                    [
                        self.qkv_mult,
                        -1,
                        self.num_heads,
                        self.window_size**2,
                        head_channels,
                    ],
                )

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "window_size": self.window_size,
                "fused": self.fused,
                "num_heads": self.num_heads,
                "qkv_mult": self.qkv_mult,
                "data_format": self.data_format,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SwinWindowReverse(layers.Layer):
    """Reconstructs the feature map from partitioned windows.

    Inverse of ``SwinWindowPartition``. Supports both standard and fused modes.

    Args:
        window_size: int, size of each window.
        fused: bool, if True, operates in fused attention mode. Default False.
        num_heads: int, number of attention heads. Required when fused=True.
        data_format: string, ``"channels_last"`` or ``"channels_first"``.
            Default ``"channels_last"``.
        **kwargs: Additional keyword arguments passed to the Layer class.
    """

    def __init__(
        self,
        window_size,
        fused=False,
        num_heads=None,
        data_format="channels_last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.fused = fused
        self.num_heads = num_heads
        self.data_format = data_format

        if self.fused and self.num_heads is None:
            raise ValueError("num_heads must be set when fused=True")

    def call(self, inputs, height=None, width=None):
        if height is None or width is None:
            raise ValueError("Height and width must be provided")

        windows_height = height // self.window_size
        windows_width = width // self.window_size
        cf = self.data_format == "channels_first"

        if not self.fused:
            if len(inputs.shape) != 3:
                raise ValueError("Expecting inputs rank to be 3.")
            channels = inputs.shape[-1]
            if channels is None:
                raise ValueError("Channel dimension must be defined.")

            x = ops.reshape(
                inputs,
                [
                    -1,
                    windows_height,
                    windows_width,
                    self.window_size,
                    self.window_size,
                    channels,
                ],
            )
            x = ops.transpose(x, [0, 1, 3, 2, 4, 5])
            # buffer is now (B, H, W, C) ordered; reshape to that grid, then
            # transpose to (B, C, H, W) for channels_first (a direct reshape to
            # (C, H, W) would scramble channels with spatial).
            x = ops.reshape(x, [-1, height, width, channels])
            if cf:
                x = ops.transpose(x, [0, 3, 1, 2])
            outputs = x
        else:
            if len(inputs.shape) != 4:
                raise ValueError("Expecting inputs rank to be 4.")
            head_channels = inputs.shape[-1]
            if head_channels is None:
                raise ValueError("Channel dimension must be defined.")
            full_channels = head_channels * self.num_heads

            x = ops.reshape(
                inputs,
                [
                    -1,
                    windows_height,
                    windows_width,
                    self.num_heads,
                    self.window_size,
                    self.window_size,
                    head_channels,
                ],
            )
            if cf:
                x = ops.transpose(x, [0, 3, 6, 1, 4, 2, 5])
                outputs = ops.reshape(x, [-1, full_channels, height, width])
            else:
                x = ops.transpose(x, [0, 1, 4, 2, 5, 3, 6])
                outputs = ops.reshape(x, [-1, height, width, full_channels])

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "window_size": self.window_size,
                "fused": self.fused,
                "num_heads": self.num_heads,
                "data_format": self.data_format,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SwinWindowAttention(layers.Layer):
    """Window-based Multi-Head Self-Attention with relative positional bias.

    Divides the input into windows, computes attention within each window,
    and reconstructs the output. Includes learnable relative positional bias
    and optional attention masking for shifted windows.

    Args:
        dim: int, input and output feature dimension.
        num_heads: int, number of attention heads.
        window_size: int, window size for attention computation.
        bias_table_window_size: int, size of the relative position bias table.
        qkv_bias: bool, whether to use bias in QKV projection. Default True.
        qk_scale: float or None, scaling factor for QK. Default None (head_dim**-0.5).
        attn_drop: float, dropout rate for attention weights. Default 0.0.
        proj_drop: float, dropout rate for output projection. Default 0.0.
        data_format: string, ``"channels_last"`` or ``"channels_first"``.
            Default ``"channels_last"``.
        block_prefix: string, prefix for layer names. Default None.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        bias_table_window_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        data_format="channels_last",
        block_prefix=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = int(window_size)
        self.bias_table_window_size = int(bias_table_window_size)
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.qkv_bias = qkv_bias
        self.data_format = data_format

        self.block_prefix = block_prefix if block_prefix is not None else "blocks"
        prefix = f"{self.block_prefix}_"

        self.qkv = layers.Dense(
            dim * 3,
            use_bias=qkv_bias,
            dtype=self.dtype_policy,
            name=prefix + "attn_qkv",
        )
        self.window_partition = SwinWindowPartition(
            window_size=self.window_size,
            fused=True,
            num_heads=num_heads,
            qkv_mult=3,
            data_format=data_format,
        )
        self.window_reverse = SwinWindowReverse(
            window_size=self.window_size,
            fused=True,
            num_heads=num_heads,
            data_format=data_format,
        )
        self.drop_attn = layers.Dropout(attn_drop, dtype=self.dtype_policy)
        self.proj = layers.Dense(
            dim,
            dtype=self.dtype_policy,
            name=prefix + "attn_proj",
        )
        self.drop_proj = layers.Dropout(proj_drop, dtype=self.dtype_policy)

        self.attn_drop_rate = attn_drop
        self.proj_drop_rate = proj_drop

    def build(self, input_shape):
        cf = self.data_format == "channels_first"
        feature_dim = input_shape[0][1] if cf else input_shape[0][-1]
        if feature_dim is None:
            raise ValueError("Channel dimension must be defined.")
        if feature_dim != self.dim:
            raise ValueError(
                f"Input feature dimension {feature_dim} must match layer dimension {self.dim}"
            )

        if cf:
            self.qkv.build(
                (input_shape[0][0], input_shape[0][2], input_shape[0][3], self.dim)
            )
        else:
            self.qkv.build(input_shape[0])

        if cf:
            self.proj.build(
                (input_shape[0][0], input_shape[0][2], input_shape[0][3], self.dim)
            )
        else:
            self.proj.build(
                (input_shape[0][0], input_shape[0][1], input_shape[0][2], self.dim)
            )

        prefix = f"{self.block_prefix}_"
        self.relative_bias = self.add_weight(
            name=prefix + "attn_relative_position_bias_table",
            shape=[(2 * self.bias_table_window_size - 1) ** 2, self.num_heads],
            trainable=True,
            dtype=self.dtype,
        )
        self.built = True

    def with_mask(self, attn, mask, length):
        mask_windows = ops.shape(mask)[1]
        attn = ops.reshape(attn, [-1, mask_windows, self.num_heads, length, length])
        attn = attn + mask
        attn = ops.reshape(attn, [-1, self.num_heads, length, length])
        return attn

    def call(self, inputs, training=None):
        inputs, _, relative_index, attention_mask = inputs
        cf = self.data_format == "channels_first"

        if cf:
            height, width = ops.shape(inputs)[2], ops.shape(inputs)[3]
            x = ops.transpose(inputs, [0, 2, 3, 1])
        else:
            height, width = ops.shape(inputs)[1], ops.shape(inputs)[2]
            x = inputs

        length = self.window_size**2

        qkv = self.qkv(x)
        if cf:
            qkv = ops.transpose(qkv, [0, 3, 1, 2])

        qkv = self.window_partition(qkv, height=height, width=width)
        q, k, v = ops.unstack(qkv, 3)

        q = q * self.scale
        k = ops.swapaxes(k, -2, -1)
        attn = ops.matmul(q, k)

        bias = ops.take(self.relative_bias, relative_index, axis=0)
        bias = ops.reshape(bias, [length, length, -1])
        bias = ops.transpose(bias, [2, 0, 1])
        attn = attn + bias[None]

        if attention_mask is not None:
            attn = self.with_mask(attn, attention_mask, length)

        attn = ops.softmax(attn)
        attn = self.drop_attn(attn, training=training)

        out = ops.matmul(attn, v)
        out = self.window_reverse(out, height=height, width=width)

        if cf:
            out = ops.transpose(out, [0, 2, 3, 1])
        out = self.proj(out)
        if cf:
            out = ops.transpose(out, [0, 3, 1, 2])
        out = self.drop_proj(out, training=training)

        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "bias_table_window_size": self.bias_table_window_size,
                "qkv_bias": self.qkv_bias,
                "qk_scale": self.scale if self.scale != self.head_dim**-0.5 else None,
                "attn_drop": self.attn_drop_rate,
                "proj_drop": self.proj_drop_rate,
                "data_format": self.data_format,
                "block_prefix": self.block_prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SwinRollLayer(layers.Layer):
    """Circular shift of tensor elements along specified axes.

    Args:
        shift: int or tuple of ints, number of positions to shift.
        axis: int or tuple of ints, axes along which to shift.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, shift, axis, **kwargs):
        super().__init__(**kwargs)
        self.shift = shift
        self.axis = axis

    def call(self, inputs):
        return ops.roll(inputs, shift=self.shift, axis=self.axis)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"shift": self.shift, "axis": self.axis})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SwinStochasticDepth(keras.layers.Layer):
    """Stochastic depth: randomly drops the residual path during training (identity at inference)."""

    def __init__(self, drop_path_rate, seed=None, **kwargs):
        super().__init__(**kwargs)
        if not 0 <= drop_path_rate <= 1:
            raise ValueError(
                f"drop_path_rate should be between 0 and 1, got {drop_path_rate}"
            )
        self.drop_path_rate = drop_path_rate
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            shape = (keras.ops.shape(x)[0],) + (1,) * (len(keras.ops.shape(x)) - 1)
            random_tensor = keep_prob + keras.random.uniform(
                shape, 0, 1, seed=self.seed_generator
            )
            random_tensor = keras.ops.cast(keras.ops.floor(random_tensor), x.dtype)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate, "seed": self.seed})
        return config
