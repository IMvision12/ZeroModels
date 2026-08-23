import math

import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="kerasformers")
class SwinV2WindowPartition(layers.Layer):
    """Partitions input tensor into non-overlapping windows.

    Divides the input feature map into non-overlapping windows of specified
    size. Supports both standard partitioning (fused=False) and fused
    attention mode (fused=True) that handles multi-headed attention and QKV
    transformations in a single reshape-transpose operation.

    For ``channels_first`` inputs, converts to ``channels_last`` layout
    internally so downstream attention operates on ``(batch, seq, dim)``
    tokens.

    Args:
        window_size: int. Size of each window (both height and width).
        fused: bool. If True, operates in fused attention mode that produces
            output shaped for multi-head attention with QKV split.
            Defaults to False.
        num_heads: int. Number of attention heads. Required when fused=True.
        qkv_mult: int. Multiplier for QKV transformations. Defaults to 3.
        data_format: str. ``"channels_last"`` or ``"channels_first"``.
            Defaults to ``"channels_last"``.
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

        cf = self.data_format == "channels_first"
        windows_height = height // self.window_size
        windows_width = width // self.window_size

        if not self.fused:
            channels = inputs.shape[1] if cf else inputs.shape[-1]
            if channels is None:
                raise ValueError(
                    "Channel dimensions of the inputs should be defined. Found `None`."
                )

            if cf:
                outputs = ops.reshape(
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
                outputs = ops.transpose(outputs, [0, 2, 4, 3, 5, 1])
                outputs = ops.reshape(outputs, [-1, self.window_size**2, channels])
            else:
                outputs = ops.reshape(
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
                outputs = ops.transpose(outputs, [0, 1, 3, 2, 4, 5])
                outputs = ops.reshape(outputs, [-1, self.window_size**2, channels])

        else:
            full_channels = inputs.shape[1] if cf else inputs.shape[-1]
            if full_channels is None:
                raise ValueError(
                    "Channel dimensions of the inputs should be defined. Found `None`."
                )

            head_channels = full_channels // (self.qkv_mult * self.num_heads)

            if cf:
                outputs = ops.reshape(
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
                outputs = ops.transpose(outputs, [1, 0, 4, 6, 2, 5, 7, 3])
                outputs = ops.reshape(
                    outputs,
                    [
                        self.qkv_mult,
                        -1,
                        self.num_heads,
                        self.window_size**2,
                        head_channels,
                    ],
                )
            else:
                outputs = ops.reshape(
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
                outputs = ops.transpose(outputs, [5, 0, 1, 3, 6, 2, 4, 7])
                outputs = ops.reshape(
                    outputs,
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


@keras.saving.register_keras_serializable(package="kerasformers")
class SwinV2WindowReverse(layers.Layer):
    """Reverses window partitioning to reconstruct the original feature map.

    Performs the inverse operation of ``SwinV2WindowPartition``, merging
    window segments back into a contiguous spatial feature map. Supports
    both standard and fused attention modes.

    Outputs in the requested ``data_format`` (``channels_first`` or
    ``channels_last``).

    Args:
        window_size: int. Size of each window (both height and width).
        fused: bool. If True, operates in fused attention mode.
            Defaults to False.
        num_heads: int. Number of attention heads. Required when fused=True.
        data_format: str. ``"channels_last"`` or ``"channels_first"``.
            Defaults to ``"channels_last"``.
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

        cf = self.data_format == "channels_first"
        windows_height = height // self.window_size
        windows_width = width // self.window_size

        if not self.fused:
            if len(inputs.shape) != 3:
                raise ValueError("Expecting inputs rank to be 3.")

            channels = inputs.shape[-1]
            if channels is None:
                raise ValueError(
                    "Channel dimensions of the inputs should be defined. Found `None`."
                )

            outputs = ops.reshape(
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
            outputs = ops.transpose(outputs, [0, 1, 3, 2, 4, 5])
            # buffer is now (B, H, W, C) ordered; reshape to that grid, then
            # transpose to (B, C, H, W) for channels_first (a direct reshape to
            # (C, H, W) would scramble channels with spatial).
            outputs = ops.reshape(outputs, [-1, height, width, channels])
            if cf:
                outputs = ops.transpose(outputs, [0, 3, 1, 2])

        else:
            if len(inputs.shape) != 4:
                raise ValueError("Expecting inputs rank to be 4.")

            head_channels = inputs.shape[-1]
            if head_channels is None:
                raise ValueError(
                    "Channel dimensions of the inputs should be defined. Found `None`."
                )

            full_channels = head_channels * self.num_heads

            outputs = ops.reshape(
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
                outputs = ops.transpose(outputs, [0, 3, 6, 1, 4, 2, 5])
                outputs = ops.reshape(outputs, [-1, full_channels, height, width])
            else:
                outputs = ops.transpose(outputs, [0, 1, 4, 2, 5, 3, 6])
                outputs = ops.reshape(outputs, [-1, height, width, full_channels])

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


@keras.saving.register_keras_serializable(package="kerasformers")
class SwinV2Attention(layers.Layer):
    """Window-based multi-head self-attention with cosine similarity (SwinV2).

    Implements the attention mechanism from Swin Transformer V2 with several
    key differences from V1:

    - **Cosine attention**: Q and K are L2-normalized before computing
      attention, with a learnable per-head ``logit_scale``.
    - **Continuous Position Bias (CPB)**: A small MLP maps log-spaced
      relative coordinates to per-head bias values, replacing the learnable
      bias table from V1.
    - **Separate Q/V bias**: Learnable ``q_bias`` and ``v_bias`` are added
      to the QKV projection; K has no bias.

    For ``channels_first`` inputs, permutes to ``channels_last`` for the
    QKV projection and permutes back after the output projection.

    Args:
        dim: int. Total dimension of input and output features.
        num_heads: int. Number of parallel attention heads.
        window_size: int. Spatial size of the attention window (W x W).
        pretrained_window_size: int. Window size used during pretraining
            for CPB coordinate normalization. Defaults to 0.
        attn_drop: float. Dropout rate for attention weights. Defaults to 0.0.
        proj_drop: float. Dropout rate for output projection. Defaults to 0.0.
        data_format: str. ``"channels_last"`` or ``"channels_first"``.
            Defaults to ``"channels_last"``.
        block_prefix: str. Prefix for naming layer components.
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        pretrained_window_size=0,
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
        self.pretrained_window_size = int(pretrained_window_size)
        self.head_dim = dim // num_heads
        self.data_format = data_format

        self.block_prefix = block_prefix if block_prefix is not None else "blocks"
        prefix = f"{self.block_prefix}_"

        self.qkv = layers.Dense(
            dim * 3,
            use_bias=False,
            dtype=self.dtype_policy,
            name=prefix + "attn_qkv",
        )

        self.window_partition = SwinV2WindowPartition(
            window_size=self.window_size,
            fused=True,
            num_heads=num_heads,
            qkv_mult=3,
            data_format=data_format,
        )
        self.window_reverse = SwinV2WindowReverse(
            window_size=self.window_size,
            fused=True,
            num_heads=num_heads,
            data_format=data_format,
        )

        self.cpb_dense1 = layers.Dense(
            512,
            use_bias=True,
            dtype=self.dtype_policy,
            name=prefix + "attn_cpb_mlp_0",
        )
        self.cpb_dense2 = layers.Dense(
            num_heads,
            use_bias=False,
            dtype=self.dtype_policy,
            name=prefix + "attn_cpb_mlp_2",
        )

        self.drop_attn = layers.Dropout(
            attn_drop,
            dtype=self.dtype_policy,
        )

        self.proj = layers.Dense(
            dim, dtype=self.dtype_policy, name=prefix + "attn_proj"
        )

        self.drop_proj = layers.Dropout(
            proj_drop,
            dtype=self.dtype_policy,
        )

        self.attn_drop_rate = attn_drop
        self.proj_drop_rate = proj_drop

    def build(self, input_shape):
        cf = self.data_format == "channels_first"
        feature_dim = input_shape[0][1] if cf else input_shape[0][-1]
        if feature_dim is None:
            raise ValueError(
                "Channel dimensions of the inputs should be defined. Found `None`."
            )

        if feature_dim != self.dim:
            raise ValueError(
                f"Input feature dimension {feature_dim} must match layer dimension {self.dim}"
            )

        if cf:
            self.qkv.build(
                (input_shape[0][0], input_shape[0][2], input_shape[0][3], self.dim)
            )
            self.proj.build(
                (input_shape[0][0], input_shape[0][2], input_shape[0][3], self.dim)
            )
        else:
            self.qkv.build(input_shape[0])
            self.proj.build(
                (input_shape[0][0], input_shape[0][1], input_shape[0][2], self.dim)
            )

        prefix = f"{self.block_prefix}_"

        self.logit_scale = self.add_weight(
            name=prefix + "attn_logit_scale",
            shape=[self.num_heads],
            initializer=keras.initializers.Constant(math.log(10.0)),
            trainable=True,
            dtype=self.dtype,
        )

        self.q_bias = self.add_weight(
            name=prefix + "attn_q_bias",
            shape=[self.dim],
            initializer="zeros",
            trainable=True,
            dtype=self.dtype,
        )
        self.v_bias = self.add_weight(
            name=prefix + "attn_v_bias",
            shape=[self.dim],
            initializer="zeros",
            trainable=True,
            dtype=self.dtype,
        )

        self.cpb_dense1.build((None, 2))
        self.cpb_dense2.build((None, 512))

        ws = self.window_size
        coords_h = ops.cast(ops.arange(-(ws - 1), ws), "float32")
        coords_w = ops.cast(ops.arange(-(ws - 1), ws), "float32")
        grid_h, grid_w = ops.meshgrid(coords_h, coords_w, indexing="ij")
        coords_table = ops.stack([grid_h, grid_w], axis=-1)

        norm = (
            (self.pretrained_window_size - 1)
            if self.pretrained_window_size > 0
            else (ws - 1)
        )
        coords_table = coords_table / norm

        coords_table = (
            ops.sign(coords_table)
            * ops.log2(1.0 + ops.abs(coords_table) * 8.0)
            / math.log2(8.0)
        )
        coords_table = ops.convert_to_numpy(coords_table)

        self.relative_coords_table = self.add_weight(
            name=prefix + "attn_relative_coords_table",
            shape=coords_table.shape,
            initializer=keras.initializers.Constant(coords_table),
            trainable=False,
            dtype=self.dtype,
        )

        coords = ops.arange(ws)
        gx, gy = ops.meshgrid(coords, coords, indexing="ij")
        flat_gx = ops.reshape(gx, [-1])
        flat_gy = ops.reshape(gy, [-1])

        rel_pos_x = flat_gx[:, None] - flat_gx[None, :]
        rel_pos_y = flat_gy[:, None] - flat_gy[None, :]

        rel_pos_x = rel_pos_x + ws - 1
        rel_pos_y = rel_pos_y + ws - 1

        rel_pos_index = rel_pos_x * (2 * ws - 1) + rel_pos_y
        rel_pos_index = ops.convert_to_numpy(ops.reshape(rel_pos_index, [-1]))
        rel_pos_index = rel_pos_index.astype("int32")

        self.relative_position_index = self.add_weight(
            name=prefix + "attn_relative_position_index",
            shape=rel_pos_index.shape,
            initializer=keras.initializers.Constant(rel_pos_index),
            trainable=False,
            dtype="int32",
        )

        self.built = True

    def with_mask(self, attn, mask, length):
        mask_windows = ops.shape(mask)[1]
        attn = ops.reshape(attn, [-1, mask_windows, self.num_heads, length, length])
        attn = attn + mask
        attn = ops.reshape(attn, [-1, self.num_heads, length, length])
        return attn

    def call(self, inputs, training=None):
        inputs, _, attention_mask = inputs
        cf = self.data_format == "channels_first"

        if cf:
            height, width = ops.shape(inputs)[2], ops.shape(inputs)[3]
            x = ops.transpose(inputs, [0, 2, 3, 1])
        else:
            height, width = ops.shape(inputs)[1], ops.shape(inputs)[2]
            x = inputs

        length = self.window_size**2

        k_bias = ops.zeros(ops.shape(self.q_bias), dtype=self.q_bias.dtype)
        qkv_bias = ops.concatenate([self.q_bias, k_bias, self.v_bias])

        # Call the (bias-free) Dense rather than reading `.kernel` directly, so a
        # quantized qkv still works (its `call` dequantizes on the fly).
        qkv = self.qkv(x)
        qkv = qkv + qkv_bias

        if cf:
            qkv = ops.transpose(qkv, [0, 3, 1, 2])

        qkv = self.window_partition(qkv, height=height, width=width)
        q, k, v = ops.unstack(qkv, 3)

        q_norm = ops.sqrt(
            ops.maximum(ops.sum(q * q, axis=-1, keepdims=True), ops.cast(1e-6, q.dtype))
        )
        k_norm = ops.sqrt(
            ops.maximum(ops.sum(k * k, axis=-1, keepdims=True), ops.cast(1e-6, k.dtype))
        )
        q = q / q_norm
        k = k / k_norm

        k = ops.swapaxes(k, -2, -1)
        attn = ops.matmul(q, k)

        log_scale = ops.minimum(self.logit_scale, math.log(100.0))
        scale = ops.exp(log_scale)
        attn = attn * scale[None, :, None, None]

        table = ops.reshape(self.relative_coords_table, [-1, 2])
        cpb = self.cpb_dense1(table)
        cpb = ops.relu(cpb)
        cpb = self.cpb_dense2(cpb)

        cpb = 16.0 * ops.sigmoid(cpb)

        bias = ops.take(cpb, self.relative_position_index, axis=0)
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
                "pretrained_window_size": self.pretrained_window_size,
                "attn_drop": self.attn_drop_rate,
                "proj_drop": self.proj_drop_rate,
                "data_format": self.data_format,
                "block_prefix": self.block_prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class SwinV2Roll(layers.Layer):
    """Circular shift of tensor elements along spatial axes.

    Wraps ``ops.roll`` as a serializable Keras layer for use in shifted
    window attention. Elements that roll past the last position are
    re-introduced at the first position.

    Args:
        shift: int or tuple of ints. Number of positions to shift.
        axis: int or tuple of ints. Axes along which to shift.
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
        config.update(
            {
                "shift": self.shift,
                "axis": self.axis,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class SwinV2StochasticDepth(keras.layers.Layer):
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
