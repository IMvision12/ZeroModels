import keras
import numpy as np
from keras import layers, ops


def beit_relative_position_index(window_size):
    window_h, window_w = window_size
    num_rel = (2 * window_h - 1) * (2 * window_w - 1) + 3
    area = window_h * window_w
    coords = np.stack(
        np.meshgrid(np.arange(window_h), np.arange(window_w), indexing="ij")
    )
    coords_flat = coords.reshape(2, -1)
    rel = coords_flat[:, :, None] - coords_flat[:, None, :]
    rel = rel.transpose(1, 2, 0)
    rel[:, :, 0] += window_h - 1
    rel[:, :, 1] += window_w - 1
    rel[:, :, 0] *= 2 * window_w - 1
    index = np.zeros((area + 1, area + 1), dtype="int32")
    index[1:, 1:] = rel.sum(-1)
    index[0, 0:] = num_rel - 3
    index[0:, 0] = num_rel - 2
    index[0, 0] = num_rel - 1
    return index


@keras.saving.register_keras_serializable(package="zeromodels")
class BeitClsToken(layers.Layer):
    """Prepends a learnable CLS token to the patch-token sequence."""

    def __init__(self, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.hidden_size),
            initializer="zeros",
            trainable=True,
            name="cls_token",
        )
        self.built = True

    def call(self, x):
        batch_size = ops.shape(x)[0]
        cls = ops.broadcast_to(self.cls_token, (batch_size, 1, self.hidden_size))
        return ops.concatenate([cls, x], axis=1)

    def compute_output_shape(self, input_shape):
        seq = None if input_shape[1] is None else input_shape[1] + 1
        return (input_shape[0], seq, self.hidden_size)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class BeitLayerScale(layers.Layer):
    """Learnable per-channel scale (x * gamma), gamma initialized to init_value."""

    def __init__(self, init_value, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer=keras.initializers.Constant(self.init_value),
            trainable=True,
            name="lambda",
        )
        self.built = True

    def call(self, x):
        return x * self.gamma

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"init_value": self.init_value})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class BeitAttention(layers.Layer):
    """Multi-head self-attention with a per-layer 2D relative position bias.

    Query and value projections carry a bias; the key projection does not, matching
    the BEiT checkpoints. The relative position bias is gathered from a learnable
    table via a fixed relative-position index (built from the patch grid) and added
    to the attention scores before the softmax.

    Args:
        hidden_size: Total model dimension. Must be divisible by ``num_heads``.
        num_heads: Number of attention heads.
        window_size: ``(grid_h, grid_w)`` patch grid, i.e. image_size // patch_size.
        block_prefix: Name prefix for the internal dense layers.
        **kwargs: Additional keyword arguments passed to the `Layer` class.
    """

    def __init__(self, hidden_size, num_heads, window_size, block_prefix, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.window_size = tuple(window_size)
        self.num_tokens = self.window_size[0] * self.window_size[1] + 1
        self.num_rel_distance = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        ) + 3
        self.block_prefix = block_prefix

        self.q_proj = layers.Dense(
            hidden_size, use_bias=True, name=f"{block_prefix}_query"
        )
        self.k_proj = layers.Dense(
            hidden_size, use_bias=False, name=f"{block_prefix}_key"
        )
        self.v_proj = layers.Dense(
            hidden_size, use_bias=True, name=f"{block_prefix}_value"
        )
        self.o_proj = layers.Dense(
            hidden_size, use_bias=True, name=f"{block_prefix}_output_dense"
        )
        self.rel_index = beit_relative_position_index(self.window_size).reshape(-1)

    def build(self, input_shape):
        self.q_proj.build(input_shape)
        self.k_proj.build(input_shape)
        self.v_proj.build(input_shape)
        self.o_proj.build(input_shape)
        self.relative_position_bias_table = self.add_weight(
            shape=(self.num_rel_distance, self.num_heads),
            initializer="zeros",
            trainable=True,
            name="relative_position_bias_table",
        )
        self.built = True

    def call(self, x):
        batch_size = ops.shape(x)[0]
        n = self.num_tokens
        shape = (batch_size, n, self.num_heads, self.head_dim)
        q = ops.transpose(ops.reshape(self.q_proj(x), shape), (0, 2, 1, 3))
        k = ops.transpose(ops.reshape(self.k_proj(x), shape), (0, 2, 1, 3))
        v = ops.transpose(ops.reshape(self.v_proj(x), shape), (0, 2, 1, 3))

        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale

        bias = ops.take(self.relative_position_bias_table, self.rel_index, axis=0)
        bias = ops.reshape(bias, (n, n, self.num_heads))
        bias = ops.transpose(bias, (2, 0, 1))  # (num_heads, n, n)
        scores = scores + ops.expand_dims(bias, 0)

        attn = ops.softmax(scores, axis=-1)
        out = ops.matmul(attn, v)  # (batch, num_heads, n, head_dim)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (batch_size, n, self.hidden_size)
        )
        return self.o_proj(out)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "block_prefix": self.block_prefix,
            }
        )
        return config
