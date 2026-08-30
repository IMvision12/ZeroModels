import itertools

import keras
import numpy as np
from keras import layers, ops

# torch BatchNorm default eps; only eps matters at inference (running stats).
BN_EPS = 1e-5


def levit_attention_bias_index(resolution):
    """(N, N) offset index + number of unique offsets for an RxR token grid."""
    points = list(itertools.product(range(resolution), range(resolution)))
    offsets, indices = {}, []
    for p1 in points:
        for p2 in points:
            offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
            if offset not in offsets:
                offsets[offset] = len(offsets)
            indices.append(offsets[offset])
    n = len(points)
    return np.array(indices, dtype="int64").reshape(n, n), len(offsets)


def levit_attention_subsample_bias_index(resolution_in, resolution_out, stride):
    """(N_out, N_in) offset index + unique-offset count for a strided subsample."""
    points = list(itertools.product(range(resolution_in), range(resolution_in)))
    points_ = list(itertools.product(range(resolution_out), range(resolution_out)))
    offsets, indices = {}, []
    for p1 in points_:
        for p2 in points:
            size = 1
            offset = (
                abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                abs(p1[1] * stride - p2[1] + (size - 1) / 2),
            )
            if offset not in offsets:
                offsets[offset] = len(offsets)
            indices.append(offsets[offset])
    return (
        np.array(indices, dtype="int64").reshape(len(points_), len(points)),
        len(offsets),
    )


def levit_subsample(x, stride, resolution):
    batch = ops.shape(x)[0]
    channels = x.shape[-1]
    x = ops.reshape(x, (batch, resolution, resolution, channels))
    x = x[:, ::stride, ::stride, :]
    return ops.reshape(x, (batch, -1, channels))


@keras.saving.register_keras_serializable(package="zeromodels")
class MLPLayerWithBN(layers.Layer):
    """A bias-free Dense fused with a BatchNorm, the LeViT linear primitive.

    torch applies ``BatchNorm1d`` to the flattened ``(B*seq, C)`` tensor; at
    inference that equals a per-channel ``BatchNormalization(axis=-1)`` over the
    last axis of ``(B, seq, C)``, so this uses the standard Keras layer.
    """

    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.linear = layers.Dense(output_dim, use_bias=False, name="linear")
        self.batch_norm = layers.BatchNormalization(
            axis=-1, epsilon=BN_EPS, momentum=0.9, name="batch_norm"
        )

    def build(self, input_shape):
        self.linear.build(input_shape)
        self.batch_norm.build((*input_shape[:-1], self.output_dim))
        self.built = True

    def call(self, x):
        return self.batch_norm(self.linear(x))

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.output_dim)

    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class LevitMLP(layers.Layer):
    """LeViT MLP: MLPLayerWithBN up, Hardswish, MLPLayerWithBN down (2x expansion)."""

    def __init__(self, hidden_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.linear_up = MLPLayerWithBN(hidden_dim, name="linear_up")
        self.linear_down = MLPLayerWithBN(output_dim, name="linear_down")

    def build(self, input_shape):
        self.linear_up.build(input_shape)
        self.linear_down.build((*input_shape[:-1], self.hidden_dim))
        self.built = True

    def call(self, x):
        return self.linear_down(ops.hard_silu(self.linear_up(x)))

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.output_dim)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim, "output_dim": self.output_dim})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class LevitAttention(layers.Layer):
    """LeViT attention with a learnable 2D relative-position bias.

    A single ``MLPLayerWithBN`` produces per-head query, key and value (the value
    is ``attention_ratio`` times wider); the bias is gathered from a learnable
    ``(num_heads, num_offsets)`` table via a fixed offset index and added to the
    scores before the softmax. The output is Hardswish-projected back to
    ``hidden_size``.
    """

    def __init__(
        self, hidden_size, key_dim, num_heads, attention_ratio, resolution, **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.attention_ratio = attention_ratio
        self.resolution = resolution
        self.scale = key_dim**-0.5
        self.out_dim_keys_values = (
            attention_ratio * key_dim * num_heads + key_dim * num_heads * 2
        )
        self.out_dim_projection = attention_ratio * key_dim * num_heads

        self.queries_keys_values = MLPLayerWithBN(
            self.out_dim_keys_values, name="queries_keys_values"
        )
        self.projection = MLPLayerWithBN(hidden_size, name="projection")
        index, self.num_offsets = levit_attention_bias_index(resolution)
        self.bias_index = index

    def build(self, input_shape):
        self.queries_keys_values.build(input_shape)
        self.projection.build((*input_shape[:-1], self.out_dim_projection))
        self.attention_biases = self.add_weight(
            shape=(self.num_heads, self.num_offsets),
            initializer="zeros",
            trainable=True,
            name="attention_biases",
        )
        self.built = True

    def call(self, x):
        batch = ops.shape(x)[0]
        per_head = self.out_dim_keys_values // self.num_heads
        qkv = ops.reshape(
            self.queries_keys_values(x), (batch, -1, self.num_heads, per_head)
        )
        query = ops.transpose(qkv[..., : self.key_dim], (0, 2, 1, 3))
        key = ops.transpose(qkv[..., self.key_dim : 2 * self.key_dim], (0, 2, 1, 3))
        value = ops.transpose(qkv[..., 2 * self.key_dim :], (0, 2, 1, 3))

        bias = ops.take(self.attention_biases, self.bias_index, axis=1)
        attention = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2)))
        attention = ops.softmax(attention * self.scale + bias, axis=-1)
        out = ops.matmul(attention, value)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (batch, -1, self.out_dim_projection)
        )
        return self.projection(ops.hard_silu(out))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "key_dim": self.key_dim,
                "num_heads": self.num_heads,
                "attention_ratio": self.attention_ratio,
                "resolution": self.resolution,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class LevitAttentionSubsample(layers.Layer):
    """Downsampling attention: strided queries attend to the full input grid.

    Keys/values come from every input token; queries are read from a strided
    subsample, so the output has ``resolution_out**2`` tokens at ``output_dim``.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        key_dim,
        num_heads,
        attention_ratio,
        stride,
        resolution_in,
        resolution_out,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.attention_ratio = attention_ratio
        self.stride = stride
        self.resolution_in = resolution_in
        self.resolution_out = resolution_out
        self.scale = key_dim**-0.5
        self.out_dim_keys_values = (
            attention_ratio * key_dim * num_heads + key_dim * num_heads
        )
        self.out_dim_projection = attention_ratio * key_dim * num_heads

        self.keys_values = MLPLayerWithBN(self.out_dim_keys_values, name="keys_values")
        self.queries = MLPLayerWithBN(key_dim * num_heads, name="queries")
        self.projection = MLPLayerWithBN(output_dim, name="projection")
        index, self.num_offsets = levit_attention_subsample_bias_index(
            resolution_in, resolution_out, stride
        )
        self.bias_index = index

    def build(self, input_shape):
        self.keys_values.build(input_shape)
        self.queries.build((*input_shape[:-1], self.input_dim))
        self.projection.build((*input_shape[:-1], self.out_dim_projection))
        self.attention_biases = self.add_weight(
            shape=(self.num_heads, self.num_offsets),
            initializer="zeros",
            trainable=True,
            name="attention_biases",
        )
        self.built = True

    def call(self, x):
        batch = ops.shape(x)[0]
        per_head = self.out_dim_keys_values // self.num_heads
        kv = ops.reshape(self.keys_values(x), (batch, -1, self.num_heads, per_head))
        key = ops.transpose(kv[..., : self.key_dim], (0, 2, 1, 3))
        value = ops.transpose(kv[..., self.key_dim :], (0, 2, 1, 3))

        query = self.queries(levit_subsample(x, self.stride, self.resolution_in))
        query = ops.reshape(
            query, (batch, self.resolution_out**2, self.num_heads, self.key_dim)
        )
        query = ops.transpose(query, (0, 2, 1, 3))

        bias = ops.take(self.attention_biases, self.bias_index, axis=1)
        attention = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2)))
        attention = ops.softmax(attention * self.scale + bias, axis=-1)
        out = ops.matmul(attention, value)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (batch, -1, self.out_dim_projection)
        )
        return self.projection(ops.hard_silu(out))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.resolution_out**2, self.output_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "key_dim": self.key_dim,
                "num_heads": self.num_heads,
                "attention_ratio": self.attention_ratio,
                "stride": self.stride,
                "resolution_in": self.resolution_in,
                "resolution_out": self.resolution_out,
            }
        )
        return config
