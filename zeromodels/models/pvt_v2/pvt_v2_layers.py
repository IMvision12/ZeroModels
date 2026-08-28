import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


def to_grid(x, height, width, channels, data_format):
    """(B, H*W, C) tokens -> spatial grid in ``data_format`` layout."""
    x = ops.reshape(x, (ops.shape(x)[0], height, width, channels))
    if data_format == "channels_first":
        x = ops.transpose(x, (0, 3, 1, 2))
    return x


def to_tokens(x, channels, data_format):
    """Spatial grid -> (B, H*W, C) tokens."""
    if data_format == "channels_first":
        x = ops.transpose(x, (0, 2, 3, 1))
    return ops.reshape(x, (ops.shape(x)[0], -1, channels))


def adaptive_pool_matrix(in_size, out_size):
    """Row-averaging matrix ``(out_size, in_size)`` matching torch AdaptiveAvgPool: output
    bin ``o`` averages input indices ``[floor(o*in/out), ceil((o+1)*in/out))``."""
    rows = []
    for o in range(out_size):
        start = (o * in_size) // out_size
        end = -(-(o + 1) * in_size // out_size)  # ceil
        weight = 1.0 / (end - start)
        rows.append([weight if start <= i < end else 0.0 for i in range(in_size)])
    return ops.convert_to_tensor(rows, dtype="float32")


@keras.saving.register_keras_serializable(package="zeromodels")
class PvtV2SelfAttention(layers.Layer):
    """PVTv2 spatial-reduction attention.

    Query is projected from the full token sequence; keys/values come from a reduced
    sequence. Standard SRA (``sr_ratio > 1``) reduces via a strided ``Conv2d`` + LayerNorm;
    the linear variant pools the grid to a fixed 7x7 (adaptive average pool), applies a 1x1
    conv + LayerNorm + GELU, so the key/value length is resolution-independent.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        sr_ratio,
        linear_attention=False,
        qkv_bias=True,
        layer_norm_eps=1e-6,
        block_prefix="block",
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.sr_ratio = sr_ratio
        self.linear_attention = linear_attention
        self.qkv_bias = qkv_bias
        self.layer_norm_eps = layer_norm_eps
        self.block_prefix = block_prefix
        self.data_format = keras.config.image_data_format()

        self.query = layers.Dense(
            hidden_size, use_bias=qkv_bias, name=f"{block_prefix}_attn_query"
        )
        self.key = layers.Dense(
            hidden_size, use_bias=qkv_bias, name=f"{block_prefix}_attn_key"
        )
        self.value = layers.Dense(
            hidden_size, use_bias=qkv_bias, name=f"{block_prefix}_attn_value"
        )
        self.proj = layers.Dense(hidden_size, name=f"{block_prefix}_attn_proj")

        if linear_attention:
            self.sr = layers.Conv2D(
                hidden_size,
                1,
                strides=1,
                data_format=self.data_format,
                name=f"{block_prefix}_attn_sr",
            )
            self.norm = layers.LayerNormalization(
                axis=-1, epsilon=layer_norm_eps, name=f"{block_prefix}_attn_norm"
            )
        elif sr_ratio > 1:
            self.sr = layers.Conv2D(
                hidden_size,
                sr_ratio,
                strides=sr_ratio,
                padding="valid",
                data_format=self.data_format,
                name=f"{block_prefix}_attn_sr",
            )
            self.norm = layers.LayerNormalization(
                axis=-1, epsilon=layer_norm_eps, name=f"{block_prefix}_attn_norm"
            )

    def split_heads(self, x):
        b = ops.shape(x)[0]
        x = ops.reshape(x, (b, -1, self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def reduce(self, x, height, width):
        grid = to_grid(x, height, width, self.hidden_size, self.data_format)
        if self.linear_attention:
            grid_cl = (
                ops.transpose(grid, (0, 2, 3, 1))
                if self.data_format == "channels_first"
                else grid
            )
            mh = adaptive_pool_matrix(height, 7)
            mw = adaptive_pool_matrix(width, 7)
            grid_cl = ops.einsum("oh,bhwc->bowc", mh, grid_cl)
            grid_cl = ops.einsum("pw,bowc->bopc", mw, grid_cl)
            grid = (
                ops.transpose(grid_cl, (0, 3, 1, 2))
                if self.data_format == "channels_first"
                else grid_cl
            )
            x = to_tokens(self.sr(grid), self.hidden_size, self.data_format)
            x = ops.gelu(self.norm(x))
        else:
            x = to_tokens(self.sr(grid), self.hidden_size, self.data_format)
            x = self.norm(x)
        return x

    def call(self, x, height, width, training=None):
        q = self.split_heads(self.query(x))
        if self.linear_attention or self.sr_ratio > 1:
            kv_in = self.reduce(x, height, width)
        else:
            kv_in = x
        k = self.split_heads(self.key(kv_in))
        v = self.split_heads(self.value(kv_in))

        out = fused_attention(q, k, v, self.scale, training=training)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (ops.shape(x)[0], ops.shape(x)[1], self.hidden_size))
        return self.proj(out)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.hidden_size)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "sr_ratio": self.sr_ratio,
                "linear_attention": self.linear_attention,
                "qkv_bias": self.qkv_bias,
                "layer_norm_eps": self.layer_norm_eps,
                "block_prefix": self.block_prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class PvtDropPath(layers.Layer):
    """Stochastic depth: drops the residual branch per-sample during training only."""

    def __init__(self, drop_prob, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, x, training=None):
        if training and self.drop_prob > 0:
            keep = 1 - self.drop_prob
            shape = (ops.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            mask = ops.floor(
                keep + keras.random.uniform(shape, 0, 1, seed=self.seed_generator)
            )
            return (x / keep) * mask
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_prob": self.drop_prob, "seed": self.seed})
        return config
