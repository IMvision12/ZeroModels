import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


@keras.saving.register_keras_serializable(package="zeromodels")
class PvtSelfAttention(layers.Layer):
    """PVT spatial-reduction attention. Query is projected from the full token sequence;
    keys/values come from a sequence reduced by a strided conv + LayerNorm (skipped when
    ``sr_ratio == 1``). ``proj`` is the attention-output Dense."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        sr_ratio,
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
        if sr_ratio > 1:
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
        return ops.transpose(
            ops.reshape(x, (b, -1, self.num_heads, self.head_dim)), (0, 2, 1, 3)
        )

    def call(self, x, height, width, training=None):
        q = self.split_heads(self.query(x))
        if self.sr_ratio > 1:
            b = ops.shape(x)[0]
            grid = ops.reshape(x, (b, height, width, self.hidden_size))
            if self.data_format == "channels_first":
                grid = ops.transpose(grid, (0, 3, 1, 2))
            grid = self.sr(grid)
            if self.data_format == "channels_first":
                grid = ops.transpose(grid, (0, 2, 3, 1))
            kv_in = self.norm(ops.reshape(grid, (b, -1, self.hidden_size)))
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
                "qkv_bias": self.qkv_bias,
                "layer_norm_eps": self.layer_norm_eps,
                "block_prefix": self.block_prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class PvtClsToken(layers.Layer):
    """Prepends a learnable class token (last stage only)."""

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, input_shape[-1]),
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def call(self, x):
        cls = ops.broadcast_to(self.cls_token, (ops.shape(x)[0], 1, ops.shape(x)[-1]))
        return ops.concatenate([cls, x], axis=1)

    def compute_output_shape(self, input_shape):
        n = None if input_shape[1] is None else input_shape[1] + 1
        return (input_shape[0], n, input_shape[2])


@keras.saving.register_keras_serializable(package="zeromodels")
class PvtStagePositions(layers.Layer):
    """Adds a per-stage learned position embedding sized to ``grid_h x grid_w`` (plus a
    class-token slot when ``has_cls``). On weight load from a different grid, the spatial
    part is bilinearly interpolated to this model's grid (the class slot is kept)."""

    def __init__(
        self, grid_h, grid_w, has_cls=False, resize_mode="bilinear", name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.has_cls = has_cls
        self.resize_mode = resize_mode

    def build(self, input_shape):
        n = self.grid_h * self.grid_w + (1 if self.has_cls else 0)
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, n, input_shape[-1]),
            initializer="random_normal",
            trainable=True,
        )
        self.built = True

    def call(self, x):
        return x + self.pos_embed

    def compute_output_shape(self, input_shape):
        return input_shape

    def save_own_variables(self, store):
        super().save_own_variables(store)
        store["grid_h"] = self.grid_h
        store["grid_w"] = self.grid_w

    def load_own_variables(self, store):
        source_h, source_w = int(store["grid_h"][...]), int(store["grid_w"][...])
        if source_h == self.grid_h and source_w == self.grid_w:
            self.pos_embed.assign(store["0"])
            return
        pe = store["0"]
        cls_pe, spatial = (pe[:, :1], pe[:, 1:]) if self.has_cls else (None, pe)
        c = spatial.shape[-1]
        spatial = ops.reshape(ops.cast(spatial, "float32"), (1, source_h, source_w, c))
        spatial = ops.image.resize(
            spatial,
            (self.grid_h, self.grid_w),
            interpolation=self.resize_mode,
            antialias=True,
        )
        spatial = ops.reshape(spatial, (1, self.grid_h * self.grid_w, c))
        pe = ops.concatenate([cls_pe, spatial], axis=1) if self.has_cls else spatial
        self.pos_embed.assign(pe)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "grid_h": self.grid_h,
                "grid_w": self.grid_w,
                "has_cls": self.has_cls,
                "resize_mode": self.resize_mode,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class PvtDropPath(layers.Layer):
    """Stochastic depth (identity at inference)."""

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
