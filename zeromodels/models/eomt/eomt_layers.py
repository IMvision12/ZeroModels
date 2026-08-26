import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


@keras.saving.register_keras_serializable(package="zeromodels")
class EoMTLayerScale(layers.Layer):
    """Learnable per-channel scaling factor for residual connections.

    Applies an element-wise multiplication by a trainable vector of
    the same dimension as the input channels. Initialised to a
    constant value so that early in training the residual branch
    contributes a controlled amount to the output. Used in every
    encoder layer of EoMT.

    Reference:
    - [Your ViT is Secretly an Image Segmentation Model](https://arxiv.org/abs/2503.19108)

    Args:
        init_value: Float, initial constant value for every element
            of the scale vector. Defaults to `1.0`.
        **kwargs: Additional keyword arguments passed to the `Layer`
            class.

    Input Shape:
        Arbitrary tensor whose last dimension is the channel
        dimension.

    Output Shape:
        Same as input shape.
    """

    def __init__(self, init_value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="lambda1",
            shape=(input_shape[-1],),
            initializer=keras.initializers.Constant(self.init_value),
            trainable=True,
        )

    def call(self, inputs):
        return inputs * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update({"init_value": self.init_value})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class EoMTPatchEmbeddings(layers.Layer):
    """Converts pixel values to patch embeddings via a Conv2D projection.

    Splits the input image into non-overlapping patches of size
    `patch_size x patch_size` using a strided convolution and projects
    each patch into a `hidden_dim`-dimensional embedding. The spatial
    grid is then flattened into a 1D token sequence.

    Reference:
    - [Your ViT is Secretly an Image Segmentation Model](https://arxiv.org/abs/2503.19108)

    Args:
        hidden_dim: Integer, output embedding dimension for each
            patch token.
        patch_size: Integer, height and width of each image patch.
            Defaults to `16`.
        num_channels: Integer, number of input image channels.
            Defaults to `3`.
        **kwargs: Additional keyword arguments passed to the `Layer`
            class.

    Input Shape:
        4D tensor: `(batch_size, height, width, num_channels)`.

    Output Shape:
        3D tensor: `(batch_size, num_patches, hidden_dim)` where
        `num_patches = (height // patch_size) * (width // patch_size)`.
    """

    def __init__(self, hidden_dim, patch_size=16, num_channels=3, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.data_format = keras.config.image_data_format()
        self.projection = layers.Conv2D(
            hidden_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            use_bias=True,
            data_format=self.data_format,
            name="projection",
        )

    def call(self, pixel_values):
        x = self.projection(pixel_values)
        shape = ops.shape(x)
        if self.data_format == "channels_first":
            # (B, C, H, W) -> (B, H*W, C)
            x = ops.transpose(x, (0, 2, 3, 1))
            x = ops.reshape(x, (shape[0], shape[2] * shape[3], self.hidden_dim))
        else:
            x = ops.reshape(x, (shape[0], shape[1] * shape[2], shape[3]))
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "patch_size": self.patch_size,
                "num_channels": self.num_channels,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class EoMTEmbeddings(layers.Layer):
    """Constructs the full input embedding for the EoMT encoder.

    Combines patch embeddings with learnable positional embeddings,
    then prepends a CLS token and a set of register tokens following
    the DINOv2 design. The output sequence has length
    `1 + num_register_tokens + num_patches`.

    Reference:
    - [Your ViT is Secretly an Image Segmentation Model](https://arxiv.org/abs/2503.19108)
    - [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)

    Args:
        hidden_dim: Integer, embedding dimension for all tokens.
        patch_size: Integer, height and width of each image patch.
            Defaults to `16`.
        image_size: Integer, spatial resolution of the input image
            (assumed square). Defaults to `640`.
        num_register_tokens: Integer, number of learnable register
            tokens prepended after the CLS token. Defaults to `4`.
        num_channels: Integer, number of input image channels.
            Defaults to `3`.
        **kwargs: Additional keyword arguments passed to the `Layer`
            class.

    Input Shape:
        4D tensor: `(batch_size, image_size, image_size, num_channels)`.

    Output Shape:
        3D tensor:
        `(batch_size, 1 + num_register_tokens + num_patches, hidden_dim)`.
    """

    def __init__(
        self,
        hidden_dim,
        patch_size=16,
        image_size=640,
        num_register_tokens=4,
        num_channels=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_register_tokens = num_register_tokens
        self.num_channels = num_channels
        self.num_patches = (image_size // patch_size) ** 2
        self.num_prefix_tokens = 1 + num_register_tokens  # CLS + register tokens

        self.patch_embeddings = EoMTPatchEmbeddings(
            hidden_dim, patch_size, num_channels, name="patch_embeddings"
        )

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.hidden_dim),
            initializer="random_normal",
            trainable=True,
        )
        self.register_tokens = self.add_weight(
            name="register_tokens",
            shape=(1, self.num_register_tokens, self.hidden_dim),
            initializer="zeros",
            trainable=True,
        )
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(1, self.num_patches, self.hidden_dim),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, pixel_values):
        batch_size = ops.shape(pixel_values)[0]

        embeddings = self.patch_embeddings(pixel_values)
        embeddings = embeddings + self.position_embeddings

        cls_tokens = ops.broadcast_to(self.cls_token, (batch_size, 1, self.hidden_dim))
        register_tokens = ops.broadcast_to(
            self.register_tokens,
            (batch_size, self.num_register_tokens, self.hidden_dim),
        )

        embeddings = ops.concatenate([cls_tokens, register_tokens, embeddings], axis=1)
        return embeddings

    def save_own_variables(self, store):
        super().save_own_variables(store)
        grid_size = self.image_size // self.patch_size
        store["grid_h"] = grid_size
        store["grid_w"] = grid_size

    def load_own_variables(self, store):
        source_h = int(store["grid_h"][...])
        source_w = int(store["grid_w"][...])
        grid_size = self.image_size // self.patch_size

        target_vars = self._trainable_variables + self._non_trainable_variables
        pos_idx = None
        for i, var in enumerate(target_vars):
            if var is self.position_embeddings:
                pos_idx = i
            else:
                var.assign(store[str(i)])

        if source_h == grid_size and source_w == grid_size:
            self.position_embeddings.assign(store[str(pos_idx)])
            return

        pos_embed = store[str(pos_idx)]
        embed_dim = pos_embed.shape[-1]

        pos_embed = ops.cast(pos_embed, dtype="float32")
        pos_embed = ops.reshape(pos_embed, (1, source_h, source_w, embed_dim))
        pos_embed = ops.image.resize(
            pos_embed,
            size=(grid_size, grid_size),
            interpolation="bilinear",
            antialias=True,
        )
        pos_embed = ops.reshape(pos_embed, (1, grid_size * grid_size, embed_dim))
        self.position_embeddings.assign(pos_embed)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "patch_size": self.patch_size,
                "image_size": self.image_size,
                "num_register_tokens": self.num_register_tokens,
                "num_channels": self.num_channels,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class EoMTAttention(layers.Layer):
    """Multi-head self-attention for the EoMT transformer encoder.

    Implements scaled dot-product multi-head self-attention with
    separate query, key, and value projections followed by an output
    projection. Each head operates on a `hidden_dim // num_heads`
    dimensional subspace.

    Reference:
    - [Your ViT is Secretly an Image Segmentation Model](https://arxiv.org/abs/2503.19108)

    Args:
        hidden_dim: Integer, total model dimension. Must be divisible
            by `num_heads`.
        num_heads: Integer, number of parallel attention heads.
        attention_dropout: Float, dropout rate applied to the
            attention weight matrix during training. Defaults to
            `0.0`.
        **kwargs: Additional keyword arguments passed to the `Layer`
            class.

    Input Shape:
        3D tensor: `(batch_size, seq_len, hidden_dim)`.

    Output Shape:
        3D tensor: `(batch_size, seq_len, hidden_dim)`.
    """

    def __init__(self, hidden_dim, num_heads, attention_dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.attention_dropout = attention_dropout
        self.attn_drop = layers.Dropout(attention_dropout)

        self.q_proj = layers.Dense(hidden_dim, name="q_proj")
        self.k_proj = layers.Dense(hidden_dim, name="k_proj")
        self.v_proj = layers.Dense(hidden_dim, name="v_proj")
        self.out_proj = layers.Dense(hidden_dim, name="out_proj")

    def call(self, hidden_states, training=False):
        batch_size = ops.shape(hidden_states)[0]
        seq_length = ops.shape(hidden_states)[1]

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = ops.reshape(
            queries, (batch_size, seq_length, self.num_heads, self.head_dim)
        )
        queries = ops.transpose(queries, (0, 2, 1, 3))

        keys = ops.reshape(
            keys, (batch_size, seq_length, self.num_heads, self.head_dim)
        )
        keys = ops.transpose(keys, (0, 2, 1, 3))

        values = ops.reshape(
            values, (batch_size, seq_length, self.num_heads, self.head_dim)
        )
        values = ops.transpose(values, (0, 2, 1, 3))

        attn_output = fused_attention(
            queries,
            keys,
            values,
            self.scale,
            dropout=self.attn_drop,
            training=training,
        )
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(
            attn_output, (batch_size, seq_length, self.hidden_dim)
        )
        attn_output = self.out_proj(attn_output)

        return attn_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "attention_dropout": self.attention_dropout,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class EoMTQueryInjection(layers.Layer):
    """Injects learned query tokens into the encoder sequence.

    Maintains a learnable weight matrix of shape
    `(num_queries, hidden_dim)` and prepends it (broadcast across
    the batch) to the existing hidden states. This enables joint
    attention between object queries and image patch tokens in the
    final encoder blocks, which is the core mechanism of EoMT.

    Reference:
    - [Your ViT is Secretly an Image Segmentation Model](https://arxiv.org/abs/2503.19108)

    Args:
        num_queries: Integer, number of learnable object query tokens.
        hidden_dim: Integer, embedding dimension matching the
            encoder hidden size.
        **kwargs: Additional keyword arguments passed to the `Layer`
            class.

    Input Shape:
        3D tensor: `(batch_size, seq_len, hidden_dim)`.

    Output Shape:
        3D tensor:
        `(batch_size, num_queries + seq_len, hidden_dim)`.
    """

    def __init__(self, num_queries, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.query_weight = self.add_weight(
            name="weight",
            shape=(self.num_queries, self.hidden_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, hidden_states):
        batch_size = ops.shape(hidden_states)[0]
        query_tokens = ops.broadcast_to(
            ops.expand_dims(self.query_weight, axis=0),
            (batch_size, self.num_queries, self.hidden_dim),
        )
        return ops.concatenate([query_tokens, hidden_states], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_queries": self.num_queries,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class EoMTStochasticDepth(keras.layers.Layer):
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
