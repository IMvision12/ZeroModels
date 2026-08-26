import keras
from keras import InputSpec, layers, ops

from zeromodels.base.base_attention import fused_attention


@keras.saving.register_keras_serializable(package="zeromodels")
class MiTEfficientMultiheadSelfAttention(layers.Layer):
    """Efficient Multi-head Self-Attention layer with hierarchical spatial reduction.

    This layer implements an efficient self-attention mechanism that uses convolutional
    downsampling of keys and values for reduced computation. The spatial reduction ratio
    can be adjusted across different stages of a network to create hierarchical feature
    representations, making it particularly suitable for vision tasks.

    Key Features:
        - Spatial reduction using Conv2D to reduce key/value sequence length
        - Multi-head attention for learning different feature relationships
        - Adaptive to different input resolutions
        - Computationally efficient for dense vision tasks

    Args:
        project_dim (int): Output dimension of the projection. Can be scaled across
            network stages for hierarchical feature learning
        sr_ratio (int): Spatial reduction ratio for keys and values. Lower values
            preserve more spatial detail while higher values improve efficiency
        block_prefix (str, optional): Prefix for naming layer components. Defaults to
            "segformer.encoder.block.0.0"
        qkv_bias (bool, optional): If True, adds learnable bias terms to the query, key,
            and value projections. Defaults to False
        num_heads (int, optional): Number of attention heads. Defaults to 8
        attn_drop (float, optional): Dropout rate applied to attention weights. Helps
            prevent overfitting. Defaults to 0.1
        proj_drop (float, optional): Dropout rate applied to the output projection.
            Provides additional regularization. Defaults to 0.1
        epsilon (float, optional): Small constant used in normalization operations for
            numerical stability. A higher value reduces precision but increases stability.
            Defaults to 1e-6
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Input shape:
        - x: (batch_size, H*W, channels) - Sequence of flattened image features

    Output shape:
        - (batch_size, H*W, project_dim)

    Notes:
        - project_dim must be divisible by num_heads
        - The spatial reduction (sr_ratio > 1) helps balance efficiency and performance
        - The Conv2D spatial reduction also helps capture local information
        - Suitable for hierarchical vision transformer architectures where different
          stages require different levels of spatial detail
    """

    def __init__(
        self,
        project_dim,
        sr_ratio,
        block_prefix=None,
        qkv_bias=False,
        num_heads=8,
        attn_drop=0.1,
        proj_drop=0.1,
        epsilon=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert project_dim % num_heads == 0, (
            f"project_dim {project_dim} should be divided by num_heads {num_heads}."
        )

        self.project_dim = project_dim
        self.num_heads = num_heads
        self.scale = (project_dim // num_heads) ** -0.5
        self.sr_ratio = sr_ratio
        self.block_prefix = block_prefix if block_prefix is not None else "block"
        self.epsilon = epsilon
        self.data_format = keras.config.image_data_format()
        self.channels_axis = -1 if self.data_format == "channels_last" else 1

        self.q = layers.Dense(
            project_dim,
            use_bias=qkv_bias,
            dtype=self.dtype_policy,
            name=f"{self.block_prefix}_attn_q",
        )
        self.k = layers.Dense(
            project_dim,
            use_bias=qkv_bias,
            dtype=self.dtype_policy,
            name=f"{self.block_prefix}_attn_k",
        )
        self.v = layers.Dense(
            project_dim,
            use_bias=qkv_bias,
            dtype=self.dtype_policy,
            name=f"{self.block_prefix}_attn_v",
        )
        self.attn_drop = layers.Dropout(attn_drop, dtype=self.dtype_policy)
        self.proj = layers.Dense(
            project_dim,
            use_bias=qkv_bias,
            dtype=self.dtype_policy,
            name=f"{self.block_prefix}_attn_proj",
        )
        self.proj_drop = layers.Dropout(proj_drop, dtype=self.dtype_policy)

        if sr_ratio > 1:
            self.sr = layers.Conv2D(
                filters=project_dim,
                kernel_size=sr_ratio,
                strides=sr_ratio,
                padding="same",
                data_format=self.data_format,
                dtype=self.dtype_policy,
                name=f"{self.block_prefix}_attn_sr",
            )
            self.norm = layers.LayerNormalization(
                axis=-1,
                epsilon=self.epsilon,
                dtype=self.dtype_policy,
                name=f"{self.block_prefix}_attn_norm",
            )

    def build(self, input_shape):
        self.input_spec = InputSpec(ndim=len(input_shape))

        if self.input_spec.ndim != 3:
            raise ValueError(
                f"MiTEfficientMultiheadSelfAttention expects 3D input tensor, but received shape: {input_shape}"
            )

        feature_dim = input_shape[-1]
        if feature_dim != self.project_dim:
            raise ValueError(
                f"Input feature dimension {feature_dim} must match layer dimension {self.project_dim}"
            )

        batch_dim = input_shape[0]
        seq_length = input_shape[1]

        self.q.build(input_shape)
        self.k.build(input_shape)
        self.v.build(input_shape)
        self.proj.build(input_shape)
        self.proj_drop.build(input_shape)

        if self.sr_ratio > 1:
            h = int(ops.cast(ops.sqrt(ops.cast(seq_length, "float32")), "int32"))
            if self.data_format == "channels_first":
                spatial_shape = (batch_dim, feature_dim, h, h)
            else:
                spatial_shape = (batch_dim, h, h, feature_dim)

            self.sr.build(spatial_shape)

            reduced_seq_length = (h // self.sr_ratio) * (h // self.sr_ratio)
            norm_shape = (batch_dim, reduced_seq_length, self.project_dim)
            self.norm.build(norm_shape)

        self._attention_head_size = self.project_dim // self.num_heads
        self._num_attention_heads = self.num_heads

        self.built = True

    def compute_output_spec(self, input_spec):
        input_shape = ops.shape(input_spec)
        return keras.KerasTensor(
            shape=(input_shape[0], input_shape[1], self.project_dim),
            dtype=self.compute_dtype,
        )

    def call(self, x, training=None):
        input_shape = ops.shape(x)
        B, N, C = input_shape[0], input_shape[1], input_shape[2]
        H = W = ops.cast(ops.sqrt(ops.cast(N, "float32")), "int32")

        q = self.q(x)
        q = ops.reshape(q, (B, N, self.num_heads, C // self.num_heads))
        q = ops.transpose(q, (0, 2, 1, 3))

        if self.sr_ratio > 1:
            x_ = ops.reshape(ops.transpose(x, (0, 2, 1)), (B, C, H, W))
            if self.data_format == "channels_last":
                x_ = ops.transpose(x_, (0, 2, 3, 1))  # BCHW -> BHWC
            x_ = self.sr(x_)
            if self.data_format == "channels_last":
                x_ = ops.transpose(x_, (0, 3, 1, 2))  # BHWC -> BCHW
            x_ = ops.reshape(x_, (B, C, -1))
            x_ = ops.transpose(x_, (0, 2, 1))
            x_ = self.norm(x_)
            k = self.k(x_)
            v = self.v(x_)
        else:
            k = self.k(x)
            v = self.v(x)

        k = ops.reshape(k, (B, -1, self.num_heads, C // self.num_heads))
        v = ops.reshape(v, (B, -1, self.num_heads, C // self.num_heads))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        x = fused_attention(
            q, k, v, self.scale, dropout=self.attn_drop, training=training
        )
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (B, N, C))

        x = self.proj(x)
        x = self.proj_drop(x, training=training)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "project_dim": self.project_dim,
                "sr_ratio": self.sr_ratio,
                "block_prefix": self.block_prefix,
                "qkv_bias": self.q.use_bias,
                "num_heads": self.num_heads,
                "proj_drop": self.proj_drop.rate,
                "attn_drop": self.attn_drop.rate,
                "epsilon": self.epsilon,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MiTStochasticDepth(keras.layers.Layer):
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
