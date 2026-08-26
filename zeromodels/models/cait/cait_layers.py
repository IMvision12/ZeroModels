import keras
from keras import InputSpec, layers, ops

from zeromodels.base.base_attention import fused_attention


@keras.saving.register_keras_serializable(package="zeromodels")
class CaiTClassDistToken(layers.Layer):
    """
    Implements learnable class and distillation tokens for Vision Transformer (ViT),
    Data-efficient image Transformer (DeiT), and Pyramid Vision Transformer (PiT) architectures.

    This layer can operate in three modes:
    1. Standard ViT mode: Only adds a class token
    2. DeiT mode: Adds separate class and distillation tokens
    3. PiT mode: Adds combined class and distillation tokens in a single weight tensor

    Args:
        use_distillation (bool): If True, adds distillation token(s) alongside class token(s).
            Defaults to False.
        combine_tokens (bool): If True, stores class and distillation tokens in a single weight
            tensor (PiT style). If False, uses separate weight tensors (ViT/DeiT style).
            Only applies when use_distillation=True. Defaults to False.
        name (str, optional): Name for the layer instance.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    References:
        - ViT: https://arxiv.org/abs/2010.11929
        - DeiT: https://arxiv.org/abs/2012.12877
        - PiT: https://arxiv.org/abs/2103.14030
    """

    def __init__(
        self, use_distillation=False, combine_tokens=False, name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.use_distillation = use_distillation
        self.combine_tokens = combine_tokens
        self.num_tokens = 2 if use_distillation else 1

    def build(self, input_shape):
        self.hidden_dim = input_shape[-1]

        if self.combine_tokens and self.use_distillation:
            self.tokens = self.add_weight(
                name="cls_token",
                shape=(1, 2, self.hidden_dim),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.cls = self.add_weight(
                name="cls_token",
                shape=(1, 1, self.hidden_dim),
                initializer="zeros",
                trainable=True,
            )
            if self.use_distillation:
                self.dist = self.add_weight(
                    name="dist_token",
                    shape=(1, 1, self.hidden_dim),
                    initializer="zeros",
                    trainable=True,
                )

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        if self.combine_tokens and self.use_distillation:
            tokens_broadcasted = ops.broadcast_to(
                self.tokens, [batch_size, 2, self.hidden_dim]
            )
            return ops.concatenate([tokens_broadcasted, inputs], axis=1)
        else:
            cls_broadcasted = ops.broadcast_to(
                self.cls, [batch_size, 1, self.hidden_dim]
            )

            if self.use_distillation:
                dist_broadcasted = ops.broadcast_to(
                    self.dist, [batch_size, 1, self.hidden_dim]
                )
                return ops.concatenate(
                    [cls_broadcasted, dist_broadcasted, inputs], axis=1
                )
            else:
                return cls_broadcasted

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "use_distillation": self.use_distillation,
                "combine_tokens": self.combine_tokens,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class CaiTAddPositionEmbs(layers.Layer):
    """Adds learnable position embeddings to input tensors.

    Supports standard mode (patches + class token) and no_embed_class mode
    (positional embeddings only on patch tokens).

    Args:
        grid_h (int): Height of the position embedding grid.
        grid_w (int): Width of the position embedding grid.
        no_embed_class (bool): If True, applies positional embeddings only to patch tokens.
        use_distillation (bool): If True, operates in DeiT mode with distillation token.
        name (str, optional): Name of the layer.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        grid_h,
        grid_w,
        no_embed_class=False,
        use_distillation=False,
        resize_mode="bilinear",
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.no_embed_class = no_embed_class
        self.use_distillation = use_distillation
        self.resize_mode = resize_mode

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape, received: {len(input_shape)}D")

        num_patches = self.grid_h * self.grid_w

        if self.no_embed_class:
            if input_shape[1] == num_patches:
                self.skip_cls = False
            elif input_shape[1] == num_patches + 1:
                self.skip_cls = True
            else:
                raise ValueError(
                    f"Input sequence length {input_shape[1]} does not match expected length "
                    f"{num_patches} or {num_patches + 1} (grid: {self.grid_h}x{self.grid_w})"
                )
            self.position_embedding = self.add_weight(
                name="pos_embed",
                shape=(1, num_patches, input_shape[2]),
                initializer="random_normal",
                trainable=True,
            )
        else:
            expected_length = num_patches + (2 if self.use_distillation else 1)
            if input_shape[1] != expected_length:
                raise ValueError(
                    f"Input sequence length {input_shape[1]} does not match expected length "
                    f"{expected_length} (grid: {self.grid_h}x{self.grid_w} + tokens)"
                )
            self.position_embedding = self.add_weight(
                name="pos_embed",
                shape=(1, expected_length, input_shape[2]),
                initializer="random_normal",
                trainable=True,
            )
        super().build(input_shape)

    def call(self, inputs):
        if self.no_embed_class:
            if hasattr(self, "skip_cls") and self.skip_cls:
                cls_token = inputs[:, :1]
                patch_tokens = inputs[:, 1:]
                patch_tokens = patch_tokens + self.position_embedding
                return ops.concatenate([cls_token, patch_tokens], axis=1)
            else:
                return inputs + self.position_embedding
        elif self.use_distillation:
            tokens = inputs[:, :2]
            patch_tokens = inputs[:, 2:]
            token_pos_embed = self.position_embedding[:, :2]
            patch_pos_embed = self.position_embedding[:, 2:]
            tokens = tokens + token_pos_embed
            patch_tokens = patch_tokens + patch_pos_embed
            return ops.concatenate([tokens, patch_tokens], axis=1)
        else:
            return inputs + self.position_embedding

    def compute_output_shape(self, input_shape):
        return input_shape

    def save_own_variables(self, store):
        super().save_own_variables(store)
        store["grid_h"] = self.grid_h
        store["grid_w"] = self.grid_w
        store["no_embed_class"] = self.no_embed_class
        store["use_distillation"] = self.use_distillation

    def load_own_variables(self, store):
        source_h = int(store["grid_h"][...])
        source_w = int(store["grid_w"][...])

        try:
            source_no_embed_class = bool(store["no_embed_class"][...])
        except KeyError:
            source_no_embed_class = False

        try:
            source_use_distillation = bool(store["use_distillation"][...])
        except KeyError:
            source_use_distillation = False

        if source_h == self.grid_h and source_w == self.grid_w:
            self.position_embedding.assign(store["0"])
            return

        pos_embed = store["0"]

        if not source_no_embed_class:
            if source_use_distillation:
                spatial_pos_embed = pos_embed[:, 2:]
                token_pos_embed = pos_embed[:, :2]
            else:
                spatial_pos_embed = pos_embed[:, 1:]
                token_pos_embed = pos_embed[:, :1]
        else:
            spatial_pos_embed = pos_embed

        embed_dim = spatial_pos_embed.shape[-1]

        spatial_pos_embed = ops.cast(spatial_pos_embed, dtype="float32")
        spatial_pos_embed = ops.reshape(
            spatial_pos_embed, [1, source_h, source_w, embed_dim]
        )

        spatial_pos_embed = ops.image.resize(
            spatial_pos_embed,
            size=[self.grid_h, self.grid_w],
            interpolation=self.resize_mode,
            antialias=True,
        )

        spatial_pos_embed = ops.reshape(
            spatial_pos_embed, [1, self.grid_h * self.grid_w, embed_dim]
        )

        if self.no_embed_class:
            pos_embed = spatial_pos_embed
        else:
            if not source_no_embed_class:
                if self.use_distillation:
                    if source_use_distillation:
                        cls_dist_pos_embed = token_pos_embed
                    else:
                        cls_dist_pos_embed = ops.concatenate(
                            [token_pos_embed, token_pos_embed], axis=1
                        )
                else:
                    if source_use_distillation:
                        cls_dist_pos_embed = token_pos_embed[:, :1]
                    else:
                        cls_dist_pos_embed = token_pos_embed
            else:
                if self.use_distillation:
                    cls_dist_pos_embed = ops.zeros((1, 2, embed_dim))
                else:
                    cls_dist_pos_embed = ops.zeros((1, 1, embed_dim))

            pos_embed = ops.concatenate([cls_dist_pos_embed, spatial_pos_embed], axis=1)

        self.position_embedding.assign(pos_embed)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "grid_h": self.grid_h,
                "grid_w": self.grid_w,
                "no_embed_class": self.no_embed_class,
                "use_distillation": self.use_distillation,
                "resize_mode": self.resize_mode,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class CaiTClassAttention(layers.Layer):
    """Class Attention layer for transformer architectures.

    This layer implements a specialized attention mechanism where queries are generated
    only from the first token (class token) of the sequence, while keys and values are
    generated from the entire sequence. This approach is particularly useful in vision
    transformers and other architectures where a special class token aggregates information
    from the entire sequence.

    Key Features:
        - Queries are derived only from the class token (first token)
        - Keys and values are derived from the entire sequence
        - Supports multiple attention heads to capture different relationship patterns
        - Configurable attention and projection dropout for regularization
        - Optional bias terms in query/key/value projections
        - Supports both channels_last (NHWC) and channels_first (NCHW) formats

    Args:
        embed_dim (int): Total dimension of the input and output features. Must be divisible
            by num_heads to ensure even distribution of features across heads
        num_heads (int): Number of parallel attention heads. Each head operates
            on embed_dim/num_heads features
        qkv_bias (bool, optional): If True, adds learnable bias terms to the query, key,
            and value projections. Defaults to True
        attn_drop (float, optional): Dropout rate applied to attention weights. Helps
            prevent overfitting. Defaults to 0.0
        proj_drop (float, optional): Dropout rate applied to the output projection.
            Provides additional regularization. Defaults to 0.0
        data_format (str, optional): Format of the input tensor. Can be either
            'channels_last' (default) or 'channels_first'. For 'channels_last', the input
            shape is (batch_size, sequence_length, feature_dim), while for 'channels_first',
            the input shape is (batch_size, feature_dim, sequence_length).
        block_prefix (str, optional): Prefix for naming layer components. Defaults to None
        **kwargs: Additional keyword arguments passed to the parent Layer class

    Input shape:
        - If data_format='channels_last': 3D tensor (batch_size, sequence_length, feature_dim)
        - If data_format='channels_first': 3D tensor (batch_size, feature_dim, sequence_length)

    Output shape:
        - If data_format='channels_last': 3D tensor (batch_size, 1, feature_dim)
        - If data_format='channels_first': 3D tensor (batch_size, feature_dim, 1)

    Notes:
        - The attention dimension (embed_dim) must be divisible by num_heads
        - Only returns attention results for the class token (first position)
        - Commonly used in Vision Transformer (ViT) architectures
        - Implements a modified scaled dot-product attention where queries come only
          from the class token position
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        data_format: str = "channels_last",
        block_prefix: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert embed_dim % num_heads == 0, "embed_dim should be divisible by num_heads"
        assert data_format in ["channels_last", "channels_first"], (
            "data_format must be either 'channels_last' or 'channels_first'"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.data_format = data_format
        self.block_prefix = block_prefix

        prefix = f"{block_prefix}_" if block_prefix else ""

        self.q = layers.Dense(
            embed_dim,
            use_bias=qkv_bias,
            dtype=self.dtype_policy,
            name=f"{prefix}q" if prefix else None,
        )
        self.k = layers.Dense(
            embed_dim,
            use_bias=qkv_bias,
            dtype=self.dtype_policy,
            name=f"{prefix}k" if prefix else None,
        )
        self.v = layers.Dense(
            embed_dim,
            use_bias=qkv_bias,
            dtype=self.dtype_policy,
            name=f"{prefix}v" if prefix else None,
        )
        self.proj = layers.Dense(
            embed_dim, dtype=self.dtype_policy, name=f"{prefix}proj" if prefix else None
        )

        self.attn_drop = layers.Dropout(
            attn_drop,
            dtype=self.dtype_policy,
        )
        self.proj_drop = layers.Dropout(
            proj_drop,
            dtype=self.dtype_policy,
        )

    def build(self, input_shape):
        self.input_spec = InputSpec(ndim=len(input_shape))

        if self.input_spec.ndim != 3:
            raise ValueError(
                f"CaiTClassAttention expects 3D input tensor, but received shape: {input_shape}"
            )

        if self.data_format == "channels_last":
            feature_dim = input_shape[-1]
        else:  # channels_first
            feature_dim = input_shape[1]

        if feature_dim != self.embed_dim:
            raise ValueError(
                f"Input feature dimension {feature_dim} must match layer dimension {self.embed_dim}"
            )

        if self.data_format == "channels_last":
            self.q.build((input_shape[0], 1, input_shape[-1]))
            self.k.build(input_shape)
            self.v.build(input_shape)
            self.proj.build((input_shape[0], 1, self.embed_dim))
        else:
            self.q.build((input_shape[0], 1, input_shape[1]))
            self.k.build((input_shape[0], input_shape[2], input_shape[1]))
            self.v.build((input_shape[0], input_shape[2], input_shape[1]))
            self.proj.build((input_shape[0], 1, self.embed_dim))

        self.built = True

    def call(self, x, training=None):
        B = ops.shape(x)[0]

        if self.data_format == "channels_first":
            x = ops.transpose(x, (0, 2, 1))

        N = ops.shape(x)[1]

        q = self.q(x[:, 0:1])
        k = self.k(x)
        v = self.v(x)

        q = ops.reshape(q, (B, 1, self.num_heads, self.head_dim))
        k = ops.reshape(k, (B, N, self.num_heads, self.head_dim))
        v = ops.reshape(v, (B, N, self.num_heads, self.head_dim))

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        x = fused_attention(
            q, k, v, self.scale, dropout=self.attn_drop, training=training
        )
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (B, 1, self.embed_dim))

        x = self.proj(x)
        x = self.proj_drop(x, training=training)

        if self.data_format == "channels_first":
            x = ops.transpose(x, (0, 2, 1))

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "qkv_bias": self.q.use_bias,
                "attn_drop": self.attn_drop.rate,
                "proj_drop": self.proj_drop.rate,
                "data_format": self.data_format,
                "block_prefix": self.block_prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class CaiTTalkingHeadAttention(layers.Layer):
    """Talking-Head Attention layer implementing enhanced attention mechanism.

    This layer implements the Talking-Head Attention mechanism described in the paper
    "Talking-Heads Attention" (https://arxiv.org/abs/2003.02436), which enhances
    multi-head self-attention by applying linear projections across attention heads.
    This allows for information exchange between attention heads, enabling more
    flexible attention patterns and potentially improved performance.

    Key Features:
        - Linear projections across attention heads for enhanced cross-head communication
        - Scaled dot-product attention with additional head-talking projections
        - Configurable attention and projection dropout
        - Optional bias terms in query/key/value projections
        - Returns both output tensor and attention weights for inspection and visualization
        - Support for both channels_last (NHWC) and channels_first (NCHW) formats

    Args:
        embed_dim (int): Total dimension of the input and output features. Must be divisible
            by num_heads to ensure even distribution of features across heads
        num_heads (int): Number of parallel attention heads. Each head operates
            on embed_dim/num_heads features
        qkv_bias (bool, optional): If True, adds learnable bias terms to the query, key,
            and value projections. Defaults to True
        attn_drop (float, optional): Dropout rate applied to attention weights. Helps
            prevent overfitting. Defaults to 0.0
        proj_drop (float, optional): Dropout rate applied to the output projection.
            Provides additional regularization. Defaults to 0.0
        data_format (str, optional): Data format, either 'channels_last' (default) or 'channels_first'.
            Determines the order of dimensions in the input tensor
        block_prefix (str, optional): Prefix for naming layer components. Defaults to None
        **kwargs: Additional keyword arguments passed to the parent Layer class

    Input shape:
        - If data_format='channels_last': 3D tensor: (batch_size, sequence_length, feature_dim)
        - If data_format='channels_first': 3D tensor: (batch_size, feature_dim, sequence_length)

    Output shape:
        - Output tensor: Same shape as input
        - Attention weights: (batch_size, num_heads, sequence_length, sequence_length)

    Notes:
        - The attention dimension (embed_dim) must be divisible by num_heads
        - Talking-Head Attention extends standard multi-head attention with two additional
          linear projections: one before softmax (proj_l) and one after softmax (proj_w)
        - These projections allow each attention head to "talk" to other heads,
          enabling more expressive attention distributions
        - Suitable for sequence data in transformer-based architectures
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        data_format: str = "channels_last",
        block_prefix: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert embed_dim % num_heads == 0, "embed_dim should be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.block_prefix = block_prefix
        self.data_format = data_format

        assert data_format in ["channels_last", "channels_first"], (
            "data_format must be either 'channels_last' or 'channels_first'"
        )

        prefix = f"{block_prefix}_" if block_prefix else ""

        self.qkv = layers.Dense(
            embed_dim * 3,
            use_bias=qkv_bias,
            dtype=self.dtype_policy,
            name=f"{prefix}qkv" if block_prefix else None,
        )
        self.proj = layers.Dense(
            embed_dim,
            dtype=self.dtype_policy,
            name=f"{prefix}proj" if block_prefix else None,
        )

        self.proj_l = layers.Dense(
            num_heads,
            dtype=self.dtype_policy,
            name=f"{prefix}proj_l" if block_prefix else None,
        )
        self.proj_w = layers.Dense(
            num_heads,
            dtype=self.dtype_policy,
            name=f"{prefix}proj_w" if block_prefix else None,
        )

        self.attn_drop = layers.Dropout(
            attn_drop,
            dtype=self.dtype_policy,
        )
        self.proj_drop = layers.Dropout(
            proj_drop,
            dtype=self.dtype_policy,
        )

    def build(self, input_shape):
        self.input_spec = InputSpec(ndim=len(input_shape))

        if self.input_spec.ndim != 3:
            raise ValueError(
                f"CaiTTalkingHeadAttention expects 3D input tensor, but received shape: {input_shape}"
            )

        feature_dim_idx = 1 if self.data_format == "channels_first" else -1
        feature_dim = input_shape[feature_dim_idx]
        if feature_dim != self.embed_dim:
            raise ValueError(
                f"Input feature dimension {feature_dim} must match layer dimension {self.embed_dim}"
            )

        if self.data_format == "channels_last":
            self.qkv.build(input_shape)
            self.proj.build((input_shape[0], input_shape[1], self.embed_dim))
            self.proj_l.build(
                (input_shape[0], input_shape[1], input_shape[1], self.num_heads)
            )
            self.proj_w.build(
                (input_shape[0], input_shape[1], input_shape[1], self.num_heads)
            )
        else:  # channels_first
            self.qkv.build((input_shape[0], input_shape[2], self.embed_dim))
            self.proj.build((input_shape[0], input_shape[2], self.embed_dim))
            self.proj_l.build(
                (input_shape[0], input_shape[2], input_shape[2], self.num_heads)
            )
            self.proj_w.build(
                (input_shape[0], input_shape[2], input_shape[2], self.num_heads)
            )

        self.built = True

    def call(self, x, training=False):
        if self.data_format == "channels_first":
            x = ops.transpose(x, (0, 2, 1))

        input_shape = ops.shape(x)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        qkv = self.qkv(x)
        qkv = ops.reshape(
            qkv, (batch_size, seq_length, 3, self.num_heads, self.head_dim)
        )
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))

        attn = ops.transpose(attn, (0, 2, 3, 1))
        attn = self.proj_l(attn)
        attn = ops.transpose(attn, (0, 3, 1, 2))

        attn = ops.softmax(attn, axis=-1)

        attn = ops.transpose(attn, (0, 2, 3, 1))
        attn = self.proj_w(attn)
        attn = ops.transpose(attn, (0, 3, 1, 2))

        attn = self.attn_drop(attn, training=training)

        x = ops.matmul(attn, v)
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (batch_size, seq_length, self.embed_dim))

        x = self.proj(x)
        x = self.proj_drop(x, training=training)

        if self.data_format == "channels_first":
            x = ops.transpose(x, (0, 2, 1))

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "qkv_bias": self.qkv.use_bias,
                "attn_drop": self.attn_drop.rate,
                "proj_drop": self.proj_drop.rate,
                "data_format": self.data_format,
                "block_prefix": self.block_prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class CaiTLayerScale(layers.Layer):
    """Learnable per-channel scale (x * gamma), gamma initialized to layer_scale_init."""

    def __init__(self, layer_scale_init, **kwargs):
        super().__init__(**kwargs)
        self.layer_scale_init = layer_scale_init

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer=keras.initializers.Constant(self.layer_scale_init),
            trainable=True,
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update({"layer_scale_init": self.layer_scale_init})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class CaiTStochasticDepth(keras.layers.Layer):
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
