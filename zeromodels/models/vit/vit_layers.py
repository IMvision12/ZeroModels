import keras
from keras import InputSpec, layers, ops

from zeromodels.base.base_attention import fused_attention


@keras.saving.register_keras_serializable(package="zeromodels")
class ViTClassDistToken(layers.Layer):
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
            # Combined tokens (PiT-style)
            self.tokens = self.add_weight(
                name="cls_token",
                shape=(1, 2, self.hidden_dim),
                initializer="zeros",
                trainable=True,
            )
        else:
            # Class token
            self.cls = self.add_weight(
                name="cls_token",
                shape=(1, 1, self.hidden_dim),
                initializer="zeros",
                trainable=True,
            )
            # Distillation token for DeiT
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
                return ops.concatenate([cls_broadcasted, inputs], axis=1)

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
class ViTAddPositionEmbs(layers.Layer):
    """
    A custom Keras layer that adds learnable position embeddings to input tensors with support for
    flexible grid sizes and optional class/distillation token handling.

    The layer supports three modes of operation:
      1. Standard mode: Creates position embeddings for both patches and a class token.
      2. FlexiViT mode: Creates embeddings only for patches, handling a class token separately.
         In this mode, the layer accepts either a pure patch token input (grid_h * grid_w tokens)
         or an input with a leading class token (grid_h * grid_w + 1 tokens).
      3. DeiT mode: Creates embeddings for patches, a class token, and a distillation token.
         In this case the expected sequence length is grid_h * grid_w + 2 tokens.

    For PiT, the recommended usage is to set `no_embed_class=True` so that the patch tokens
    are first augmented with positional embeddings. Then, the class (or combined class/distillation)
    token is added later in the model pipeline.

    Features:
      - Configurable grid dimensions for position embeddings.
      - Dynamic resizing of position embeddings during loading through bilinear interpolation.
      - Compatible with standard Vision Transformer, FlexiViT, DeiT, and PiT architectures.
      - Handles class and distillation token embeddings appropriately based on mode.
      - When no_embed_class=True, the layer applies positional embeddings only to patch tokens.
        If a class token is present at the beginning, it is preserved and concatenated back after
        the patch tokens are positionally embedded.

    Args:
        grid_h (int): Height of the position embedding grid.
        grid_w (int): Width of the position embedding grid.
        no_embed_class (bool):
            - If False (default), operates in standard mode where position embeddings are learned for
              the entire input (patches plus token(s)).
            - If True, operates in FlexiViT or PiT mode where positional embeddings are applied only to
              the patch tokens. In this mode the input can be either a sequence of patch tokens only
              (grid_h * grid_w tokens) or a sequence with a leading class token (grid_h * grid_w + 1 tokens).
        use_distillation (bool):
            If True, operates in DeiT mode with an additional distillation token.
            When no_embed_class is False, the expected input sequence length is grid_h * grid_w + 2.
            Defaults to False.
        name (str, optional): Name of the layer.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Input Shape:
        3D tensor with shape: `(batch_size, sequence_length, embed_dim)`, where sequence_length should be:
          * Standard mode: grid_h * grid_w + 1 (patch tokens + class token)
          * FlexiViT mode / PiT mode: either grid_h * grid_w (patch tokens only) or grid_h * grid_w + 1
          * DeiT mode: grid_h * grid_w + 2 (patch tokens + class token + distillation token)

    Output Shape:
        Same as the input shape: `(batch_size, sequence_length, embed_dim)`. In FlexiViT or PiT mode,
        if a class token is present at the beginning, it is preserved and positional embeddings are added
        only to the patch tokens.

    The layer supports loading weights from different grid sizes through bilinear interpolation,
    making it flexible for transfer learning and model adaptation scenarios. When loading weights,
    it automatically handles the conversion between different modes (standard / FlexiViT / DeiT),
    adjusting the position embeddings appropriately.
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
class ViTMultiHeadSelfAttention(layers.Layer):
    """Multi-Head Self-Attention layer implementing scaled dot-product attention.

    This layer implements the standard multi-head self-attention mechanism where input is split
    into multiple attention heads operating in parallel. Each head performs scaled dot-product
    attention independently, after which results are concatenated and projected back to the
    original dimension.

    Key Features:
        - Independent parallel attention heads for capturing different relationship patterns
        - Scaled dot-product attention with optional layer normalization
        - Configurable attention and projection dropout
        - Optional bias terms in query/key/value projections
        - Support for both 3D and 4D input tensors

    Args:
        embed_dim (int): Total dimension of the input and output features. Must be divisible
            by num_heads to ensure even distribution of features across heads
        num_heads (int, optional): Number of parallel attention heads. Defaults to 8
        qkv_bias (bool, optional): If True, adds learnable bias terms to the query, key,
            and value projections. Defaults to False
        qk_norm (bool, optional): If True, applies layer normalization to query and key
            tensors before attention computation. Defaults to False
        attn_drop (float, optional): Dropout rate applied to attention weights. Defaults to 0.0
        proj_drop (float, optional): Dropout rate applied to the output projection. Defaults to 0.0
        epsilon (float, optional): Small constant used in normalization. Defaults to 1e-6
        block_prefix (str, optional): Prefix for naming layer components. Defaults to None
        **kwargs: Additional keyword arguments passed to the parent Layer class

    Input shape:
        - 3D tensor: (batch_size, sequence_length, feature_dim)
        - 4D tensor: (batch_size, height, width, feature_dim)

    Output shape:
        - Same as input shape
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        epsilon=1e-6,
        block_prefix=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.block_prefix = block_prefix if block_prefix is not None else "blocks"
        prefix = f"{self.block_prefix}_"

        assert embed_dim % num_heads == 0, "embed_dim should be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.epsilon = epsilon

        self.qkv = layers.Dense(
            embed_dim * 3,
            use_bias=qkv_bias,
            dtype=self.dtype_policy,
            name=prefix + "attn_qkv",
        )

        self.q_norm = (
            layers.LayerNormalization(
                epsilon=self.epsilon,
                dtype=self.dtype_policy,
                name=prefix + "attn_norm1",
            )
            if qk_norm
            else None
        )
        self.k_norm = (
            layers.LayerNormalization(
                epsilon=self.epsilon,
                dtype=self.dtype_policy,
                name=prefix + "attn_norm2",
            )
            if qk_norm
            else None
        )

        self.attn_drop = layers.Dropout(
            attn_drop,
            dtype=self.dtype_policy,
        )
        self.proj = layers.Dense(
            embed_dim, dtype=self.dtype_policy, name=prefix + "attn_proj"
        )
        self.proj_drop = layers.Dropout(
            proj_drop,
            dtype=self.dtype_policy,
        )

    def build(self, input_shape):
        self.input_spec = InputSpec(ndim=len(input_shape))

        if self.input_spec.ndim not in (3, 4):
            raise ValueError(
                f"ViTMultiHeadSelfAttention expects 3D or 4D input tensor, but received shape: {input_shape}"
            )

        feature_dim = input_shape[-1]
        if feature_dim != self.embed_dim:
            raise ValueError(
                f"Input feature dimension {feature_dim} must match layer dimension {self.embed_dim}"
            )

        self.qkv.build(input_shape)
        self.proj.build(input_shape)

        if self.q_norm is not None:
            norm_shape = (input_shape[-1],)
            self.q_norm.build(norm_shape)
        if self.k_norm is not None:
            norm_shape = (input_shape[-1],)
            self.k_norm.build(norm_shape)

        self.built = True

    def call(self, inputs, training=None):
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        ndim = len(inputs.shape)

        qkv = self.qkv(inputs)

        qkv_split = ops.split(qkv, 3, axis=-1)
        q, k, v = qkv_split

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q = q * self.scale

        if ndim == 3:
            q = ops.reshape(q, [batch_size, -1, self.num_heads, self.head_dim])
            k = ops.reshape(k, [batch_size, -1, self.num_heads, self.head_dim])
            v = ops.reshape(v, [batch_size, -1, self.num_heads, self.head_dim])

            q = ops.transpose(q, [0, 2, 1, 3])
            k = ops.transpose(k, [0, 2, 1, 3])
            v = ops.transpose(v, [0, 2, 1, 3])

            x = fused_attention(q, k, v, 1.0, dropout=self.attn_drop, training=training)

            x = ops.transpose(x, [0, 2, 1, 3])
            x = ops.reshape(x, input_shape)
        else:
            q = ops.reshape(
                q,
                [
                    batch_size,
                    input_shape[1],
                    input_shape[2],
                    self.num_heads,
                    self.head_dim,
                ],
            )
            k = ops.reshape(
                k,
                [
                    batch_size,
                    input_shape[1],
                    input_shape[2],
                    self.num_heads,
                    self.head_dim,
                ],
            )
            v = ops.reshape(
                v,
                [
                    batch_size,
                    input_shape[1],
                    input_shape[2],
                    self.num_heads,
                    self.head_dim,
                ],
            )

            q = ops.transpose(q, [0, 1, 3, 2, 4])
            k = ops.transpose(k, [0, 1, 3, 2, 4])
            v = ops.transpose(v, [0, 1, 3, 2, 4])

            q = ops.reshape(q, [-1, self.num_heads, input_shape[2], self.head_dim])
            k = ops.reshape(k, [-1, self.num_heads, input_shape[2], self.head_dim])
            v = ops.reshape(v, [-1, self.num_heads, input_shape[2], self.head_dim])

            x = fused_attention(q, k, v, 1.0, dropout=self.attn_drop, training=training)

            x = ops.reshape(
                x,
                [
                    batch_size,
                    input_shape[1],
                    self.num_heads,
                    input_shape[2],
                    self.head_dim,
                ],
            )
            x = ops.transpose(x, [0, 1, 3, 2, 4])
            x = ops.reshape(x, input_shape)

        x = self.proj(x)
        x = self.proj_drop(x, training=training)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "qkv_bias": self.qkv.use_bias,
                "qk_norm": self.q_norm is not None,
                "attn_drop": self.attn_drop.rate,
                "proj_drop": self.proj_drop.rate,
                "epsilon": self.epsilon,
                "block_prefix": self.block_prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class ViTLayerScale(layers.Layer):
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
class ViTSwiGLUGate(layers.Layer):
    """SwiGLU gate: split the fused projection in half and gate, silu(x1) * x2.

    Pairs with a single ``weights_in`` Dense that outputs ``2 * hidden`` features,
    matching HF's ``Dinov2SwiGLUFFN`` used by the DINOv2 giant.
    """

    def call(self, x):
        x1, x2 = ops.split(x, 2, axis=-1)
        return ops.silu(x1) * x2

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], input_shape[-1] // 2)
