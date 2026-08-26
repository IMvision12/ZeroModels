import keras
from keras import InputSpec, layers, ops

from zeromodels.base.base_attention import fused_attention


@keras.saving.register_keras_serializable(package="zeromodels")
class MobileViTImageToPatchesLayer(layers.Layer):
    """A Keras layer that converts images into patches.

    This layer takes a batch of images and converts each image into a sequence of patches.
    The patches are created by dividing the image into non-overlapping squares of size
    `patch_size` x `patch_size`. If the image dimensions are not perfectly divisible by
    the patch size, the image is resized to the nearest larger size that is divisible.

    Args:
        patch_size (int): The size of each square patch (both height and width).
        **kwargs: Additional layer arguments.

    Input shape:
        4D tensor with shape:
        - If data_format='channels_last': `(batch_size, height, width, channels)`
        - If data_format='channels_first': `(batch_size, channels, height, width)`

    Output shape:
        4D tensor with shape:
        - If data_format='channels_last': `(batch_size, patch_size*patch_size, num_patches, channels)`
        - If data_format='channels_first': `(batch_size, channels, patch_size*patch_size, num_patches)`
        where num_patches = ceil(height/patch_size) * ceil(width/patch_size)
    """

    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.resize = False
        self.data_format = keras.config.image_data_format()

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch_size, height, width, channels) or "
                f"(batch_size, channels, height, width), got {len(input_shape)}"
            )
        super().build(input_shape)

    def call(self, inputs):
        from keras import ops

        x = inputs

        if self.data_format == "channels_last":
            h, w, c = x.shape[-3], x.shape[-2], x.shape[-1]
        else:
            c, h, w = x.shape[-3], x.shape[-2], x.shape[-1]

        new_h = ((h + self.patch_size - 1) // self.patch_size) * self.patch_size
        new_w = ((w + self.patch_size - 1) // self.patch_size) * self.patch_size
        num_patches_h = new_h // self.patch_size
        num_patches_w = new_w // self.patch_size
        num_patches = num_patches_h * num_patches_w

        self.resize = False
        if new_h != h or new_w != w:
            x = ops.image.resize(x, size=(new_h, new_w), data_format=self.data_format)
            self.resize = True

        if self.data_format == "channels_last":
            x = ops.reshape(
                x,
                [-1, num_patches_h, self.patch_size, num_patches_w, self.patch_size, c],
            )
            x = ops.transpose(x, [0, 2, 4, 1, 3, 5])
            x = ops.reshape(x, [-1, self.patch_size * self.patch_size, num_patches, c])
        else:
            x = ops.reshape(
                x,
                [-1, c, num_patches_h, self.patch_size, num_patches_w, self.patch_size],
            )
            x = ops.transpose(x, [0, 1, 3, 5, 2, 4])
            x = ops.reshape(x, [-1, c, self.patch_size * self.patch_size, num_patches])

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MobileViTPatchesToImageLayer(layers.Layer):
    """A Keras layer that reconstructs images from patches.

    This layer takes a sequence of image patches and reconstructs the original image by
    placing the patches back in their original positions. It can handle both cases where
    the original image dimensions are known or unknown, and can optionally resize the
    output to match the original image dimensions.

    Args:
        patch_size (int): The size of each square patch (both height and width).
        **kwargs: Additional layer arguments.

    Input shape:
        4D tensor with shape:
        - If data_format='channels_last': `(batch_size, patch_size*patch_size, num_patches, channels)`
        - If data_format='channels_first': `(batch_size, channels, patch_size*patch_size, num_patches)`

    Output shape:
        4D tensor with shape:
        - If data_format='channels_last': `(batch_size, height, width, channels)`
        - If data_format='channels_first': `(batch_size, channels, height, width)`
    """

    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.data_format = keras.config.image_data_format()

    def build(self, input_shape):
        self.h = None
        self.w = None
        self.c = (
            input_shape[-1] if self.data_format == "channels_last" else input_shape[1]
        )
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        c = input_shape[-1] if self.data_format == "channels_last" else input_shape[1]
        num_patches = (
            input_shape[2] if self.data_format == "channels_last" else input_shape[-1]
        )
        side_patches = int(num_patches**0.5)
        h = w = side_patches * self.patch_size

        if self.data_format == "channels_last":
            return input_shape[0], h, w, c
        else:
            return input_shape[0], c, h, w

    def compute_output_spec(self, inputs, original_size=None, resize=False):
        input_spec = keras.KerasTensor(inputs.shape, dtype=inputs.dtype)
        batch_size = input_spec.shape[0]
        c = (
            input_spec.shape[-1]
            if self.data_format == "channels_last"
            else input_spec.shape[1]
        )

        if original_size is None:
            num_patches = (
                input_spec.shape[2]
                if self.data_format == "channels_last"
                else input_spec.shape[-1]
            )
            side_patches = int(num_patches**0.5)
            h = w = side_patches * self.patch_size
        else:
            h, w = original_size

            h = ((h + self.patch_size - 1) // self.patch_size) * self.patch_size
            w = ((w + self.patch_size - 1) // self.patch_size) * self.patch_size

            if resize:
                h, w = original_size

        if self.data_format == "channels_last":
            return keras.KerasTensor((batch_size, h, w, c), dtype=inputs.dtype)
        else:
            return keras.KerasTensor((batch_size, c, h, w), dtype=inputs.dtype)

    def call(self, inputs, original_size=None, resize=False):
        x = inputs

        if original_size is not None:
            self.h, self.w = original_size

        if self.h is None or self.w is None:
            num_patches = (
                inputs.shape[2]
                if self.data_format == "channels_last"
                else inputs.shape[-1]
            )
            side_patches = int(num_patches**0.5)
            self.h = self.w = side_patches * self.patch_size

        new_h = ((self.h + self.patch_size - 1) // self.patch_size) * self.patch_size
        new_w = ((self.w + self.patch_size - 1) // self.patch_size) * self.patch_size
        num_patches_h = new_h // self.patch_size
        num_patches_w = new_w // self.patch_size

        if self.data_format == "channels_last":
            x = ops.reshape(
                x,
                [
                    -1,
                    self.patch_size,
                    self.patch_size,
                    num_patches_h,
                    num_patches_w,
                    self.c,
                ],
            )
            x = ops.transpose(x, [0, 3, 1, 4, 2, 5])
            x = ops.reshape(x, [-1, new_h, new_w, self.c])
        else:
            x = ops.reshape(
                x,
                [
                    -1,
                    self.c,
                    self.patch_size,
                    self.patch_size,
                    num_patches_h,
                    num_patches_w,
                ],
            )
            x = ops.transpose(x, [0, 1, 4, 2, 5, 3])
            x = ops.reshape(x, [-1, self.c, new_h, new_w])

        if resize:
            x = ops.image.resize(x, size=(self.h, self.w), data_format=self.data_format)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MobileViTMultiHeadSelfAttention(layers.Layer):
    """Multi-Head Self-Attention layer implementing scaled dot-product attention.

    Args:
        dim (int): Total dimension of the input and output features.
        num_heads (int, optional): Number of parallel attention heads. Defaults to 8.
        qkv_bias (bool, optional): If True, adds bias to q/k/v projections. Defaults to False.
        qk_norm (bool, optional): If True, applies layer normalization to q and k. Defaults to False.
        attn_drop (float, optional): Dropout rate for attention weights. Defaults to 0.0.
        proj_drop (float, optional): Dropout rate for output projection. Defaults to 0.0.
        epsilon (float, optional): Small constant for normalization. Defaults to 1e-6.
        block_prefix (str, optional): Prefix for naming layer components. Defaults to None.
        **kwargs: Additional keyword arguments.

    Input shape:
        - 3D tensor: (batch_size, sequence_length, feature_dim)
        - 4D tensor: (batch_size, height, width, feature_dim)

    Output shape:
        - Same as input shape.
    """

    def __init__(
        self,
        dim: int,
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

        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.epsilon = epsilon

        self.qkv = layers.Dense(
            dim * 3,
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
            dim, dtype=self.dtype_policy, name=prefix + "attn_proj"
        )
        self.proj_drop = layers.Dropout(
            proj_drop,
            dtype=self.dtype_policy,
        )

    def build(self, input_shape):
        self.input_spec = InputSpec(ndim=len(input_shape))

        if self.input_spec.ndim not in (3, 4):
            raise ValueError(
                f"MobileViTMultiHeadSelfAttention expects 3D or 4D input tensor, but received shape: {input_shape}"
            )

        feature_dim = input_shape[-1]
        if feature_dim != self.dim:
            raise ValueError(
                f"Input feature dimension {feature_dim} must match layer dimension {self.dim}"
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
                "dim": self.dim,
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
