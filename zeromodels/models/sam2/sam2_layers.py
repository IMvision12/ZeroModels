import math

import keras
import numpy as np
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM2NoMemoryEmbedding(layers.Layer):
    """Learnable bias added to image embeddings indicating no memory.

    This layer adds a trainable zero-initialized bias to image embeddings,
    signaling that no memory conditioning is available. The bias is broadcast
    across the spatial dimensions of the input feature map.

    Args:
        hidden_dim (int): Channel dimension.
            Defaults to ``256``.
        data_format (str): Image data format.
            Defaults to ``"channels_last"``.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    References:
        - SAM 2: https://arxiv.org/abs/2408.00714
    """

    def __init__(self, hidden_dim=256, data_format="channels_last", **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.data_format = data_format

    def build(self, input_shape):
        self.embedding = self.add_weight(
            name="embedding",
            shape=(self.hidden_dim,),
            initializer="zeros",
        )
        self.built = True

    def call(self, image_embeddings):
        if self.data_format == "channels_first":
            bias = ops.reshape(self.embedding, (1, self.hidden_dim, 1, 1))
        else:
            bias = ops.reshape(self.embedding, (1, 1, 1, self.hidden_dim))
        return image_embeddings + bias

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim, "data_format": self.data_format})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM2SinePositionEmbedding(layers.Layer):
    """2-D sine-cosine positional encoding for FPN feature maps.

    Generates fixed positional embeddings by computing sine and cosine
    functions over normalized spatial coordinates. The encoding
    concatenates y and x components to produce a channels-first tensor
    of shape ``(batch, num_pos_feats * 2, height, width)``.

    Args:
        num_pos_feats (int): Number of positional features per spatial
            dimension. Defaults to ``128``.
        temperature (int): Temperature scaling for the sinusoidal
            frequencies. Defaults to ``10000``.
        normalize (bool): Whether to normalize coordinates before
            encoding. Defaults to ``True``.
        scale (float): Coordinate scaling factor applied after
            normalization. Defaults to ``2 * pi``.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    References:
        - SAM 2: https://arxiv.org/abs/2408.00714
    """

    def __init__(
        self,
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        scale=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if scale is None:
            scale = 2 * math.pi
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def call(self, inputs):
        shape = ops.shape(inputs)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        not_mask = ops.ones((batch_size, height, width), dtype="float32")
        y_embed = ops.cumsum(not_mask, axis=1)
        x_embed = ops.cumsum(not_mask, axis=2)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = ops.cast(ops.arange(self.num_pos_feats), dtype="float32")
        dim_t = self.temperature ** (2 * ops.floor(dim_t / 2) / self.num_pos_feats)

        pos_x = ops.expand_dims(x_embed, axis=-1) / dim_t
        pos_y = ops.expand_dims(y_embed, axis=-1) / dim_t

        pos_x_sin = ops.sin(pos_x[:, :, :, 0::2])
        pos_x_cos = ops.cos(pos_x[:, :, :, 1::2])
        pos_y_sin = ops.sin(pos_y[:, :, :, 0::2])
        pos_y_cos = ops.cos(pos_y[:, :, :, 1::2])

        pos_x = ops.reshape(
            ops.stack([pos_x_sin, pos_x_cos], axis=4),
            (batch_size, height, width, self.num_pos_feats),
        )
        pos_y = ops.reshape(
            ops.stack([pos_y_sin, pos_y_cos], axis=4),
            (batch_size, height, width, self.num_pos_feats),
        )

        pos = ops.concatenate([pos_y, pos_x], axis=-1)
        pos = ops.transpose(pos, (0, 3, 1, 2))
        return pos

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_pos_feats": self.num_pos_feats,
                "temperature": self.temperature,
                "normalize": self.normalize,
                "scale": self.scale,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM2MultiScaleAttention(layers.Layer):
    """Multi-head attention with optional query pooling for Hiera.

    Computes multi-head self-attention over spatial feature maps. When
    ``query_stride`` is provided, queries are spatially downsampled via
    max-pooling before the attention computation, enabling progressive
    resolution reduction across Hiera stages.

    Args:
        dim (int): Input feature dimension.
        dim_out (int): Output feature dimension.
        num_heads (int): Number of attention heads.
        query_stride (int or None): Spatial stride for query pooling. When
            set, queries are max-pooled before attention.
            Defaults to ``None``.
        data_format (str): Image data format.
            Defaults to ``"channels_last"``.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    References:
        - SAM 2: https://arxiv.org/abs/2408.00714
    """

    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        query_stride=None,
        data_format="channels_last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.query_stride = query_stride
        self.head_dim = dim_out // num_heads
        self.scale = self.head_dim**-0.5
        self.data_format = data_format

    def build(self, input_shape):
        if self.data_format == "channels_first":
            nhwc_shape = (
                *input_shape[:-3],
                input_shape[-2],
                input_shape[-1],
                input_shape[-3],
            )
        else:
            nhwc_shape = input_shape
        self.qkv = layers.Dense(self.dim_out * 3, use_bias=True, name="qkv")
        self.qkv.build(nhwc_shape)
        self.proj = layers.Dense(self.dim_out, use_bias=True, name="proj")
        self.proj.build((*nhwc_shape[:-1], self.dim_out))
        if self.query_stride is not None:
            self._q_pool = layers.MaxPool2D(
                pool_size=self.query_stride,
                strides=self.query_stride,
                data_format=self.data_format,
                name="q_pool",
            )
        self.built = True

    def call(self, hidden_states):
        cf = self.data_format == "channels_first"
        shape = ops.shape(hidden_states)
        batch_size = shape[0]
        if cf:
            height = shape[2]
            width = shape[3]
        else:
            height = shape[1]
            width = shape[2]

        if cf:
            hidden_states = ops.transpose(hidden_states, (0, 2, 3, 1))

        qkv = self.qkv(hidden_states)
        qkv = ops.reshape(
            qkv,
            (batch_size, -1, 3, self.num_heads, self.head_dim),
        )

        q = qkv[:, :, 0, :, :]
        k = qkv[:, :, 1, :, :]
        v = qkv[:, :, 2, :, :]

        if self.query_stride is not None:
            q = ops.reshape(
                q, (batch_size, height, width, self.num_heads * self.head_dim)
            )
            if cf:
                q = ops.transpose(q, (0, 3, 1, 2))
            q = self._q_pool(q)
            if cf:
                new_h = ops.shape(q)[2]
                new_w = ops.shape(q)[3]
                q = ops.transpose(q, (0, 2, 3, 1))
            else:
                new_h = ops.shape(q)[1]
                new_w = ops.shape(q)[2]
            q = ops.reshape(q, (batch_size, -1, self.num_heads, self.head_dim))
        else:
            new_h = height
            new_w = width

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        attn_output = fused_attention(q, k, v, self.scale)

        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(attn_output, (batch_size, new_h, new_w, self.dim_out))

        attn_output = self.proj(attn_output)

        if cf:
            attn_output = ops.transpose(attn_output, (0, 3, 1, 2))

        return attn_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "dim_out": self.dim_out,
                "num_heads": self.num_heads,
                "query_stride": self.query_stride,
                "data_format": self.data_format,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM2MultiScaleBlock(layers.Layer):
    """Hiera transformer block with windowed or global attention.

    Implements a single Hiera transformer block consisting of layer
    normalization, multi-scale attention, and a two-layer MLP with GELU
    activation. Attention can be restricted to local windows when
    ``window_size > 0`` or applied globally when ``window_size`` is ``0``.

    Args:
        dim (int): Input feature dimension.
        dim_out (int): Output feature dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dimension to ``dim_out``.
            Defaults to ``4.0``.
        window_size (int): Spatial window size for local attention. Use
            ``0`` for global attention.
            Defaults to ``0``.
        query_stride (int or None): Spatial stride for query pooling.
            Defaults to ``None``.
        layer_norm_eps (float): Epsilon for layer normalization.
            Defaults to ``1e-6``.
        data_format (str): Image data format.
            Defaults to ``"channels_last"``.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    References:
        - SAM 2: https://arxiv.org/abs/2408.00714
    """

    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        window_size=0,
        query_stride=None,
        layer_norm_eps=1e-6,
        data_format="channels_last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.query_stride = query_stride
        self.layer_norm_eps = layer_norm_eps
        self.data_format = data_format

    def build(self, input_shape):
        self.layer_norm1 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm1"
        )
        if self.data_format == "channels_first":
            ln_shape = (
                *input_shape[:-3],
                input_shape[-2],
                input_shape[-1],
                input_shape[-3],
            )
        else:
            ln_shape = input_shape
        self.layer_norm1.build(ln_shape)
        self.attn = SAM2MultiScaleAttention(
            self.dim,
            self.dim_out,
            self.num_heads,
            query_stride=self.query_stride,
            data_format=self.data_format,
            name="attn",
        )
        self.attn.build(input_shape)
        mlp_dim = int(self.dim_out * self.mlp_ratio)
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="layer_norm2"
        )
        if self.data_format == "channels_first":
            nhwc_prefix = (*input_shape[:-3], input_shape[-2], input_shape[-1])
        else:
            nhwc_prefix = input_shape[:-1]
        self.layer_norm2.build((*nhwc_prefix, self.dim_out))
        self.mlp_lin1 = layers.Dense(mlp_dim, name="mlp_proj_in")
        self.mlp_lin1.build((*nhwc_prefix, self.dim_out))
        self.mlp_lin2 = layers.Dense(self.dim_out, name="mlp_proj_out")
        self.mlp_lin2.build((*nhwc_prefix, mlp_dim))

        if self.dim != self.dim_out:
            self.proj = layers.Dense(self.dim_out, name="proj")
            self.proj.build((*nhwc_prefix, self.dim))

        if self.query_stride is not None:
            self._residual_pool = layers.MaxPool2D(
                pool_size=self.query_stride,
                strides=self.query_stride,
                data_format=self.data_format,
                name="residual_pool",
            )

        self.built = True

    def _window_partition(self, hidden_states, window_size):
        cf = self.data_format == "channels_first"
        shape = ops.shape(hidden_states)
        batch_size = shape[0]
        if cf:
            channels = shape[1]
            height = shape[2]
            width = shape[3]
        else:
            height = shape[1]
            width = shape[2]
            channels = shape[3]

        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size

        if pad_h > 0 or pad_w > 0:
            if cf:
                hidden_states = ops.pad(
                    hidden_states, [[0, 0], [0, 0], [0, pad_h], [0, pad_w]]
                )
            else:
                hidden_states = ops.pad(
                    hidden_states, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]]
                )

        padded_h = height + pad_h
        padded_w = width + pad_w

        if cf:
            hidden_states = ops.transpose(hidden_states, (0, 2, 3, 1))

        hidden_states = ops.reshape(
            hidden_states,
            (
                batch_size,
                padded_h // window_size,
                window_size,
                padded_w // window_size,
                window_size,
                channels,
            ),
        )
        hidden_states = ops.transpose(hidden_states, (0, 1, 3, 2, 4, 5))
        hidden_states = ops.reshape(
            hidden_states, (-1, window_size, window_size, channels)
        )

        if cf:
            hidden_states = ops.transpose(hidden_states, (0, 3, 1, 2))

        return hidden_states, (padded_h, padded_w)

    def _window_unpartition(self, windows, window_size, pad_hw, original_hw):
        cf = self.data_format == "channels_first"
        padded_h, padded_w = pad_hw
        height, width = original_hw
        num_windows_h = padded_h // window_size
        num_windows_w = padded_w // window_size

        if cf:
            channels = ops.shape(windows)[1]
            batch_size = ops.shape(windows)[0] // (num_windows_h * num_windows_w)
            windows = ops.transpose(windows, (0, 2, 3, 1))
        else:
            channels = ops.shape(windows)[-1]
            batch_size = ops.shape(windows)[0] // (num_windows_h * num_windows_w)

        x = ops.reshape(
            windows,
            (
                batch_size,
                num_windows_h,
                num_windows_w,
                window_size,
                window_size,
                channels,
            ),
        )
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        x = ops.reshape(x, (batch_size, padded_h, padded_w, channels))
        x = x[:, :height, :width, :]

        if cf:
            x = ops.transpose(x, (0, 3, 1, 2))

        return x

    def call(self, hidden_states):
        cf = self.data_format == "channels_first"
        residual = hidden_states

        if cf:
            hidden_states = ops.transpose(hidden_states, (0, 2, 3, 1))
        hidden_states = self.layer_norm1(hidden_states)
        if cf:
            hidden_states = ops.transpose(hidden_states, (0, 3, 1, 2))

        if self.dim != self.dim_out:
            if cf:
                hidden_states_nhwc = ops.transpose(hidden_states, (0, 2, 3, 1))
            else:
                hidden_states_nhwc = hidden_states
            residual_nhwc = self.proj(hidden_states_nhwc)
            if cf:
                residual = ops.transpose(residual_nhwc, (0, 3, 1, 2))
            else:
                residual = residual_nhwc
            if self.query_stride is not None:
                residual = self._residual_pool(residual)

        window_size = self.window_size
        if window_size > 0:
            if cf:
                H = ops.shape(hidden_states)[2]
                W = ops.shape(hidden_states)[3]
            else:
                H = ops.shape(hidden_states)[1]
                W = ops.shape(hidden_states)[2]
            hidden_states, pad_hw = self._window_partition(hidden_states, window_size)

        hidden_states = self.attn(hidden_states)

        if self.query_stride is not None and window_size > 0:
            window_size = window_size // self.query_stride
            if cf:
                H_new = ops.shape(residual)[2]
                W_new = ops.shape(residual)[3]
            else:
                H_new = ops.shape(residual)[1]
                W_new = ops.shape(residual)[2]

            pad_h = (window_size - H_new % window_size) % window_size
            pad_w = (window_size - W_new % window_size) % window_size
            pad_hw = (H_new + pad_h, W_new + pad_w)
            H = H_new
            W = W_new

        if window_size > 0:
            hidden_states = self._window_unpartition(
                hidden_states, window_size, pad_hw, (H, W)
            )

        hidden_states = residual + hidden_states

        if cf:
            ln_out = ops.transpose(hidden_states, (0, 2, 3, 1))
        else:
            ln_out = hidden_states
        ln_out = self.layer_norm2(ln_out)
        mlp_out = self.mlp_lin1(ln_out)
        mlp_out = ops.nn.gelu(mlp_out, approximate=False)
        mlp_out = self.mlp_lin2(mlp_out)
        if cf:
            mlp_out = ops.transpose(mlp_out, (0, 3, 1, 2))
        hidden_states = hidden_states + mlp_out

        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "dim_out": self.dim_out,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "window_size": self.window_size,
                "query_stride": self.query_stride,
                "layer_norm_eps": self.layer_norm_eps,
                "data_format": self.data_format,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM2PositionalEmbedding(layers.Layer):
    """Random Fourier feature positional encoding for 2-D coordinates.

    Projects normalized 2-D coordinates through a fixed random Gaussian
    matrix and applies sine and cosine transformations to produce a
    continuous positional encoding of dimension ``num_pos_feats * 4``.

    Args:
        num_pos_feats (int): Number of positional features (output
            dimension is ``num_pos_feats * 4``).
            Defaults to ``128``.
        scale (float): Standard deviation of the random projection
            matrix.
            Defaults to ``1.0``.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    References:
        - SAM 2: https://arxiv.org/abs/2408.00714
    """

    def __init__(self, num_pos_feats=128, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.num_pos_feats = num_pos_feats
        self.scale = scale

    def build(self, input_shape):
        self.positional_embedding = self.add_weight(
            name="positional_embedding",
            shape=(2, self.num_pos_feats),
            initializer=keras.initializers.RandomNormal(stddev=self.scale),
            trainable=False,
        )
        self.built = True

    def call(self, coordinates):
        coordinates = 2 * coordinates - 1
        coordinates = ops.cast(coordinates, dtype=self.positional_embedding.dtype)
        coordinates = ops.matmul(coordinates, self.positional_embedding)
        coordinates = 2 * np.pi * coordinates
        return ops.concatenate([ops.sin(coordinates), ops.cos(coordinates)], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_pos_feats": self.num_pos_feats,
                "scale": self.scale,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM2ImagePositionalEmbeddings(layers.Layer):
    """Grid-based positional embeddings for the image feature map.

    Constructs a uniform coordinate grid over the image embedding space
    and encodes it using a shared ``SAM2PositionalEmbedding`` layer to
    produce dense positional embeddings for the image encoder output.

    Args:
        image_embedding_size (int): Spatial size of the image embedding
            grid (assumes square).
        shared_embedding (SAM2PositionalEmbedding): A ``SAM2PositionalEmbedding``
            instance used to encode grid coordinates.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    References:
        - SAM 2: https://arxiv.org/abs/2408.00714
    """

    def __init__(self, image_embedding_size, shared_embedding, **kwargs):
        super().__init__(**kwargs)
        self.image_embedding_size = image_embedding_size
        self.shared_embedding = shared_embedding

    def call(self, inputs):
        size = self.image_embedding_size
        grid = ops.ones((size, size), dtype="float32")
        y_embed = ops.cumsum(grid, axis=0) - 0.5
        x_embed = ops.cumsum(grid, axis=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size
        coords = ops.stack([x_embed, y_embed], axis=-1)
        pe = self.shared_embedding(coords)
        pe = ops.expand_dims(pe, axis=0)
        return pe

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_embedding_size": self.image_embedding_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM2PromptEncoderLayer(layers.Layer):
    """Encodes sparse and dense prompts for SAM2 mask prediction.

    Converts point and label inputs into sparse token embeddings using
    a shared positional encoding and learnable point-type embeddings.
    Also produces a dense no-mask embedding broadcast to the image
    embedding spatial dimensions when no mask prompt is provided.

    Args:
        hidden_dim (int): Embedding dimension for prompt tokens.
            Defaults to ``256``.
        image_embedding_size (int): Spatial size of the image embedding
            grid.
            Defaults to ``64``.
        image_size (int): Input image resolution used to normalize point
            coordinates.
            Defaults to ``1024``.
        num_point_embeddings (int): Number of learnable point-type
            embeddings.
            Defaults to ``4``.
        shared_embedding (SAM2PositionalEmbedding or None): A
            ``SAM2PositionalEmbedding`` instance shared with the image
            encoder.
            Defaults to ``None``.
        data_format (str): Image data format.
            Defaults to ``"channels_last"``.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    References:
        - SAM 2: https://arxiv.org/abs/2408.00714
    """

    def __init__(
        self,
        hidden_dim=256,
        image_embedding_size=64,
        image_size=1024,
        num_point_embeddings=4,
        shared_embedding=None,
        data_format="channels_last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.image_embedding_size = image_embedding_size
        self.image_size = image_size
        self.num_point_embeddings = num_point_embeddings
        self.shared_embedding = shared_embedding
        self.data_format = data_format

    def build(self, input_shape):
        embed_init = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)

        self.point_embeddings = []
        for i in range(self.num_point_embeddings):
            w = self.add_weight(
                name=f"point_embed_{i}",
                shape=(1, self.hidden_dim),
                initializer=embed_init,
            )
            self.point_embeddings.append(w)
        self.not_a_point_embed = self.add_weight(
            name="not_a_point_embed",
            shape=(1, self.hidden_dim),
            initializer=embed_init,
        )
        self.no_mask_embed = self.add_weight(
            name="no_mask_embed",
            shape=(1, self.hidden_dim),
            initializer=embed_init,
        )
        self.built = True

    def _embed_points(self, points, labels, pad=True):
        points = points + 0.5
        if pad:
            points = ops.pad(points, [[0, 0], [0, 0], [0, 1], [0, 0]])
            labels = ops.pad(labels, [[0, 0], [0, 0], [0, 1]], constant_values=-1)
        point_embedding = self.shared_embedding(
            points / ops.cast(self.image_size, dtype=points.dtype)
        )

        point_embedding = ops.where(
            ops.expand_dims(labels, axis=-1) == -1,
            ops.broadcast_to(self.not_a_point_embed, ops.shape(point_embedding)),
            point_embedding,
        )
        point_embedding = ops.where(
            ops.expand_dims(labels, axis=-1) != -10,
            point_embedding,
            ops.zeros_like(point_embedding),
        )

        for i in range(self.num_point_embeddings):
            mask = ops.expand_dims(
                ops.cast(labels == i, dtype=point_embedding.dtype), axis=-1
            )
            point_embedding = point_embedding + mask * ops.broadcast_to(
                self.point_embeddings[i], ops.shape(point_embedding)
            )

        return point_embedding

    def _embed_boxes(self, boxes):
        """Encode box prompts as three-token (top-left, bottom-right, pad).

        Reproduces ``Sam2PromptEncoder._embed_boxes`` exactly:

        - Input shape ``(batch_size, num_boxes, 4)`` with corner
          layout ``(x1, y1, x2, y2)``.
        - Shift by +0.5 to pixel centers, reshape to
          ``(batch, num_boxes, 2, 2)``, then pad with an all-zero
          third "not-a-point" row so the final layout is
          ``(batch, num_boxes, 3, 2)``.
        - Run the shared Fourier positional encoding once on the
          padded coord grid, which produces
          ``(batch, num_boxes, 3, hidden)``.
        - Add the top-left type embedding (``point_embeddings[2]``)
          to slot 0, the bottom-right type embedding
          (``point_embeddings[3]``) to slot 1, and **replace** slot 2
          with ``not_a_point_embed`` broadcast to full shape.

        The returned shape ``(batch, num_boxes, 3, hidden)`` slots
        into the sparse-embedding concat where ``num_boxes`` plays
        the role of ``point_batch_size``.
        """
        boxes = boxes + 0.5
        batch_size = ops.shape(boxes)[0]
        num_boxes = ops.shape(boxes)[1]
        coords = ops.reshape(boxes, (batch_size, num_boxes, 2, 2))
        coords = ops.pad(coords, [[0, 0], [0, 0], [0, 1], [0, 0]])
        corner_embedding = self.shared_embedding(
            coords / ops.cast(self.image_size, dtype=coords.dtype)
        )

        tl_type = ops.reshape(self.point_embeddings[2], (1, 1, 1, self.hidden_dim))
        br_type = ops.reshape(self.point_embeddings[3], (1, 1, 1, self.hidden_dim))
        pad_type = ops.broadcast_to(
            ops.reshape(self.not_a_point_embed, (1, 1, 1, self.hidden_dim)),
            (batch_size, num_boxes, 1, self.hidden_dim),
        )

        tl = corner_embedding[:, :, 0:1, :] + tl_type
        br = corner_embedding[:, :, 1:2, :] + br_type
        corner_embedding = ops.concatenate([tl, br, pad_type], axis=2)
        return corner_embedding

    def _no_mask_dense(self, batch_size):
        cf = self.data_format == "channels_first"
        if cf:
            return ops.broadcast_to(
                ops.reshape(self.no_mask_embed, (1, self.hidden_dim, 1, 1)),
                (
                    batch_size,
                    self.hidden_dim,
                    self.image_embedding_size,
                    self.image_embedding_size,
                ),
            )
        return ops.broadcast_to(
            ops.reshape(self.no_mask_embed, (1, 1, 1, self.hidden_dim)),
            (
                batch_size,
                self.image_embedding_size,
                self.image_embedding_size,
                self.hidden_dim,
            ),
        )

    def call(self, inputs):
        input_points = inputs[0]
        input_labels = inputs[1]
        input_boxes = inputs[2] if len(inputs) >= 3 else None
        mask_dense = inputs[3] if len(inputs) >= 5 else None
        has_mask = inputs[4] if len(inputs) >= 5 else None

        pad_points = input_boxes is None
        sparse_embeddings = self._embed_points(
            input_points, input_labels, pad=pad_points
        )

        if input_boxes is not None:
            box_embeddings = self._embed_boxes(input_boxes)
            sparse_embeddings = ops.concatenate(
                [sparse_embeddings, box_embeddings], axis=2
            )

        batch_size = ops.shape(input_points)[0]
        no_mask_dense = self._no_mask_dense(batch_size)

        if mask_dense is not None and has_mask is not None:
            gate = ops.reshape(ops.cast(has_mask, no_mask_dense.dtype), (-1, 1, 1, 1))
            dense_embeddings = gate * mask_dense + (1.0 - gate) * no_mask_dense
        else:
            dense_embeddings = no_mask_dense

        return {
            "sparse_embeddings": sparse_embeddings,
            "dense_embeddings": dense_embeddings,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "image_embedding_size": self.image_embedding_size,
                "image_size": self.image_size,
                "num_point_embeddings": self.num_point_embeddings,
                "data_format": self.data_format,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM2TwoWayAttention(layers.Layer):
    """Attention layer for the two-way mask decoder transformer.

    Performs multi-head attention with separate query, key, and value
    projections. The internal projection dimension can be reduced by
    ``downsample_rate`` to lower computation in cross-attention layers.
    An optional additive attention similarity bias is supported.

    Args:
        hidden_dim (int): Input and output feature dimension.
            Defaults to ``256``.
        num_heads (int): Number of attention heads.
            Defaults to ``8``.
        downsample_rate (int): Factor by which the internal projection
            dimension is reduced from ``hidden_dim``.
            Defaults to ``1``.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    References:
        - SAM 2: https://arxiv.org/abs/2408.00714
    """

    def __init__(self, hidden_dim=256, num_heads=8, downsample_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.downsample_rate = downsample_rate
        self.internal_dim = hidden_dim // downsample_rate
        self.head_dim = self.internal_dim // num_heads

    def build(self, input_shape):
        self.q_proj = layers.Dense(self.internal_dim, name="q_proj")
        self.q_proj.build(input_shape)
        self.k_proj = layers.Dense(self.internal_dim, name="k_proj")
        self.k_proj.build(input_shape)
        self.v_proj = layers.Dense(self.internal_dim, name="v_proj")
        self.v_proj.build(input_shape)
        self.out_proj = layers.Dense(self.hidden_dim, name="out_proj")
        self.out_proj.build((*input_shape[:-1], self.internal_dim))
        self.built = True

    def call(self, query, key, value, attention_similarity=None):
        batch_size = ops.shape(query)[0]
        point_batch = ops.shape(query)[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        new_shape = (
            batch_size * point_batch,
            -1,
            self.num_heads,
            self.head_dim,
        )
        q = ops.reshape(q, new_shape)
        k = ops.reshape(k, new_shape)
        v = ops.reshape(v, new_shape)

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        scale = self.head_dim**-0.5
        attn_output = fused_attention(q, k, v, scale, attention_similarity)

        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(
            attn_output,
            (batch_size, point_batch, -1, self.internal_dim),
        )
        return self.out_proj(attn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "downsample_rate": self.downsample_rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM2MaskDecoderLayer(layers.Layer):
    """Mask decoder with two-way transformer and object scoring.

    Combines image embeddings and prompt tokens through a stack of
    two-way transformer layers with self-attention and cross-attention.
    The decoded features are upscaled and projected through per-mask
    hypernetwork MLPs to predict multiple masks, IoU confidence scores,
    and an object presence score.

    Args:
        hidden_dim (int): Channel dimension for embeddings and
            transformer layers.
            Defaults to ``256``.
        num_hidden_layers (int): Number of two-way transformer layers.
            Defaults to ``2``.
        num_heads (int): Number of attention heads in each
            transformer layer.
            Defaults to ``8``.
        mlp_dim (int): Hidden dimension of the feed-forward network
            inside each transformer layer.
            Defaults to ``2048``.
        num_multimask_outputs (int): Number of additional mask
            predictions beyond the single-mask output.
            Defaults to ``3``.
        iou_head_depth (int): Number of layers in the IoU prediction
            MLP.
            Defaults to ``3``.
        iou_head_hidden_dim (int): Hidden dimension of the IoU
            prediction MLP.
            Defaults to ``256``.
        attention_downsample_rate (int): Downsample rate for
            cross-attention internal projections.
            Defaults to ``2``.
        layer_norm_eps (float): Epsilon for layer normalization.
            Defaults to ``1e-6``.
        data_format (str): Image data format.
            Defaults to ``"channels_last"``.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    References:
        - SAM 2: https://arxiv.org/abs/2408.00714
    """

    def __init__(
        self,
        hidden_dim=256,
        num_hidden_layers=2,
        num_heads=8,
        mlp_dim=2048,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        attention_downsample_rate=2,
        layer_norm_eps=1e-6,
        data_format="channels_last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.attention_downsample_rate = attention_downsample_rate
        self.layer_norm_eps = layer_norm_eps
        self.data_format = data_format

    def build(self, input_shape):
        hs = self.hidden_dim
        nm = self.num_mask_tokens
        ds = self.attention_downsample_rate

        embed_init = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)

        self.obj_score_token = self.add_weight(
            name="obj_score_token",
            shape=(1, hs),
            initializer=embed_init,
        )
        self.iou_token = self.add_weight(
            name="iou_token",
            shape=(1, hs),
            initializer=embed_init,
        )
        self.mask_tokens = self.add_weight(
            name="mask_tokens",
            shape=(nm, hs),
            initializer=embed_init,
        )

        self.transformer_self_attns = []
        self.transformer_layer_norm1s = []
        self.transformer_cross_attn_token_to_images = []
        self.transformer_layer_norm2s = []
        self.transformer_mlp_lin1s = []
        self.transformer_mlp_lin2s = []
        self.transformer_layer_norm3s = []
        self.transformer_cross_attn_image_to_tokens = []
        self.transformer_layer_norm4s = []

        dummy_shape = (None, None, None, hs)
        for i in range(self.num_hidden_layers):
            self_attn = SAM2TwoWayAttention(
                hs,
                self.num_heads,
                downsample_rate=1,
                name=f"transformer_self_attns_{i}",
            )
            self_attn.build(dummy_shape)
            self.transformer_self_attns.append(self_attn)

            ln1 = layers.LayerNormalization(
                epsilon=self.layer_norm_eps,
                name=f"transformer_layer_norm1s_{i}",
            )
            ln1.build(dummy_shape)
            self.transformer_layer_norm1s.append(ln1)

            cross_t2i = SAM2TwoWayAttention(
                hs,
                self.num_heads,
                downsample_rate=ds,
                name=f"transformer_cross_attn_token_to_images_{i}",
            )
            cross_t2i.build(dummy_shape)
            self.transformer_cross_attn_token_to_images.append(cross_t2i)

            ln2 = layers.LayerNormalization(
                epsilon=self.layer_norm_eps,
                name=f"transformer_layer_norm2s_{i}",
            )
            ln2.build(dummy_shape)
            self.transformer_layer_norm2s.append(ln2)

            mlp1 = layers.Dense(self.mlp_dim, name=f"transformer_mlp_lin1s_{i}")
            mlp1.build(dummy_shape)
            self.transformer_mlp_lin1s.append(mlp1)

            mlp2 = layers.Dense(hs, name=f"transformer_mlp_lin2s_{i}")
            mlp2.build((*dummy_shape[:-1], self.mlp_dim))
            self.transformer_mlp_lin2s.append(mlp2)

            ln3 = layers.LayerNormalization(
                epsilon=self.layer_norm_eps,
                name=f"transformer_layer_norm3s_{i}",
            )
            ln3.build(dummy_shape)
            self.transformer_layer_norm3s.append(ln3)

            cross_i2t = SAM2TwoWayAttention(
                hs,
                self.num_heads,
                downsample_rate=ds,
                name=f"transformer_cross_attn_image_to_tokens_{i}",
            )
            cross_i2t.build(dummy_shape)
            self.transformer_cross_attn_image_to_tokens.append(cross_i2t)

            ln4 = layers.LayerNormalization(
                epsilon=self.layer_norm_eps,
                name=f"transformer_layer_norm4s_{i}",
            )
            ln4.build(dummy_shape)
            self.transformer_layer_norm4s.append(ln4)

        self.final_attn_token_to_image = SAM2TwoWayAttention(
            hs,
            self.num_heads,
            downsample_rate=ds,
            name="final_attn_token_to_image",
        )
        self.final_attn_token_to_image.build(dummy_shape)
        self.layer_norm_final_attn = layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            name="layer_norm_final_attn",
        )
        self.layer_norm_final_attn.build(dummy_shape)

        self.upscale_conv1 = layers.Conv2DTranspose(
            hs // 4,
            kernel_size=2,
            strides=2,
            data_format=self.data_format,
            name="upscale_conv1",
        )
        if self.data_format == "channels_first":
            self.upscale_conv1.build((None, hs, None, None))
        else:
            self.upscale_conv1.build((None, None, None, hs))
        self.upscale_layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="upscale_layer_norm"
        )
        self.upscale_layer_norm.build((None, None, None, hs // 4))
        self.upscale_conv2 = layers.Conv2DTranspose(
            hs // 8,
            kernel_size=2,
            strides=2,
            data_format=self.data_format,
            name="upscale_conv2",
        )
        if self.data_format == "channels_first":
            self.upscale_conv2.build((None, hs // 4, None, None))
        else:
            self.upscale_conv2.build((None, None, None, hs // 4))

        self.conv_s0 = layers.Conv2D(
            hs // 8,
            kernel_size=1,
            data_format=self.data_format,
            name="conv_s0",
        )
        if self.data_format == "channels_first":
            self.conv_s0.build((None, hs, None, None))
        else:
            self.conv_s0.build((None, None, None, hs))
        self.conv_s1 = layers.Conv2D(
            hs // 4,
            kernel_size=1,
            data_format=self.data_format,
            name="conv_s1",
        )
        if self.data_format == "channels_first":
            self.conv_s1.build((None, hs, None, None))
        else:
            self.conv_s1.build((None, None, None, hs))

        out_dim = hs // 8
        self.output_hypernetworks_mlps_proj_ins = []
        self.output_hypernetworks_mlps_hidden_layers = []
        self.output_hypernetworks_mlps_proj_outs = []
        for i in range(nm):
            pi = layers.Dense(hs, name=f"output_hypernetworks_mlps_proj_ins_{i}")
            pi.build((None, None, hs))
            self.output_hypernetworks_mlps_proj_ins.append(pi)

            hl = layers.Dense(hs, name=f"output_hypernetworks_mlps_layers_{i}")
            hl.build((None, None, hs))
            self.output_hypernetworks_mlps_hidden_layers.append(hl)

            po = layers.Dense(
                out_dim,
                name=f"output_hypernetworks_mlps_proj_outs_{i}",
            )
            po.build((None, None, hs))
            self.output_hypernetworks_mlps_proj_outs.append(po)

        self.iou_head_proj_in = layers.Dense(
            self.iou_head_hidden_dim, name="iou_head_proj_in"
        )
        self.iou_head_proj_in.build((None, None, hs))
        self.iou_head_hidden_layers = []
        for i in range(self.iou_head_depth - 2):
            hl = layers.Dense(
                self.iou_head_hidden_dim,
                name=f"iou_head_hidden_layers_{i}",
            )
            hl.build((None, None, self.iou_head_hidden_dim))
            self.iou_head_hidden_layers.append(hl)
        self.iou_head_proj_out = layers.Dense(nm, name="iou_head_proj_out")
        self.iou_head_proj_out.build((None, None, self.iou_head_hidden_dim))

        self.obj_score_proj_in = layers.Dense(hs, name="obj_score_proj_in")
        self.obj_score_proj_in.build((None, None, hs))
        self.obj_score_hidden_layers = []
        for i in range(1):
            hl = layers.Dense(hs, name=f"obj_score_hidden_layers_{i}")
            hl.build((None, None, hs))
            self.obj_score_hidden_layers.append(hl)
        self.obj_score_proj_out = layers.Dense(1, name="obj_score_proj_out")
        self.obj_score_proj_out.build((None, None, hs))

        self.built = True

    def call(self, inputs):
        cf = self.data_format == "channels_first"
        (
            image_embeddings,
            image_pe,
            sparse_embeddings,
            dense_embeddings,
            high_res_feat_s0,
            high_res_feat_s1,
        ) = inputs

        batch_size = ops.shape(image_embeddings)[0]
        if cf:
            num_channels = ops.shape(image_embeddings)[1]
            height = ops.shape(image_embeddings)[2]
            width = ops.shape(image_embeddings)[3]
        else:
            num_channels = ops.shape(image_embeddings)[3]
            height = ops.shape(image_embeddings)[1]
            width = ops.shape(image_embeddings)[2]
        point_batch_size = ops.shape(sparse_embeddings)[1]

        output_tokens = ops.concatenate(
            [self.obj_score_token, self.iou_token, self.mask_tokens],
            axis=0,
        )
        output_tokens = ops.broadcast_to(
            ops.reshape(
                output_tokens,
                (1, 1, 2 + self.num_mask_tokens, self.hidden_dim),
            ),
            (
                batch_size,
                point_batch_size,
                2 + self.num_mask_tokens,
                self.hidden_dim,
            ),
        )
        tokens = ops.concatenate([output_tokens, sparse_embeddings], axis=2)

        image_embeddings_with_dense = image_embeddings + dense_embeddings

        if cf:
            ie_nhwc = ops.transpose(image_embeddings_with_dense, (0, 2, 3, 1))
        else:
            ie_nhwc = image_embeddings_with_dense
        ie_flat = ops.reshape(
            ie_nhwc,
            (batch_size, height * width, num_channels),
        )
        ie_flat = ops.expand_dims(ie_flat, axis=1)
        ie_flat = ops.broadcast_to(
            ie_flat,
            (batch_size, point_batch_size, height * width, num_channels),
        )

        image_pe_nhwc = ops.broadcast_to(
            image_pe, (batch_size, height, width, num_channels)
        )
        pe_flat = ops.reshape(image_pe_nhwc, (batch_size, height * width, num_channels))
        pe_flat = ops.expand_dims(pe_flat, axis=1)
        pe_flat = ops.broadcast_to(
            pe_flat,
            (batch_size, point_batch_size, height * width, num_channels),
        )

        queries = tokens
        keys = ie_flat

        for i in range(self.num_hidden_layers):
            if i == 0:
                queries = self.transformer_self_attns[i](queries, queries, queries)
            else:
                queries_with_pe = queries + tokens
                attn_out = self.transformer_self_attns[i](
                    queries_with_pe, queries_with_pe, queries
                )
                queries = queries + attn_out
            queries = self.transformer_layer_norm1s[i](queries)

            queries_with_pe = queries + tokens
            keys_with_pe = keys + pe_flat

            attn_out = self.transformer_cross_attn_token_to_images[i](
                queries_with_pe, keys_with_pe, keys
            )
            queries = queries + attn_out
            queries = self.transformer_layer_norm2s[i](queries)

            mlp_out = self.transformer_mlp_lin1s[i](queries)
            mlp_out = ops.nn.relu(mlp_out)
            mlp_out = self.transformer_mlp_lin2s[i](mlp_out)
            queries = queries + mlp_out
            queries = self.transformer_layer_norm3s[i](queries)

            queries_with_pe = queries + tokens
            keys_with_pe = keys + pe_flat

            attn_out = self.transformer_cross_attn_image_to_tokens[i](
                keys_with_pe, queries_with_pe, queries
            )
            keys = keys + attn_out
            keys = self.transformer_layer_norm4s[i](keys)

        queries_with_pe = queries + tokens
        keys_with_pe = keys + pe_flat
        attn_out = self.final_attn_token_to_image(queries_with_pe, keys_with_pe, keys)
        queries = queries + attn_out
        queries = self.layer_norm_final_attn(queries)

        iou_token_out = queries[:, :, 1, :]
        mask_tokens_out = queries[:, :, 2 : 2 + self.num_mask_tokens, :]

        keys_spatial = ops.reshape(
            keys,
            (batch_size * point_batch_size, height, width, num_channels),
        )
        if cf:
            keys_spatial = ops.transpose(keys_spatial, (0, 3, 1, 2))

        feat_s1 = ops.expand_dims(high_res_feat_s1, axis=1)
        if cf:
            feat_s1 = ops.broadcast_to(
                feat_s1,
                (
                    batch_size,
                    point_batch_size,
                    ops.shape(feat_s1)[2],
                    ops.shape(feat_s1)[3],
                    ops.shape(feat_s1)[4],
                ),
            )
            feat_s1 = ops.reshape(
                feat_s1,
                (
                    -1,
                    ops.shape(feat_s1)[2],
                    ops.shape(feat_s1)[3],
                    ops.shape(feat_s1)[4],
                ),
            )
        else:
            feat_s1 = ops.broadcast_to(
                feat_s1,
                (
                    batch_size,
                    point_batch_size,
                    ops.shape(feat_s1)[2],
                    ops.shape(feat_s1)[3],
                    ops.shape(feat_s1)[4],
                ),
            )
            feat_s1 = ops.reshape(
                feat_s1,
                (
                    -1,
                    ops.shape(feat_s1)[2],
                    ops.shape(feat_s1)[3],
                    ops.shape(feat_s1)[4],
                ),
            )

        feat_s0 = ops.expand_dims(high_res_feat_s0, axis=1)
        if cf:
            feat_s0 = ops.broadcast_to(
                feat_s0,
                (
                    batch_size,
                    point_batch_size,
                    ops.shape(feat_s0)[2],
                    ops.shape(feat_s0)[3],
                    ops.shape(feat_s0)[4],
                ),
            )
            feat_s0 = ops.reshape(
                feat_s0,
                (
                    -1,
                    ops.shape(feat_s0)[2],
                    ops.shape(feat_s0)[3],
                    ops.shape(feat_s0)[4],
                ),
            )
        else:
            feat_s0 = ops.broadcast_to(
                feat_s0,
                (
                    batch_size,
                    point_batch_size,
                    ops.shape(feat_s0)[2],
                    ops.shape(feat_s0)[3],
                    ops.shape(feat_s0)[4],
                ),
            )
            feat_s0 = ops.reshape(
                feat_s0,
                (
                    -1,
                    ops.shape(feat_s0)[2],
                    ops.shape(feat_s0)[3],
                    ops.shape(feat_s0)[4],
                ),
            )

        upscaled = self.upscale_conv1(keys_spatial) + self.conv_s1(feat_s1)
        if cf:
            upscaled = ops.transpose(upscaled, (0, 2, 3, 1))
        upscaled = ops.nn.gelu(self.upscale_layer_norm(upscaled), approximate=False)
        if cf:
            upscaled = ops.transpose(upscaled, (0, 3, 1, 2))
        upscaled_2 = self.upscale_conv2(upscaled) + self.conv_s0(feat_s0)
        upscaled = ops.nn.gelu(upscaled_2, approximate=False)

        up_shape = ops.shape(upscaled)
        if cf:
            up_c = up_shape[1]
            up_h = up_shape[2]
            up_w = up_shape[3]
        else:
            up_h = up_shape[1]
            up_w = up_shape[2]
            up_c = up_shape[3]

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            h = self.output_hypernetworks_mlps_proj_ins[i](mask_tokens_out[:, :, i, :])
            h = ops.nn.relu(h)
            h = ops.nn.relu(self.output_hypernetworks_mlps_hidden_layers[i](h))
            h = self.output_hypernetworks_mlps_proj_outs[i](h)
            hyper_in_list.append(h)
        hyper_in = ops.stack(hyper_in_list, axis=2)

        if cf:
            upscaled_nhwc = ops.transpose(upscaled, (0, 2, 3, 1))
        else:
            upscaled_nhwc = upscaled
        upscaled_flat = ops.reshape(
            upscaled_nhwc,
            (batch_size, point_batch_size, up_h * up_w, up_c),
        )
        masks = ops.matmul(hyper_in, ops.transpose(upscaled_flat, (0, 1, 3, 2)))
        masks = ops.reshape(
            masks,
            (batch_size, point_batch_size, -1, up_h, up_w),
        )

        iou_out = self.iou_head_proj_in(iou_token_out)
        iou_out = ops.nn.relu(iou_out)
        for hl in self.iou_head_hidden_layers:
            iou_out = ops.nn.relu(hl(iou_out))
        iou_pred = ops.sigmoid(self.iou_head_proj_out(iou_out))

        obj_score = self.obj_score_proj_in(queries[:, :, 0, :])
        obj_score = ops.nn.relu(obj_score)
        for hl in self.obj_score_hidden_layers:
            obj_score = ops.nn.relu(hl(obj_score))
        object_score_logits = self.obj_score_proj_out(obj_score)

        return {
            "pred_masks": masks,
            "iou_scores": iou_pred,
            "object_score_logits": object_score_logits,
            "mask_tokens_out": mask_tokens_out,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_hidden_layers": self.num_hidden_layers,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "num_multimask_outputs": self.num_multimask_outputs,
                "iou_head_depth": self.iou_head_depth,
                "iou_head_hidden_dim": self.iou_head_hidden_dim,
                "attention_downsample_rate": self.attention_downsample_rate,
                "layer_norm_eps": self.layer_norm_eps,
                "data_format": self.data_format,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM2HieraPositionEmbedding(layers.Layer):
    """Windowed positional embedding for the Hiera backbone.

    Combines a global background positional grid, resized via bicubic
    interpolation, with a tiled local window positional embedding. The
    two components are summed and added to the input feature map to
    provide both coarse and fine-grained spatial information.

    Args:
        hidden_dim (int): Embedding channel dimension.
        spatial_size (tuple of int): Spatial height and width of the
            feature map.
        window_size (int): Local window size for the tiled positional
            component.
        bg_size (tuple of int): Spatial size of the learnable background
            positional grid that is resized to ``spatial_size``.
            Defaults to ``(7, 7)``.
        data_format (str): Image data format.
            Defaults to ``"channels_last"``.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    References:
        - SAM 2: https://arxiv.org/abs/2408.00714
    """

    def __init__(
        self,
        hidden_dim,
        spatial_size,
        window_size,
        bg_size=(7, 7),
        data_format="channels_last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.spatial_size = tuple(spatial_size)
        self.window_size = window_size
        self.bg_size = tuple(bg_size)
        self.data_format = data_format

    def build(self, input_shape):
        h, w = self.spatial_size
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, self.bg_size[0], self.bg_size[1], self.hidden_dim),
            initializer="zeros",
        )
        self.pos_embed_window = self.add_weight(
            name="pos_embed_window",
            shape=(1, self.window_size, self.window_size, self.hidden_dim),
            initializer="zeros",
        )
        self._full_pos = self.add_weight(
            name="full_pos",
            shape=(1, h, w, self.hidden_dim),
            initializer="zeros",
            trainable=False,
        )
        self.built = True

    def _recompute_full_pos(self):
        h, w = self.spatial_size
        pos = ops.image.resize(
            ops.convert_to_tensor(self.pos_embed),
            size=(h, w),
            interpolation="bicubic",
            antialias=False,
            data_format="channels_last",
        )
        tile_h = h // self.window_size
        tile_w = w // self.window_size
        window_pos = ops.tile(self.pos_embed_window, (1, tile_h, tile_w, 1))
        self._full_pos.assign(pos + window_pos)

    def load_own_variables(self, store):
        self.pos_embed.assign(store["0"])
        self.pos_embed_window.assign(store["1"])
        self._recompute_full_pos()

    def call(self, hidden_states):
        pos = ops.convert_to_tensor(self._full_pos)
        if self.data_format == "channels_first":
            pos = ops.transpose(pos, (0, 3, 1, 2))
        return hidden_states + pos

    def compute_output_spec(self, hidden_states):
        return keras.KerasTensor(hidden_states.shape, dtype=hidden_states.dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "spatial_size": self.spatial_size,
                "window_size": self.window_size,
                "bg_size": self.bg_size,
                "data_format": self.data_format,
            }
        )
        return config
