import math

import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="zeromodels")
class DETRExpandQueryEmbedding(layers.Layer):
    """Expands learned query embeddings to match the batch dimension.

    Wraps a standard `Embedding` layer and broadcasts its output along
    a new batch axis so that each sample in the batch receives the same
    set of learned object queries. Used to produce the positional part
    of the object queries fed into the DETR decoder.

    Reference:
    - [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

    Args:
        num_queries: Integer, number of object queries (maximum
            detections per image).
        hidden_dim: Integer, embedding dimension for each query.
        **kwargs: Additional keyword arguments passed to the `Layer`
            class.

    Input Shape:
        Any tensor whose first dimension is the batch size. Only
        `batch_size` is read from the input; the content is unused.

    Output Shape:
        3D tensor: `(batch_size, num_queries, hidden_dim)`.
    """

    def __init__(self, num_queries, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.embedding = layers.Embedding(
            num_queries,
            hidden_dim,
            name="embedding",
        )

    def call(self, batch_ref):
        batch_size = ops.shape(batch_ref)[0]
        indices = ops.arange(self.num_queries)
        query_embed = self.embedding(indices)
        query_embed = ops.expand_dims(query_embed, axis=0)
        query_embed = ops.tile(query_embed, [batch_size, 1, 1])
        return query_embed

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
class DETRFlattenFeatures(layers.Layer):
    """Flattens spatial feature maps into a 1D token sequence.

    Reshapes a 4D spatial tensor into a 3D sequence tensor suitable
    for transformer input by collapsing the height and width
    dimensions into a single sequence dimension.

    Reference:
    - [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

    Args:
        hidden_dim: Integer, channel dimension of the input feature
            map. Used as the last dimension in the reshape target.
        **kwargs: Additional keyword arguments passed to the `Layer`
            class.

    Input Shape:
        4D tensor: `(batch_size, height, width, hidden_dim)`.

    Output Shape:
        3D tensor: `(batch_size, height * width, hidden_dim)`.
    """

    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def call(self, inputs):
        shape = ops.shape(inputs)
        data_format = keras.config.image_data_format()
        if data_format == "channels_first":
            x = ops.transpose(inputs, [0, 2, 3, 1])
            return ops.reshape(x, [shape[0], shape[2] * shape[3], self.hidden_dim])
        return ops.reshape(inputs, [shape[0], shape[1] * shape[2], self.hidden_dim])

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DETRPositionEmbeddingSine(layers.Layer):
    """Fixed sinusoidal 2D positional embedding for spatial feature maps.

    Generates non-learnable sine/cosine positional encodings that
    encode the row and column position of each spatial location. Half
    of the embedding dimension encodes the vertical position and the
    other half encodes the horizontal position, using sinusoidal
    functions at geometrically spaced frequencies. Matches the
    positional encoding used in the original DETR implementation.

    Reference:
    - [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

    Args:
        hidden_dim: Integer, total embedding dimension. Half is
            allocated to row embeddings and half to column embeddings.
            Defaults to `256`.
        temperature: Integer, temperature scaling factor for the
            sinusoidal frequencies. Defaults to `10000`.
        normalize: Boolean, whether to normalize position coordinates
            to the range `[0, 2*pi]` before computing the encoding.
            Defaults to `True`.
        eps: Float, small constant added during normalization to
            prevent division by zero. Defaults to `1e-6`.
        **kwargs: Additional keyword arguments passed to the `Layer`
            class.

    Input Shape:
        4D tensor: `(batch_size, height, width, channels)`. Only the
        spatial dimensions are used; the channel dimension is ignored.

    Output Shape:
        4D tensor: `(batch_size, height, width, hidden_dim)`.
    """

    def __init__(
        self,
        hidden_dim=256,
        temperature=10000,
        normalize=True,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.normalize = normalize
        self.eps = eps
        self.num_pos_feats = hidden_dim // 2

    def call(self, inputs):
        shape = ops.shape(inputs)
        batch_size = shape[0]
        data_format = keras.config.image_data_format()
        if data_format == "channels_first":
            h = shape[2]
            w = shape[3]
        else:
            h = shape[1]
            w = shape[2]

        y_embed = ops.cast(
            ops.repeat(
                ops.expand_dims(ops.arange(1, h + 1, dtype="float32"), axis=1),
                w,
                axis=1,
            ),
            dtype="float32",
        )
        x_embed = ops.cast(
            ops.repeat(
                ops.expand_dims(ops.arange(1, w + 1, dtype="float32"), axis=0),
                h,
                axis=0,
            ),
            dtype="float32",
        )

        if self.normalize:
            y_embed = y_embed / (y_embed[-1:, :] + self.eps) * 2 * math.pi
            x_embed = x_embed / (x_embed[:, -1:] + self.eps) * 2 * math.pi

        dim_t = ops.arange(self.num_pos_feats, dtype="float32")
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = ops.expand_dims(x_embed, axis=-1) / dim_t
        pos_y = ops.expand_dims(y_embed, axis=-1) / dim_t

        pos_x_sin = ops.sin(pos_x[:, :, 0::2])
        pos_x_cos = ops.cos(pos_x[:, :, 1::2])
        pos_x = ops.reshape(
            ops.stack([pos_x_sin, pos_x_cos], axis=-1),
            [h, w, self.num_pos_feats],
        )

        pos_y_sin = ops.sin(pos_y[:, :, 0::2])
        pos_y_cos = ops.cos(pos_y[:, :, 1::2])
        pos_y = ops.reshape(
            ops.stack([pos_y_sin, pos_y_cos], axis=-1),
            [h, w, self.num_pos_feats],
        )

        pos = ops.concatenate([pos_y, pos_x], axis=-1)
        pos = ops.expand_dims(pos, axis=0)
        pos = ops.broadcast_to(pos, [batch_size, h, w, self.hidden_dim])

        if data_format == "channels_first":
            pos = ops.transpose(pos, [0, 3, 1, 2])

        return pos

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "temperature": self.temperature,
                "normalize": self.normalize,
                "eps": self.eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DETRMultiHeadAttention(layers.Layer):
    """Multi-head attention layer for the DETR transformer.

    Implements scaled dot-product multi-head attention with separate
    query, key, and value projections followed by an output projection.
    The projection naming matches the reference DETR layout to
    simplify weight transfer from pretrained models. Used in both the
    encoder (self-attention) and decoder (self-attention and
    cross-attention) stages of DETR.

    Reference:
    - [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

    Args:
        hidden_dim: Integer, total model dimension. Must be divisible
            by `num_heads`.
        num_heads: Integer, number of parallel attention heads.
        dropout_rate: Float, dropout rate applied to the attention
            weight matrix. Defaults to `0.0`.
        block_prefix: String, name prefix for the internal dense
            layers (`q_proj`, `k_proj`, `v_proj`, `out_proj`).
            Defaults to `""`.
        **kwargs: Additional keyword arguments passed to the `Layer`
            class.

    Input Shape:
        Three 3D tensors:
        - `query`: `(batch_size, seq_len_q, hidden_dim)`
        - `key`:   `(batch_size, seq_len_k, hidden_dim)`
        - `value`: `(batch_size, seq_len_k, hidden_dim)`

    Output Shape:
        3D tensor: `(batch_size, seq_len_q, hidden_dim)`.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        dropout_rate=0.0,
        block_prefix="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout_rate = dropout_rate
        self.block_prefix = block_prefix

        self.q_proj = layers.Dense(
            hidden_dim,
            name=f"{block_prefix}_q_proj",
        )
        self.k_proj = layers.Dense(
            hidden_dim,
            name=f"{block_prefix}_k_proj",
        )
        self.v_proj = layers.Dense(
            hidden_dim,
            name=f"{block_prefix}_v_proj",
        )
        self.out_proj = layers.Dense(
            hidden_dim,
            name=f"{block_prefix}_out_proj",
        )
        self.attn_dropout = layers.Dropout(dropout_rate)

    def call(self, query, key, value, training=None):
        batch_size = ops.shape(query)[0]
        seq_len_q = ops.shape(query)[1]
        seq_len_k = ops.shape(key)[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = ops.reshape(q, [batch_size, seq_len_q, self.num_heads, self.head_dim])
        k = ops.reshape(k, [batch_size, seq_len_k, self.num_heads, self.head_dim])
        v = ops.reshape(v, [batch_size, seq_len_k, self.num_heads, self.head_dim])

        q = ops.transpose(q, [0, 2, 1, 3])
        k = ops.transpose(k, [0, 2, 1, 3])
        v = ops.transpose(v, [0, 2, 1, 3])

        attn_weights = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2])) * self.scale
        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        attn_output = ops.matmul(attn_weights, v)
        attn_output = ops.transpose(attn_output, [0, 2, 1, 3])
        attn_output = ops.reshape(attn_output, [batch_size, seq_len_q, self.hidden_dim])
        attn_output = self.out_proj(attn_output)

        return attn_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "block_prefix": self.block_prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DETRMHAttentionMap(layers.Layer):
    """Multi-head attention map between decoder queries and encoder features.

    Computes per-head, per-query attention weights over the encoder's
    spatial feature map: used by :class:`DETRPanopticSegment` as the seed for
    the mask head. Matches the reference ``DetrMHAttentionMap`` weights
    and forward semantics.

    Args:
        hidden_dim: Query / key projection dimension. Must equal the
            DETR transformer ``hidden_dim``.
        num_heads: Attention head count. Must divide ``hidden_dim``.
        **kwargs: Additional keyword arguments passed to the `Layer`
            class.

    Input Shapes:
        ``q``: ``(B, num_queries, hidden_dim)``, decoder hidden states.
        ``k``: ``(B, H, W, hidden_dim)``, projected encoder feature map
        in channels-last form (after the 1×1 ``input_projection`` and
        reshape from the encoder's flattened output).

    Output Shape:
        5D tensor ``(B, num_queries, num_heads, H, W)``: softmax-
        normalized over the joint ``num_heads × H × W`` axis per query,
        matching the reference behaviour exactly.
    """

    def __init__(self, hidden_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_linear = layers.Dense(hidden_dim, name="q_linear")
        self.k_linear = layers.Dense(hidden_dim, name="k_linear")

    def call(self, q, k):
        b = ops.shape(q)[0]
        num_queries = ops.shape(q)[1]
        h = ops.shape(k)[1]
        w = ops.shape(k)[2]

        q_proj = self.q_linear(q)
        k_proj = self.k_linear(k)

        q_proj = ops.reshape(q_proj, (b, num_queries, self.num_heads, self.head_dim))
        k_proj = ops.reshape(k_proj, (b, h, w, self.num_heads, self.head_dim))

        weights = ops.einsum("bqnc,bhwnc->bqnhw", q_proj * self.scale, k_proj)

        weights_flat = ops.reshape(weights, (b, num_queries, self.num_heads * h * w))
        weights = ops.softmax(weights_flat, axis=-1)
        weights = ops.reshape(weights, (b, num_queries, self.num_heads, h, w))
        return weights

    def compute_output_spec(self, q, k):
        return keras.KerasTensor(
            (q.shape[0], q.shape[1], self.num_heads, k.shape[1], k.shape[2]),
            dtype=q.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DETRMaskHeadSmallConv(layers.Layer):
    """Small convolutional FPN-style mask head used by DETR segmentation.

    Fuses the projected encoder feature map and the per-query
    attention maps from :class:`DETRMHAttentionMap` with multi-scale
    backbone features (C4/C3/C2) through three nearest-neighbour
    upsampling stages, producing a per-query mask logit at stride 4.

    Reproduces the ``DetrMaskHeadSmallConv`` exactly: five
    ``Conv2D + GroupNorm + ReLU`` stages, three 1×1 ``adapter`` convs
    that align backbone channel counts, and a final ``out_lay`` 1-
    channel conv. The intermediate channel widths are
    ``[dim, context_dim/2, context_dim/4, context_dim/8,
    context_dim/16, context_dim/64]``.

    Args:
        dim: Number of input channels, ``hidden_dim + num_heads``.
            Must be divisible by 8 (the ``GroupNormalization`` group
            count for the first stage).
        fpn_dims: Three integers ``(C4_channels, C3_channels,
            C2_channels)`` of the backbone stages that will be fused
            in order. For ResNet-50 / ResNet-101 this is
            ``(1024, 512, 256)``.
        context_dim: DETR transformer ``hidden_dim``. Drives the
            ``inter_dims`` channel sequence.
        **kwargs: Additional keyword arguments passed to the `Layer`
            class.

    Input Shapes:
        ``x``: ``(B, H, W, context_dim)``, projected encoder feature
        map reshaped back to spatial form.
        ``bbox_mask``: ``(B, num_queries, num_heads, H, W)``, output of
        :class:`DETRMHAttentionMap`.
        ``fpns``: list of three 4D tensors at strides 16, 8, 4 in
        channels-last form.

    Output Shape:
        4D tensor ``(B, num_queries, H_out, W_out)`` where ``H_out`` /
        ``W_out`` match the C2 (stride-4) backbone feature map.
    """

    def __init__(self, dim, fpn_dims, context_dim, **kwargs):
        super().__init__(**kwargs)
        if dim % 8 != 0:
            raise ValueError(
                "Mask head 'dim' (hidden_dim + num_heads) must be divisible by "
                f"8 for GroupNorm. Received: dim={dim}."
            )
        self.dim = dim
        self.fpn_dims = tuple(fpn_dims)
        self.context_dim = context_dim

        inter_dims = [
            dim,
            context_dim // 2,
            context_dim // 4,
            context_dim // 8,
            context_dim // 16,
            context_dim // 64,
        ]
        self.inter_dims = tuple(inter_dims)

        # The mask head always operates on channels-last tensors (the panoptic
        # model permutes every input to channels-last before calling it), so the
        # convs are pinned to channels_last regardless of the global image format.
        self.lay1 = layers.Conv2D(
            dim, 3, padding="same", data_format="channels_last", name="lay1"
        )
        self.gn1 = layers.GroupNormalization(
            groups=8, axis=-1, epsilon=1e-5, name="gn1"
        )
        self.lay2 = layers.Conv2D(
            inter_dims[1], 3, padding="same", data_format="channels_last", name="lay2"
        )
        self.gn2 = layers.GroupNormalization(
            groups=min(8, inter_dims[1]), axis=-1, epsilon=1e-5, name="gn2"
        )
        self.lay3 = layers.Conv2D(
            inter_dims[2], 3, padding="same", data_format="channels_last", name="lay3"
        )
        self.gn3 = layers.GroupNormalization(
            groups=min(8, inter_dims[2]), axis=-1, epsilon=1e-5, name="gn3"
        )
        self.lay4 = layers.Conv2D(
            inter_dims[3], 3, padding="same", data_format="channels_last", name="lay4"
        )
        self.gn4 = layers.GroupNormalization(
            groups=min(8, inter_dims[3]), axis=-1, epsilon=1e-5, name="gn4"
        )
        self.lay5 = layers.Conv2D(
            inter_dims[4], 3, padding="same", data_format="channels_last", name="lay5"
        )
        self.gn5 = layers.GroupNormalization(
            groups=min(8, inter_dims[4]), axis=-1, epsilon=1e-5, name="gn5"
        )
        self.out_lay = layers.Conv2D(
            1, 3, padding="same", data_format="channels_last", name="out_lay"
        )

        self.adapter1 = layers.Conv2D(
            inter_dims[1], 1, data_format="channels_last", name="adapter1"
        )
        self.adapter2 = layers.Conv2D(
            inter_dims[2], 1, data_format="channels_last", name="adapter2"
        )
        self.adapter3 = layers.Conv2D(
            inter_dims[3], 1, data_format="channels_last", name="adapter3"
        )

    def call(self, x, bbox_mask, fpns):
        b = ops.shape(x)[0]
        h = ops.shape(x)[1]
        w = ops.shape(x)[2]
        q_dim = ops.shape(bbox_mask)[1]

        x = ops.expand_dims(x, axis=1)
        x = ops.tile(x, (1, q_dim, 1, 1, 1))
        x = ops.reshape(x, (-1, h, w, self.context_dim))

        bbox_mask = ops.transpose(bbox_mask, (0, 1, 3, 4, 2))
        bbox_mask = ops.reshape(bbox_mask, (-1, h, w, self.dim - self.context_dim))

        x = ops.concatenate([x, bbox_mask], axis=-1)

        x = ops.relu(self.gn1(self.lay1(x)))
        x = ops.relu(self.gn2(self.lay2(x)))

        x = self._merge_fpn(x, self.adapter1(fpns[0]), b, q_dim)
        x = ops.relu(self.gn3(self.lay3(x)))

        x = self._merge_fpn(x, self.adapter2(fpns[1]), b, q_dim)
        x = ops.relu(self.gn4(self.lay4(x)))

        x = self._merge_fpn(x, self.adapter3(fpns[2]), b, q_dim)
        x = ops.relu(self.gn5(self.lay5(x)))

        x = self.out_lay(x)

        out_h = ops.shape(x)[1]
        out_w = ops.shape(x)[2]
        x = ops.reshape(x, (b, q_dim, out_h, out_w))
        return x

    def _merge_fpn(self, x, fpn, batch, q_dim):
        fpn_h = ops.shape(fpn)[1]
        fpn_w = ops.shape(fpn)[2]
        fpn_c = fpn.shape[-1]

        fpn = ops.expand_dims(fpn, axis=1)
        fpn = ops.tile(fpn, (1, q_dim, 1, 1, 1))
        fpn = ops.reshape(fpn, (-1, fpn_h, fpn_w, fpn_c))

        x_resized = ops.image.resize(
            x,
            size=(fpn_h, fpn_w),
            interpolation="nearest",
            data_format="channels_last",
        )
        return fpn + x_resized

    def compute_output_spec(self, x, bbox_mask, fpns):
        return keras.KerasTensor(
            (
                x.shape[0],
                bbox_mask.shape[1],
                fpns[-1].shape[1],
                fpns[-1].shape[2],
            ),
            dtype=x.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "fpn_dims": list(self.fpn_dims),
                "context_dim": self.context_dim,
            }
        )
        return config
