import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="zeromodels")
class MaxViTWindowPartition(layers.Layer):
    """Partition a spatial tensor into non-overlapping local windows.

    Rearranges a 4D spatial tensor into a batch of contiguous windows that can
    be processed independently by attention layers. Each window contains a
    ``wh x ww`` patch of spatially adjacent pixels.

    Supports both ``channels_last`` and ``channels_first`` inputs. The output
    is always ``channels_last`` so that downstream Dense / attention layers
    operate on the channel dimension directly.

    Args:
        window_size: Tuple ``(wh, ww)`` or int for the window dimensions.
            Spatial dimensions of the input must be divisible by the window
            size.
        data_format: ``"channels_last"`` or ``"channels_first"``.
            Defaults to ``"channels_last"``.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    Input Shape:
        - ``channels_last``: ``(batch_size, height, width, channels)``
        - ``channels_first``: ``(batch_size, channels, height, width)``

    Output Shape:
        4D tensor (always channels_last):
        ``(batch_size * nH * nW, wh, ww, channels)``
        where ``nH = height // wh`` and ``nW = width // ww``.
    """

    def __init__(self, window_size, data_format="channels_last", **kwargs):
        super().__init__(**kwargs)
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        self.window_size = window_size
        self.data_format = data_format

    def call(self, x):
        if self.data_format == "channels_first":
            x = ops.transpose(x, [0, 2, 3, 1])
        wh, ww = self.window_size
        x_shape = ops.shape(x)
        B, H, W, C = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        nH = H // wh
        nW = W // ww
        x = ops.reshape(x, [B, nH, wh, nW, ww, C])
        x = ops.transpose(x, [0, 1, 3, 2, 4, 5])
        x = ops.reshape(x, [B * nH * nW, wh, ww, C])
        return x

    def compute_output_shape(self, input_shape):
        wh, ww = self.window_size
        if self.data_format == "channels_first":
            C = input_shape[1]
        else:
            C = input_shape[-1]
        return (None, wh, ww, C)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"window_size": self.window_size, "data_format": self.data_format}
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MaxViTWindowReverse(layers.Layer):
    """Reverse window partition back to a spatial tensor.

    Inverse of ``MaxViTWindowPartition``. Reassembles a batch of contiguous
    windows back into the original spatial layout. The input is always
    ``channels_last`` (produced by the partition layer), and the output format
    matches the configured ``data_format``.

    Args:
        window_size: Tuple ``(wh, ww)`` or int for the window dimensions.
        img_size: Tuple ``(H, W)`` of the original spatial dimensions before
            partitioning.
        data_format: ``"channels_last"`` or ``"channels_first"``.
            Defaults to ``"channels_last"``.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    Input Shape:
        4D tensor (always channels_last):
        ``(batch_size * nH * nW, wh, ww, channels)``.

    Output Shape:
        - ``channels_last``: ``(batch_size, H, W, channels)``
        - ``channels_first``: ``(batch_size, channels, H, W)``
    """

    def __init__(self, window_size, img_size, data_format="channels_last", **kwargs):
        super().__init__(**kwargs)
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        self.window_size = window_size
        self.img_size = img_size
        self.data_format = data_format

    def call(self, windows):
        wh, ww = self.window_size
        H, W = self.img_size
        C = ops.shape(windows)[-1]
        nH = H // wh
        nW = W // ww
        total_windows = ops.shape(windows)[0]
        B = total_windows // (nH * nW)
        x = ops.reshape(windows, [B, nH, nW, wh, ww, C])
        x = ops.transpose(x, [0, 1, 3, 2, 4, 5])
        x = ops.reshape(x, [B, H, W, C])
        if self.data_format == "channels_first":
            x = ops.transpose(x, [0, 3, 1, 2])
        return x

    def compute_output_shape(self, input_shape):
        H, W = self.img_size
        C = input_shape[-1]
        if self.data_format == "channels_first":
            return (None, C, H, W)
        return (None, H, W, C)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "window_size": self.window_size,
                "img_size": self.img_size,
                "data_format": self.data_format,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MaxViTGridPartition(layers.Layer):
    """Partition a spatial tensor into dilated (grid) windows.

    Unlike ``MaxViTWindowPartition`` which extracts contiguous spatial blocks,
    grid partition samples every ``gh``-th / ``gw``-th element along the
    height / width axes. This produces windows whose pixels are uniformly
    spaced apart in the original image, enabling the subsequent attention
    layer to capture long-range global dependencies.

    Supports both ``channels_last`` and ``channels_first`` inputs. The output
    is always ``channels_last``.

    Args:
        grid_size: Tuple ``(gh, gw)`` or int for the grid dimensions.
            Spatial dimensions of the input must be divisible by the grid
            size.
        data_format: ``"channels_last"`` or ``"channels_first"``.
            Defaults to ``"channels_last"``.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    Input Shape:
        - ``channels_last``: ``(batch_size, height, width, channels)``
        - ``channels_first``: ``(batch_size, channels, height, width)``

    Output Shape:
        4D tensor (always channels_last):
        ``(batch_size * nH * nW, gh, gw, channels)``
        where ``nH = height // gh`` and ``nW = width // gw``.
    """

    def __init__(self, grid_size, data_format="channels_last", **kwargs):
        super().__init__(**kwargs)
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)
        self.grid_size = grid_size
        self.data_format = data_format

    def call(self, x):
        if self.data_format == "channels_first":
            x = ops.transpose(x, [0, 2, 3, 1])
        gh, gw = self.grid_size
        x_shape = ops.shape(x)
        B, H, W, C = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        nH = H // gh
        nW = W // gw
        x = ops.reshape(x, [B, gh, nH, gw, nW, C])
        x = ops.transpose(x, [0, 2, 4, 1, 3, 5])
        x = ops.reshape(x, [B * nH * nW, gh, gw, C])
        return x

    def compute_output_shape(self, input_shape):
        gh, gw = self.grid_size
        if self.data_format == "channels_first":
            C = input_shape[1]
        else:
            C = input_shape[-1]
        return (None, gh, gw, C)

    def get_config(self):
        config = super().get_config()
        config.update({"grid_size": self.grid_size, "data_format": self.data_format})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MaxViTGridReverse(layers.Layer):
    """Reverse grid partition back to a spatial tensor.

    Inverse of ``MaxViTGridPartition``. Reassembles a batch of dilated grid
    windows back into the original spatial layout. The input is always
    ``channels_last`` (produced by the partition layer), and the output format
    matches the configured ``data_format``.

    Args:
        grid_size: Tuple ``(gh, gw)`` or int for the grid dimensions.
        img_size: Tuple ``(H, W)`` of the original spatial dimensions before
            partitioning.
        data_format: ``"channels_last"`` or ``"channels_first"``.
            Defaults to ``"channels_last"``.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    Input Shape:
        4D tensor (always channels_last):
        ``(batch_size * nH * nW, gh, gw, channels)``.

    Output Shape:
        - ``channels_last``: ``(batch_size, H, W, channels)``
        - ``channels_first``: ``(batch_size, channels, H, W)``
    """

    def __init__(self, grid_size, img_size, data_format="channels_last", **kwargs):
        super().__init__(**kwargs)
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)
        self.grid_size = grid_size
        self.img_size = img_size
        self.data_format = data_format

    def call(self, windows):
        gh, gw = self.grid_size
        H, W = self.img_size
        C = ops.shape(windows)[-1]
        nH = H // gh
        nW = W // gw
        total_windows = ops.shape(windows)[0]
        B = total_windows // (nH * nW)
        x = ops.reshape(windows, [B, nH, nW, gh, gw, C])
        x = ops.transpose(x, [0, 3, 1, 4, 2, 5])
        x = ops.reshape(x, [B, H, W, C])
        if self.data_format == "channels_first":
            x = ops.transpose(x, [0, 3, 1, 2])
        return x

    def compute_output_shape(self, input_shape):
        H, W = self.img_size
        C = input_shape[-1]
        if self.data_format == "channels_first":
            return (None, C, H, W)
        return (None, H, W, C)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "grid_size": self.grid_size,
                "img_size": self.img_size,
                "data_format": self.data_format,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MaxViTRelPosBias(layers.Layer):
    """Learnable relative position bias for MaxViT attention.

    Maintains a learnable bias table of shape ``(num_heads, 2*wh-1, 2*ww-1)``
    and reindexes it at every forward pass into a dense bias matrix of shape
    ``(num_heads, window_area, window_area)`` using one-hot lookup tensors.
    This matches the TensorFlow-style log-spaced implementation from the
    original MaxViT codebase.

    The layer supports loading weights from different window sizes through
    bicubic interpolation of the bias table, making it flexible for transfer
    learning across different input resolutions. When loading weights, it
    automatically resizes the bias table if the source and target window sizes
    differ.

    Args:
        window_size: Tuple ``(wh, ww)`` or int for the attention window
            dimensions.
        num_heads: Number of attention heads.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    Input Shape:
        4D attention logit tensor:
        ``(batch_size, num_heads, window_area, window_area)``.

    Output Shape:
        Same as input (bias is added element-wise).
    """

    def __init__(self, window_size, num_heads, **kwargs):
        super().__init__(**kwargs)
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        self.window_area = window_size[0] * window_size[1]
        self.vocab_height = 2 * window_size[0] - 1
        self.vocab_width = 2 * window_size[1] - 1

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            name="relative_position_bias_table",
            shape=(self.num_heads, self.vocab_height, self.vocab_width),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        self.built = True

    def _generate_lookup(self, length, max_rel_pos):
        vocab_size = 2 * max_rel_pos + 1
        indices = []
        for i in range(length):
            for x in range(length):
                v = x - i + max_rel_pos
                if abs(x - i) <= max_rel_pos:
                    indices.append([i, x, v])
        indices_array = ops.convert_to_tensor(indices, dtype="int32")
        ones = ops.ones((len(indices),))
        lookup = ops.scatter_update(
            ops.zeros((length, length, vocab_size)),
            indices_array,
            ones,
        )
        return lookup

    def call(self, attn):
        height_lookup = self._generate_lookup(
            self.window_size[0], self.window_size[0] - 1
        )
        width_lookup = self._generate_lookup(
            self.window_size[1], self.window_size[1] - 1
        )

        reindexed = ops.einsum(
            "nhw,ixh->nixw", self.relative_position_bias_table, height_lookup
        )
        reindexed = ops.einsum("nixw,jyw->nijxy", reindexed, width_lookup)
        bias = ops.reshape(
            reindexed, (self.num_heads, self.window_area, self.window_area)
        )
        return attn + bias

    def save_own_variables(self, store):
        super().save_own_variables(store)
        store["window_size_h"] = self.window_size[0]
        store["window_size_w"] = self.window_size[1]

    def load_own_variables(self, store):
        source_h = int(store["window_size_h"][...])
        source_w = int(store["window_size_w"][...])

        if source_h == self.window_size[0] and source_w == self.window_size[1]:
            self.relative_position_bias_table.assign(store["0"])
            return

        # Interpolate bias table from source size to target size
        # Shape: (num_heads, 2*src_h-1, 2*src_w-1) -> (num_heads, 2*tgt_h-1, 2*tgt_w-1)
        bias = store["0"]
        bias = ops.cast(bias, "float32")
        # Add batch dim for resize: (num_heads, H, W) -> (num_heads, H, W, 1)
        bias = ops.expand_dims(bias, axis=-1)
        bias = ops.image.resize(
            bias,
            size=(self.vocab_height, self.vocab_width),
            interpolation="bicubic",
            antialias=True,
        )
        bias = ops.squeeze(bias, axis=-1)
        self.relative_position_bias_table.assign(bias)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "window_size": self.window_size,
                "num_heads": self.num_heads,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MaxViTAttention(layers.Layer):
    """Multi-head self-attention with relative position bias for MaxViT.

    Implements scaled dot-product attention with a fused QKV projection and
    learnable ``MaxViTRelPosBias`` added to the attention logits before softmax.
    The layer operates on channels-last ``(B, N, C)`` sequences; upstream
    partition layers handle any necessary spatial format conversion.

    Key Features:
        - Fused query/key/value projection via a single Dense layer.
        - Learnable relative position bias supporting cross-resolution weight
          loading via bicubic interpolation.
        - Configurable attention and projection dropout.

    Args:
        dim: Total input / output feature dimension. Must be divisible by
            ``num_heads``.
        num_heads: Number of parallel attention heads.
        window_size: Spatial window size used for the relative position bias
            table. Int or tuple ``(wh, ww)``. Defaults to ``7``.
        attn_drop: Dropout rate applied to attention weights. Defaults to ``0.0``.
        proj_drop: Dropout rate applied after the output projection. Defaults
            to ``0.0``.
        prefix: String prefix prepended to sub-layer names. Defaults to ``""``.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.

    Input Shape:
        3D tensor: ``(batch_size, sequence_length, dim)``.

    Output Shape:
        Same as input: ``(batch_size, sequence_length, dim)``.
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        attn_drop=0.0,
        proj_drop=0.0,
        prefix="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = self.dim_head**-0.5
        self.window_size = (
            window_size
            if isinstance(window_size, tuple)
            else (window_size, window_size)
        )

        self.qkv = layers.Dense(
            dim * 3,
            use_bias=True,
            name=prefix + "attn_qkv",
        )
        self.rel_pos = MaxViTRelPosBias(
            window_size=self.window_size,
            num_heads=num_heads,
            name=prefix + "attn_rel_pos",
        )
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(
            dim,
            use_bias=True,
            name=prefix + "attn_proj",
        )
        self.proj_drop = layers.Dropout(proj_drop)

    def call(self, x, training=None):
        input_shape = ops.shape(x)
        B = input_shape[0]

        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, [B, -1, 3, self.num_heads, self.dim_head])
        qkv = ops.transpose(qkv, [0, 3, 2, 1, 4])
        q = qkv[:, :, 0, :, :]
        k = qkv[:, :, 1, :, :]
        v = qkv[:, :, 2, :, :]

        q = q * self.scale
        attn = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2]))
        attn = self.rel_pos(attn)
        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x = ops.matmul(attn, v)
        x = ops.transpose(x, [0, 2, 1, 3])
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
                "window_size": self.window_size,
                "attn_drop": self.attn_drop.rate,
                "proj_drop": self.proj_drop.rate,
            }
        )
        return config
