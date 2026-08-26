import keras
import numpy as np
from keras import layers, ops


def window_partition(x, window_size):
    """Partition a feature-map grid into non-overlapping square windows.

    Args:
        x: Feature map of shape ``(b, h, w, c)`` in channels-last layout;
            ``h`` and ``w`` must be divisible by ``window_size``.
        window_size: Edge length of each square window.

    Returns:
        Tensor of shape ``(b * num_windows, window_size * window_size, c)``:
        each window flattened to a token sequence for windowed self-attention.
    """
    b = ops.shape(x)[0]
    h = ops.shape(x)[1]
    w = ops.shape(x)[2]
    c = x.shape[-1]
    x = ops.reshape(
        x, (b, h // window_size, window_size, w // window_size, window_size, c)
    )
    windows = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    return ops.reshape(windows, (-1, window_size * window_size, c))


def window_reverse(windows, window_size, h, w):
    """Reassemble windowed tokens back into a feature-map grid.

    Inverse of :func:`window_partition`.

    Args:
        windows: Tensor of shape
            ``(b * num_windows, window_size * window_size, c)``.
        window_size: Edge length of each square window.
        h: Height of the reconstructed grid.
        w: Width of the reconstructed grid.

    Returns:
        Feature map of shape ``(b, h, w, c)`` in channels-last layout.
    """
    c = windows.shape[-1]
    b = ops.shape(windows)[0] // ((h // window_size) * (w // window_size))
    x = ops.reshape(
        windows,
        (b, h // window_size, w // window_size, window_size, window_size, c),
    )
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    return ops.reshape(x, (b, h, w, c))


@keras.saving.register_keras_serializable(package="zeromodels")
class Mask2FormerSwinPatchEmbeddings(layers.Layer):
    """Patch-embedding stem: a strided convolution flattened to tokens.

    Projects the input image with a ``patch_size`` strided convolution and
    flattens the spatial grid to a ``(b, h*w, embed_dim)`` token sequence:
    the layout the windowed attention operates on. The convolution honours the
    configured data format; a channels-first output is transposed to
    channels-last before flattening.

    Args:
        embed_dim: Token embedding dimension produced by the projection.
        patch_size: Convolution kernel size and stride (the patch edge length).
        data_format: ``"channels_last"`` or ``"channels_first"``. Defaults to
            ``keras.config.image_data_format()``.
        **kwargs: Additional keyword arguments forwarded to ``keras.layers.Layer``.
    """

    def __init__(self, embed_dim, patch_size=4, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.data_format = data_format or keras.config.image_data_format()
        # The conv honours the data format; the windowed attention that follows
        # operates on flattened (b, h*w, c) tokens, so the conv output is
        # converted to a channels-last grid before flattening.
        self.projection = layers.Conv2D(
            embed_dim,
            patch_size,
            strides=patch_size,
            data_format=self.data_format,
            name="projection",
        )

    def call(self, pixel_values):
        x = self.projection(pixel_values)
        if self.data_format == "channels_first":
            x = ops.transpose(x, (0, 2, 3, 1))  # (b, c, h, w) -> (b, h, w, c)
        b = ops.shape(x)[0]
        h = ops.shape(x)[1]
        w = ops.shape(x)[2]
        x = ops.reshape(x, (b, h * w, self.embed_dim))
        return x, h, w

    def get_config(self):
        c = super().get_config()
        c.update(
            {
                "embed_dim": self.embed_dim,
                "patch_size": self.patch_size,
                "data_format": self.data_format,
            }
        )
        return c


@keras.saving.register_keras_serializable(package="zeromodels")
class Mask2FormerSwinPatchMerging(layers.Layer):
    """Patch-merging downsample between stages.

    Gathers each 2x2 neighbourhood of tokens, concatenates them along the
    channel axis (``4C``), applies LayerNorm, then a bias-free linear
    projection to ``2C``, halving the spatial resolution while doubling the
    channel count.

    Args:
        dim: Input channel dimension ``C`` (the output dimension is ``2C``).
        **kwargs: Additional keyword arguments forwarded to ``keras.layers.Layer``.
    """

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.reduction = layers.Dense(2 * dim, use_bias=False, name="reduction")
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="norm")

    def call(self, x, h=None, w=None):
        b = ops.shape(x)[0]
        c = x.shape[-1]
        x = ops.reshape(x, (b, h, w, c))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = ops.concatenate([x0, x1, x2, x3], axis=-1)
        new_h = h // 2
        new_w = w // 2
        x = ops.reshape(x, (b, new_h * new_w, 4 * c))
        x = self.norm(x)
        x = self.reduction(x)
        return x, new_h, new_w

    def get_config(self):
        c = super().get_config()
        c.update({"dim": self.dim})
        return c


@keras.saving.register_keras_serializable(package="zeromodels")
class Mask2FormerSwinSelfAttention(layers.Layer):
    """Windowed multi-head self-attention with relative position bias.

    Computes self-attention within each window using separate query/key/value
    projections plus a learned relative-position bias
    table. An optional additive mask supports shifted-window attention.

    Args:
        dim: Input/output feature dimension.
        num_heads: Number of attention heads (``dim`` must be divisible by it).
        window_size: Edge length of the attention window.
        **kwargs: Additional keyword arguments forwarded to ``keras.layers.Layer``.
    """

    def __init__(self, dim, num_heads, window_size, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.query = layers.Dense(dim, name="query")
        self.key = layers.Dense(dim, name="key")
        self.value = layers.Dense(dim, name="value")

    def build(self, input_shape):
        ws = self.window_size
        self.relative_position_bias_table = self.add_weight(
            name="relative_position_bias_table",
            shape=((2 * ws - 1) * (2 * ws - 1), self.num_heads),
            initializer="zeros",
            trainable=True,
        )
        coords_h = np.arange(ws)
        coords_w = np.arange(ws)
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = coords.reshape(2, -1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = np.transpose(rel, (1, 2, 0))
        rel[..., 0] += ws - 1
        rel[..., 1] += ws - 1
        rel[..., 0] *= 2 * ws - 1
        idx = rel.sum(-1).astype(np.int32)
        self.relative_position_index = self.add_weight(
            name="relative_position_index",
            shape=(ws * ws, ws * ws),
            initializer=keras.initializers.Constant(idx),
            trainable=False,
            dtype="int32",
        )
        super().build(input_shape)

    def call(self, x, attention_mask=None):
        b = ops.shape(x)[0]
        n = ops.shape(x)[1]
        q = ops.reshape(self.query(x), (b, n, self.num_heads, self.head_dim))
        k = ops.reshape(self.key(x), (b, n, self.num_heads, self.head_dim))
        v = ops.reshape(self.value(x), (b, n, self.num_heads, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        ws = self.window_size
        idx_flat = ops.reshape(self.relative_position_index, (-1,))
        rpb = ops.take(self.relative_position_bias_table, idx_flat, axis=0)
        rpb = ops.reshape(rpb, (ws * ws, ws * ws, self.num_heads))
        rpb = ops.transpose(rpb, (2, 0, 1))
        attn = attn + ops.expand_dims(rpb, axis=0)

        if attention_mask is not None:
            nW = ops.shape(attention_mask)[0]
            attn = ops.reshape(attn, (b // nW, nW, self.num_heads, n, n))
            attn = attn + ops.reshape(attention_mask, (1, nW, 1, n, n))
            attn = ops.reshape(attn, (b, self.num_heads, n, n))

        attn = ops.softmax(attn, axis=-1)
        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, n, self.dim))
        return out

    def get_config(self):
        c = super().get_config()
        c.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
            }
        )
        return c


@keras.saving.register_keras_serializable(package="zeromodels")
class Mask2FormerSwinAttention(layers.Layer):
    """Windowed self-attention followed by an output projection.

    Thin wrapper composing :class:`Mask2FormerSwinSelfAttention` with the
    block's dense ``output`` projection.

    Args:
        dim: Input/output feature dimension.
        num_heads: Number of attention heads.
        window_size: Edge length of the attention window.
        **kwargs: Additional keyword arguments forwarded to ``keras.layers.Layer``.
    """

    def __init__(self, dim, num_heads, window_size, **kwargs):
        super().__init__(**kwargs)
        self.self_attn = Mask2FormerSwinSelfAttention(
            dim, num_heads, window_size, name="self"
        )
        self.output_dense = layers.Dense(dim, name="output_dense")

    def call(self, x, attention_mask=None):
        x = self.self_attn(x, attention_mask=attention_mask)
        return self.output_dense(x)


def get_shift_mask(h, w, window_size, shift_size):
    """Build the additive attention mask for shifted-window attention.

    Assigns a region id to every location of the cyclically-shifted ``(h, w)``
    grid, partitions it into windows, and produces a per-window mask that is
    ``0`` between tokens of the same region and ``-100`` between tokens of
    different regions, so cross-region attention is suppressed.

    Args:
        h: Padded feature-map height.
        w: Padded feature-map width.
        window_size: Edge length of each attention window.
        shift_size: Cyclic-shift offset used for this block.

    Returns:
        Additive mask of shape
        ``(num_windows, window_size**2, window_size**2)``.
    """
    # Region id (0..8) per location matching the cyclic-shift partition, built
    # with ops so symbolic h/w (e.g. traced call-kwargs on JAX) don't break. The
    # absolute ids are arbitrary: only same/different region matters below.
    hi = ops.arange(h)
    wi = ops.arange(w)
    h_idx = ops.cast(hi >= h - window_size, "int32") + ops.cast(
        hi >= h - shift_size, "int32"
    )
    w_idx = ops.cast(wi >= w - window_size, "int32") + ops.cast(
        wi >= w - shift_size, "int32"
    )
    region = h_idx[:, None] * 3 + w_idx[None, :]
    img_mask = ops.cast(region, "float32")[None, :, :, None]
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = ops.squeeze(mask_windows, axis=-1)
    attn_mask = ops.expand_dims(mask_windows, axis=1) - ops.expand_dims(
        mask_windows, axis=2
    )
    attn_mask = ops.where(ops.cast(attn_mask, "bool"), ops.cast(-100.0, "float32"), 0.0)
    return attn_mask


@keras.saving.register_keras_serializable(package="zeromodels")
class Mask2FormerSwinBlock(layers.Layer):
    """A single Swin Transformer block.

    Pre-norm (shifted) windowed self-attention with a residual connection,
    followed by a pre-norm two-layer GELU MLP with a residual connection. The
    token grid is padded to a multiple of ``window_size``, optionally
    cyclically shifted, partitioned into windows for attention, then reversed
    (and un-shifted / un-padded).

    Args:
        dim: Token feature dimension.
        num_heads: Number of attention heads.
        window_size: Edge length of the attention window.
        shift_size: Cyclic-shift offset, ``0`` for a regular block,
            ``window_size // 2`` for a shifted block.
        mlp_ratio: Hidden-dimension expansion ratio of the MLP.
        **kwargs: Additional keyword arguments forwarded to ``keras.layers.Layer``.
    """

    def __init__(
        self, dim, num_heads, window_size, shift_size, mlp_ratio=4.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.layernorm_before = layers.LayerNormalization(
            epsilon=1e-5, name="layernorm_before"
        )
        self.attention = Mask2FormerSwinAttention(
            dim, num_heads, window_size, name="attention"
        )
        self.layernorm_after = layers.LayerNormalization(
            epsilon=1e-5, name="layernorm_after"
        )
        self.intermediate_dense = layers.Dense(
            int(dim * mlp_ratio), name="intermediate_dense"
        )
        self.output_dense = layers.Dense(dim, name="output_dense")

    def call(self, x, h=None, w=None, attn_mask=None):
        b = ops.shape(x)[0]
        c = x.shape[-1]
        shortcut = x
        x = self.layernorm_before(x)
        x = ops.reshape(x, (b, h, w, c))

        ws = self.window_size
        pad_r = (ws - w % ws) % ws
        pad_b = (ws - h % ws) % ws
        x = ops.pad(x, ((0, 0), (0, pad_b), (0, pad_r), (0, 0)))
        hp = h + pad_b
        wp = w + pad_r

        if self.shift_size > 0:
            shifted = ops.roll(
                x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2)
            )
        else:
            shifted = x

        windows = window_partition(shifted, ws)
        attn_out = self.attention(windows, attention_mask=attn_mask)
        attn_out = ops.reshape(attn_out, (-1, ws, ws, c))
        merged = window_reverse(ops.reshape(attn_out, (-1, ws * ws, c)), ws, hp, wp)
        if self.shift_size > 0:
            merged = ops.roll(
                merged, shift=(self.shift_size, self.shift_size), axis=(1, 2)
            )
        if pad_r > 0 or pad_b > 0:
            merged = merged[:, :h, :w, :]
        merged = ops.reshape(merged, (b, h * w, c))
        x = shortcut + merged

        shortcut2 = x
        y = self.layernorm_after(x)
        y = self.intermediate_dense(y)
        y = ops.gelu(y, approximate=False)
        y = self.output_dense(y)
        return shortcut2 + y

    def get_config(self):
        c = super().get_config()
        c.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "shift_size": self.shift_size,
                "mlp_ratio": self.mlp_ratio,
            }
        )
        return c


@keras.saving.register_keras_serializable(package="zeromodels")
class Mask2FormerSwinStage(layers.Layer):
    """One hierarchical Swin stage: ``depth`` blocks then optional downsample.

    Stacks ``depth`` :class:`Mask2FormerSwinBlock` layers with alternating
    regular / shifted windows, precomputing the shifted-window attention mask
    once per forward call, and optionally appends a
    :class:`Mask2FormerSwinPatchMerging` downsample after the blocks.

    Args:
        dim: Token feature dimension at this stage.
        depth: Number of Swin blocks in the stage.
        num_heads: Number of attention heads.
        window_size: Edge length of the attention window.
        downsample: Whether to append a patch-merging downsample after the blocks.
        **kwargs: Additional keyword arguments forwarded to ``keras.layers.Layer``.
    """

    def __init__(self, dim, depth, num_heads, window_size, downsample, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.do_downsample = downsample

        self.blocks = [
            Mask2FormerSwinBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                name=f"blocks_{i}",
            )
            for i in range(depth)
        ]
        if downsample:
            self.downsample = Mask2FormerSwinPatchMerging(dim, name="downsample")
        else:
            self.downsample = None

    def call(self, x, h=None, w=None):
        ws = self.window_size
        if any(b.shift_size > 0 for b in self.blocks):
            pad_r = (ws - w % ws) % ws
            pad_b = (ws - h % ws) % ws
            hp = h + pad_b
            wp = w + pad_r
            shift_mask = get_shift_mask(hp, wp, ws, ws // 2)
        else:
            shift_mask = None

        for blk in self.blocks:
            mask = shift_mask if blk.shift_size > 0 else None
            x = blk(x, h=h, w=w, attn_mask=mask)
        stage_features = (x, h, w)
        if self.downsample is not None:
            x, h, w = self.downsample(x, h=h, w=w)
        return stage_features, x, h, w

    def get_config(self):
        c = super().get_config()
        c.update(
            {
                "dim": self.dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "downsample": self.do_downsample,
            }
        )
        return c


@keras.saving.register_keras_serializable(package="zeromodels")
class Mask2FormerSwinBackbone(layers.Layer):
    """Hierarchical Swin backbone producing per-stage feature maps.

    Embeds the input image into patch tokens, runs four Swin stages with
    progressive patch merging, and applies a per-stage LayerNorm
    (``hidden_states_norms``) to each stage output. Returns the four feature
    maps in the configured data format, consumed by the Mask2Former pixel
    decoder.

    Args:
        embed_dim: Stage-0 token embedding dimension.
        depths: Number of Swin blocks per stage (length-4 sequence).
        num_heads: Number of attention heads per stage (length-4 sequence).
        window_size: Edge length of the attention window.
        data_format: ``"channels_last"`` or ``"channels_first"``. Defaults to
            ``keras.config.image_data_format()``.
        **kwargs: Additional keyword arguments forwarded to ``keras.layers.Layer``.
    """

    def __init__(
        self, embed_dim, depths, num_heads, window_size, data_format=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.data_format = data_format or keras.config.image_data_format()

        self.patch_embeddings = Mask2FormerSwinPatchEmbeddings(
            embed_dim,
            data_format=self.data_format,
            name="embeddings_patch_embeddings",
        )
        self.embeddings_norm = layers.LayerNormalization(
            epsilon=1e-5, name="embeddings_norm"
        )

        self.stages = []
        for i, (depth, nh) in enumerate(zip(depths, num_heads)):
            dim = embed_dim * (2**i)
            self.stages.append(
                Mask2FormerSwinStage(
                    dim=dim,
                    depth=depth,
                    num_heads=nh,
                    window_size=window_size,
                    downsample=(i < len(depths) - 1),
                    name=f"layers_{i}",
                )
            )

        self.hidden_states_norms = [
            layers.LayerNormalization(
                epsilon=1e-5, name=f"hidden_states_norms_stage{i + 1}"
            )
            for i in range(len(depths))
        ]

    def call(self, pixel_values):
        x, h, w = self.patch_embeddings(pixel_values)
        x = self.embeddings_norm(x)

        # Windowed attention runs on channels-last token grids internally; the
        # per-stage feature maps are emitted in the configured data format.
        channels_first = self.data_format == "channels_first"
        outs = []
        for i, stage in enumerate(self.stages):
            (stage_feat, sh, sw), x, h, w = stage(x, h=h, w=w)
            normed = self.hidden_states_norms[i](stage_feat)
            b = ops.shape(normed)[0]
            c = normed.shape[-1]
            normed = ops.reshape(normed, (b, sh, sw, c))
            if channels_first:
                normed = ops.transpose(normed, (0, 3, 1, 2))  # -> (b, c, h, w)
            outs.append(normed)
        return outs

    def get_config(self):
        c = super().get_config()
        c.update(
            {
                "embed_dim": self.embed_dim,
                "depths": self.depths,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "data_format": self.data_format,
            }
        )
        return c
