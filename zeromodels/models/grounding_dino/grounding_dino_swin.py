import keras
import numpy as np
from keras import layers, ops


def window_partition(x, window_size):
    # x: (B, H, W, C) -> (num_windows*B, ws, ws, C)
    b = ops.shape(x)[0]
    h = int(x.shape[1])
    w = int(x.shape[2])
    c = int(x.shape[3])
    x = ops.reshape(
        x, (b, h // window_size, window_size, w // window_size, window_size, c)
    )
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    return ops.reshape(x, (-1, window_size, window_size, c))


def window_reverse(windows, window_size, h, w):
    c = int(windows.shape[-1])
    x = ops.reshape(
        windows, (-1, h // window_size, w // window_size, window_size, window_size, c)
    )
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    return ops.reshape(x, (-1, h, w, c))


def build_relative_position_index(window_size):
    coords_h = np.arange(window_size)
    coords_w = np.arange(window_size)
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, ws, ws
    coords_flatten = coords.reshape(2, -1)  # 2, ws*ws
    rel = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
    rel = rel.transpose(1, 2, 0)  # N, N, 2
    rel[:, :, 0] += window_size - 1
    rel[:, :, 1] += window_size - 1
    rel[:, :, 0] *= 2 * window_size - 1
    return rel.sum(-1).reshape(-1)  # N*N


def build_shift_attn_mask(height, width, window_size, shift_size):
    # Cyclic-shift attention mask: (num_windows, ws*ws, ws*ws) additive (0 / -100).
    img_mask = np.zeros((1, height, width, 1), dtype="float32")
    slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    count = 0
    for hs in slices:
        for ws in slices:
            img_mask[:, hs, ws, :] = count
            count += 1
    mask = ops.convert_to_tensor(img_mask)
    mask_windows = window_partition(mask, window_size)
    mask_windows = ops.reshape(mask_windows, (-1, window_size * window_size))
    attn_mask = ops.expand_dims(mask_windows, 1) - ops.expand_dims(mask_windows, 2)
    attn_mask = ops.where(attn_mask != 0, -100.0, 0.0)
    return attn_mask


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoSwinSelfAttention(layers.Layer):
    """Window self-attention with learned relative position bias (HF Swin)."""

    def __init__(self, dim, num_heads, window_size, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = layers.Dense(dim, name="q_proj")
        self.k_proj = layers.Dense(dim, name="k_proj")
        self.v_proj = layers.Dense(dim, name="v_proj")
        self.o_proj = layers.Dense(dim, name="o_proj")
        self.rel_index = build_relative_position_index(window_size)

    def build(self, input_shape):
        n = (2 * self.window_size - 1) ** 2
        self.relative_position_bias_table = self.add_weight(
            name="relative_position_bias_table",
            shape=(n, self.num_heads),
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def relative_position_bias(self):
        area = self.window_size * self.window_size
        bias = ops.take(self.relative_position_bias_table, self.rel_index, axis=0)
        bias = ops.reshape(bias, (area, area, self.num_heads))
        return ops.expand_dims(ops.transpose(bias, (2, 0, 1)), 0)  # 1, nh, N, N

    def call(self, hidden_states, attention_mask=None):
        bn = ops.shape(hidden_states)[0]
        n = int(hidden_states.shape[1])

        def split(t):
            return ops.transpose(
                ops.reshape(t, (bn, n, self.num_heads, self.head_dim)), (0, 2, 1, 3)
            )

        q = split(self.q_proj(hidden_states))
        k = split(self.k_proj(hidden_states))
        v = split(self.v_proj(hidden_states))
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scaling
        bias = self.relative_position_bias()
        if attention_mask is not None:
            num_windows = int(attention_mask.shape[0])
            batch = bn // num_windows
            mask = ops.reshape(attention_mask, (1, num_windows, 1, n, n))
            mask = ops.broadcast_to(mask, (batch, num_windows, 1, n, n))
            mask = ops.reshape(mask, (-1, 1, n, n))
            attn = attn + bias + mask
        else:
            attn = attn + bias
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), q.dtype)
        out = ops.matmul(attn, v)
        out = ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (bn, n, self.dim))
        return self.o_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoSwinLayer(layers.Layer):
    """One Swin block: (shifted) window attention + MLP, both pre-norm."""

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
        self.attention = GroundingDinoSwinSelfAttention(
            dim, num_heads, window_size, name="attention"
        )
        self.layernorm_after = layers.LayerNormalization(
            epsilon=1e-5, name="layernorm_after"
        )
        self.fc1 = layers.Dense(int(dim * mlp_ratio), name="mlp_fc1")
        self.fc2 = layers.Dense(dim, name="mlp_fc2")

    def call(self, hidden_states, height=None, width=None):
        window_size = self.window_size
        shift_size = self.shift_size
        if min(height, width) <= window_size:
            shift_size = 0
            window_size = min(height, width)
        channels = int(hidden_states.shape[-1])
        shortcut = hidden_states
        x = self.layernorm_before(hidden_states)
        x = ops.reshape(x, (-1, height, width, channels))

        pad_r = (window_size - width % window_size) % window_size
        pad_b = (window_size - height % window_size) % window_size
        if pad_r or pad_b:
            x = ops.pad(x, [[0, 0], [0, pad_b], [0, pad_r], [0, 0]])
        hp, wp = height + pad_b, width + pad_r

        if shift_size > 0:
            x = ops.roll(x, shift=(-shift_size, -shift_size), axis=(1, 2))
        x_windows = window_partition(x, window_size)
        x_windows = ops.reshape(x_windows, (-1, window_size * window_size, channels))
        attn_mask = (
            build_shift_attn_mask(hp, wp, window_size, shift_size)
            if shift_size > 0
            else None
        )
        attn_windows = self.attention(x_windows, attention_mask=attn_mask)
        attn_windows = ops.reshape(
            attn_windows, (-1, window_size, window_size, channels)
        )
        x = window_reverse(attn_windows, window_size, hp, wp)
        if shift_size > 0:
            x = ops.roll(x, shift=(shift_size, shift_size), axis=(1, 2))
        if pad_r or pad_b:
            x = x[:, :height, :width, :]
        x = ops.reshape(x, (-1, height * width, channels))
        hidden_states = shortcut + x

        residual = hidden_states
        x = self.layernorm_after(hidden_states)
        x = self.fc2(ops.gelu(self.fc1(x)))
        return residual + x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "shift_size": self.shift_size,
                "mlp_ratio": self.mlp_ratio,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoSwinPatchMerging(layers.Layer):
    """Patch merging: concat 2x2 neighborhood, LayerNorm, linear to 2*dim."""

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="norm")
        self.reduction = layers.Dense(2 * dim, use_bias=False, name="reduction")

    def call(self, x, height=None, width=None):
        channels = int(x.shape[-1])
        x = ops.reshape(x, (-1, height, width, channels))
        if height % 2 or width % 2:
            x = ops.pad(x, [[0, 0], [0, height % 2], [0, width % 2], [0, 0]])
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = ops.concatenate([x0, x1, x2, x3], axis=-1)
        x = ops.reshape(x, (-1, ((height + 1) // 2) * ((width + 1) // 2), 4 * channels))
        return self.reduction(self.norm(x))

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoSwinStage(layers.Layer):
    """A Swin stage: ``depth`` blocks (alternating shift) + optional downsample."""

    def __init__(self, dim, depth, num_heads, window_size, downsample, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.has_downsample = downsample
        self.blocks = [
            GroundingDinoSwinLayer(
                dim,
                num_heads,
                window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                name=f"blocks_{i}",
            )
            for i in range(depth)
        ]
        self.downsample = (
            GroundingDinoSwinPatchMerging(dim, name="downsample")
            if downsample
            else None
        )

    def call(self, hidden_states, height=None, width=None):
        for block in self.blocks:
            hidden_states = block(hidden_states, height=height, width=width)
        before_downsample = hidden_states
        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states, height=height, width=width)
            out_h, out_w = (height + 1) // 2, (width + 1) // 2
        else:
            out_h, out_w = height, width
        return hidden_states, before_downsample, out_h, out_w

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "downsample": self.has_downsample,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoSwinBackbone(layers.Layer):
    """Swin backbone returning the 3 normed multi-scale feature maps (out_indices).

    Patch-embed (4x4 conv) -> LayerNorm -> 4 stages with patch merging; the
    pre-downsampling outputs of stages 1, 2, 3 (channels ``2/4/8 * embed_dim``)
    are LayerNorm'd (``hidden_states_norms``) and returned as multi-scale
    feature maps in the configured ``data_format``.
    """

    def __init__(
        self,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        out_indices=(2, 3, 4),
        patch_size=4,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.depths = tuple(depths)
        self.num_heads = tuple(num_heads)
        self.window_size = window_size
        self.out_indices = tuple(out_indices)
        self.patch_size = patch_size
        self.data_format = data_format or keras.config.image_data_format()
        self.num_stages = len(depths)

        self.patch_embed = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            data_format=self.data_format,
            name="patch_embeddings_projection",
        )
        self.embed_norm = layers.LayerNormalization(epsilon=1e-5, name="embed_norm")
        self.stages = []
        for i in range(self.num_stages):
            dim = embed_dim * (2**i)
            self.stages.append(
                GroundingDinoSwinStage(
                    dim,
                    depths[i],
                    num_heads[i],
                    window_size,
                    downsample=i < self.num_stages - 1,
                    name=f"stage_{i}",
                )
            )
        # HF hidden-states are [stem, stage0_out, ...]; out_index k -> 0-based stage k-1.
        self.out_stage_idx = [idx - 1 for idx in out_indices]
        self.out_norms = {
            i: layers.LayerNormalization(
                epsilon=1e-5, name=f"hidden_states_norms_stage{i + 1}"
            )
            for i in self.out_stage_idx
        }

    def call(self, pixel_values):
        x = self.patch_embed(pixel_values)
        # Window attention runs channels-last; normalize a channels-first
        # patch-embed output before the sequence reshape.
        if self.data_format == "channels_first":
            x = ops.transpose(x, (0, 2, 3, 1))
        h = int(x.shape[1])
        w = int(x.shape[2])
        c = int(x.shape[3])
        x = ops.reshape(x, (-1, h * w, c))
        x = self.embed_norm(x)
        feature_maps = []
        for i, stage in enumerate(self.stages):
            x, before, out_h, out_w = stage(x, height=h, width=w)
            if i in self.out_stage_idx:
                normed = self.out_norms[i](before)
                dim = self.embed_dim * (2**i)
                fm = ops.reshape(normed, (-1, h, w, dim))
                if self.data_format == "channels_first":
                    fm = ops.transpose(fm, (0, 3, 1, 2))
                feature_maps.append(fm)
            h, w = out_h, out_w
        return feature_maps

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "depths": self.depths,
                "num_heads": self.num_heads,
                "window_size": self.window_size,
                "out_indices": self.out_indices,
                "patch_size": self.patch_size,
                "data_format": self.data_format,
            }
        )
        return config
