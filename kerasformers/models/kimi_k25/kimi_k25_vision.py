import keras
from keras import layers, ops


def vision_rope_angles(height, width, head_dim, theta_base=10000.0):
    """2D-RoPE angles for a height x width grid, row-major flattened.

    The angle vector is built per frequency as ``[w*f0, h*f0, w*f1, h*f1, ...]``
    (the two spatial axes interleaved *within* each frequency, width first: the
    source flips the ``(h, w)`` position columns before the rotary), then
    duplicated to ``head_dim`` for the split-half ``rotate_half`` rotation. This
    is the Qwen2-VL-style convention, **not** the adjacent-pair rotation the
    original MoonViT uses; the two are not interchangeable.
    """
    spatial_dim = head_dim // 2
    m = ops.arange(0, spatial_dim, 2, dtype="float32") / spatial_dim
    freqs = 1.0 / ops.power(theta_base, m)  # (head_dim/4,)
    rows = ops.cast(ops.repeat(ops.arange(height), width), "float32")  # h index
    cols = ops.cast(ops.tile(ops.arange(width), (height,)), "float32")  # w index
    w_ang = cols[:, None] * freqs[None, :]
    h_ang = rows[:, None] * freqs[None, :]
    angles = ops.reshape(
        ops.stack([w_ang, h_ang], axis=-1), (height * width, spatial_dim)
    )
    return ops.concatenate([angles, angles], axis=-1)  # (L, head_dim)


def time_position_embeddings(num_frames, dim):
    """Additive sinusoidal table over the temporal axis, ``(num_frames, dim)``."""
    pos = ops.arange(num_frames, dtype="float32")
    inv_freq = 1.0 / ops.power(10000.0, ops.arange(0, dim, 2, dtype="float32") / dim)
    freqs = pos[:, None] * inv_freq[None, :]
    return ops.concatenate([ops.sin(freqs), ops.cos(freqs)], axis=-1)


def rotate_half(x):
    half = int(x.shape[-1]) // 2
    return ops.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope_vision(x, cos, sin):
    """Split-half rotation. ``x`` (L, heads, head_dim); ``cos``/``sin`` (L, head_dim)."""
    return x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]


@keras.saving.register_keras_serializable(package="kerasformers")
class KimiK25VisionMLP(layers.Layer):
    """Two-layer MLP with tanh-approximate GELU (``gelu_pytorch_tanh``)."""

    def __init__(self, hidden_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.fc1 = layers.Dense(mlp_dim, use_bias=True, name="fc1")
        self.fc2 = layers.Dense(hidden_dim, use_bias=True, name="fc2")

    def call(self, x):
        return self.fc2(ops.gelu(self.fc1(x), approximate=True))

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim, "mlp_dim": self.mlp_dim})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class KimiK25VisionAttention(layers.Layer):
    """Bias-ful q/k/v/proj self-attention with 2D rope, over one packed clip."""

    def __init__(self, hidden_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = layers.Dense(hidden_dim, use_bias=True, name="q_proj")
        self.k_proj = layers.Dense(hidden_dim, use_bias=True, name="k_proj")
        self.v_proj = layers.Dense(hidden_dim, use_bias=True, name="v_proj")
        self.proj = layers.Dense(hidden_dim, use_bias=True, name="proj")

    def call(self, x, cos, sin):
        leng = int(x.shape[0])
        shape = (leng, self.num_heads, self.head_dim)
        q = apply_rope_vision(ops.reshape(self.q_proj(x), shape), cos, sin)
        k = apply_rope_vision(ops.reshape(self.k_proj(x), shape), cos, sin)
        v = ops.reshape(self.v_proj(x), shape)
        q = ops.transpose(q, (1, 0, 2))  # (heads, L, head_dim)
        k = ops.transpose(k, (1, 0, 2))
        v = ops.transpose(v, (1, 0, 2))
        scale = self.head_dim**-0.5
        attn = ops.softmax(ops.matmul(q, ops.transpose(k, (0, 2, 1))) * scale, axis=-1)
        out = ops.matmul(attn, v)
        out = ops.reshape(ops.transpose(out, (1, 0, 2)), (leng, self.hidden_dim))
        return self.proj(out)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim, "num_heads": self.num_heads})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class KimiK25VisionEncoderLayer(layers.Layer):
    """Pre-norm block: rope self-attention + MLP, both residual."""

    def __init__(self, hidden_dim, num_heads, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.norm1 = layers.LayerNormalization(epsilon=1e-5, name="norm1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-5, name="norm2")
        self.attn = KimiK25VisionAttention(hidden_dim, num_heads, name="attn")
        self.mlp = KimiK25VisionMLP(hidden_dim, mlp_dim, name="mlp")

    def call(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class KimiK25VisionModel(layers.Layer):
    """MoonViT with a temporal axis: native-resolution packed ViT for images and video.

    Patch-embeds each ``(t, h, w)`` clip, adds a bicubic-interpolated learnable 2D
    position embedding (plus an additive sinusoidal *time* table when ``t > 1``),
    runs 27 pre-norm 2D-rotary blocks over the whole clip at once (each clip
    attended independently: the source's block-diagonal ``cu_seqlens`` packing),
    then temporally averages the frames and merges ``merge_kernel`` (2x2)
    spatial neighborhoods. Returns ``(total_merged, kh*kw, embed_dim)``: the
    ``kh*kw`` axis is kept separate because the projector layer-norms over
    ``embed_dim`` *before* flattening it.
    """

    def __init__(
        self,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_dim=4304,
        patch_size=14,
        pos_emb_height=64,
        pos_emb_width=64,
        pos_emb_time=4,
        merge_kernel=(2, 2),
        in_channels=3,
        rope_theta=10000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.patch_size = patch_size
        self.pos_emb_height = pos_emb_height
        self.pos_emb_width = pos_emb_width
        self.pos_emb_time = pos_emb_time
        self.merge_kernel = tuple(merge_kernel)
        self.in_channels = in_channels
        self.rope_theta = rope_theta
        self.head_dim = embed_dim // num_heads

        # A stride-P, PxP convolution over a PxP patch is exactly a dense
        # projection of the flattened patch. Dense is used instead of Conv2D
        # because keras' torch-backend Conv2D computes this at ~1e-4 relative
        # precision (vs ~1e-7 for matmul), which compounds across the 27 blocks.
        self.patch_proj = layers.Dense(embed_dim, use_bias=True, name="patch_proj")
        self.blocks = [
            KimiK25VisionEncoderLayer(embed_dim, num_heads, mlp_dim, name=f"block_{i}")
            for i in range(depth)
        ]
        self.final_norm = layers.LayerNormalization(epsilon=1e-5, name="final_norm")

    def build(self, _):
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(self.pos_emb_height, self.pos_emb_width, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.patch_proj.build((None, self.in_channels * self.patch_size**2))
        self.built = True

    def interp_pos_emb(self, height, width):
        pe = ops.convert_to_tensor(self.pos_emb)
        if height == self.pos_emb_height and width == self.pos_emb_width:
            return ops.reshape(pe, (height * width, self.embed_dim))
        # Bicubic resample of the learned grid: Keys cubic (a = -0.75),
        # half-pixel centers, replicated borders -- exactly
        # F.interpolate(mode="bicubic", align_corners=False). Spelled out
        # because ops.image.resize is backend divergent (jax/tf use a = -0.5
        # and disagree with torch/the reference by ~0.3).
        mats = []
        for out_size, in_size in (
            (height, self.pos_emb_height),
            (width, self.pos_emb_width),
        ):
            scale = in_size / out_size
            center = (ops.arange(out_size, dtype="float32") + 0.5) * scale - 0.5
            start = ops.floor(center)
            frac = center - start
            a = -0.75
            d0, d1, d2, d3 = frac + 1.0, frac, 1.0 - frac, 2.0 - frac
            taps = (
                ((a * d0 - 5.0 * a) * d0 + 8.0 * a) * d0 - 4.0 * a,
                ((a + 2.0) * d1 - (a + 3.0)) * d1 * d1 + 1.0,
                ((a + 2.0) * d2 - (a + 3.0)) * d2 * d2 + 1.0,
                ((a * d3 - 5.0 * a) * d3 + 8.0 * a) * d3 - 4.0 * a,
            )
            matrix = ops.zeros((out_size, in_size), dtype="float32")
            for offset, tap in zip((-1, 0, 1, 2), taps):
                index = ops.clip(ops.cast(start, "int32") + offset, 0, in_size - 1)
                # one_hot + add so clamped duplicate border taps accumulate
                onehot = ops.one_hot(index, in_size, dtype="float32")
                matrix = matrix + onehot * tap[:, None]
            mats.append(matrix)
        resized = ops.einsum("hi,ijc->hjc", mats[0], ops.cast(pe, "float32"))
        resized = ops.einsum("wj,hjc->hwc", mats[1], resized)
        return ops.reshape(resized, (height * width, self.embed_dim))

    def clip_pos_emb(self, frames, height, width):
        pos = self.interp_pos_emb(height, width)  # (h*w, D)
        if frames == 1:
            return pos
        pos = ops.tile(pos[None, :, :], (frames, 1, 1))  # (t, h*w, D)
        time_pe = time_position_embeddings(self.pos_emb_time, self.embed_dim)
        pos = pos + time_pe[:frames][:, None, :]
        return ops.reshape(pos, (frames * height * width, self.embed_dim))

    def embed_patches(self, patches):
        # (L, C, P, P) -> (L, embed_dim). A channels-last (L, P, P, C) batch is
        # transposed first so the flatten order matches the source conv kernel.
        if (
            int(patches.shape[1]) != self.in_channels
            and int(patches.shape[-1]) == self.in_channels
        ):
            patches = ops.transpose(patches, (0, 3, 1, 2))
        x = ops.reshape(
            ops.cast(patches, self.compute_dtype), (int(patches.shape[0]), -1)
        )
        return self.patch_proj(x)

    def temporal_patch_merge(self, x, frames, height, width):
        kh, kw = self.merge_kernel
        nh, nw = height // kh, width // kw
        x = ops.reshape(x, (frames, nh, kh, nw, kw, self.embed_dim))
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))  # (t, nh, nw, kh, kw, D)
        x = ops.mean(x, axis=0)  # temporal pooling
        return ops.reshape(x, (nh * nw, kh * kw, self.embed_dim))

    def call(self, pixel_values, grid_thw):
        grids = [(int(t), int(h), int(w)) for t, h, w in grid_thw]
        outputs = []
        start = 0
        for frames, height, width in grids:
            leng = frames * height * width
            patches = pixel_values[start : start + leng]
            start += leng
            x = self.embed_patches(patches) + ops.cast(
                self.clip_pos_emb(frames, height, width), self.compute_dtype
            )
            angles = vision_rope_angles(height, width, self.head_dim, self.rope_theta)
            if frames > 1:
                angles = ops.tile(angles, (frames, 1))
            cos = ops.cast(ops.cos(angles), self.compute_dtype)
            sin = ops.cast(ops.sin(angles), self.compute_dtype)
            for block in self.blocks:
                x = block(x, cos, sin)
            x = self.final_norm(x)
            outputs.append(self.temporal_patch_merge(x, frames, height, width))
        return ops.concatenate(outputs, axis=0)  # (total_merged, kh*kw, embed_dim)

    def compute_output_spec(self, pixel_values, grid_thw):
        # ``call`` iterates the concrete grid (Python int(t/h/w)), so it cannot be
        # traced symbolically; an explicit spec makes the tower a functional-graph
        # node whose ``call`` runs at (eager) runtime with the real grid. The
        # merged-token count is data-dependent -> leave the leading axis dynamic.
        kh, kw = self.merge_kernel
        return keras.KerasTensor(
            (None, kh * kw, self.embed_dim), dtype=self.compute_dtype
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "patch_size": self.patch_size,
                "pos_emb_height": self.pos_emb_height,
                "pos_emb_width": self.pos_emb_width,
                "pos_emb_time": self.pos_emb_time,
                "merge_kernel": self.merge_kernel,
                "in_channels": self.in_channels,
                "rope_theta": self.rope_theta,
            }
        )
        return config
