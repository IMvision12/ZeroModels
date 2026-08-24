import keras
from keras import layers, ops

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class Glm4vRMSNorm(layers.Layer):
    """RMSNorm (plain learned weight, ones init), float32 statistics."""

    def __init__(self, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight", shape=(input_shape[-1],), initializer="ones", trainable=True
        )
        self.built = True

    def call(self, x):
        dtype = x.dtype
        x = ops.cast(x, "float32")
        variance = ops.mean(ops.square(x), axis=-1, keepdims=True)
        x = x * ops.rsqrt(variance + self.eps)
        return self.weight * ops.cast(x, dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


def vision_grid_rows(grid_thw):
    return [
        tuple(int(v) for v in row)
        for row in ops.convert_to_numpy(ops.convert_to_tensor(grid_thw))
    ]


def vision_position_coords(grid_thw, spatial_merge_size):
    """Block-ordered (h, w) patch coordinates for the flattened sequence."""
    m = spatial_merge_size
    pieces = []
    for t, h, w in vision_grid_rows(grid_thw):
        hpos = ops.broadcast_to(ops.arange(h)[:, None], (h, w))
        hpos = ops.reshape(
            ops.transpose(ops.reshape(hpos, (h // m, m, w // m, m)), (0, 2, 1, 3)),
            (-1,),
        )
        wpos = ops.broadcast_to(ops.arange(w)[None, :], (h, w))
        wpos = ops.reshape(
            ops.transpose(ops.reshape(wpos, (h // m, m, w // m, m)), (0, 2, 1, 3)),
            (-1,),
        )
        pieces.append(ops.tile(ops.stack([hpos, wpos], axis=-1), [t, 1]))
    return ops.concatenate(pieces, axis=0)


def vision_rotary_cos_sin(grid_thw, head_dim, spatial_merge_size, theta=10000.0):
    """2D vision rotary tables ``(total_patches, head_dim)`` (cat(rotary, rotary))."""
    rotary_dim = head_dim // 2
    inv_freq = 1.0 / ops.power(
        theta, ops.arange(0, rotary_dim, 2, dtype="float32") / rotary_dim
    )
    pos_ids = vision_position_coords(grid_thw, spatial_merge_size)
    max_grid = max(max(h, w) for _, h, w in vision_grid_rows(grid_thw))
    freqs = ops.arange(max_grid, dtype="float32")[:, None] * inv_freq
    total = pos_ids.shape[0]
    rotary = ops.reshape(ops.take(freqs, pos_ids, axis=0), (total, -1))
    emb = ops.concatenate([rotary, rotary], axis=-1)
    return ops.cos(emb), ops.sin(emb)


def vision_block_mask(grid_thw):
    """Block-diagonal additive mask so patches attend only within their image."""
    seqlens = [t * h * w for t, h, w in vision_grid_rows(grid_thw)]
    if len(seqlens) <= 1:
        return None
    seg = ops.concatenate(
        [ops.full((n,), i, dtype="int32") for i, n in enumerate(seqlens)], axis=0
    )
    mask = ops.where(seg[:, None] == seg[None, :], 0.0, MASK_NEG)
    return ops.cast(mask, "float32")[None, None]


@keras.saving.register_keras_serializable(package="kerasformers")
class Glm4vVisionPatchEmbed(layers.Layer):
    """Conv3d patch embed as a bias-carrying ``Dense`` over flattened patches.

    HF's ``Conv3d`` kernel equals its stride, so each ``(temporal, patch,
    patch)`` patch is projected once; the processor already flattens each patch
    to ``in_channels * temporal_patch_size * patch_size**2``.
    """

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.proj = layers.Dense(embed_dim, use_bias=True, name="proj")

    def call(self, x):
        return self.proj(x)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Glm4vVisionEmbeddings(layers.Layer):
    """Learned position embedding interpolated (bicubic) to each image's grid.

    The ``num_positions = (image_size // patch_size)**2`` learned positions are
    reshaped to a square grid and resized to each image's ``(grid_h, grid_w)``
    with bicubic interpolation, then gathered per patch (block order) and added.

    Args:
        embed_dim: Vision hidden width.
        num_positions: Number of learned positions (square count).
    """

    def __init__(self, embed_dim, num_positions, spatial_merge_size=2, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_positions = num_positions
        self.spatial_merge_size = spatial_merge_size
        self.orig_size = int(round(num_positions**0.5))
        self.position_embedding = layers.Embedding(
            num_positions, embed_dim, name="position_embedding"
        )

    def build(self, input_shape):
        self.position_embedding.build((1, 1))
        self.built = True

    def call(self, embeddings, grid_thw):
        spatial_merge_size = self.spatial_merge_size
        weight = ops.cast(self.position_embedding.embeddings, "float32")
        pos_2d = ops.reshape(weight, (self.orig_size, self.orig_size, self.embed_dim))
        adapted = []
        for t, h, w in vision_grid_rows(grid_thw):
            # Bicubic resample of the learned grid: Keys cubic (a = -0.75),
            # half-pixel centers, replicated borders -- exactly
            # F.interpolate(mode="bicubic", align_corners=False). Spelled out
            # because ops.image.resize is backend divergent (jax/tf use a = -0.5
            # and disagree with torch/the reference by ~0.3).
            mats = []
            for out_size, in_size in ((h, self.orig_size), (w, self.orig_size)):
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
            resized = ops.einsum("hi,ijc->hjc", mats[0], ops.cast(pos_2d, "float32"))
            resized = ops.einsum("wj,hjc->hwc", mats[1], resized)
            flat = ops.reshape(resized, (h * w, self.embed_dim))
            coords = vision_position_coords([[t, h, w]], spatial_merge_size)
            gather_idx = coords[:, 0] * w + coords[:, 1]
            adapted.append(ops.take(flat, gather_idx, axis=0))
        adapted = ops.concatenate(adapted, axis=0)
        return embeddings + ops.cast(adapted, embeddings.dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_positions": self.num_positions,
                "spatial_merge_size": self.spatial_merge_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Glm4VisionMlp(layers.Layer):
    """Vision SwiGLU MLP: ``down(silu(gate(x)) * up(x))`` (bias-free).

    The hidden width is ``out_hidden_size`` (the LLM width), per GLM-4V.
    """

    def __init__(self, hidden_size, intermediate_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = layers.Dense(intermediate_size, use_bias=False, name="gate")
        self.up_proj = layers.Dense(intermediate_size, use_bias=False, name="up")
        self.down_proj = layers.Dense(hidden_size, use_bias=False, name="down")

    def call(self, x):
        return self.down_proj(ops.silu(self.gate_proj(x)) * self.up_proj(x))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Glm4vVisionAttention(layers.Layer):
    """Full (non-causal) packed vision attention with 2D rotary positions.

    Fused bias-free ``qkv`` and a bias-free ``proj``; an additive block mask
    (from ``cu_seqlens``) keeps attention within each image.

    Call args:
        hidden_states: ``(seq, embed_dim)``.
        cos, sin: vision rotary tables ``(seq, head_dim)``.
        attention_mask: additive ``(1, 1, seq, seq)`` mask or ``None``.
    """

    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = layers.Dense(embed_dim * 3, use_bias=False, name="qkv")
        self.proj = layers.Dense(embed_dim, use_bias=False, name="proj")

    def call(self, hidden_states, cos, sin, attention_mask=None):
        seq = ops.shape(hidden_states)[0]
        qkv = self.qkv(hidden_states)
        qkv = ops.reshape(qkv, (seq, 3, self.num_heads, self.head_dim))
        qkv = ops.transpose(qkv, (1, 0, 2, 3))
        query, key, value = qkv[0], qkv[1], qkv[2]

        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        half = self.head_dim // 2
        query = query * cos + (
            ops.concatenate([-query[..., half:], query[..., :half]], axis=-1) * sin
        )
        key = key * cos + (
            ops.concatenate([-key[..., half:], key[..., :half]], axis=-1) * sin
        )

        query = ops.expand_dims(ops.transpose(query, (1, 0, 2)), axis=0)
        key = ops.expand_dims(ops.transpose(key, (1, 0, 2)), axis=0)
        value = ops.expand_dims(ops.transpose(value, (1, 0, 2)), axis=0)

        attn = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2))) * self.scaling
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), query.dtype)
        out = ops.matmul(attn, value)
        out = ops.transpose(out[0], (1, 0, 2))
        out = ops.reshape(out, (seq, self.embed_dim))
        return self.proj(out)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Glm4vVisionBlock(layers.Layer):
    """Pre-norm vision block (RMSNorm): ``h += attn(n1(h)); h += mlp(n2(h))``."""

    def __init__(
        self, embed_dim, num_heads, intermediate_size, norm_eps=1e-5, **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.norm_eps = norm_eps
        self.norm1 = Glm4vRMSNorm(eps=norm_eps, name="norm1")
        self.norm2 = Glm4vRMSNorm(eps=norm_eps, name="norm2")
        self.attn = Glm4vVisionAttention(embed_dim, num_heads, name="attn")
        self.mlp = Glm4VisionMlp(embed_dim, intermediate_size, name="mlp")

    def call(self, hidden_states, cos, sin, attention_mask=None):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cos, sin, attention_mask=attention_mask
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Glm4vVisionPatchMerger(layers.Layer):
    """GLM-4V merger: ``proj -> LayerNorm -> GELU -> SwiGLU(gate/up/down)``."""

    def __init__(self, dim, context_dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.context_dim = context_dim
        self.proj = layers.Dense(dim, use_bias=False, name="proj")
        self.post_projection_norm = layers.LayerNormalization(
            epsilon=1e-5, name="post_projection_norm"
        )
        self.gate_proj = layers.Dense(context_dim, use_bias=False, name="gate")
        self.up_proj = layers.Dense(context_dim, use_bias=False, name="up")
        self.down_proj = layers.Dense(dim, use_bias=False, name="down")

    def call(self, x):
        x = self.proj(x)
        x = ops.gelu(self.post_projection_norm(x), approximate=False)
        return self.down_proj(ops.silu(self.gate_proj(x)) * self.up_proj(x))

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "context_dim": self.context_dim})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Glm4vVisionModel(layers.Layer):
    """GLM-4V vision tower.

    Conv3d-as-Dense patch embed -> RMSNorm -> learned (bicubic-interpolated)
    positions -> ``depth`` packed-attention rotary blocks -> RMSNorm ->
    2x2 downsample conv (to ``out_hidden_size``) -> SwiGLU merger.

    Call args:
        pixel_values: ``(num_patches, patch_dim)`` flattened patches.
        grid_thw: per-image ``(t, h, w)`` patch-grid sizes.

    Returns:
        ``(num_merged_tokens, out_hidden_size)`` image embeddings.
    """

    def __init__(
        self,
        embed_dim=1536,
        depth=24,
        num_heads=12,
        out_hidden_size=4096,
        intermediate_size=13696,
        image_size=336,
        patch_size=14,
        spatial_merge_size=2,
        norm_eps=1e-5,
        rope_theta=10000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.out_hidden_size = out_hidden_size
        self.intermediate_size = intermediate_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.head_dim = embed_dim // num_heads
        self.num_positions = (image_size // patch_size) ** 2

        self.patch_embed = Glm4vVisionPatchEmbed(embed_dim, name="patch_embed")
        self.post_conv_layernorm = Glm4vRMSNorm(
            eps=norm_eps, name="post_conv_layernorm"
        )
        self.embeddings = Glm4vVisionEmbeddings(
            embed_dim, self.num_positions, spatial_merge_size, name="embeddings"
        )
        self.blocks = [
            Glm4vVisionBlock(
                embed_dim, num_heads, out_hidden_size, norm_eps, name=f"blocks_{i}"
            )
            for i in range(depth)
        ]
        self.post_layernorm = Glm4vRMSNorm(eps=norm_eps, name="post_layernorm")
        self.downsample = layers.Conv2D(
            out_hidden_size,
            kernel_size=spatial_merge_size,
            strides=spatial_merge_size,
            data_format="channels_last",
            name="downsample",
        )
        self.merger = Glm4vVisionPatchMerger(
            out_hidden_size, intermediate_size, name="merger"
        )

    def call(self, pixel_values, grid_thw):
        m = self.spatial_merge_size
        cos, sin = vision_rotary_cos_sin(grid_thw, self.head_dim, m, self.rope_theta)
        mask = vision_block_mask(grid_thw)
        hidden = self.patch_embed(pixel_values)
        hidden = self.post_conv_layernorm(hidden)
        hidden = self.embeddings(hidden, grid_thw)
        for block in self.blocks:
            hidden = block(hidden, cos, sin, attention_mask=mask)
        hidden = self.post_layernorm(hidden)
        hidden = ops.reshape(hidden, (-1, m, m, self.embed_dim))
        hidden = self.downsample(hidden)
        hidden = ops.reshape(hidden, (-1, self.out_hidden_size))
        return self.merger(hidden)

    def compute_output_spec(self, pixel_values, grid_thw):
        # Merged-token count is grid-dependent (dynamic); the grid-iterating call
        # runs eagerly at runtime while the functional build uses this spec.
        return keras.KerasTensor((None, self.out_hidden_size), dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "out_hidden_size": self.out_hidden_size,
                "intermediate_size": self.intermediate_size,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "spatial_merge_size": self.spatial_merge_size,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
            }
        )
        return config
