import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


def rotate_half(x):
    half = ops.shape(x)[-1] // 2
    return ops.concatenate([-x[..., half:], x[..., :half]], axis=-1)


@keras.saving.register_keras_serializable(package="zeromodels")
class Mistral3RMSNorm(layers.Layer):
    """Root-mean-square layer norm shared by the Pixtral tower (eps 1e-5), the
    Mistral text decoder (eps 1e-5), and the projector input norm.

    Normalizes the last axis by its RMS in float32, casts back to the input
    dtype, then scales by a learned per-channel weight. No mean subtraction,
    no bias.

    Args:
        eps: Variance epsilon. Defaults to ``1e-5``.
    """

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


@keras.saving.register_keras_serializable(package="zeromodels")
class Mistral3VisionMLP(layers.Layer):
    """Pixtral SwiGLU feed-forward: ``down(silu(gate(x)) * up(x))``, bias-free."""

    def __init__(self, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.gate = layers.Dense(mlp_dim, use_bias=False, name="gate")
        self.up = layers.Dense(mlp_dim, use_bias=False, name="up")
        self.down = layers.Dense(embed_dim, use_bias=False, name="down")

    def call(self, x):
        return self.down(ops.silu(self.gate(x)) * self.up(x))

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "mlp_dim": self.mlp_dim})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Mistral3VisionAttention(layers.Layer):
    """Pixtral full (bidirectional) self-attention with 2D rotary positions.

    Bias-free ``query`` / ``key`` / ``value`` / ``output_proj`` projections;
    the caller supplies per-position 2D rotary tables (gathered from the
    height/width meshgrid) and a block-diagonal mask so packed patches only
    attend within their own image.

    Args:
        embed_dim: Vision hidden width.
        num_heads: Attention heads.

    Call args:
        hidden_states: ``(1, total_patches, embed_dim)`` packed sequence.
        cos, sin: rotary tables ``(1, total_patches, head_dim)``.
        attention_mask: additive block-diagonal mask, or ``None``.

    Returns:
        ``(1, total_patches, embed_dim)``.
    """

    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.query = layers.Dense(embed_dim, use_bias=False, name="query")
        self.key = layers.Dense(embed_dim, use_bias=False, name="key")
        self.value = layers.Dense(embed_dim, use_bias=False, name="value")
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")

    def call(self, hidden_states, cos, sin, attention_mask=None):
        b = ops.shape(hidden_states)[0]
        seq = ops.shape(hidden_states)[1]
        q = ops.transpose(
            ops.reshape(
                self.query(hidden_states), (b, seq, self.num_heads, self.head_dim)
            ),
            (0, 2, 1, 3),
        )
        k = ops.transpose(
            ops.reshape(
                self.key(hidden_states), (b, seq, self.num_heads, self.head_dim)
            ),
            (0, 2, 1, 3),
        )
        v = ops.transpose(
            ops.reshape(
                self.value(hidden_states), (b, seq, self.num_heads, self.head_dim)
            ),
            (0, 2, 1, 3),
        )
        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        out = fused_attention(q, k, v, self.scaling, attention_mask)
        out = ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (b, seq, self.embed_dim))
        return self.output_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Mistral3VisionLayer(layers.Layer):
    """One Pixtral block: pre-norm 2D-rotary attention, then pre-norm SwiGLU.

    Computes ``h = x + attention(attention_norm(x))`` followed by
    ``h = h + feed_forward(ffn_norm(h))`` with RMSNorms (eps 1e-5, the
    hardcoded Pixtral value).

    Args:
        embed_dim: Vision hidden width.
        mlp_dim: Vision MLP hidden width.
        num_heads: Attention heads.
    """

    def __init__(self, embed_dim, mlp_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.attention_norm = Mistral3RMSNorm(eps=1e-5, name="attention_norm")
        self.attention = Mistral3VisionAttention(embed_dim, num_heads, name="attention")
        self.ffn_norm = Mistral3RMSNorm(eps=1e-5, name="ffn_norm")
        self.feed_forward = Mistral3VisionMLP(embed_dim, mlp_dim, name="feed_forward")

    def call(self, hidden_states, cos, sin, attention_mask=None):
        hidden_states = hidden_states + self.attention(
            self.attention_norm(hidden_states), cos, sin, attention_mask
        )
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Mistral3PatchMerger(layers.Layer):
    """Learned 2x2 spatial merge of vision patches.

    Per image, the ``(h_patches, w_patches)`` token grid is grouped into
    ``spatial_merge_size``-square blocks whose features are concatenated
    channel-major (matching torch ``unfold``) and projected back to the
    vision width by a bias-free linear.

    Args:
        embed_dim: Vision hidden width.
        spatial_merge_size: Merge factor (2 for the released models).
    """

    def __init__(self, embed_dim, spatial_merge_size=2, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.spatial_merge_size = spatial_merge_size
        self.merging_layer = layers.Dense(
            embed_dim, use_bias=False, name="merging_layer"
        )

    def merge_one(self, tokens, grid_h, grid_w):
        m = self.spatial_merge_size
        d = self.embed_dim
        h2, w2 = grid_h // m, grid_w // m
        grid = ops.reshape(tokens, (grid_h, grid_w, d))[: h2 * m, : w2 * m]
        grid = ops.reshape(grid, (h2, m, w2, m, d))
        grid = ops.transpose(grid, (0, 2, 4, 1, 3))  # (h2, w2, d, m, m)
        return ops.reshape(grid, (h2 * w2, d * m * m))

    def call(self, image_features, grid_sizes):
        # image_features: (total_patches, embed_dim); grid_sizes: host list of
        # (h_patches, w_patches) per image.
        pieces = []
        offset = 0
        for grid_h, grid_w in grid_sizes:
            n = grid_h * grid_w
            pieces.append(
                self.merge_one(image_features[offset : offset + n], grid_h, grid_w)
            )
            offset += n
        return self.merging_layer(ops.concatenate(pieces, axis=0))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "spatial_merge_size": self.spatial_merge_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Mistral3MultiModalProjector(layers.Layer):
    """Vision-to-text projection: RMS norm -> 2x2 patch merge -> linear /
    gelu / linear.

    Args:
        vision_dim: Vision hidden width.
        text_dim: Text decoder hidden width.
        spatial_merge_size: Patch-merge factor.
        norm_eps: Epsilon of the input RMS norm (the text ``rms_norm_eps``).
        use_bias: Whether the two projection linears carry biases
            (``multimodal_projector_bias``; released models: ``False``).
    """

    def __init__(
        self,
        vision_dim,
        text_dim,
        spatial_merge_size=2,
        norm_eps=1e-5,
        use_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.spatial_merge_size = spatial_merge_size
        self.norm_eps = norm_eps
        self.use_bias = use_bias
        self.norm = Mistral3RMSNorm(eps=norm_eps, name="norm")
        self.patch_merger = Mistral3PatchMerger(
            vision_dim, spatial_merge_size, name="patch_merger"
        )
        self.linear_1 = layers.Dense(text_dim, use_bias=use_bias, name="linear_1")
        self.linear_2 = layers.Dense(text_dim, use_bias=use_bias, name="linear_2")

    def call(self, image_features, grid_sizes):
        x = self.norm(image_features)
        x = self.patch_merger(x, grid_sizes=grid_sizes)
        return self.linear_2(ops.gelu(self.linear_1(x), approximate=False))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_dim": self.vision_dim,
                "text_dim": self.text_dim,
                "spatial_merge_size": self.spatial_merge_size,
                "norm_eps": self.norm_eps,
                "use_bias": self.use_bias,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Mistral3TextAttention(layers.Layer):
    """Mistral grouped-query causal self-attention (text decoder).

    Bias-free projections, half-rotation rotary, K/V head repetition for GQA,
    ``head_dim`` decoupled from ``embed_dim // num_heads``. A KV cache can be
    threaded through ``past_key_value``.

    Args:
        embed_dim: Text width (output dim of ``output_proj``).
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (GQA).
        head_dim: Per-head dim.
    """

    def __init__(self, embed_dim, num_heads, num_kv_heads, head_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim**-0.5
        self.query = layers.Dense(num_heads * head_dim, use_bias=False, name="query")
        self.key = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="key")
        self.value = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="value")
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")

    def call(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        b = ops.shape(hidden_states)[0]
        q_len = ops.shape(hidden_states)[1]
        q = ops.reshape(
            self.query(hidden_states), (b, q_len, self.num_heads, self.head_dim)
        )
        k = ops.reshape(
            self.key(hidden_states), (b, q_len, self.num_kv_heads, self.head_dim)
        )
        v = ops.reshape(
            self.value(hidden_states), (b, q_len, self.num_kv_heads, self.head_dim)
        )
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = ops.concatenate([past_k, k], axis=2)
            v = ops.concatenate([past_v, v], axis=2)
        new_kv = (k, v) if use_cache else None

        if self.num_kv_groups > 1:
            k = ops.repeat(k, self.num_kv_groups, axis=1)
            v = ops.repeat(v, self.num_kv_groups, axis=1)

        out = fused_attention(q, k, v, self.scaling, attention_mask)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, q_len, self.num_heads * self.head_dim)
        )
        out = self.output_proj(out)
        return (out, new_kv) if use_cache else out

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        # Single-token attention against a fixed-size KV cache written in
        # place at ``write_pos``.
        b = ops.shape(hidden_states)[0]
        q = ops.reshape(
            self.query(hidden_states), (b, 1, self.num_heads, self.head_dim)
        )
        k = ops.reshape(
            self.key(hidden_states), (b, 1, self.num_kv_heads, self.head_dim)
        )
        v = ops.reshape(
            self.value(hidden_states), (b, 1, self.num_kv_heads, self.head_dim)
        )
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))
        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        # rope runs in float32; match the cache dtype before writing
        k = ops.cast(k, cache_k.dtype)
        v = ops.cast(v, cache_v.dtype)
        cache_k = ops.slice_update(cache_k, (0, 0, write_pos, 0), k)
        cache_v = ops.slice_update(cache_v, (0, 0, write_pos, 0), v)
        kk, vv = cache_k, cache_v
        if self.num_kv_groups > 1:
            kk = ops.repeat(kk, self.num_kv_groups, axis=1)
            vv = ops.repeat(vv, self.num_kv_groups, axis=1)
        out = fused_attention(q, kk, vv, self.scaling, key_mask)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, 1, self.num_heads * self.head_dim)
        )
        return self.output_proj(out), cache_k, cache_v

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Mistral3TextMLP(layers.Layer):
    """SwiGLU feed-forward block: ``down(silu(gate(x)) * up(x))``, bias-free."""

    def __init__(self, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.gate = layers.Dense(mlp_dim, use_bias=False, name="gate")
        self.up = layers.Dense(mlp_dim, use_bias=False, name="up")
        self.down = layers.Dense(embed_dim, use_bias=False, name="down")

    def call(self, x):
        return self.down(ops.silu(self.gate(x)) * self.up(x))

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "mlp_dim": self.mlp_dim})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Mistral3DecoderLayer(layers.Layer):
    """One Mistral text block: pre-norm attention, then pre-norm SwiGLU.

    Args:
        embed_dim: Text / residual-stream width.
        mlp_dim: SwiGLU hidden width.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (GQA).
        head_dim: Per-head dim.
        norm_eps: Epsilon shared by both RMSNorms.
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.attention_norm = Mistral3RMSNorm(eps=norm_eps, name="attention_norm")
        self.attention = Mistral3TextAttention(
            embed_dim, num_heads, num_kv_heads, head_dim, name="attention"
        )
        self.mlp_norm = Mistral3RMSNorm(eps=norm_eps, name="mlp_norm")
        self.mlp = Mistral3TextMLP(embed_dim, mlp_dim, name="mlp")

    def call(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        attn_out = self.attention(
            hidden_states,
            cos,
            sin,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        new_kv = None
        if use_cache:
            attn_out, new_kv = attn_out
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return (hidden_states, new_kv) if use_cache else hidden_states

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        residual = hidden_states
        x = self.attention_norm(hidden_states)
        attn_out, cache_k, cache_v = self.attention.decode_step(
            x, cos, sin, cache_k, cache_v, write_pos, key_mask
        )
        hidden_states = residual + attn_out
        residual = hidden_states
        x = self.mlp_norm(hidden_states)
        hidden_states = residual + self.mlp(x)
        return hidden_states, cache_k, cache_v

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "norm_eps": self.norm_eps,
            }
        )
        return config
