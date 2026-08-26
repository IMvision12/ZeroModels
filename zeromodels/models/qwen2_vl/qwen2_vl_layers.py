import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


def quick_gelu(x):
    """``x * sigmoid(1.702 * x)``: the GELU approximation Qwen2-VL's vision
    MLP uses (``hidden_act="quick_gelu"``)."""
    return x * ops.sigmoid(1.702 * x)


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen2VLRMSNorm(layers.Layer):
    """RMSNorm (Llama/Qwen style): normalize by RMS in float32, then scale.

    Args:
        eps: Variance epsilon. Defaults to ``1e-6``.
    """

    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True,
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
class Qwen2VLMLP(layers.Layer):
    """SwiGLU MLP: ``down(silu(gate(x)) * up(x))``.

    The Qwen2 text decoder uses this bias-free; Qwen2.5-VL's vision MLP reuses
    it with ``use_bias=True``.
    """

    def __init__(self, embed_dim, mlp_dim, use_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.use_bias = use_bias
        self.gate = layers.Dense(mlp_dim, use_bias=use_bias, name="gate")
        self.up = layers.Dense(mlp_dim, use_bias=use_bias, name="up")
        self.down = layers.Dense(embed_dim, use_bias=use_bias, name="down")

    def call(self, x):
        return self.down(ops.silu(self.gate(x)) * self.up(x))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "use_bias": self.use_bias,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen2VLAttention(layers.Layer):
    """Grouped-query causal self-attention with multimodal rotary positions.

    ``query`` / ``key`` / ``value`` carry a bias (Qwen2), ``output_proj`` does
    not. ``num_kv_heads`` may be smaller than ``num_heads`` (GQA); the K/V heads
    are repeated to match Q. The merged M-RoPE ``cos`` / ``sin`` (shape
    ``(batch, seq, head_dim)``) are computed by the model and passed in. A KV
    cache can be threaded through ``past_key_value`` for incremental decoding.

    Call args:
        hidden_states: ``(batch, seq, hidden)``.
        cos, sin: merged rotary tables ``(batch, seq, head_dim)``.
        attention_mask: additive mask broadcastable to
            ``(batch, 1, q_len, kv_len)`` (``0`` keep / large-negative mask).
        past_key_value: optional ``(past_k, past_v)`` each
            ``(batch, kv_heads, past_len, head_dim)``.
        use_cache: when ``True``, also return the updated ``(k, v)``.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads,
        head_dim=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or embed_dim // num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = self.head_dim**-0.5

        self.query = layers.Dense(
            num_heads * self.head_dim, use_bias=True, name="query"
        )
        self.key = layers.Dense(num_kv_heads * self.head_dim, use_bias=True, name="key")
        self.value = layers.Dense(
            num_kv_heads * self.head_dim, use_bias=True, name="value"
        )
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")

    def _split_heads(self, x, num_heads):
        b = ops.shape(x)[0]
        s = ops.shape(x)[1]
        x = ops.reshape(x, (b, s, num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

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

        query = self._split_heads(self.query(hidden_states), self.num_heads)
        key = self._split_heads(self.key(hidden_states), self.num_kv_heads)
        value = self._split_heads(self.value(hidden_states), self.num_kv_heads)

        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        half = self.head_dim // 2
        query = query * cos + (
            ops.concatenate([-query[..., half:], query[..., :half]], axis=-1) * sin
        )
        key = key * cos + (
            ops.concatenate([-key[..., half:], key[..., :half]], axis=-1) * sin
        )

        if past_key_value is not None:
            past_k, past_v = past_key_value
            key = ops.concatenate([past_k, key], axis=2)
            value = ops.concatenate([past_v, value], axis=2)
        new_key_value = (key, value) if use_cache else None

        if self.num_kv_groups > 1:
            key = ops.repeat(key, self.num_kv_groups, axis=1)
            value = ops.repeat(value, self.num_kv_groups, axis=1)

        out = fused_attention(query, key, value, self.scaling, attention_mask)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, q_len, self.num_heads * self.head_dim))
        out = self.output_proj(out)
        return (out, new_key_value) if use_cache else out

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        # Single-token attention against a fixed-size KV cache written at ``write_pos``.
        # ``cos``/``sin`` are the merged M-RoPE tables for this token; ``key_mask``
        # blocks the still-empty cache slots.
        b = ops.shape(hidden_states)[0]
        query = self._split_heads(self.query(hidden_states), self.num_heads)
        key = self._split_heads(self.key(hidden_states), self.num_kv_heads)
        value = self._split_heads(self.value(hidden_states), self.num_kv_heads)
        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        half = self.head_dim // 2
        query = query * cos + (
            ops.concatenate([-query[..., half:], query[..., :half]], axis=-1) * sin
        )
        key = key * cos + (
            ops.concatenate([-key[..., half:], key[..., :half]], axis=-1) * sin
        )
        # rope runs in float32; match the cache dtype before writing
        key = ops.cast(key, cache_k.dtype)
        value = ops.cast(value, cache_v.dtype)
        cache_k = ops.slice_update(cache_k, (0, 0, write_pos, 0), key)
        cache_v = ops.slice_update(cache_v, (0, 0, write_pos, 0), value)
        kk, vv = cache_k, cache_v
        if self.num_kv_groups > 1:
            kk = ops.repeat(kk, self.num_kv_groups, axis=1)
            vv = ops.repeat(vv, self.num_kv_groups, axis=1)
        out = fused_attention(query, kk, vv, self.scaling, key_mask)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, 1, self.num_heads * self.head_dim))
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
class Qwen2VLDecoderLayer(layers.Layer):
    """One Qwen2 decoder block: pre-norm GQA attention then pre-norm SwiGLU."""

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_heads,
        num_kv_heads,
        head_dim=None,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or embed_dim // num_heads
        self.norm_eps = norm_eps

        self.attention_norm = Qwen2VLRMSNorm(eps=norm_eps, name="attention_norm")
        self.attention = Qwen2VLAttention(
            embed_dim,
            num_heads,
            num_kv_heads,
            head_dim=self.head_dim,
            name="attention",
        )
        self.mlp_norm = Qwen2VLRMSNorm(eps=norm_eps, name="mlp_norm")
        self.mlp = Qwen2VLMLP(embed_dim, mlp_dim, name="mlp")

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
        if use_cache:
            attn_out, new_key_value = attn_out
        else:
            new_key_value = None
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return (hidden_states, new_key_value) if use_cache else hidden_states

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


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen2VLPatchEmbed(layers.Layer):
    """Patch embedding for Qwen2-VL's vision tower.

    HF uses a ``Conv3d`` whose kernel equals its stride and tiles each
    ``(temporal_patch_size, patch_size, patch_size)`` patch exactly once, i.e.
    a per-patch linear projection. The processor already flattens every patch
    to a ``in_channels * temporal_patch_size * patch_size**2`` vector, so this
    is just a bias-free ``Dense`` (no spatial axes, hence layout-agnostic).

    Call args:
        x: ``(num_patches, in_channels * temporal_patch_size * patch_size**2)``.
    Returns:
        ``(num_patches, embed_dim)``.
    """

    def __init__(self, embed_dim, use_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.use_bias = use_bias
        self.proj = layers.Dense(embed_dim, use_bias=use_bias, name="proj")

    def call(self, x):
        return self.proj(x)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "use_bias": self.use_bias})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen2VLVisionAttention(layers.Layer):
    """Full (non-causal) vision attention with 2D rotary positions.

    Operates on the flattened patch sequence of (possibly) several images. An
    additive ``attention_mask`` makes attention block-diagonal per image
    (built from ``cu_seqlens`` by the vision model). A fused ``qkv`` projection
    and an output ``proj`` both carry a bias (matching HF).

    Call args:
        hidden_states: ``(seq, embed_dim)``.
        cos, sin: vision rotary tables ``(seq, head_dim)``.
        attention_mask: additive mask ``(1, 1, seq, seq)`` or ``None``.
    """

    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = layers.Dense(embed_dim * 3, use_bias=True, name="qkv")
        self.proj = layers.Dense(embed_dim, use_bias=True, name="proj")

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

        out = fused_attention(query, key, value, self.scaling, attention_mask)
        out = ops.transpose(out[0], (1, 0, 2))
        out = ops.reshape(out, (seq, self.embed_dim))
        return self.proj(out)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen2VLVisionMLP(layers.Layer):
    """Vision MLP: ``fc2(quick_gelu(fc1(x)))`` (both projections biased)."""

    def __init__(self, embed_dim, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.fc1 = layers.Dense(hidden_dim, use_bias=True, name="fc1")
        self.fc2 = layers.Dense(embed_dim, use_bias=True, name="fc2")

    def call(self, x):
        return self.fc2(quick_gelu(self.fc1(x)))

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "hidden_dim": self.hidden_dim})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen2VLVisionBlock(layers.Layer):
    """Pre-norm vision block: ``h += attn(norm1(h)); h += mlp(norm2(h))``.

    Norms are ``LayerNorm`` with ``eps=1e-6`` (Qwen2-VL; Qwen2.5-VL switches
    these to RMSNorm).
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        self.attn = Qwen2VLVisionAttention(embed_dim, num_heads, name="attn")
        self.mlp = Qwen2VLVisionMLP(embed_dim, int(embed_dim * mlp_ratio), name="mlp")

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
                "mlp_ratio": self.mlp_ratio,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen2VLPatchMerger(layers.Layer):
    """Merge each 2x2 (``spatial_merge_size``) patch group and project to the
    LLM hidden size: ``mlp_fc2(gelu(mlp_fc1(layernorm(x).reshape(-1, ctx*m^2))))``.

    ``ln_q`` is a ``LayerNorm`` over the vision ``context_dim``; ``gelu`` is the
    exact GELU (HF uses ``nn.GELU()`` here, not quick-gelu).
    """

    def __init__(
        self, dim, context_dim, spatial_merge_size=2, use_rmsnorm=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.context_dim = context_dim
        self.spatial_merge_size = spatial_merge_size
        self.use_rmsnorm = use_rmsnorm
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = (
            Qwen2VLRMSNorm(eps=1e-6, name="ln_q")
            if use_rmsnorm
            else layers.LayerNormalization(epsilon=1e-6, name="ln_q")
        )
        self.mlp_fc1 = layers.Dense(self.hidden_size, use_bias=True, name="mlp_fc1")
        self.mlp_fc2 = layers.Dense(dim, use_bias=True, name="mlp_fc2")

    def call(self, x):
        x = self.ln_q(x)
        x = ops.reshape(x, (-1, self.hidden_size))
        x = self.mlp_fc1(x)
        x = ops.gelu(x, approximate=False)
        return self.mlp_fc2(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "context_dim": self.context_dim,
                "spatial_merge_size": self.spatial_merge_size,
                "use_rmsnorm": self.use_rmsnorm,
            }
        )
        return config
