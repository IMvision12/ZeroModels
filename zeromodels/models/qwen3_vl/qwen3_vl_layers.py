import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen3VLRMSNorm(layers.Layer):
    """RMSNorm: normalize by RMS in float32, then scale."""

    def __init__(self, eps=1e-6, **kwargs):
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
class Qwen3VLMLP(layers.Layer):
    """SwiGLU MLP: ``down(silu(gate(x)) * up(x))`` (bias-free; Qwen3 text)."""

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
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen3VLTextAttention(layers.Layer):
    """Qwen3 GQA attention: no qkv bias, per-head RMSNorm on q and k (pre-RoPE)."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim**-0.5
        self.query = layers.Dense(num_heads * head_dim, use_bias=False, name="query")
        self.key = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="key")
        self.value = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="value")
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")
        self.query_norm = Qwen3VLRMSNorm(eps=norm_eps, name="query_norm")
        self.key_norm = Qwen3VLRMSNorm(eps=norm_eps, name="key_norm")

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
        q = self.query_norm(
            ops.reshape(
                self.query(hidden_states),
                (b, q_len, self.num_heads, self.head_dim),
            )
        )
        k = self.key_norm(
            ops.reshape(
                self.key(hidden_states),
                (b, q_len, self.num_kv_heads, self.head_dim),
            )
        )
        v = ops.reshape(
            self.value(hidden_states),
            (b, q_len, self.num_kv_heads, self.head_dim),
        )
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        half = self.head_dim // 2
        q = q * cos + (ops.concatenate([-q[..., half:], q[..., :half]], axis=-1) * sin)
        k = k * cos + (ops.concatenate([-k[..., half:], k[..., :half]], axis=-1) * sin)

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
            ops.transpose(out, (0, 2, 1, 3)),
            (b, q_len, self.num_heads * self.head_dim),
        )
        out = self.output_proj(out)
        return (out, new_kv) if use_cache else out

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        # Single-token QK-norm GQA attention against a fixed KV cache; merged M-RoPE
        # ``cos``/``sin`` in, ``key_mask`` blocks the still-empty cache slots.
        b = ops.shape(hidden_states)[0]
        q = self.query_norm(
            ops.reshape(
                self.query(hidden_states), (b, 1, self.num_heads, self.head_dim)
            )
        )
        k = self.key_norm(
            ops.reshape(
                self.key(hidden_states), (b, 1, self.num_kv_heads, self.head_dim)
            )
        )
        v = ops.reshape(
            self.value(hidden_states), (b, 1, self.num_kv_heads, self.head_dim)
        )
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))
        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        half = self.head_dim // 2
        q = q * cos + (ops.concatenate([-q[..., half:], q[..., :half]], axis=-1) * sin)
        k = k * cos + (ops.concatenate([-k[..., half:], k[..., :half]], axis=-1) * sin)
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
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen3VLTextDecoderLayer(layers.Layer):
    """Qwen3 decoder block: pre-norm QK-norm attention then pre-norm SwiGLU."""

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.attention_norm = Qwen3VLRMSNorm(eps=norm_eps, name="attention_norm")
        self.attention = Qwen3VLTextAttention(
            embed_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            norm_eps,
            name="attention",
        )
        self.mlp_norm = Qwen3VLRMSNorm(eps=norm_eps, name="mlp_norm")
        self.mlp = Qwen3VLMLP(embed_dim, mlp_dim, name="mlp")

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


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen3VLVisionPatchEmbed(layers.Layer):
    """Per-patch linear projection with bias (HF's biased Conv3d as a Dense)."""

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


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen3VLVisionAttention(layers.Layer):
    """Full vision attention with 2D rotary positions + block-diagonal mask."""

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
        qkv = ops.reshape(
            self.qkv(hidden_states), (seq, 3, self.num_heads, self.head_dim)
        )
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
class Qwen3VLVisionMLP(layers.Layer):
    """Non-gated vision MLP: ``fc2(gelu_tanh(fc1(x)))`` (biased)."""

    def __init__(self, hidden_size, intermediate_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.linear_fc1 = layers.Dense(
            intermediate_size, use_bias=True, name="linear_fc1"
        )
        self.linear_fc2 = layers.Dense(hidden_size, use_bias=True, name="linear_fc2")

    def call(self, x):
        return self.linear_fc2(ops.gelu(self.linear_fc1(x), approximate=True))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen3VLVisionBlock(layers.Layer):
    """Pre-norm vision block: LayerNorm + full attention + GELU MLP."""

    def __init__(self, embed_dim, num_heads, intermediate_size, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        self.attn = Qwen3VLVisionAttention(embed_dim, num_heads, name="attn")
        self.mlp = Qwen3VLVisionMLP(embed_dim, intermediate_size, name="mlp")

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
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen3VLVisionPatchMerger(layers.Layer):
    """Merge 2x2 patches and project to ``out_hidden_size``.

    ``use_postshuffle_norm`` controls whether the LayerNorm runs after the
    merge (DeepStack mergers) or before it (the main merger).
    """

    def __init__(
        self,
        out_hidden_size,
        context_dim,
        spatial_merge_size=2,
        use_postshuffle_norm=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.out_hidden_size = out_hidden_size
        self.context_dim = context_dim
        self.spatial_merge_size = spatial_merge_size
        self.use_postshuffle_norm = use_postshuffle_norm
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")
        self.linear_fc1 = layers.Dense(
            self.hidden_size, use_bias=True, name="linear_fc1"
        )
        self.linear_fc2 = layers.Dense(
            out_hidden_size, use_bias=True, name="linear_fc2"
        )

    def call(self, x):
        if self.use_postshuffle_norm:
            x = self.norm(ops.reshape(x, (-1, self.hidden_size)))
        else:
            x = ops.reshape(self.norm(x), (-1, self.hidden_size))
        return self.linear_fc2(ops.gelu(self.linear_fc1(x), approximate=False))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_hidden_size": self.out_hidden_size,
                "context_dim": self.context_dim,
                "spatial_merge_size": self.spatial_merge_size,
                "use_postshuffle_norm": self.use_postshuffle_norm,
            }
        )
        return config
