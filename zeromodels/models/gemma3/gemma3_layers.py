import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3RMSNorm(layers.Layer):
    """Gemma root-mean-square layer norm with the ``(1 + weight)`` scale.

    Normalizes the last axis by its RMS in float32 and multiplies by
    ``(1 + weight)`` before casting back (zero-initialized weights). Used for
    the decoder norms, the per-head QK norms, and the projector soft-token
    norm.

    Args:
        eps: Variance epsilon. Defaults to ``1e-6``.
    """

    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight", shape=(input_shape[-1],), initializer="zeros", trainable=True
        )
        self.built = True

    def call(self, x):
        dtype = x.dtype
        x = ops.cast(x, "float32")
        variance = ops.mean(ops.square(x), axis=-1, keepdims=True)
        x = x * ops.rsqrt(variance + self.eps)
        x = x * (1.0 + ops.cast(self.weight, "float32"))
        return ops.cast(x, dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3MLP(layers.Layer):
    """Gemma GeGLU feed-forward block: ``down(gelu_tanh(gate(x)) * up(x))``.

    Bias-free GeGLU: the ``gate`` branch uses the tanh ``gelu`` approximation,
    is multiplied elementwise by the ``up`` projection, and ``down`` projects
    the result back to ``embed_dim``.

    Args:
        embed_dim: Model width (input and output dimension).
        mlp_dim: Hidden width of the ``gate`` / ``up`` projections.
    """

    def __init__(self, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.gate = layers.Dense(mlp_dim, use_bias=False, name="gate")
        self.up = layers.Dense(mlp_dim, use_bias=False, name="up")
        self.down = layers.Dense(embed_dim, use_bias=False, name="down")

    def build(self, input_shape):
        self.gate.build(input_shape)
        self.up.build(input_shape)
        self.down.build(tuple(input_shape[:-1]) + (self.mlp_dim,))
        self.built = True

    def call(self, x):
        return self.down(ops.gelu(self.gate(x), approximate=True) * self.up(x))

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "mlp_dim": self.mlp_dim})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3Attention(layers.Layer):
    """Gemma 3 grouped-query causal self-attention with per-head QK norms.

    Bias-free projections; the reshaped per-head query and key are
    ``(1 + w)`` RMS-normed *before* rotary; scores are scaled by
    ``query_pre_attn_scalar**-0.5``. The (optionally sliding-window, possibly
    image-bidirectional) additive mask and the layer-type-specific rotary
    tables are supplied by the caller.

    Args:
        embed_dim: Model width (output dim of ``output_proj``).
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (GQA; the 1B is MQA).
        head_dim: Per-head dim.
        query_pre_attn_scalar: Scaling denominator for the attention scores.
        norm_eps: Epsilon of the QK norms.

    Call args:
        hidden_states, cos, sin, attention_mask, past_key_value, use_cache:
            standard decoder-attention arguments.

    Returns:
        Output ``(batch, q_len, embed_dim)``, or ``(output, (key, value))``
        when ``use_cache`` is set.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        query_pre_attn_scalar=256.0,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.norm_eps = norm_eps
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = query_pre_attn_scalar**-0.5
        self.query = layers.Dense(num_heads * head_dim, use_bias=False, name="query")
        self.key = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="key")
        self.value = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="value")
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")
        self.query_norm = Gemma3RMSNorm(eps=norm_eps, name="query_norm")
        self.key_norm = Gemma3RMSNorm(eps=norm_eps, name="key_norm")

    def build(self, input_shape):
        self.query.build(input_shape)
        self.key.build(input_shape)
        self.value.build(input_shape)
        self.output_proj.build(tuple(input_shape[:-1]) + (self.num_heads * self.head_dim,))
        self.query_norm.build(tuple(input_shape[:-1]) + (self.num_heads, self.head_dim))
        self.key_norm.build(tuple(input_shape[:-1]) + (self.num_kv_heads, self.head_dim))
        self.built = True

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
                self.query(hidden_states), (b, q_len, self.num_heads, self.head_dim)
            )
        )
        k = self.key_norm(
            ops.reshape(
                self.key(hidden_states), (b, q_len, self.num_kv_heads, self.head_dim)
            )
        )
        v = ops.reshape(
            self.value(hidden_states), (b, q_len, self.num_kv_heads, self.head_dim)
        )
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        half = self.head_dim // 2
        q = q * cos + ops.concatenate([-q[..., half:], q[..., :half]], axis=-1) * sin
        k = k * cos + ops.concatenate([-k[..., half:], k[..., :half]], axis=-1) * sin

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
        q = q * cos + ops.concatenate([-q[..., half:], q[..., :half]], axis=-1) * sin
        k = k * cos + ops.concatenate([-k[..., half:], k[..., :half]], axis=-1) * sin
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
                "query_pre_attn_scalar": self.query_pre_attn_scalar,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3DecoderLayer(layers.Layer):
    """One Gemma 3 block with the four-norm sandwich and QK-normed attention.

    Computes ``h = x + post_attention_norm(attention(attention_norm(x)))``
    followed by
    ``h = h + post_feedforward_norm(mlp(pre_feedforward_norm(h)))``.

    Args:
        embed_dim: Model / residual-stream width.
        mlp_dim: GeGLU hidden width.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.
        head_dim: Per-head dim.
        query_pre_attn_scalar: Attention scaling denominator.
        norm_eps: Epsilon shared by all norms.
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        query_pre_attn_scalar=256.0,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.norm_eps = norm_eps
        self.attention_norm = Gemma3RMSNorm(eps=norm_eps, name="attention_norm")
        self.attention = Gemma3Attention(
            embed_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            query_pre_attn_scalar,
            norm_eps,
            name="attention",
        )
        self.post_attention_norm = Gemma3RMSNorm(
            eps=norm_eps, name="post_attention_norm"
        )
        self.pre_feedforward_norm = Gemma3RMSNorm(
            eps=norm_eps, name="pre_feedforward_norm"
        )
        self.mlp = Gemma3MLP(embed_dim, mlp_dim, name="mlp")
        self.post_feedforward_norm = Gemma3RMSNorm(
            eps=norm_eps, name="post_feedforward_norm"
        )

    def build(self, input_shape):
        self.attention_norm.build(input_shape)
        self.attention.build(input_shape)
        self.post_attention_norm.build(input_shape)
        self.pre_feedforward_norm.build(input_shape)
        self.mlp.build(input_shape)
        self.post_feedforward_norm.build(input_shape)
        self.built = True

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
        hidden_states = residual + self.post_attention_norm(attn_out)
        residual = hidden_states
        hidden_states = self.pre_feedforward_norm(hidden_states)
        hidden_states = residual + self.post_feedforward_norm(self.mlp(hidden_states))
        return (hidden_states, new_kv) if use_cache else hidden_states

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        residual = hidden_states
        x = self.attention_norm(hidden_states)
        attn_out, cache_k, cache_v = self.attention.decode_step(
            x, cos, sin, cache_k, cache_v, write_pos, key_mask
        )
        hidden_states = residual + self.post_attention_norm(attn_out)
        residual = hidden_states
        x = self.pre_feedforward_norm(hidden_states)
        hidden_states = residual + self.post_feedforward_norm(self.mlp(x))
        return hidden_states, cache_k, cache_v

    def compute_output_spec(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        return keras.KerasTensor(hidden_states.shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "query_pre_attn_scalar": self.query_pre_attn_scalar,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3VisionAttention(layers.Layer):
    """SigLIP full (bidirectional) self-attention: all projections biased."""

    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.query = layers.Dense(embed_dim, name="query")
        self.key = layers.Dense(embed_dim, name="key")
        self.value = layers.Dense(embed_dim, name="value")
        self.output_proj = layers.Dense(embed_dim, name="output_proj")

    def call(self, hidden_states):
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
        out = fused_attention(q, k, v, self.scaling)
        out = ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (b, seq, self.embed_dim))
        return self.output_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3VisionLayer(layers.Layer):
    """One SigLIP encoder block: pre-LN attention + pre-LN gelu-tanh MLP.

    Args:
        embed_dim: Vision hidden width.
        mlp_dim: Vision MLP hidden width.
        num_heads: Attention heads.
        norm_eps: LayerNorm epsilon (1e-6).
    """

    def __init__(self, embed_dim, mlp_dim, num_heads, norm_eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.norm_eps = norm_eps
        self.layer_norm1 = layers.LayerNormalization(
            epsilon=norm_eps, name="layer_norm1"
        )
        self.attention = Gemma3VisionAttention(embed_dim, num_heads, name="attention")
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=norm_eps, name="layer_norm2"
        )
        self.fc1 = layers.Dense(mlp_dim, name="fc1")
        self.fc2 = layers.Dense(embed_dim, name="fc2")

    def call(self, hidden_states):
        hidden_states = hidden_states + self.attention(self.layer_norm1(hidden_states))
        x = self.layer_norm2(hidden_states)
        x = self.fc2(ops.gelu(self.fc1(x), approximate=True))
        return hidden_states + x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_heads": self.num_heads,
                "norm_eps": self.norm_eps,
            }
        )
        return config
