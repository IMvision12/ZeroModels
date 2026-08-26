import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen2RMSNorm(layers.Layer):
    """Root-mean-square layer norm (Llama / Qwen style).

    Normalizes the last axis by its RMS in float32 (for numerical stability),
    casts back to the input dtype, then scales by a learned per-channel weight.
    There is no mean subtraction and no bias. Shape-preserving:
    ``(..., dim) -> (..., dim)``.

    Args:
        eps: Variance epsilon added before the reciprocal square root.
            Defaults to ``1e-6``.
    """

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
class Qwen2MLP(layers.Layer):
    """SwiGLU feed-forward block: ``down(silu(gate(x)) * up(x))``.

    Two parallel bias-free projections to ``mlp_dim``: a SiLU-gated ``gate`` and
    a linear ``up``: are multiplied elementwise, then projected back to
    ``embed_dim`` by ``down``. Shape-preserving on the last axis.

    Args:
        embed_dim: Model / residual-stream width (input and output dim).
        mlp_dim: Hidden expansion width of the ``gate`` / ``up`` projections.
    """

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
class Qwen2Attention(layers.Layer):
    """Grouped-query causal self-attention with 1D rotary positions (Qwen2).

    ``query`` / ``key`` / ``value`` are biased projections; ``output_proj`` is
    bias-free. When ``num_kv_heads < num_heads`` (GQA) the K/V heads are repeated
    to match the query heads. Rotary embeddings are applied to Q and K from the
    ``cos`` / ``sin`` tables computed by the model. A KV cache can be threaded
    through ``past_key_value`` for O(1)-per-token incremental decoding.

    Args:
        embed_dim: Model width (output dim of ``output_proj``).
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (``<= num_heads`` for GQA).
        head_dim: Per-head dim; defaults to ``embed_dim // num_heads``.

    Call args:
        hidden_states: ``(batch, q_len, embed_dim)``.
        cos, sin: rotary tables ``(batch, q_len, head_dim)``.
        attention_mask: additive mask broadcastable to
            ``(batch, 1, q_len, kv_len)`` (``0`` keep / large-negative block), or
            ``None``.
        past_key_value: optional ``(past_k, past_v)``, each
            ``(batch, num_kv_heads, past_len, head_dim)``.
        use_cache: when ``True``, also return the updated ``(key, value)``.

    Returns:
        Output ``(batch, q_len, embed_dim)``, or ``(output, (key, value))`` when
        ``use_cache`` is set.
    """

    def __init__(self, embed_dim, num_heads, num_kv_heads, head_dim=None, **kwargs):
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
        return ops.transpose(
            ops.reshape(x, (b, s, num_heads, self.head_dim)), (0, 2, 1, 3)
        )

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
        # Single-token attention against a fixed-size KV cache written in place at
        # ``write_pos`` (constant cache shape -> compilable decode loop). ``key_mask``
        # (additive, (.., max_len)) blocks the still-empty cache slots.
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
class Qwen2DecoderLayer(layers.Layer):
    """One Qwen2 transformer block: pre-norm GQA attention, then pre-norm SwiGLU.

    Computes ``h = x + attention(attention_norm(x))`` followed by
    ``h = h + mlp(mlp_norm(h))``: RMSNorm pre-normalization with residual adds.
    The rotary tables, mask, and KV cache pass straight through to the attention.

    Args:
        embed_dim: Model / residual-stream width.
        mlp_dim: SwiGLU hidden width.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (GQA).
        head_dim: Per-head dim; defaults to ``embed_dim // num_heads``.
        norm_eps: Epsilon shared by both RMSNorms.

    Call args:
        hidden_states, cos, sin, attention_mask, past_key_value, use_cache: as in
            :class:`Qwen2Attention`.

    Returns:
        The block output, or ``(output, (key, value))`` when ``use_cache`` is set.
    """

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
        self.attention_norm = Qwen2RMSNorm(eps=norm_eps, name="attention_norm")
        self.attention = Qwen2Attention(
            embed_dim, num_heads, num_kv_heads, head_dim=self.head_dim, name="attention"
        )
        self.mlp_norm = Qwen2RMSNorm(eps=norm_eps, name="mlp_norm")
        self.mlp = Qwen2MLP(embed_dim, mlp_dim, name="mlp")

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
        new_key_value = None
        if use_cache:
            attn_out, new_key_value = attn_out
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return (hidden_states, new_key_value) if use_cache else hidden_states

    def compute_output_spec(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        # The residual stream keeps its shape; an explicit spec stops the
        # functional builder from tracing the dynamic-shape reshapes / rope in
        # attention (which resolve only at run time).
        return keras.KerasTensor(hidden_states.shape, dtype=self.compute_dtype)

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
