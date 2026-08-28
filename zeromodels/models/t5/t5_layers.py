import math

import keras
from keras import layers, ops

MASK_NEG = -1e9


def relative_position_bucket(
    relative_position, bidirectional, num_buckets, max_distance
):
    """Bucket relative positions for T5 relative attention bias (memory - query)."""
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets = ops.cast(relative_position > 0, "int32") * num_buckets
        relative_position = ops.abs(relative_position)
    else:
        relative_position = -ops.minimum(
            relative_position, ops.zeros_like(relative_position)
        )
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    # guard log(0): small positions are masked out by `is_small` anyway
    safe = ops.cast(ops.maximum(relative_position, 1), "float32")
    relative_if_large = max_exact + ops.cast(
        ops.log(safe / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact),
        "int32",
    )
    relative_if_large = ops.minimum(relative_if_large, num_buckets - 1)
    relative_buckets = relative_buckets + ops.where(
        is_small, ops.cast(relative_position, "int32"), relative_if_large
    )
    return relative_buckets


def compute_relative_bias(
    embedding, query_length, key_length, bidirectional, num_buckets, max_distance
):
    """Learned relative position bias ``(1, num_heads, query_length, key_length)``.

    ``embedding`` is an ``Embedding(num_buckets, num_heads)`` owned by the stack (T5
    shares one bias across all layers of an encoder / decoder).
    """
    context = ops.arange(query_length)[:, None]
    memory = ops.arange(key_length)[None, :]
    bucket = relative_position_bucket(
        memory - context, bidirectional, num_buckets, max_distance
    )
    values = embedding(bucket)  # (query, key, heads)
    return ops.transpose(values, (2, 0, 1))[None]  # (1, heads, query, key)


class T5LayerNorm(layers.Layer):
    """T5-style RMSNorm: scale only, no mean subtraction, no bias."""

    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            shape=(input_shape[-1],), initializer="ones", name="weight"
        )
        self.built = True

    def call(self, hidden_states):
        variance = ops.mean(
            ops.square(ops.cast(hidden_states, "float32")), axis=-1, keepdims=True
        )
        hidden_states = hidden_states * ops.cast(
            ops.rsqrt(variance + self.eps), hidden_states.dtype
        )
        return self.weight * hidden_states


class T5Attention(layers.Layer):
    """T5 multi-head attention: bias-free q/k/v/o, no 1/sqrt(d) scaling. The additive
    relative position bias (plus mask) is computed by the stack and passed in. The
    q/k/v/o projections are named ``{prefix}_{q,k,v,o}`` so every weight's last-two path
    segments are unique across blocks (no cross-block collision)."""

    def __init__(self, embed_dim, key_value_dim, num_heads, prefix, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.key_value_dim = key_value_dim
        self.num_heads = num_heads
        self.inner_dim = num_heads * key_value_dim
        self.q = layers.Dense(self.inner_dim, use_bias=False, name=f"{prefix}_q")
        self.k = layers.Dense(self.inner_dim, use_bias=False, name=f"{prefix}_k")
        self.v = layers.Dense(self.inner_dim, use_bias=False, name=f"{prefix}_v")
        self.o = layers.Dense(embed_dim, use_bias=False, name=f"{prefix}_o")

    def build(self, input_shape):
        self.q.build(input_shape)
        self.k.build(input_shape)
        self.v.build(input_shape)
        self.o.build(tuple(input_shape[:-1]) + (self.inner_dim,))
        self.built = True

    def split_heads(self, x, batch):
        x = ops.reshape(x, (batch, -1, self.num_heads, self.key_value_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def call(self, hidden_states, position_bias, key_value_states=None):
        batch = int(ops.shape(hidden_states)[0])
        current = key_value_states if key_value_states is not None else hidden_states
        query = self.split_heads(self.q(hidden_states), batch)
        key = self.split_heads(self.k(current), batch)
        value = self.split_heads(self.v(current), batch)

        scores = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2)))
        scores = scores + ops.cast(position_bias, scores.dtype)
        weights = ops.softmax(ops.cast(scores, "float32"), axis=-1)
        weights = ops.cast(weights, hidden_states.dtype)

        out = ops.matmul(weights, value)  # (batch, heads, query, d_kv)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (batch, -1, self.inner_dim))
        return self.o(out)


class T5SelfAttentionLayer(layers.Layer):
    def __init__(self, embed_dim, key_value_dim, num_heads, eps, prefix, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = T5LayerNorm(eps, name=f"{prefix}_ln")
        self.attention = T5Attention(embed_dim, key_value_dim, num_heads, prefix=prefix)

    def build(self, input_shape):
        self.layer_norm.build(input_shape)
        self.attention.build(input_shape)
        self.built = True

    def call(self, hidden_states, position_bias):
        normed = self.layer_norm(hidden_states)
        return hidden_states + self.attention(normed, position_bias)


class T5CrossAttentionLayer(layers.Layer):
    def __init__(self, embed_dim, key_value_dim, num_heads, eps, prefix, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = T5LayerNorm(eps, name=f"{prefix}_ln")
        self.attention = T5Attention(embed_dim, key_value_dim, num_heads, prefix=prefix)

    def build(self, input_shape):
        self.layer_norm.build(input_shape)
        self.attention.build(input_shape)
        self.built = True

    def call(self, hidden_states, encoder_hidden_states, position_bias):
        normed = self.layer_norm(hidden_states)
        attn = self.attention(
            normed, position_bias, key_value_states=encoder_hidden_states
        )
        return hidden_states + attn


class T5FeedForwardLayer(layers.Layer):
    def __init__(self, embed_dim, mlp_dim, hidden_act, eps, prefix, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = T5LayerNorm(eps, name=f"{prefix}_ln")
        self.wi = layers.Dense(mlp_dim, use_bias=False, name=f"{prefix}_wi")
        self.act = layers.Activation(hidden_act, name=f"{prefix}_act")
        self.wo = layers.Dense(embed_dim, use_bias=False, name=f"{prefix}_wo")

    def build(self, input_shape):
        self.layer_norm.build(input_shape)
        self.wi.build(input_shape)
        inter_shape = tuple(input_shape[:-1]) + (self.wi.units,)
        self.act.build(inter_shape)
        self.wo.build(inter_shape)
        self.built = True

    def call(self, hidden_states):
        normed = self.layer_norm(hidden_states)
        forwarded = self.wo(self.act(self.wi(normed)))
        return hidden_states + forwarded


class T5EncoderBlock(layers.Layer):
    def __init__(
        self,
        embed_dim,
        key_value_dim,
        num_heads,
        mlp_dim,
        hidden_act,
        eps,
        prefix,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.self_attention = T5SelfAttentionLayer(
            embed_dim, key_value_dim, num_heads, eps, prefix=f"{prefix}_attn"
        )
        self.ff = T5FeedForwardLayer(
            embed_dim, mlp_dim, hidden_act, eps, prefix=f"{prefix}_ff"
        )

    def build(self, input_shape):
        self.self_attention.build(input_shape)
        self.ff.build(input_shape)
        self.built = True

    def call(self, hidden_states, position_bias):
        hidden_states = self.self_attention(hidden_states, position_bias)
        return self.ff(hidden_states)

    def compute_output_spec(self, hidden_states, position_bias):
        return keras.KerasTensor(hidden_states.shape, dtype=hidden_states.dtype)


class T5DecoderBlock(layers.Layer):
    def __init__(
        self,
        embed_dim,
        key_value_dim,
        num_heads,
        mlp_dim,
        hidden_act,
        eps,
        prefix,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.self_attention = T5SelfAttentionLayer(
            embed_dim, key_value_dim, num_heads, eps, prefix=f"{prefix}_self"
        )
        self.cross_attention = T5CrossAttentionLayer(
            embed_dim, key_value_dim, num_heads, eps, prefix=f"{prefix}_cross"
        )
        self.ff = T5FeedForwardLayer(
            embed_dim, mlp_dim, hidden_act, eps, prefix=f"{prefix}_ff"
        )

    def build(self, input_shape):
        self.self_attention.build(input_shape)
        self.cross_attention.build(input_shape)
        self.ff.build(input_shape)
        self.built = True

    def call(
        self,
        hidden_states,
        self_position_bias,
        encoder_hidden_states,
        cross_position_bias,
    ):
        hidden_states = self.self_attention(hidden_states, self_position_bias)
        hidden_states = self.cross_attention(
            hidden_states, encoder_hidden_states, cross_position_bias
        )
        return self.ff(hidden_states)

    def compute_output_spec(
        self,
        hidden_states,
        self_position_bias,
        encoder_hidden_states,
        cross_position_bias,
    ):
        return keras.KerasTensor(hidden_states.shape, dtype=hidden_states.dtype)
