import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


def rotate_half_interleaved(x):
    # GLM interleaved rotate over channel pairs: [-x1, x0, -x3, x2, ...].
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    stacked = ops.stack([-x2, x1], axis=-1)
    return ops.reshape(stacked, ops.shape(x))


def apply_glm_rope(x, cos, sin, rotary_dim):
    # Partial interleaved rope: rotate the first ``rotary_dim`` channels (cos /
    # sin are repeat-interleaved, one angle per pair) and pass the rest through.
    dtype = x.dtype
    x_rot = ops.cast(x[..., :rotary_dim], "float32")
    x_pass = x[..., rotary_dim:]
    cos = ops.cast(cos, "float32")
    sin = ops.cast(sin, "float32")
    out = x_rot * cos + rotate_half_interleaved(x_rot) * sin
    return ops.concatenate([ops.cast(out, dtype), x_pass], axis=-1)


@keras.saving.register_keras_serializable(package="zeromodels")
class Glm4RMSNorm(layers.Layer):
    """Root-mean-square norm (plain learned weight, ones init).

    Args:
        eps: Variance epsilon.
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
class Glm4MLP(layers.Layer):
    """SwiGLU with a fused gate/up projection: ``down(up * silu(gate))``."""

    def __init__(self, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.gate_up = layers.Dense(2 * mlp_dim, use_bias=False, name="gate_up")
        self.down = layers.Dense(embed_dim, use_bias=False, name="down")

    def call(self, x):
        gate_up = self.gate_up(x)
        gate = gate_up[..., : self.mlp_dim]
        up = gate_up[..., self.mlp_dim :]
        return self.down(up * ops.silu(gate))

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "mlp_dim": self.mlp_dim})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Glm4Attention(layers.Layer):
    """GLM-4 grouped-query attention with partial interleaved rope.

    Args:
        embed_dim: Model width.
        num_heads / num_kv_heads / head_dim: Attention geometry.
        rotary_dim: Channels of each head that receive rotary embeddings.
        attention_bias: Whether the q/k/v projections carry bias.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        attention_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.attention_bias = attention_bias
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim**-0.5
        self.query = layers.Dense(
            num_heads * head_dim, use_bias=attention_bias, name="query"
        )
        self.key = layers.Dense(
            num_kv_heads * head_dim, use_bias=attention_bias, name="key"
        )
        self.value = layers.Dense(
            num_kv_heads * head_dim, use_bias=attention_bias, name="value"
        )
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")

    def project(self, hidden_states):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        q = ops.reshape(
            self.query(hidden_states), (b, s, self.num_heads, self.head_dim)
        )
        k = ops.reshape(
            self.key(hidden_states), (b, s, self.num_kv_heads, self.head_dim)
        )
        v = ops.reshape(
            self.value(hidden_states), (b, s, self.num_kv_heads, self.head_dim)
        )
        return (
            ops.transpose(q, (0, 2, 1, 3)),
            ops.transpose(k, (0, 2, 1, 3)),
            ops.transpose(v, (0, 2, 1, 3)),
        )

    def attend(self, q, k, v, attention_mask):
        if self.num_kv_groups > 1:
            k = ops.repeat(k, self.num_kv_groups, axis=1)
            v = ops.repeat(v, self.num_kv_groups, axis=1)
        return fused_attention(q, k, v, self.scaling, attention_mask)

    def call(self, hidden_states, cos, sin, attention_mask=None, use_cache=False):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        q, k, v = self.project(hidden_states)
        cos_e = ops.expand_dims(cos, axis=1)
        sin_e = ops.expand_dims(sin, axis=1)
        q = apply_glm_rope(q, cos_e, sin_e, self.rotary_dim)
        k = apply_glm_rope(k, cos_e, sin_e, self.rotary_dim)
        new_kv = (k, v) if use_cache else None
        out = self.attend(q, k, v, attention_mask)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, s, self.num_heads * self.head_dim)
        )
        out = self.output_proj(out)
        return (out, new_kv) if use_cache else out

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        b = ops.shape(hidden_states)[0]
        q, k, v = self.project(hidden_states)
        cos_e = ops.expand_dims(cos, axis=1)
        sin_e = ops.expand_dims(sin, axis=1)
        q = apply_glm_rope(q, cos_e, sin_e, self.rotary_dim)
        k = apply_glm_rope(k, cos_e, sin_e, self.rotary_dim)
        # rope runs in float32; match the cache dtype before writing
        k = ops.cast(k, cache_k.dtype)
        v = ops.cast(v, cache_v.dtype)
        cache_k = ops.slice_update(cache_k, (0, 0, write_pos, 0), k)
        cache_v = ops.slice_update(cache_v, (0, 0, write_pos, 0), v)
        out = self.attend(q, cache_k, cache_v, key_mask)
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
                "rotary_dim": self.rotary_dim,
                "attention_bias": self.attention_bias,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Glm4DecoderLayer(layers.Layer):
    """One GLM-4 block with sandwich norms.

    Both the attention and MLP outputs are RMSNorm'd *before* the residual add
    (``post_self_attn_layernorm`` / ``post_mlp_layernorm``), in addition to the
    usual pre-norms: the GLM-4-0414 formulation.

    Args:
        embed_dim / mlp_dim / num_heads / num_kv_heads / head_dim / rotary_dim:
        Geometry forwarded to the sub-layers.
        norm_eps: RMSNorm epsilon.
        attention_bias: Attention projection bias.
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        norm_eps=1e-6,
        attention_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.norm_eps = norm_eps
        self.attention_bias = attention_bias
        self.input_layernorm = Glm4RMSNorm(eps=norm_eps, name="input_layernorm")
        self.attention = Glm4Attention(
            embed_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            attention_bias=attention_bias,
            name="attention",
        )
        self.post_self_attn_layernorm = Glm4RMSNorm(
            eps=norm_eps, name="post_self_attn_layernorm"
        )
        self.post_attention_layernorm = Glm4RMSNorm(
            eps=norm_eps, name="post_attention_layernorm"
        )
        self.mlp = Glm4MLP(embed_dim, mlp_dim, name="mlp")
        self.post_mlp_layernorm = Glm4RMSNorm(eps=norm_eps, name="post_mlp_layernorm")

    def call(self, hidden_states, cos, sin, attention_mask=None, use_cache=False):
        residual = hidden_states
        attn_out = self.attention(
            self.input_layernorm(hidden_states),
            cos,
            sin,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )
        new_kv = None
        if use_cache:
            attn_out, new_kv = attn_out
        hidden_states = residual + self.post_self_attn_layernorm(attn_out)
        residual = hidden_states
        mlp_out = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = residual + self.post_mlp_layernorm(mlp_out)
        return (hidden_states, new_kv) if use_cache else hidden_states

    def compute_output_spec(
        self, hidden_states, cos, sin, attention_mask=None, use_cache=False
    ):
        # Residual stream keeps its shape; an explicit spec stops the functional
        # builder from tracing the dynamic-shape reshapes / rope in attention.
        return keras.KerasTensor(hidden_states.shape, dtype=self.compute_dtype)

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        residual = hidden_states
        attn_out, cache_k, cache_v = self.attention.decode_step(
            self.input_layernorm(hidden_states),
            cos,
            sin,
            cache_k,
            cache_v,
            write_pos,
            key_mask,
        )
        hidden_states = residual + self.post_self_attn_layernorm(attn_out)
        residual = hidden_states
        mlp_out = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = residual + self.post_mlp_layernorm(mlp_out)
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
                "rotary_dim": self.rotary_dim,
                "norm_eps": self.norm_eps,
                "attention_bias": self.attention_bias,
            }
        )
        return config
