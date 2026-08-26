import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


@keras.saving.register_keras_serializable(package="zeromodels")
class GptAttention(layers.Layer):
    """GPT (original) causal multi-head self-attention.

    Fused ``c_attn`` query/key/value projection + ``c_proj`` output projection,
    both with bias and GPT's ``Conv1D`` ``(in, out)`` weight layout (copied
    without transpose). Supports a KV cache via ``past_key_value``.

    Args:
        embed_dim: Model width.
        num_heads: Number of attention heads.
    """

    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.c_attn = layers.Dense(3 * embed_dim, name="c_attn")
        self.c_proj = layers.Dense(embed_dim, name="c_proj")

    def call(
        self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False
    ):
        b = ops.shape(hidden_states)[0]
        q_len = ops.shape(hidden_states)[1]
        q, k, v = ops.split(self.c_attn(hidden_states), 3, axis=-1)
        shape = (b, q_len, self.num_heads, self.head_dim)
        q = ops.transpose(ops.reshape(q, shape), (0, 2, 1, 3))
        k = ops.transpose(ops.reshape(k, shape), (0, 2, 1, 3))
        v = ops.transpose(ops.reshape(v, shape), (0, 2, 1, 3))

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = ops.concatenate([past_k, k], axis=2)
            v = ops.concatenate([past_v, v], axis=2)
        new_kv = (k, v) if use_cache else None

        out = fused_attention(q, k, v, self.scaling, attention_mask)
        out = ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (b, q_len, self.embed_dim))
        out = self.c_proj(out)
        return (out, new_kv) if use_cache else out

    def decode_step(self, hidden_states, cache_k, cache_v, write_pos, key_mask):
        # Single-token attention against a fixed-size KV cache (no rotary; GPT uses
        # learned positions added at the embedding). ``key_mask`` blocks empty slots.
        b = ops.shape(hidden_states)[0]
        q, k, v = ops.split(self.c_attn(hidden_states), 3, axis=-1)
        shape = (b, 1, self.num_heads, self.head_dim)
        q = ops.transpose(ops.reshape(q, shape), (0, 2, 1, 3))
        k = ops.transpose(ops.reshape(k, shape), (0, 2, 1, 3))
        v = ops.transpose(ops.reshape(v, shape), (0, 2, 1, 3))
        # rope runs in float32; match the cache dtype before writing
        k = ops.cast(k, cache_k.dtype)
        v = ops.cast(v, cache_v.dtype)
        cache_k = ops.slice_update(cache_k, (0, 0, write_pos, 0), k)
        cache_v = ops.slice_update(cache_v, (0, 0, write_pos, 0), v)
        out = fused_attention(q, cache_k, cache_v, self.scaling, key_mask)
        out = ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (b, 1, self.embed_dim))
        return self.c_proj(out), cache_k, cache_v

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GptMLP(layers.Layer):
    """GPT feed-forward block: ``c_proj(gelu_new(c_fc(x)))`` (Conv1D layout)."""

    def __init__(self, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.c_fc = layers.Dense(mlp_dim, name="c_fc")
        self.c_proj = layers.Dense(embed_dim, name="c_proj")

    def call(self, x):
        return self.c_proj(ops.gelu(self.c_fc(x), approximate=True))

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "mlp_dim": self.mlp_dim})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GptBlock(layers.Layer):
    """One GPT transformer block: post-LayerNorm attention, then post-LN MLP.

    Matches the original GPT: ``n = ln_1(x + attn(x))`` then
    ``h = ln_2(n + mlp(n))`` (LayerNorm *after* each residual add).

    Args:
        embed_dim: Model width.
        mlp_dim: Feed-forward hidden width.
        num_heads: Number of attention heads.
        norm_eps: LayerNorm epsilon.
    """

    def __init__(self, embed_dim, mlp_dim, num_heads, norm_eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.norm_eps = norm_eps
        self.attn = GptAttention(embed_dim, num_heads, name="attn")
        self.ln_1 = layers.LayerNormalization(epsilon=norm_eps, name="ln_1")
        self.mlp = GptMLP(embed_dim, mlp_dim, name="mlp")
        self.ln_2 = layers.LayerNormalization(epsilon=norm_eps, name="ln_2")

    def call(
        self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False
    ):
        attn_out = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        new_kv = None
        if use_cache:
            attn_out, new_kv = attn_out
        n = self.ln_1(hidden_states + attn_out)
        h = self.ln_2(n + self.mlp(n))
        return (h, new_kv) if use_cache else h

    def decode_step(self, hidden_states, cache_k, cache_v, write_pos, key_mask):
        attn_out, cache_k, cache_v = self.attn.decode_step(
            hidden_states, cache_k, cache_v, write_pos, key_mask
        )
        n = self.ln_1(hidden_states + attn_out)
        h = self.ln_2(n + self.mlp(n))
        return h, cache_k, cache_v

    def compute_output_spec(
        self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False
    ):
        # Residual stream keeps its shape; an explicit spec keeps the functional
        # builder from tracing the dynamic-shape reshapes in attention.
        return keras.KerasTensor(hidden_states.shape, dtype=self.compute_dtype)

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
