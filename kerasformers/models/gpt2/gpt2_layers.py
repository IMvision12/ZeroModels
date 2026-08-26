import keras
from keras import layers, ops

from kerasformers.base.base_attention import fused_attention

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="kerasformers")
class GPT2Attention(layers.Layer):
    """GPT-2 causal multi-head self-attention.

    A single fused ``c_attn`` projects to query/key/value (each ``embed_dim``),
    scaled dot-product causal attention is applied, and ``c_proj`` projects back.
    Both projections carry a bias and use GPT-2's ``Conv1D`` ``(in, out)`` weight
    layout (so the converter copies them without transposing). A KV cache can be
    threaded through ``past_key_value`` for incremental decoding.

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
        # Single-token attention against a fixed-size KV cache (no rotary; GPT-2 uses
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


@keras.saving.register_keras_serializable(package="kerasformers")
class GPT2MLP(layers.Layer):
    """GPT-2 feed-forward block: ``c_proj(gelu_new(c_fc(x)))`` (Conv1D layout)."""

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


@keras.saving.register_keras_serializable(package="kerasformers")
class GPT2Block(layers.Layer):
    """One GPT-2 transformer block: pre-LayerNorm attention, then pre-LN MLP.

    ``h = x + attn(ln_1(x))`` followed by ``h = h + mlp(ln_2(h))`` (LayerNorm with
    bias, GPT-2's pre-normalization).

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
        self.ln_1 = layers.LayerNormalization(epsilon=norm_eps, name="ln_1")
        self.attn = GPT2Attention(embed_dim, num_heads, name="attn")
        self.ln_2 = layers.LayerNormalization(epsilon=norm_eps, name="ln_2")
        self.mlp = GPT2MLP(embed_dim, mlp_dim, name="mlp")

    def call(
        self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False
    ):
        attn_out = self.attn(
            self.ln_1(hidden_states),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        new_kv = None
        if use_cache:
            attn_out, new_kv = attn_out
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return (hidden_states, new_kv) if use_cache else hidden_states

    def decode_step(self, hidden_states, cache_k, cache_v, write_pos, key_mask):
        attn_out, cache_k, cache_v = self.attn.decode_step(
            self.ln_1(hidden_states), cache_k, cache_v, write_pos, key_mask
        )
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states, cache_k, cache_v

    def compute_output_spec(
        self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False
    ):
        # The residual stream keeps its shape; giving the block an explicit spec
        # stops the functional builder from tracing the dynamic-shape reshapes in
        # attention (which only resolve at run time, not symbolic-build time).
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
