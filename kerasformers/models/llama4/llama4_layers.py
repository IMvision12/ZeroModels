import keras
from keras import layers, ops

from kerasformers.base.base_attention import fused_attention


def apply_interleaved_rope(x, cos, sin):
    # Llama 4 rotates interleaved (re, im) pairs: the original Llama codebase's
    # complex rope, unlike the half-rotation layout used by Llama 2/3 HF
    # checkpoints. x: (B, L, H, D); cos/sin: (B, L, D/2) broadcast over heads.
    b = ops.shape(x)[0]
    q_len = ops.shape(x)[1]
    heads = ops.shape(x)[2]
    half = ops.shape(x)[-1] // 2
    x = ops.reshape(x, (b, q_len, heads, half, 2))
    re, im = x[..., 0], x[..., 1]
    cos = ops.expand_dims(cos, axis=2)
    sin = ops.expand_dims(sin, axis=2)
    out_re = re * cos - im * sin
    out_im = re * sin + im * cos
    return ops.reshape(ops.stack([out_re, out_im], axis=-1), (b, q_len, heads, -1))


@keras.saving.register_keras_serializable(package="kerasformers")
class Llama4RMSNorm(layers.Layer):
    """Root-mean-square layer norm (Llama 4 text style).

    Normalizes the last axis by its RMS in float32 (for numerical stability),
    casts back to the input dtype, then scales by a learned per-channel weight.
    No mean subtraction, no bias. Shape-preserving: ``(..., dim) -> (..., dim)``.

    Args:
        eps: Variance epsilon added before the reciprocal square root.
            Defaults to ``1e-5``.
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


@keras.saving.register_keras_serializable(package="kerasformers")
class Llama4L2Norm(layers.Layer):
    """Weightless RMS normalization (Llama 4's QK-norm).

    Like RMSNorm but with no learned scale: normalizes the last axis by its
    RMS in float32 and casts back. Applied to the rotated per-head query and
    key on the rope layers of the Scout (16E) model.

    Args:
        eps: Variance epsilon. Defaults to ``1e-5`` (``rms_norm_eps``).
    """

    def __init__(self, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, x):
        dtype = x.dtype
        x = ops.cast(x, "float32")
        variance = ops.mean(ops.square(x), axis=-1, keepdims=True)
        return ops.cast(x * ops.rsqrt(variance + self.eps), dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Llama4MLP(layers.Layer):
    """SwiGLU feed-forward block: ``down(silu(gate(x)) * up(x))``.

    Used both as the dense layers' feed-forward (width
    ``intermediate_size_mlp``) and as every MoE layer's always-active shared
    expert (width ``intermediate_size``). Bias-free projections.

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


@keras.saving.register_keras_serializable(package="kerasformers")
class Llama4Experts(layers.Layer):
    """Llama 4 fused mixture-of-experts feed-forward bank (dense evaluation).

    Holds the packed per-expert parameters in Hugging Face's layout
    (``gate_up_proj`` ``(E, H, 2I)`` with *contiguous* gate/up halves,
    ``down_proj`` ``(E, I, H)``, no biases) and, given per-token per-expert
    pre-scaled inputs ``(E, T, H)`` (each expert sees the token scaled by its
    sigmoid routing score: Llama 4 scales the expert *input*, not output),
    computes every expert's SwiGLU and returns the per-expert outputs summed
    over experts. Backend-agnostic ``einsum``: mathematically identical to
    the sparse top-1 routing but compute is O(num_experts).

    Args:
        num_experts: Number of experts ``E``.
        embed_dim: Model width ``H``.
        mlp_dim: Per-expert hidden width ``I``.
    """

    def __init__(self, num_experts, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim

    def build(self, input_shape):
        e, h, i = self.num_experts, self.embed_dim, self.mlp_dim
        self.gate_up_proj = self.add_weight(
            name="gate_up_proj",
            shape=(e, h, 2 * i),
            initializer="zeros",
            trainable=True,
        )
        self.down_proj = self.add_weight(
            name="down_proj", shape=(e, i, h), initializer="zeros", trainable=True
        )
        self.built = True

    def call(self, scaled_inputs):
        gate_up = ops.einsum("eth,ehi->eti", scaled_inputs, self.gate_up_proj)
        gate = gate_up[..., : self.mlp_dim]
        up = gate_up[..., self.mlp_dim :]
        expert_out = ops.einsum(
            "eti,eih->eth", up * ops.silu(gate), self.down_proj
        )  # (E, T, H)
        return ops.sum(expert_out, axis=0)  # (T, H)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Llama4MoE(layers.Layer):
    """Llama 4 sparse MoE block: sigmoid top-k router + experts + shared expert.

    The bias-free router scores every expert; the top-``num_experts_per_tok``
    (1 in the released models) logits are kept, everything else is set to
    ``-inf``, and a sigmoid turns the kept logits into per-expert scores in
    ``[0, 1]`` (non-selected experts get exactly 0). Each expert processes the
    token *input scaled by its score* (the score multiplies the expert input,
    not its output), the expert outputs are summed, and the always-active
    shared expert's output is added.

    Args:
        num_experts: Number of routed experts.
        num_experts_per_tok: Top-k experts kept per token (released: 1).
        embed_dim: Model width.
        mlp_dim: Per-expert (and shared-expert) hidden width.
    """

    def __init__(self, num_experts, num_experts_per_tok, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.router = layers.Dense(num_experts, use_bias=False, name="router")
        self.experts = Llama4Experts(num_experts, embed_dim, mlp_dim, name="experts")
        self.shared_expert = Llama4MLP(embed_dim, mlp_dim, name="shared_expert")

    def call(self, hidden_states):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        x = ops.reshape(hidden_states, (-1, self.embed_dim))  # (T, H)
        router_logits = ops.cast(self.router(x), "float32")  # (T, E)
        top_vals, top_idx = ops.top_k(router_logits, self.num_experts_per_tok)
        one_hot = ops.one_hot(top_idx, self.num_experts)  # (T, k, E)
        scattered = ops.sum(one_hot * top_vals[..., None], axis=1)  # (T, E)
        selected = ops.sum(one_hot, axis=1) > 0
        scores = ops.sigmoid(ops.where(selected, scattered, float("-inf")))
        scores = ops.cast(scores, x.dtype)
        scaled = ops.expand_dims(ops.transpose(scores), axis=-1) * x[None]  # (E, T, H)
        routed_out = self.experts(scaled)  # (T, H)
        out = self.shared_expert(x) + routed_out
        return ops.reshape(out, (b, s, self.embed_dim))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Llama4Attention(layers.Layer):
    """Llama 4 grouped-query causal self-attention (iRoPE).

    Bias-free projections with the interleaved-pair (complex-style) rotary
    embedding on rope layers; NoPE layers (every
    ``no_rope_layer_interval``-th) skip rotary entirely and instead scale the
    query by a position-dependent "attention temperature"
    (``log1p(floor((pos + 1) / floor_scale)) * attn_scale + 1``), supplied by
    the model as ``attn_scales``. On rope layers of models with
    ``use_qk_norm`` (Scout) the rotated query/key are L2-normalized. The
    additive mask (full causal on NoPE layers, chunked causal on rope layers)
    is supplied by the caller.

    Args:
        embed_dim: Model width (output dim of ``output_proj``).
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (GQA).
        head_dim: Per-head dim.
        use_rope: Whether this layer applies rotary (False on NoPE layers).
        use_qk_norm: Whether to L2-normalize the rotated query/key (rope
            layers of the Scout model only).
        norm_eps: Epsilon of the QK L2 norm.

    Call args:
        hidden_states: ``(batch, q_len, embed_dim)``.
        cos, sin: interleaved rotary tables ``(batch, q_len, head_dim // 2)``
            (ignored on NoPE layers).
        attn_scales: temperature-tuning scales broadcastable to
            ``(batch, q_len, 1, 1)``, or ``None`` (ignored on rope layers).
        attention_mask: additive mask broadcastable to
            ``(batch, 1, q_len, kv_len)``, or ``None``.
        past_key_value: optional ``(past_k, past_v)``, each
            ``(batch, num_kv_heads, past_len, head_dim)``.
        use_cache: when ``True``, also return the updated ``(key, value)``.

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
        use_rope=True,
        use_qk_norm=True,
        norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.norm_eps = norm_eps
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim**-0.5
        self.query = layers.Dense(num_heads * head_dim, use_bias=False, name="query")
        self.key = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="key")
        self.value = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="value")
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")
        self.qk_norm = (
            Llama4L2Norm(eps=norm_eps, name="qk_norm")
            if (use_qk_norm and use_rope)
            else None
        )

    def project_qkv(self, hidden_states, q_len, cos, sin, attn_scales):
        # Shared q/k/v head-splitting + rope / qk-norm / temperature-tuning
        # pipeline (HF order: rope -> qk-norm -> temperature), on (B, L, H, D)
        # before the head transpose.
        b = ops.shape(hidden_states)[0]
        q = ops.reshape(
            self.query(hidden_states), (b, q_len, self.num_heads, self.head_dim)
        )
        k = ops.reshape(
            self.key(hidden_states), (b, q_len, self.num_kv_heads, self.head_dim)
        )
        v = ops.reshape(
            self.value(hidden_states), (b, q_len, self.num_kv_heads, self.head_dim)
        )
        if self.use_rope:
            q = apply_interleaved_rope(q, cos, sin)
            k = apply_interleaved_rope(k, cos, sin)
        if self.qk_norm is not None:
            q = self.qk_norm(q)
            k = self.qk_norm(k)
        if not self.use_rope and attn_scales is not None:
            q = ops.cast(ops.cast(q, "float32") * attn_scales, q.dtype)
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))
        return q, k, v

    def call(
        self,
        hidden_states,
        cos,
        sin,
        attn_scales=None,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        b = ops.shape(hidden_states)[0]
        q_len = ops.shape(hidden_states)[1]
        q, k, v = self.project_qkv(hidden_states, q_len, cos, sin, attn_scales)

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
        self,
        hidden_states,
        cos,
        sin,
        attn_scales,
        cache_k,
        cache_v,
        write_pos,
        key_mask,
    ):
        # Single-token attention against a fixed-size KV cache written in place
        # at ``write_pos``. ``key_mask`` (additive, (.., max_len)) is the full
        # or chunked mask for this layer and also blocks empty cache slots.
        b = ops.shape(hidden_states)[0]
        q, k, v = self.project_qkv(hidden_states, 1, cos, sin, attn_scales)
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
                "use_rope": self.use_rope,
                "use_qk_norm": self.use_qk_norm,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Llama4DecoderLayer(layers.Layer):
    """One Llama 4 text block: pre-norm iRoPE attention, then pre-norm feed-forward.

    Computes ``h = x + attention(attention_norm(x))`` followed by
    ``h = h + feed_forward(mlp_norm(h))``, where ``feed_forward`` is a
    :class:`Llama4MoE` on MoE layers (all layers on Scout; the odd layers on
    Maverick) and a dense :class:`Llama4MLP` of width ``dense_mlp_dim``
    otherwise. Rotary tables, temperature scales, mask, and KV cache pass
    straight through to the attention.

    Args:
        embed_dim: Model / residual-stream width.
        mlp_dim: Per-expert / shared-expert hidden width (MoE layers).
        dense_mlp_dim: Dense feed-forward hidden width (non-MoE layers).
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (GQA).
        head_dim: Per-head dim.
        is_moe: Whether this layer's feed-forward is the MoE block.
        num_experts: Routed expert count (MoE layers).
        num_experts_per_tok: Top-k experts per token (MoE layers).
        use_rope: Whether the attention applies rotary (False on NoPE layers).
        use_qk_norm: Whether the attention L2-normalizes rotated q/k.
        norm_eps: Epsilon of both RMSNorms and the QK norm.

    Call args:
        hidden_states, cos, sin, attn_scales, attention_mask, past_key_value,
        use_cache: as in :class:`Llama4Attention`.

    Returns:
        The block output, or ``(output, (key, value))`` when ``use_cache`` is
        set.
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        dense_mlp_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        is_moe,
        num_experts,
        num_experts_per_tok,
        use_rope,
        use_qk_norm,
        norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.dense_mlp_dim = dense_mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.is_moe = is_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.norm_eps = norm_eps
        self.attention_norm = Llama4RMSNorm(eps=norm_eps, name="attention_norm")
        self.attention = Llama4Attention(
            embed_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            use_rope,
            use_qk_norm,
            norm_eps,
            name="attention",
        )
        self.mlp_norm = Llama4RMSNorm(eps=norm_eps, name="mlp_norm")
        self.feed_forward = (
            Llama4MoE(
                num_experts,
                num_experts_per_tok,
                embed_dim,
                mlp_dim,
                name="feed_forward",
            )
            if is_moe
            else Llama4MLP(embed_dim, dense_mlp_dim, name="feed_forward")
        )

    def call(
        self,
        hidden_states,
        cos,
        sin,
        attn_scales=None,
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
            attn_scales=attn_scales,
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
        hidden_states = residual + self.feed_forward(hidden_states)
        return (hidden_states, new_kv) if use_cache else hidden_states

    def decode_step(
        self,
        hidden_states,
        cos,
        sin,
        attn_scales,
        cache_k,
        cache_v,
        write_pos,
        key_mask,
    ):
        residual = hidden_states
        x = self.attention_norm(hidden_states)
        attn_out, cache_k, cache_v = self.attention.decode_step(
            x, cos, sin, attn_scales, cache_k, cache_v, write_pos, key_mask
        )
        hidden_states = residual + attn_out
        residual = hidden_states
        x = self.mlp_norm(hidden_states)
        hidden_states = residual + self.feed_forward(x)
        return hidden_states, cache_k, cache_v

    def compute_output_spec(
        self,
        hidden_states,
        cos,
        sin,
        attn_scales=None,
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
                "dense_mlp_dim": self.dense_mlp_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "is_moe": self.is_moe,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "use_rope": self.use_rope,
                "use_qk_norm": self.use_qk_norm,
                "norm_eps": self.norm_eps,
            }
        )
        return config
