import keras
from keras import layers, ops


def l2norm(x, eps=1e-6):
    """L2-normalize over the last axis (used on the Gated-DeltaNet q/k)."""
    return x * ops.rsqrt(ops.sum(ops.square(x), axis=-1, keepdims=True) + eps)


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3NextRMSNorm(layers.Layer):
    """Zero-centered root-mean-square layer norm (Qwen3.5).

    Like a standard RMSNorm but the learned scale is stored *zero-centered*: the
    weight is initialized to ``0`` and the effective per-channel gain is
    ``1 + weight``. Normalizes the last axis by its RMS in float32, scales by
    ``1 + weight`` *still in float32*, then casts back to the input dtype (matching
    HF Qwen3-Next, which differs from Llama's cast-then-scale order). No mean
    subtraction, no bias. Shape-preserving ``(..., dim) -> (..., dim)``.

    Args:
        eps: Variance epsilon added before the reciprocal square root.
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
        x = x * ops.rsqrt(ops.mean(ops.square(x), axis=-1, keepdims=True) + self.eps)
        # Match HF Qwen3-Next: scale by (1 + weight) in float32, THEN cast back to
        # the input dtype. (Llama instead casts first, then multiplies: this order
        # reduces fp16/bf16 drift. See HF Qwen3NextRMSNorm / transformers#29402.)
        x = x * (1.0 + ops.cast(self.weight, "float32"))
        return ops.cast(x, dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3NextRMSNormGated(layers.Layer):
    """Gated RMSNorm used inside Gated-DeltaNet read-out.

    RMS-normalizes ``x`` in float32, scales by a learned per-channel ``weight``
    (init ``1``), then gates the result elementwise by ``silu(gate)``. ``x`` and
    ``gate`` share the same shape; computes ``weight * (x / rms(x)) * silu(gate)``.

    Args:
        eps: Variance epsilon added before the reciprocal square root.
    """

    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight", shape=(input_shape[-1],), initializer="ones", trainable=True
        )
        self.built = True

    def call(self, x, gate):
        dtype = x.dtype
        x = ops.cast(x, "float32")
        x = x * ops.rsqrt(ops.mean(ops.square(x), axis=-1, keepdims=True) + self.eps)
        x = self.weight * ops.cast(x, dtype)
        return x * ops.silu(ops.cast(gate, "float32"))

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3NextMLP(layers.Layer):
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


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3NextAttention(layers.Layer):
    """Gated grouped-query full attention (Qwen3.5): QK-norm + partial rotary.

    The ``query`` projection emits both the query and an output gate
    (``num_heads * head_dim * 2``, split per head), the per-head query and key are
    zero-centered-RMSNorm'd (``query_norm`` / ``key_norm``), then *partial* rotary
    is applied to only the first ``rotary_dim`` channels (the rest pass through).
    After scaled-dot-product attention with GQA K/V repetition, the output is
    multiplied by ``sigmoid(gate)`` before ``output_proj``. A KV cache can be
    threaded through ``past_key_value`` for incremental decoding.

    Args:
        embed_dim: Model width (output dim of ``output_proj``).
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (``<= num_heads`` for GQA).
        head_dim: Per-head dim.
        rotary_dim: Number of leading per-head channels that receive rotary.
        norm_eps: Epsilon for the per-head ``query_norm`` / ``key_norm``.

    Call args:
        hidden_states: ``(batch, q_len, embed_dim)``.
        cos, sin: partial-rotary tables ``(batch, q_len, rotary_dim)``.
        attention_mask: additive mask broadcastable to
            ``(batch, 1, q_len, kv_len)``, or ``None``.
        past_key_value: optional ``(past_k, past_v)``.
        use_cache: when ``True``, also return the updated ``(key, value)``.

    Returns:
        Output ``(batch, q_len, embed_dim)``, or ``(output, (key, value))`` when
        ``use_cache`` is set.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.norm_eps = norm_eps
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim**-0.5
        self.query = layers.Dense(
            num_heads * head_dim * 2, use_bias=False, name="query"
        )
        self.key = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="key")
        self.value = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="value")
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")
        self.query_norm = Qwen3NextRMSNorm(eps=norm_eps, name="query_norm")
        self.key_norm = Qwen3NextRMSNorm(eps=norm_eps, name="key_norm")

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
        qg = ops.reshape(
            self.query(hidden_states),
            (b, q_len, self.num_heads, self.head_dim * 2),
        )
        query, gate = qg[..., : self.head_dim], qg[..., self.head_dim :]
        gate = ops.reshape(gate, (b, q_len, self.num_heads * self.head_dim))
        query = self.query_norm(query)
        key = self.key_norm(
            ops.reshape(
                self.key(hidden_states), (b, q_len, self.num_kv_heads, self.head_dim)
            )
        )
        value = ops.reshape(
            self.value(hidden_states), (b, q_len, self.num_kv_heads, self.head_dim)
        )

        query = ops.transpose(query, (0, 2, 1, 3))
        key = ops.transpose(key, (0, 2, 1, 3))
        value = ops.transpose(value, (0, 2, 1, 3))

        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        rd = self.rotary_dim
        half = rd // 2
        q_rot = query[..., :rd]
        q_rot = q_rot * cos + (
            ops.concatenate([-q_rot[..., half:], q_rot[..., :half]], axis=-1) * sin
        )
        query = ops.concatenate([q_rot, query[..., rd:]], axis=-1)
        k_rot = key[..., :rd]
        k_rot = k_rot * cos + (
            ops.concatenate([-k_rot[..., half:], k_rot[..., :half]], axis=-1) * sin
        )
        key = ops.concatenate([k_rot, key[..., rd:]], axis=-1)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            key = ops.concatenate([past_k, key], axis=2)
            value = ops.concatenate([past_v, value], axis=2)
        new_kv = (key, value) if use_cache else None

        if self.num_kv_groups > 1:
            key = ops.repeat(key, self.num_kv_groups, axis=1)
            value = ops.repeat(value, self.num_kv_groups, axis=1)

        attn = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2))) * self.scaling
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), query.dtype)
        out = ops.matmul(attn, value)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)),
            (b, q_len, self.num_heads * self.head_dim),
        )
        out = out * ops.sigmoid(gate)
        out = self.output_proj(out)
        return (out, new_kv) if use_cache else out

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        # Single-token gated full attention against a fixed-size KV cache (QK-norm +
        # partial rotary + sigmoid output gate). ``key_mask`` blocks empty slots.
        b = ops.shape(hidden_states)[0]
        qg = ops.reshape(
            self.query(hidden_states), (b, 1, self.num_heads, self.head_dim * 2)
        )
        query, gate = qg[..., : self.head_dim], qg[..., self.head_dim :]
        gate = ops.reshape(gate, (b, 1, self.num_heads * self.head_dim))
        query = self.query_norm(query)
        key = self.key_norm(
            ops.reshape(
                self.key(hidden_states), (b, 1, self.num_kv_heads, self.head_dim)
            )
        )
        value = ops.reshape(
            self.value(hidden_states), (b, 1, self.num_kv_heads, self.head_dim)
        )
        query = ops.transpose(query, (0, 2, 1, 3))
        key = ops.transpose(key, (0, 2, 1, 3))
        value = ops.transpose(value, (0, 2, 1, 3))
        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        rd, half = self.rotary_dim, self.rotary_dim // 2
        q_rot = query[..., :rd]
        q_rot = q_rot * cos + (
            ops.concatenate([-q_rot[..., half:], q_rot[..., :half]], axis=-1) * sin
        )
        query = ops.concatenate([q_rot, query[..., rd:]], axis=-1)
        k_rot = key[..., :rd]
        k_rot = k_rot * cos + (
            ops.concatenate([-k_rot[..., half:], k_rot[..., :half]], axis=-1) * sin
        )
        key = ops.concatenate([k_rot, key[..., rd:]], axis=-1)
        # rope runs in float32; match the cache dtype before writing
        key = ops.cast(key, cache_k.dtype)
        value = ops.cast(value, cache_v.dtype)
        cache_k = ops.slice_update(cache_k, (0, 0, write_pos, 0), key)
        cache_v = ops.slice_update(cache_v, (0, 0, write_pos, 0), value)
        kk, vv = cache_k, cache_v
        if self.num_kv_groups > 1:
            kk = ops.repeat(kk, self.num_kv_groups, axis=1)
            vv = ops.repeat(vv, self.num_kv_groups, axis=1)
        attn = ops.matmul(query, ops.transpose(kk, (0, 1, 3, 2))) * self.scaling
        attn = attn + key_mask
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), query.dtype)
        out = ops.matmul(attn, vv)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, 1, self.num_heads * self.head_dim)
        )
        out = out * ops.sigmoid(gate)
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
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3NextGatedDeltaNet(layers.Layer):
    """Gated-DeltaNet linear-attention token mixer (Qwen3.5 / Qwen3-Next).

    A short causal depthwise conv1d mixes the projected query/key/value, followed
    by a gated delta-rule recurrence that maintains a per-head
    ``(head_k_dim, head_v_dim)`` state ``S``: each step ``S`` decays by a
    data-dependent gate ``exp(g)``, a delta correction ``beta * (v - S^T k)`` is
    written along ``k``, and the read-out is ``S^T q`` (with L2-normalized,
    scaled ``q`` / ``k``). The read-out is gated-RMSNorm'd by ``z`` and projected
    out. Runs in O(seq) with no quadratic attention matrix.

    The cache state is ``(conv_state, recurrent_state)``: the last
    ``conv_kernel_dim - 1`` conv inputs and the recurrent ``S``, so decoding
    consumes one token at a time. ``cos`` / ``sin`` / ``attention_mask`` are
    unused (no rotary, no explicit mask: causality is inherent to the recurrence).

    Args:
        embed_dim: Model width (input and output dim of the block).
        num_k_heads: Query/key head count.
        num_v_heads: Value head count (``num_v_heads % num_k_heads == 0``; q/k are
            repeated ``num_v_heads // num_k_heads`` times for GQA).
        head_k_dim: Per-head query/key dim.
        head_v_dim: Per-head value dim.
        conv_kernel_dim: Causal conv1d kernel width.
        norm_eps: Epsilon for the gated output RMSNorm.

    Call args:
        hidden_states: ``(batch, seq, embed_dim)``.
        past_key_value: optional ``(conv_state, recurrent_state)``.
        use_cache: when ``True``, also return the updated state tuple.

    Returns:
        Output ``(batch, seq, embed_dim)``, or ``(output, (conv_state,
        recurrent_state))`` when ``use_cache`` is set.
    """

    def __init__(
        self,
        embed_dim,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_kernel_dim=4,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.conv_kernel_dim = conv_kernel_dim
        self.norm_eps = norm_eps
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.kv_ratio = num_v_heads // num_k_heads

        self.in_proj_qkv = layers.Dense(
            self.conv_dim, use_bias=False, name="in_proj_qkv"
        )
        self.in_proj_z = layers.Dense(self.value_dim, use_bias=False, name="in_proj_z")
        self.in_proj_b = layers.Dense(num_v_heads, use_bias=False, name="in_proj_b")
        self.in_proj_a = layers.Dense(num_v_heads, use_bias=False, name="in_proj_a")
        self.norm = Qwen3NextRMSNormGated(eps=norm_eps, name="norm")
        self.out_proj = layers.Dense(embed_dim, use_bias=False, name="out_proj")

    def build(self, input_shape):
        self.conv_weight = self.add_weight(
            name="conv_weight",
            shape=(self.conv_dim, self.conv_kernel_dim),
            initializer="zeros",
            trainable=True,
        )
        self.dt_bias = self.add_weight(
            name="dt_bias",
            shape=(self.num_v_heads,),
            initializer="zeros",
            trainable=True,
        )
        self.A_log = self.add_weight(
            name="A_log", shape=(self.num_v_heads,), initializer="zeros", trainable=True
        )
        self.built = True

    def _causal_conv(self, x, conv_state=None):
        k = self.conv_kernel_dim
        if conv_state is not None:
            x_pad = ops.concatenate([conv_state, x], axis=1)
        else:
            x_pad = ops.concatenate(
                [ops.zeros((ops.shape(x)[0], k - 1, self.conv_dim), dtype=x.dtype), x],
                axis=1,
            )
        seq = ops.shape(x)[1]
        out = 0.0
        for j in range(k):
            out = out + x_pad[:, j : j + seq, :] * self.conv_weight[:, j]
        new_state = x_pad[:, -(k - 1) :, :] if (k - 1) > 0 else None
        return ops.silu(out), new_state

    def _delta_rule(self, q, k, v, g, beta, init_state=None):
        q = l2norm(q) * (self.head_k_dim**-0.5)
        k = l2norm(k)
        b = ops.shape(q)[0]
        seq = q.shape[1]
        state = (
            init_state
            if init_state is not None
            else ops.zeros(
                (b, self.num_v_heads, self.head_k_dim, self.head_v_dim), dtype=q.dtype
            )
        )
        outs = []
        for t in range(seq):
            q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]
            g_t = ops.exp(g[:, t])[:, :, None, None]
            beta_t = beta[:, t][:, :, None]
            state = state * g_t
            kv_mem = ops.sum(state * k_t[..., None], axis=-2)
            delta = (v_t - kv_mem) * beta_t
            state = state + k_t[..., None] * delta[:, :, None, :]
            outs.append(ops.sum(state * q_t[..., None], axis=-2))
        core = ops.stack(outs, axis=1)
        return core, state

    def call(self, hidden_states, past_key_value=None, use_cache=False, pad_mask=None):
        b = ops.shape(hidden_states)[0]
        seq = ops.shape(hidden_states)[1]
        conv_state = past_key_value[0] if past_key_value is not None else None
        rec_state = past_key_value[1] if past_key_value is not None else None

        # Zero padding positions before the causal conv / delta-rule recurrence so
        # padded tokens don't leak into real ones (mirrors HF
        # apply_mask_to_padding_states); the recurrence has no additive mask.
        if pad_mask is not None:
            hidden_states = hidden_states * ops.cast(pad_mask, hidden_states.dtype)

        mixed = self.in_proj_qkv(hidden_states)
        mixed, new_conv_state = self._causal_conv(mixed, conv_state)
        query = ops.reshape(
            mixed[..., : self.key_dim], (b, seq, self.num_k_heads, self.head_k_dim)
        )
        key = ops.reshape(
            mixed[..., self.key_dim : self.key_dim * 2],
            (b, seq, self.num_k_heads, self.head_k_dim),
        )
        value = ops.reshape(
            mixed[..., self.key_dim * 2 :], (b, seq, self.num_v_heads, self.head_v_dim)
        )

        z = ops.reshape(
            self.in_proj_z(hidden_states), (b, seq, self.num_v_heads, self.head_v_dim)
        )
        beta = ops.sigmoid(self.in_proj_b(hidden_states))
        a = self.in_proj_a(hidden_states)
        g = -ops.exp(ops.cast(self.A_log, "float32")) * ops.softplus(
            ops.cast(a, "float32") + ops.cast(self.dt_bias, "float32")
        )

        if self.kv_ratio > 1:
            query = ops.repeat(query, self.kv_ratio, axis=2)
            key = ops.repeat(key, self.kv_ratio, axis=2)

        core, new_rec_state = self._delta_rule(query, key, value, g, beta, rec_state)
        core = self.norm(core, z)
        core = ops.reshape(core, (b, seq, self.value_dim))
        out = self.out_proj(core)
        return (out, (new_conv_state, new_rec_state)) if use_cache else out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_k_heads": self.num_k_heads,
                "num_v_heads": self.num_v_heads,
                "head_k_dim": self.head_k_dim,
                "head_v_dim": self.head_v_dim,
                "conv_kernel_dim": self.conv_kernel_dim,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3NextExperts(layers.Layer):
    """Dense bank of SwiGLU experts evaluated for every token via einsum.

    Weights are stored fused as the HF layout: ``gate_up_proj`` ``(E, 2I, H)``
    (gate stacked over up) and ``down_proj`` ``(E, H, I)``. Routing weights
    are a dense ``(T, E)`` matrix (zero for unrouted experts), so the result
    equals sparse top-k dispatch.

    Args:
        num_experts: Number of routed experts ``E``.
        embed_dim: Model width ``H``.
        mlp_dim: Per-expert hidden width ``I`` (``moe_intermediate_size``).
    """

    def __init__(self, num_experts, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim

    def build(self, input_shape):
        self.gate_up_proj = self.add_weight(
            name="gate_up_proj",
            shape=(self.num_experts, 2 * self.mlp_dim, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.down_proj = self.add_weight(
            name="down_proj",
            shape=(self.num_experts, self.embed_dim, self.mlp_dim),
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def call(self, hidden_states, routing_weights):
        gate_up = ops.einsum("th,eoh->teo", hidden_states, self.gate_up_proj)
        gate = gate_up[..., : self.mlp_dim]
        up = gate_up[..., self.mlp_dim :]
        expert_out = ops.einsum("tei,ehi->teh", ops.silu(gate) * up, self.down_proj)
        return ops.einsum("te,teh->th", routing_weights, expert_out)

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
class Qwen3NextSparseBlock(layers.Layer):
    """Qwen2-MoE sparse block: softmax top-k router + sigmoid-gated shared expert.

    The bias-free router softmaxes over all experts in float32, keeps the
    top-``num_experts_per_tok`` (optionally renormalized to sum to one), and
    applies them to the expert bank. A full-width shared expert runs on every
    token, scaled by ``sigmoid(shared_expert_gate(x))``, and is added to the
    routed output.

    Args:
        num_experts / num_experts_per_tok: Routing shape.
        embed_dim: Model width.
        moe_mlp_dim: Per-routed-expert hidden width (``moe_intermediate_size``).
        shared_mlp_dim: Shared-expert hidden width.
        norm_topk_prob: Renormalize the selected weights to sum to one.
    """

    def __init__(
        self,
        num_experts,
        num_experts_per_tok,
        embed_dim,
        moe_mlp_dim,
        shared_mlp_dim,
        norm_topk_prob=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.embed_dim = embed_dim
        self.moe_mlp_dim = moe_mlp_dim
        self.shared_mlp_dim = shared_mlp_dim
        self.norm_topk_prob = norm_topk_prob
        self.experts = Qwen3NextExperts(
            num_experts, embed_dim, moe_mlp_dim, name="experts"
        )
        self.shared_expert = Qwen3NextMLP(
            embed_dim, shared_mlp_dim, name="shared_expert"
        )
        self.shared_expert_gate = layers.Dense(
            1, use_bias=False, name="shared_expert_gate"
        )

    def build(self, input_shape):
        self.gate_weight = self.add_weight(
            name="gate_weight",
            shape=(self.num_experts, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def call(self, hidden_states):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        x = ops.reshape(hidden_states, (-1, self.embed_dim))
        logits = ops.matmul(x, ops.transpose(self.gate_weight))
        probs = ops.softmax(ops.cast(logits, "float32"), axis=-1)
        top_vals, top_idx = ops.top_k(probs, self.num_experts_per_tok)
        if self.norm_topk_prob:
            top_vals = top_vals / ops.sum(top_vals, axis=-1, keepdims=True)
        top_vals = ops.cast(top_vals, x.dtype)
        one_hot = ops.one_hot(top_idx, self.num_experts, dtype=x.dtype)
        routing = ops.sum(one_hot * top_vals[..., None], axis=1)
        routed = self.experts(x, routing)
        shared = ops.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        out = routed + shared
        return ops.reshape(out, (b, s, self.embed_dim))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "embed_dim": self.embed_dim,
                "moe_mlp_dim": self.moe_mlp_dim,
                "shared_mlp_dim": self.shared_mlp_dim,
                "norm_topk_prob": self.norm_topk_prob,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen3NextDecoderLayer(layers.Layer):
    """One hybrid Qwen3.5 decoder block: token mixer then pre-norm SwiGLU.

    Pre-norm residual block ``h = x + mixer(attention_norm(x))`` then
    ``h = h + mlp(mlp_norm(h))``, where the token ``mixer`` is selected by
    ``layer_type``: :class:`Qwen3NextAttention` (gated full attention) for
    ``"full_attention"``, otherwise :class:`Qwen3NextGatedDeltaNet` (linear
    attention). Rotary tables and the mask are forwarded only to the full-attention
    mixer; the cache state threaded through ``past_key_value`` is whatever the
    chosen mixer produces.

    Args:
        config: Dict of shared layer hyperparameters (``embed_dim``, ``mlp_dim``,
            ``num_heads``, ``num_kv_heads``, ``head_dim``, ``rotary_dim``,
            ``norm_eps``, and the ``linear_*`` Gated-DeltaNet dims).
        layer_type: ``"full_attention"`` or ``"linear_attention"``.

    Call args:
        hidden_states, cos, sin, attention_mask, past_key_value, use_cache: as in
            the selected mixer.

    Returns:
        The block output, or ``(output, state)`` when ``use_cache`` is set.
    """

    def __init__(self, config, layer_type, use_moe=True, **kwargs):
        super().__init__(**kwargs)
        self.config_dict = dict(config)
        self.layer_type = layer_type
        self.use_moe = use_moe
        c = config
        eps = c["norm_eps"]
        self.attention_norm = Qwen3NextRMSNorm(eps=eps, name="attention_norm")
        self.mlp_norm = Qwen3NextRMSNorm(eps=eps, name="mlp_norm")
        if use_moe:
            self.mlp = Qwen3NextSparseBlock(
                c["num_experts"],
                c["num_experts_per_tok"],
                c["embed_dim"],
                c["moe_mlp_dim"],
                c["shared_mlp_dim"],
                c["norm_topk_prob"],
                name="mlp",
            )
        else:
            self.mlp = Qwen3NextMLP(c["embed_dim"], c["mlp_dim"], name="mlp")
        if layer_type == "full_attention":
            self.attention = Qwen3NextAttention(
                c["embed_dim"],
                c["num_heads"],
                c["num_kv_heads"],
                c["head_dim"],
                c["rotary_dim"],
                eps,
                name="attention",
            )
        else:
            self.linear_attn = Qwen3NextGatedDeltaNet(
                c["embed_dim"],
                c["linear_num_key_heads"],
                c["linear_num_value_heads"],
                c["linear_key_head_dim"],
                c["linear_value_head_dim"],
                c["linear_conv_kernel_dim"],
                eps,
                name="linear_attn",
            )

    def call(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
        pad_mask=None,
    ):
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        new_state = None
        if self.layer_type == "full_attention":
            out = self.attention(
                hidden_states,
                cos,
                sin,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
        else:
            out = self.linear_attn(
                hidden_states,
                past_key_value=past_key_value,
                use_cache=use_cache,
                pad_mask=pad_mask,
            )
        if use_cache:
            out, new_state = out
        hidden_states = residual + out
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return (hidden_states, new_state) if use_cache else hidden_states

    def compute_output_spec(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
        pad_mask=None,
    ):
        # Residual stream keeps its shape; an explicit spec stops the functional
        # builder from tracing the conv1d / delta-rule recurrence (which only
        # resolves at run time, not symbolic-build time).
        return keras.KerasTensor(hidden_states.shape, dtype=self.compute_dtype)

    def decode_step(self, hidden_states, cos, sin, state, write_pos, key_mask):
        # One decode step; ``state`` is the per-layer cache: (cache_k, cache_v) for a
        # full-attention layer (fixed-slot KV), or (conv_state, recurrent_state) for a
        # Gated-DeltaNet layer (the recurrence is identical to prefill). cos/sin/
        # write_pos/key_mask are used only by the full-attention path.
        residual = hidden_states
        x = self.attention_norm(hidden_states)
        if self.layer_type == "full_attention":
            out, ck, cv = self.attention.decode_step(
                x, cos, sin, state[0], state[1], write_pos, key_mask
            )
            new_state = (ck, cv)
        else:
            out, new_state = self.linear_attn(x, past_key_value=state, use_cache=True)
        hidden_states = residual + out
        residual = hidden_states
        x = self.mlp_norm(hidden_states)
        hidden_states = residual + self.mlp(x)
        return hidden_states, new_state

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "config": self.config_dict,
                "layer_type": self.layer_type,
                "use_moe": self.use_moe,
            }
        )
        return config
