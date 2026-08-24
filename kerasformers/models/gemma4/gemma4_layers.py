import math

import keras
from keras import layers, ops

from kerasformers.base.base_attention import fused_attention


def apply_rope(x, cos, sin):
    # Full-width half-rotation rope on (B, L, H, D); partial ("proportional")
    # rotary is realized upstream by zero-padding the inverse frequencies to
    # head_dim // 2 (cos(0) = 1, sin(0) = 0 leaves those dims untouched),
    # exactly like HF.
    cos = ops.expand_dims(cos, axis=2)
    sin = ops.expand_dims(sin, axis=2)
    half = ops.shape(x)[-1] // 2
    rot = ops.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rot * sin


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4RMSNorm(layers.Layer):
    """Gemma 4 root-mean-square layer norm: unlike earlier Gemmas, a *plain*
    ``* weight`` scale (ones-initialized), optionally weightless
    (``with_scale=False``: the value norm and the router input norm carry no
    checkpoint parameters).

    Args:
        eps: Variance epsilon. Defaults to ``1e-6``.
        with_scale: Whether a learned scale is applied.
    """

    def __init__(self, eps=1e-6, with_scale=True, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.with_scale = with_scale

    def build(self, input_shape):
        if self.with_scale:
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
        if self.with_scale:
            x = x * ops.cast(self.weight, "float32")
        return ops.cast(x, dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps, "with_scale": self.with_scale})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4MLP(layers.Layer):
    """Gemma 4 GeGLU feed-forward block: ``down(gelu_tanh(gate(x)) * up(x))``.

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

    def call(self, x):
        return self.down(ops.gelu(self.gate(x), approximate=True) * self.up(x))

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "mlp_dim": self.mlp_dim})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4Experts(layers.Layer):
    """Gemma 4 fused routed-expert bank (dense evaluation, GeGLU experts).

    Hugging Face fused layout: ``gate_up_proj`` ``(E, 2I, H)`` (contiguous
    halves), ``down_proj`` ``(E, H, I)``, no biases: with the
    ``gelu_pytorch_tanh`` activation. Given per-token per-expert routing
    weights ``(T, E)``, computes every expert and combines the outputs.

    Args:
        num_experts: Number of routed experts ``E``.
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
            shape=(e, 2 * i, h),
            initializer="zeros",
            trainable=True,
        )
        self.down_proj = self.add_weight(
            name="down_proj", shape=(e, h, i), initializer="zeros", trainable=True
        )
        self.built = True

    def call(self, hidden_states, routing_weights):
        gate_up = ops.einsum("th,eoh->teo", hidden_states, self.gate_up_proj)
        gate = gate_up[..., : self.mlp_dim]
        up = gate_up[..., self.mlp_dim :]
        act = ops.gelu(gate, approximate=True) * up
        expert_out = ops.einsum("tei,ehi->teh", act, self.down_proj)
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
class Gemma4Router(layers.Layer):
    """Gemma 4 expert router.

    The input is RMS-normalized (weightless), scaled by a learned per-channel
    ``scale`` times ``hidden**-0.5``, projected to expert logits, softmaxed;
    the top-k weights are renormalized and multiplied by a learned
    ``per_expert_scale``. Returns dense ``(T, E)`` routing weights.

    Args:
        num_experts: Routed expert count.
        top_k: Experts kept per token.
        embed_dim: Model width.
        norm_eps: Epsilon of the input norm.
    """

    def __init__(self, num_experts, top_k, embed_dim, norm_eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.top_k = top_k
        self.embed_dim = embed_dim
        self.norm_eps = norm_eps
        self.norm = Gemma4RMSNorm(eps=norm_eps, with_scale=False, name="norm")
        self.proj = layers.Dense(num_experts, use_bias=False, name="proj")

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale", shape=(self.embed_dim,), initializer="ones", trainable=True
        )
        self.per_expert_scale = self.add_weight(
            name="per_expert_scale",
            shape=(self.num_experts,),
            initializer="ones",
            trainable=True,
        )
        self.built = True

    def call(self, x):
        x = self.norm(x)
        x = x * self.scale * ops.cast(self.embed_dim**-0.5, x.dtype)
        probs = ops.softmax(self.proj(x), axis=-1)  # (T, E)
        top_vals, top_idx = ops.top_k(probs, self.top_k)
        top_vals = top_vals / ops.sum(top_vals, axis=-1, keepdims=True)
        top_vals = top_vals * ops.take(self.per_expert_scale, top_idx)
        one_hot = ops.one_hot(top_idx, self.num_experts)
        return ops.sum(one_hot * top_vals[..., None], axis=1)  # (T, E)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "embed_dim": self.embed_dim,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4Attention(layers.Layer):
    """Gemma 4 self-attention with per-layer geometry and K=V global layers.

    Sliding layers: ``head_dim`` (256), ``num_kv_heads`` K/V heads, separate
    value projection, full-width default rope (theta 1e4). Global layers:
    ``global_head_dim`` (512), ``num_global_kv_heads`` (MQA-ish) and: when
    ``k_eq_v`` means *no value projection*: the value is the raw key projection,
    normalized by a weightless RMS norm. Per-head q/k ``(1 + w)`` RMS norms
    are applied before rope; global layers rotate only the first
    ``partial_rotary_factor`` fraction of the head ("proportional" rope,
    theta 1e6). Attention scores are unscaled (``scaling = 1.0``).

    Args:
        embed_dim: Model width.
        num_heads: Query heads.
        num_kv_heads: K/V heads for this layer.
        head_dim: Per-head dim for this layer.
        k_eq_v: Whether value = normalized key projection (global layers).
        norm_eps: Epsilon of the q/k/v norms.

    Call args:
        hidden_states, cos, sin (width = rotated dims), attention_mask,
        past_key_value, use_cache: standard decoder-attention arguments.

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
        k_eq_v=False,
        is_kv_shared=False,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.k_eq_v = k_eq_v
        self.is_kv_shared = is_kv_shared
        self.norm_eps = norm_eps
        self.num_kv_groups = num_heads // num_kv_heads
        self.query = layers.Dense(num_heads * head_dim, use_bias=False, name="query")
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")
        self.query_norm = Gemma4RMSNorm(eps=norm_eps, name="query_norm")
        # Shared-KV layers (the last num_kv_shared_layers) carry no key/value
        # weights; they reuse the K/V of the last non-shared layer of their type.
        if is_kv_shared:
            self.key = self.value = self.key_norm = self.value_norm = None
        else:
            self.key = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="key")
            self.value = (
                None
                if k_eq_v
                else layers.Dense(num_kv_heads * head_dim, use_bias=False, name="value")
            )
            self.key_norm = Gemma4RMSNorm(eps=norm_eps, name="key_norm")
            self.value_norm = Gemma4RMSNorm(
                eps=norm_eps, with_scale=False, name="value_norm"
            )

    def project_q(self, hidden_states, q_len, cos, sin):
        b = ops.shape(hidden_states)[0]
        q = ops.reshape(
            self.query(hidden_states), (b, q_len, self.num_heads, self.head_dim)
        )
        q = self.query_norm(q)
        q = apply_rope(q, cos, sin)
        return ops.transpose(q, (0, 2, 1, 3))

    def project_kv(self, hidden_states, q_len, cos, sin):
        b = ops.shape(hidden_states)[0]
        k_raw = ops.reshape(
            self.key(hidden_states), (b, q_len, self.num_kv_heads, self.head_dim)
        )
        if self.value is not None:
            v = ops.reshape(
                self.value(hidden_states),
                (b, q_len, self.num_kv_heads, self.head_dim),
            )
        else:
            v = k_raw
        k = self.key_norm(k_raw)
        k = apply_rope(k, cos, sin)
        k = ops.transpose(k, (0, 2, 1, 3))
        v = self.value_norm(v)
        v = ops.transpose(v, (0, 2, 1, 3))
        return k, v

    def attend(self, q, k, v, attention_mask, b, q_len):
        if self.num_kv_groups > 1:
            k = ops.repeat(k, self.num_kv_groups, axis=1)
            v = ops.repeat(v, self.num_kv_groups, axis=1)
        out = fused_attention(q, k, v, 1.0, attention_mask)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, q_len, self.num_heads * self.head_dim)
        )
        return self.output_proj(out)

    def call(self, hidden_states, cos, sin, attention_mask=None, shared_kv=None):
        # Returns (output, (k, v)); shared-KV layers reuse ``shared_kv`` and pass
        # it back so the caller keeps threading it, others compute and expose it.
        b = ops.shape(hidden_states)[0]
        q_len = ops.shape(hidden_states)[1]
        q = self.project_q(hidden_states, q_len, cos, sin)
        k, v = (
            shared_kv
            if self.is_kv_shared
            else self.project_kv(hidden_states, q_len, cos, sin)
        )
        out = self.attend(q, k, v, attention_mask, b, q_len)
        return out, (k, v)

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        # Single-token attention against fixed-size caches. Shared-KV layers get
        # the storing layer's (already written) cache and do not write their own.
        b = ops.shape(hidden_states)[0]
        q = self.project_q(hidden_states, 1, cos, sin)
        if not self.is_kv_shared:
            k, v = self.project_kv(hidden_states, 1, cos, sin)
            # rope runs in float32; match the cache dtype before writing
            k = ops.cast(k, cache_k.dtype)
            v = ops.cast(v, cache_v.dtype)
            cache_k = ops.slice_update(cache_k, (0, 0, write_pos, 0), k)
            cache_v = ops.slice_update(cache_v, (0, 0, write_pos, 0), v)
        kk, vv = cache_k, cache_v
        if self.num_kv_groups > 1:
            kk = ops.repeat(kk, self.num_kv_groups, axis=1)
            vv = ops.repeat(vv, self.num_kv_groups, axis=1)
        out = fused_attention(q, kk, vv, 1.0, key_mask)
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
                "k_eq_v": self.k_eq_v,
                "is_kv_shared": self.is_kv_shared,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4DecoderLayer(layers.Layer):
    """One Gemma 4 text block: four-norm sandwich, optional parallel MoE
    branch, and a learned ``layer_scalar`` output multiplier.

    Dense layers: ``h = res + post_ff(mlp(pre_ff(h)))``. MoE layers (26B-A4B):
    the dense branch is normed (``post_ff_1``), a routed-expert branch is
    computed from the *residual* (``pre_ff_2`` -> experts -> ``post_ff_2``),
    the two are summed, then ``post_ff`` + residual as usual.

    Args:
        embed_dim: Model / residual-stream width.
        mlp_dim: Dense GeGLU hidden width.
        num_heads: Query heads.
        num_kv_heads: K/V heads for this layer.
        head_dim: Per-head dim for this layer.
        k_eq_v: Whether the attention is the global K=V kind.
        is_moe: Whether this layer carries the parallel expert branch.
        num_experts / num_experts_per_tok / moe_mlp_dim: MoE parameters.
        norm_eps: Epsilon of all norms.
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        k_eq_v=False,
        is_kv_shared=False,
        hidden_size_per_layer_input=0,
        is_moe=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_mlp_dim=0,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.k_eq_v = k_eq_v
        self.is_kv_shared = is_kv_shared
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.is_moe = is_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_mlp_dim = moe_mlp_dim
        self.norm_eps = norm_eps
        self.attention_norm = Gemma4RMSNorm(eps=norm_eps, name="attention_norm")
        self.attention = Gemma4Attention(
            embed_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            k_eq_v=k_eq_v,
            is_kv_shared=is_kv_shared,
            norm_eps=norm_eps,
            name="attention",
        )
        if hidden_size_per_layer_input:
            self.per_layer_input_gate = layers.Dense(
                hidden_size_per_layer_input, use_bias=False, name="per_layer_input_gate"
            )
            self.per_layer_projection = layers.Dense(
                embed_dim, use_bias=False, name="per_layer_projection"
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(
                eps=norm_eps, name="post_per_layer_input_norm"
            )
        self.post_attention_norm = Gemma4RMSNorm(
            eps=norm_eps, name="post_attention_norm"
        )
        self.pre_feedforward_norm = Gemma4RMSNorm(
            eps=norm_eps, name="pre_feedforward_norm"
        )
        self.mlp = Gemma4MLP(embed_dim, mlp_dim, name="mlp")
        self.post_feedforward_norm = Gemma4RMSNorm(
            eps=norm_eps, name="post_feedforward_norm"
        )
        if is_moe:
            self.router = Gemma4Router(
                num_experts, num_experts_per_tok, embed_dim, norm_eps, name="router"
            )
            self.experts = Gemma4Experts(
                num_experts, embed_dim, moe_mlp_dim, name="experts"
            )
            self.post_feedforward_norm_1 = Gemma4RMSNorm(
                eps=norm_eps, name="post_feedforward_norm_1"
            )
            self.pre_feedforward_norm_2 = Gemma4RMSNorm(
                eps=norm_eps, name="pre_feedforward_norm_2"
            )
            self.post_feedforward_norm_2 = Gemma4RMSNorm(
                eps=norm_eps, name="post_feedforward_norm_2"
            )

    def build(self, input_shape):
        self.layer_scalar = self.add_weight(
            name="layer_scalar", shape=(1,), initializer="ones", trainable=True
        )
        self.built = True

    def feed_forward(self, residual):
        h = self.pre_feedforward_norm(residual)
        h = self.mlp(h)
        if self.is_moe:
            h1 = self.post_feedforward_norm_1(h)
            flat = ops.reshape(residual, (-1, self.embed_dim))
            routing = self.router(flat)
            h2 = self.pre_feedforward_norm_2(flat)
            h2 = self.experts(h2, ops.cast(routing, h2.dtype))
            h2 = ops.reshape(h2, ops.shape(residual))
            h2 = self.post_feedforward_norm_2(h2)
            h = h1 + h2
        h = self.post_feedforward_norm(h)
        return residual + h

    def apply_per_layer_input(self, hidden_states, per_layer_input):
        # PLE residual: gate the block output, multiply by the per-layer input
        # embedding, project back, norm, and add.
        residual = hidden_states
        h = self.per_layer_input_gate(hidden_states)
        h = ops.gelu(h, approximate=True)
        h = h * ops.cast(per_layer_input, h.dtype)
        h = self.per_layer_projection(h)
        h = self.post_per_layer_input_norm(h)
        return residual + h

    def call(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        shared_kv=None,
        per_layer_input=None,
    ):
        # Always returns (hidden, (k, v)); the caller threads (k, v) for KV sharing.
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        attn_out, kv = self.attention(
            hidden_states, cos, sin, attention_mask=attention_mask, shared_kv=shared_kv
        )
        hidden_states = residual + self.post_attention_norm(attn_out)
        hidden_states = self.feed_forward(hidden_states)
        if self.hidden_size_per_layer_input:
            hidden_states = self.apply_per_layer_input(hidden_states, per_layer_input)
        hidden_states = hidden_states * ops.cast(self.layer_scalar, hidden_states.dtype)
        return hidden_states, kv

    def decode_step(
        self,
        hidden_states,
        cos,
        sin,
        cache_k,
        cache_v,
        write_pos,
        key_mask,
        per_layer_input=None,
    ):
        residual = hidden_states
        x = self.attention_norm(hidden_states)
        attn_out, cache_k, cache_v = self.attention.decode_step(
            x, cos, sin, cache_k, cache_v, write_pos, key_mask
        )
        hidden_states = residual + self.post_attention_norm(attn_out)
        hidden_states = self.feed_forward(hidden_states)
        if self.hidden_size_per_layer_input:
            hidden_states = self.apply_per_layer_input(hidden_states, per_layer_input)
        hidden_states = hidden_states * ops.cast(self.layer_scalar, hidden_states.dtype)
        return hidden_states, cache_k, cache_v

    def compute_output_spec(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        shared_kv=None,
        per_layer_input=None,
    ):
        # ``call`` returns (hidden, (k, v)); k/v are (batch, num_kv_heads, seq,
        # head_dim). An explicit spec lets the functional builder skip tracing the
        # dynamic-shape attention on backends that can't infer it.
        b, s = hidden_states.shape[0], hidden_states.shape[1]
        kv_shape = (b, self.num_kv_heads, s, self.head_dim)
        k = keras.KerasTensor(kv_shape, dtype=self.compute_dtype)
        v = keras.KerasTensor(kv_shape, dtype=self.compute_dtype)
        out = keras.KerasTensor(hidden_states.shape, dtype=self.compute_dtype)
        return out, (k, v)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "k_eq_v": self.k_eq_v,
                "is_kv_shared": self.is_kv_shared,
                "hidden_size_per_layer_input": self.hidden_size_per_layer_input,
                "is_moe": self.is_moe,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "moe_mlp_dim": self.moe_mlp_dim,
                "norm_eps": self.norm_eps,
            }
        )
        return config


def gelu_pytorch_tanh(x):
    return keras.activations.gelu(x, approximate=True)


def rotate_half(x):
    half = ops.shape(x)[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return ops.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=2):
    cos = ops.expand_dims(cos, unsqueeze_dim)
    sin = ops.expand_dims(sin, unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def apply_multidimensional_rope(x, cos, sin, ndim=2, unsqueeze_dim=2):
    # head_dim is split into ndim equal parts; each part is rotated with the
    # cos/sin for its spatial dimension, then concatenated back.
    channels = ops.shape(x)[-1]
    per = 2 * (channels // (2 * ndim))
    out = []
    for k in range(ndim):
        sl = slice(k * per, (k + 1) * per)
        out.append(
            apply_rotary_pos_emb(x[..., sl], cos[..., sl], sin[..., sl], unsqueeze_dim)
        )
    return ops.concatenate(out, axis=-1)


@keras.saving.register_keras_serializable(package="kerasformers")
class ClippableDense(layers.Layer):
    """A bias-free Dense with optional input/output clamping.

    Gemma 4 stores per-projection clip bounds (default +-inf, a no-op). When a
    checkpoint provides finite bounds the activations are clamped before and
    after the matmul.
    """

    def __init__(self, units, use_clipped_linears=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_clipped_linears = use_clipped_linears

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            name="kernel",
        )
        if self.use_clipped_linears:
            inf = float("inf")
            self.input_min = self.add_weight(
                shape=(),
                initializer=keras.initializers.Constant(-inf),
                trainable=False,
                name="input_min",
            )
            self.input_max = self.add_weight(
                shape=(),
                initializer=keras.initializers.Constant(inf),
                trainable=False,
                name="input_max",
            )
            self.output_min = self.add_weight(
                shape=(),
                initializer=keras.initializers.Constant(-inf),
                trainable=False,
                name="output_min",
            )
            self.output_max = self.add_weight(
                shape=(),
                initializer=keras.initializers.Constant(inf),
                trainable=False,
                name="output_max",
            )

    def call(self, x):
        if self.use_clipped_linears:
            x = ops.clip(
                x, ops.cast(self.input_min, x.dtype), ops.cast(self.input_max, x.dtype)
            )
        x = ops.matmul(x, ops.cast(self.kernel, x.dtype))
        if self.use_clipped_linears:
            x = ops.clip(
                x,
                ops.cast(self.output_min, x.dtype),
                ops.cast(self.output_max, x.dtype),
            )
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {"units": self.units, "use_clipped_linears": self.use_clipped_linears}
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4VisionRotaryEmbedding(layers.Layer):
    """2-D rotary embedding: computes cos/sin per spatial dim from (x, y) patch ids."""

    def __init__(self, head_dim, rope_theta=100.0, ndim=2, **kwargs):
        super().__init__(**kwargs)
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.ndim = ndim
        spatial_dim = head_dim // 2
        idx = ops.arange(0, spatial_dim, 2, dtype="float32")
        self.inv_freq = 1.0 / (rope_theta ** (idx / spatial_dim))

    def call(self, position_ids):
        # position_ids: (batch, num_patches, ndim)
        inv = ops.cast(self.inv_freq, "float32")[None, :, None]
        inv = ops.broadcast_to(inv, (ops.shape(position_ids)[0], ops.shape(inv)[1], 1))
        all_cos, all_sin = [], []
        for i in range(self.ndim):
            pos = ops.cast(position_ids[:, :, i], "float32")[:, None, :]
            freqs = ops.transpose(ops.matmul(inv, pos), (0, 2, 1))
            emb = ops.concatenate([freqs, freqs], axis=-1)
            all_cos.append(ops.cos(emb))
            all_sin.append(ops.sin(emb))
        cos = ops.concatenate(all_cos, axis=-1)
        sin = ops.concatenate(all_sin, axis=-1)
        return cos, sin

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "head_dim": self.head_dim,
                "rope_theta": self.rope_theta,
                "ndim": self.ndim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4VisionPatchEmbedder(layers.Layer):
    """Projects flattened patches and adds a 2-D (x, y) learned position embedding."""

    def __init__(self, hidden_size, patch_size, position_embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.position_embedding_size = position_embedding_size
        self.input_proj = layers.Dense(hidden_size, use_bias=False, name="input_proj")

    def build(self, input_shape):
        self.position_embedding_table = self.add_weight(
            shape=(2, self.position_embedding_size, self.hidden_size),
            initializer="ones",
            name="position_embedding_table",
        )

    def call(self, pixel_values, pixel_position_ids, padding_positions):
        pixel_values = 2.0 * (pixel_values - 0.5)
        hidden = self.input_proj(ops.cast(pixel_values, self.input_proj.compute_dtype))
        clamped = ops.maximum(pixel_position_ids, 0)
        x_emb = ops.take(self.position_embedding_table[0], clamped[..., 0], axis=0)
        y_emb = ops.take(self.position_embedding_table[1], clamped[..., 1], axis=0)
        pos = x_emb + y_emb
        pos = ops.where(ops.expand_dims(padding_positions, -1), 0.0, pos)
        return hidden + ops.cast(pos, hidden.dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "patch_size": self.patch_size,
                "position_embedding_size": self.position_embedding_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4VisionMLP(layers.Layer):
    """Gated SwiGLU MLP with clippable linears."""

    def __init__(self, intermediate_size, use_clipped_linears=True, **kwargs):
        super().__init__(**kwargs)
        self.intermediate_size = intermediate_size
        self.gate_proj = ClippableDense(
            intermediate_size, use_clipped_linears, name="gate_proj"
        )
        self.up_proj = ClippableDense(
            intermediate_size, use_clipped_linears, name="up_proj"
        )

    def build(self, input_shape):
        self.down_proj = ClippableDense(
            input_shape[-1], self.gate_proj.use_clipped_linears, name="down_proj"
        )

    def call(self, x):
        return self.down_proj(gelu_pytorch_tanh(self.gate_proj(x)) * self.up_proj(x))

    def get_config(self):
        config = super().get_config()
        config.update({"intermediate_size": self.intermediate_size})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4VisionAttention(layers.Layer):
    """Bidirectional MHA with q/k/v RMSNorm and 2-D rotary embedding."""

    def __init__(
        self,
        num_heads,
        num_kv_heads,
        head_dim,
        eps=1e-6,
        use_clipped_linears=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = ClippableDense(
            num_heads * head_dim, use_clipped_linears, name="q_proj"
        )
        self.k_proj = ClippableDense(
            num_kv_heads * head_dim, use_clipped_linears, name="k_proj"
        )
        self.v_proj = ClippableDense(
            num_kv_heads * head_dim, use_clipped_linears, name="v_proj"
        )
        self.o_proj = ClippableDense(
            num_heads * head_dim, use_clipped_linears, name="o_proj"
        )
        self.q_norm = Gemma4RMSNorm(eps=eps, name="q_norm")
        self.k_norm = Gemma4RMSNorm(eps=eps, name="k_norm")
        self.v_norm = Gemma4RMSNorm(eps=eps, with_scale=False, name="v_norm")

    def call(self, hidden_states, cos, sin, attention_mask=None):
        b = ops.shape(hidden_states)[0]
        n = ops.shape(hidden_states)[1]
        q = ops.reshape(
            self.q_proj(hidden_states), (b, n, self.num_heads, self.head_dim)
        )
        q = apply_multidimensional_rope(self.q_norm(q), cos, sin)
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.reshape(
            self.k_proj(hidden_states), (b, n, self.num_kv_heads, self.head_dim)
        )
        k = apply_multidimensional_rope(self.k_norm(k), cos, sin)
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.reshape(
            self.v_proj(hidden_states), (b, n, self.num_kv_heads, self.head_dim)
        )
        v = ops.transpose(self.v_norm(v), (0, 2, 1, 3))

        # scaling is 1.0 in Gemma4 vision (baked into the q/k norms)
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), q.dtype)
        out = ops.matmul(attn, v)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, n, self.num_heads * self.head_dim)
        )
        return self.o_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4VisionEncoderLayer(layers.Layer):
    """Sandwich-norm transformer block (4 RMSNorms around attention and MLP)."""

    def __init__(
        self,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        eps=1e-6,
        use_clipped_linears=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.self_attn = Gemma4VisionAttention(
            num_heads,
            num_kv_heads,
            head_dim,
            eps,
            use_clipped_linears,
            name="self_attn",
        )
        self.mlp = Gemma4VisionMLP(intermediate_size, use_clipped_linears, name="mlp")
        self.input_layernorm = Gemma4RMSNorm(eps=eps, name="input_layernorm")
        self.post_attention_layernorm = Gemma4RMSNorm(
            eps=eps, name="post_attention_layernorm"
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            eps=eps, name="pre_feedforward_layernorm"
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            eps=eps, name="post_feedforward_layernorm"
        )

    def call(self, hidden_states, cos, sin, attention_mask=None):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.self_attn(h, cos, sin, attention_mask)
        h = self.post_attention_layernorm(h)
        hidden_states = residual + h

        residual = hidden_states
        h = self.pre_feedforward_layernorm(hidden_states)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        return residual + h


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4VisionPooler(layers.Layer):
    """2-D spatial average pooling of patches, then sqrt(hidden) scaling in float32."""

    def __init__(self, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.root_hidden_size = hidden_size**0.5

    def call(self, hidden_states, pixel_position_ids, padding_positions, output_length):
        hidden_states = ops.where(
            ops.expand_dims(padding_positions, -1), 0.0, hidden_states
        )
        seq = ops.shape(hidden_states)[1]
        if int(seq) != int(output_length):
            hidden_states, _ = self.avg_pool_by_positions(
                hidden_states, pixel_position_ids, output_length
            )
        return ops.cast(hidden_states, "float32") * self.root_hidden_size

    def avg_pool_by_positions(self, hidden_states, pixel_position_ids, length):
        seq = int(ops.shape(hidden_states)[1])
        k = int((seq // int(length)) ** 0.5)
        clamped = ops.maximum(pixel_position_ids, 0)
        max_x = ops.max(clamped[..., 0], axis=-1, keepdims=True) + 1
        kernel = ops.floor_divide(clamped, k)
        kernel = kernel[..., 0] + ops.floor_divide(max_x, k) * kernel[..., 1]
        weights = ops.cast(
            ops.one_hot(ops.cast(kernel, "int32"), int(length)), "float32"
        ) / (k * k)
        out = ops.matmul(
            ops.transpose(weights, (0, 2, 1)), ops.cast(hidden_states, "float32")
        )
        mask = ops.logical_not(ops.all(weights == 0, axis=1))
        return ops.cast(out, hidden_states.dtype), mask

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4MultimodalEmbedder(layers.Layer):
    """Projects vision/audio soft tokens into the text embedding space."""

    def __init__(self, text_hidden_size, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.text_hidden_size = text_hidden_size
        self.embedding_pre_projection_norm = Gemma4RMSNorm(
            eps=eps, with_scale=False, name="embedding_pre_projection_norm"
        )
        self.embedding_projection = layers.Dense(
            text_hidden_size, use_bias=False, name="embedding_projection"
        )

    def call(self, inputs_embeds):
        return self.embedding_projection(
            self.embedding_pre_projection_norm(inputs_embeds)
        )

    def get_config(self):
        config = super().get_config()
        config.update({"text_hidden_size": self.text_hidden_size})
        return config


def glu(x, axis=-1):
    a, b = ops.split(x, 2, axis=axis)
    return a * ops.sigmoid(b)


def block_indices(num_blocks, context_size, chunk_size):
    starts = ops.arange(num_blocks) * chunk_size
    offsets = ops.arange(context_size)
    return starts[:, None] + offsets[None, :]


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4AudioRelPositionalEncoding(layers.Layer):
    """Sinusoidal relative position table of shape (1, context // 2 + 1, hidden)."""

    def __init__(self, hidden_size, context_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.context_size = context_size
        num_timescales = hidden_size // 2
        log_increment = math.log(10000.0) / max(num_timescales - 1, 1)
        self.inv_timescales = ops.exp(
            ops.arange(num_timescales, dtype="float32") * -log_increment
        )

    def compute(self, dtype="float32"):
        position_ids = ops.cast(ops.arange(self.context_size // 2, -1, -1), "float32")[
            :, None
        ]
        scaled = position_ids * self.inv_timescales[None, :]
        emb = ops.concatenate([ops.sin(scaled), ops.cos(scaled)], axis=-1)
        return ops.cast(emb[None], dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"hidden_size": self.hidden_size, "context_size": self.context_size}
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4AudioAttention(layers.Layer):
    """Chunked local attention with relative-position bias and tanh logit cap."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        chunk_size,
        context_left,
        context_right,
        logit_cap=50.0,
        invalid_logits=-1e9,
        use_clipped_linears=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.chunk_size = chunk_size
        self.max_past_horizon = context_left - 1
        self.max_future_horizon = context_right
        self.context_size = chunk_size + self.max_past_horizon + self.max_future_horizon
        self.logit_cap = logit_cap
        self.invalid_logits = invalid_logits
        self.q_scale = (self.head_dim**-0.5) / math.log(2)
        self.k_scale = math.log(1 + math.e) / math.log(2)

        units = num_heads * self.head_dim
        self.q_proj = ClippableDense(units, use_clipped_linears, name="q_proj")
        self.k_proj = ClippableDense(units, use_clipped_linears, name="k_proj")
        self.v_proj = ClippableDense(units, use_clipped_linears, name="v_proj")
        self.post = ClippableDense(hidden_size, use_clipped_linears, name="post")
        self.relative_k_proj = layers.Dense(
            units, use_bias=False, name="relative_k_proj"
        )

    def build(self, input_shape):
        self.per_dim_scale = self.add_weight(
            shape=(self.head_dim,), initializer="zeros", name="per_dim_scale"
        )

    def to_block(self, x, seq_len, num_blocks):
        pad = num_blocks * self.chunk_size - seq_len
        x = ops.pad(x, [[0, 0], [0, pad], [0, 0], [0, 0]])
        b = ops.shape(x)[0]
        return ops.reshape(
            x, (b, num_blocks, self.chunk_size, self.num_heads, self.head_dim)
        )

    def extract_context(self, x, num_blocks):
        x = ops.pad(
            x,
            [
                [0, 0],
                [self.max_past_horizon, self.max_future_horizon + self.chunk_size - 1],
                [0, 0],
                [0, 0],
            ],
        )
        idx = block_indices(num_blocks, self.context_size, self.chunk_size)
        # take with a [num_blocks, context] index over the seq axis yields
        # [B, num_blocks, context, H, D], matching HF's unfold + movedim.
        return ops.take(x, idx, axis=1)

    def rel_shift(self, x, num_blocks, position_length):
        b = ops.shape(x)[0]
        x = ops.pad(
            x,
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, self.context_size + 1 - position_length],
            ],
        )
        x = ops.reshape(
            x,
            (b, self.num_heads, num_blocks, self.chunk_size * (self.context_size + 1)),
        )
        x = x[..., : self.chunk_size * self.context_size]
        return ops.reshape(
            x, (b, self.num_heads, num_blocks, self.chunk_size, self.context_size)
        )

    def call(self, hidden_states, position_embeddings, attention_mask=None):
        b = ops.shape(hidden_states)[0]
        seq_len = int(ops.shape(hidden_states)[1])
        num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
        shape = (b, seq_len, self.num_heads, self.head_dim)

        q = ops.cast(ops.reshape(self.q_proj(hidden_states), shape), "float32")
        k = ops.cast(ops.reshape(self.k_proj(hidden_states), shape), "float32")
        v = ops.cast(ops.reshape(self.v_proj(hidden_states), shape), "float32")
        q = q * self.q_scale * ops.softplus(ops.cast(self.per_dim_scale, "float32"))
        k = k * self.k_scale

        q = self.to_block(q, seq_len, num_blocks)
        k = self.extract_context(k, num_blocks)
        v = self.extract_context(v, num_blocks)

        rel_k = self.relative_k_proj(position_embeddings)
        rel_k = ops.cast(
            ops.reshape(rel_k, (-1, self.num_heads, self.head_dim)), "float32"
        )
        position_length = int(ops.shape(rel_k)[0])

        queries = ops.transpose(q, (0, 3, 1, 2, 4))
        matrix_ac = ops.matmul(queries, ops.transpose(k, (0, 3, 1, 4, 2)))

        queries_flat = ops.reshape(queries, (b, self.num_heads, -1, self.head_dim))
        matrix_bd = ops.matmul(queries_flat, ops.transpose(rel_k, (1, 2, 0)))
        matrix_bd = ops.reshape(
            matrix_bd, (b, self.num_heads, num_blocks, self.chunk_size, position_length)
        )
        matrix_bd = self.rel_shift(matrix_bd, num_blocks, position_length)

        attn = (matrix_ac + matrix_bd) / self.logit_cap
        attn = ops.tanh(attn) * self.logit_cap
        if attention_mask is not None:
            attn = ops.where(attention_mask, attn, self.invalid_logits)
        attn = ops.softmax(attn, axis=-1)
        out = ops.matmul(attn, ops.transpose(v, (0, 3, 1, 2, 4)))
        out = ops.transpose(out, (0, 2, 3, 1, 4))
        out = ops.reshape(out, (b, num_blocks * self.chunk_size, self.hidden_size))
        out = out[:, :seq_len]
        return self.post(ops.cast(out, hidden_states.dtype))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "chunk_size": self.chunk_size,
                "context_left": self.max_past_horizon + 1,
                "context_right": self.max_future_horizon,
                "logit_cap": self.logit_cap,
                "invalid_logits": self.invalid_logits,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4AudioSubSampleConvLayer(layers.Layer):
    """3x3 stride-2 conv (pad 1) with channel LayerNorm and ReLU; halves time/freq."""

    def __init__(self, out_channels, norm_eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.norm_eps = norm_eps
        self.pad = layers.ZeroPadding2D(padding=1)
        self.conv = layers.Conv2D(
            out_channels, 3, strides=2, padding="valid", use_bias=False, name="conv"
        )
        self.norm = layers.LayerNormalization(
            epsilon=norm_eps, center=False, scale=True, name="norm"
        )

    def call(self, x, mask=None):
        if mask is not None:
            x = x * ops.cast(mask[:, :, None, None], x.dtype)
        x = self.conv(self.pad(x))
        x = keras.activations.relu(self.norm(x))
        if mask is not None:
            mask = mask[:, ::2]
        return x, mask

    def get_config(self):
        config = super().get_config()
        config.update({"out_channels": self.out_channels, "norm_eps": self.norm_eps})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4AudioSubSampleConvProjection(layers.Layer):
    """Two stride-2 conv layers then a linear projection to hidden_size."""

    def __init__(self, conv_channels, hidden_size, norm_eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.conv_channels = list(conv_channels)
        self.hidden_size = hidden_size
        self.norm_eps = norm_eps
        self.layer0 = Gemma4AudioSubSampleConvLayer(
            conv_channels[0], norm_eps, name="layer0"
        )
        self.layer1 = Gemma4AudioSubSampleConvLayer(
            conv_channels[1], norm_eps, name="layer1"
        )
        self.input_proj_linear = layers.Dense(
            hidden_size, use_bias=False, name="input_proj_linear"
        )

    def call(self, input_features, mask=None):
        x = input_features[..., None]
        x, mask = self.layer0(x, mask)
        x, mask = self.layer1(x, mask)
        b = ops.shape(x)[0]
        seq = int(ops.shape(x)[1])
        x = ops.reshape(x, (b, seq, -1))
        return self.input_proj_linear(x), mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "conv_channels": self.conv_channels,
                "hidden_size": self.hidden_size,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4AudioFeedForward(layers.Layer):
    """Half-step conformer feed-forward (pre/post RMSNorm, SiLU, residual weight)."""

    def __init__(
        self,
        hidden_size,
        norm_eps=1e-6,
        residual_weight=0.5,
        use_clipped_linears=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.residual_weight = residual_weight
        self.ffw_layer_1 = ClippableDense(
            hidden_size * 4, use_clipped_linears, name="ffw_layer_1"
        )
        self.ffw_layer_2 = ClippableDense(
            hidden_size, use_clipped_linears, name="ffw_layer_2"
        )
        self.pre_layer_norm = Gemma4RMSNorm(eps=norm_eps, name="pre_layer_norm")
        self.post_layer_norm = Gemma4RMSNorm(eps=norm_eps, name="post_layer_norm")

    def call(self, hidden_states):
        residual = hidden_states
        h = self.pre_layer_norm(hidden_states)
        h = self.ffw_layer_1(h)
        h = keras.activations.silu(h)
        h = self.ffw_layer_2(h)
        h = self.post_layer_norm(h)
        return h * self.residual_weight + residual

    def get_config(self):
        config = super().get_config()
        config.update(
            {"hidden_size": self.hidden_size, "residual_weight": self.residual_weight}
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4AudioLightConv1d(layers.Layer):
    """GLU + causal depthwise conv1d conformer module."""

    def __init__(
        self,
        hidden_size,
        conv_kernel_size=5,
        norm_eps=1e-6,
        use_clipped_linears=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.conv_kernel_size = conv_kernel_size
        self.left_pad = conv_kernel_size - 1
        self.linear_start = ClippableDense(
            hidden_size * 2, use_clipped_linears, name="linear_start"
        )
        self.linear_end = ClippableDense(
            hidden_size, use_clipped_linears, name="linear_end"
        )
        self.pre_layer_norm = Gemma4RMSNorm(eps=norm_eps, name="pre_layer_norm")
        self.conv_norm = Gemma4RMSNorm(eps=norm_eps, name="conv_norm")

    def build(self, input_shape):
        self.depthwise_kernel = self.add_weight(
            shape=(self.conv_kernel_size, self.hidden_size, 1),
            initializer="glorot_uniform",
            name="depthwise_conv1d_kernel",
        )

    def call(self, hidden_states):
        residual = hidden_states
        h = self.pre_layer_norm(hidden_states)
        h = self.linear_start(h)
        h = glu(h, axis=-1)
        h = ops.pad(h, [[0, 0], [self.left_pad, 0], [0, 0]])
        h = ops.depthwise_conv(
            h, ops.cast(self.depthwise_kernel, h.dtype), strides=1, padding="valid"
        )
        h = self.conv_norm(h)
        h = keras.activations.silu(h)
        h = self.linear_end(h)
        return h + residual

    def get_config(self):
        config = super().get_config()
        config.update(
            {"hidden_size": self.hidden_size, "conv_kernel_size": self.conv_kernel_size}
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Gemma4AudioLayer(layers.Layer):
    """Conformer block: FF1, attention, light conv1d, FF2, sandwiched by RMSNorms."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        chunk_size,
        context_left,
        context_right,
        conv_kernel_size=5,
        norm_eps=1e-6,
        residual_weight=0.5,
        logit_cap=50.0,
        invalid_logits=-1e9,
        use_clipped_linears=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feed_forward1 = Gemma4AudioFeedForward(
            hidden_size,
            norm_eps,
            residual_weight,
            use_clipped_linears,
            name="feed_forward1",
        )
        self.feed_forward2 = Gemma4AudioFeedForward(
            hidden_size,
            norm_eps,
            residual_weight,
            use_clipped_linears,
            name="feed_forward2",
        )
        self.self_attn = Gemma4AudioAttention(
            hidden_size,
            num_heads,
            chunk_size,
            context_left,
            context_right,
            logit_cap,
            invalid_logits,
            use_clipped_linears,
            name="self_attn",
        )
        self.lconv1d = Gemma4AudioLightConv1d(
            hidden_size, conv_kernel_size, norm_eps, use_clipped_linears, name="lconv1d"
        )
        self.norm_pre_attn = Gemma4RMSNorm(eps=norm_eps, name="norm_pre_attn")
        self.norm_post_attn = Gemma4RMSNorm(eps=norm_eps, name="norm_post_attn")
        self.norm_out = Gemma4RMSNorm(eps=norm_eps, name="norm_out")

    def call(self, hidden_states, position_embeddings, attention_mask=None):
        hidden_states = self.feed_forward1(hidden_states)
        residual = hidden_states
        h = self.norm_pre_attn(hidden_states)
        h = self.self_attn(h, position_embeddings, attention_mask)
        h = self.norm_post_attn(h)
        hidden_states = h + residual
        hidden_states = self.lconv1d(hidden_states)
        hidden_states = self.feed_forward2(hidden_states)
        return self.norm_out(hidden_states)
