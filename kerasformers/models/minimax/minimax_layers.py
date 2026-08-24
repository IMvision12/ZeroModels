import keras
import numpy as np
from keras import layers, ops


def rotate_half(x):
    half = ops.shape(x)[-1] // 2
    return ops.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_partial_rope(x, cos, sin):
    # Rotate the first ``cos.shape[-1]`` head channels; pass the rest through.
    rot = cos.shape[-1]
    if rot == x.shape[-1]:
        return x * cos + rotate_half(x) * sin
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    x_rot = x_rot * cos + rotate_half(x_rot) * sin
    return ops.concatenate([x_rot, x_pass], axis=-1)


@keras.saving.register_keras_serializable(package="kerasformers")
class MiniMaxRMSNorm(layers.Layer):
    """Root-mean-square norm (plain learned weight, ones init).

    Args:
        eps: Variance epsilon. Defaults to ``1e-5`` (the MiniMax-Text-01
            checkpoints' value).
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
class MiniMaxExperts(layers.Layer):
    """Dense bank of SwiGLU experts evaluated for every token via einsum.

    Weights are stored fused exactly as the HF layout: ``gate_up_proj``
    ``(E, 2I, H)`` (gate stacked over up) and ``down_proj`` ``(E, H, I)``.
    Routing weights are a dense ``(T, E)`` matrix (zeros for unrouted
    experts), so the result equals sparse top-k dispatch.

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
class MiniMaxMoE(layers.Layer):
    """MiniMax sparse MoE block (Mixtral recipe): softmax top-k router.

    The bias-free router scores every expert, softmaxes in float32 over all
    experts, keeps the top-``num_experts_per_tok`` and renormalizes them to
    sum to one.

    Args:
        num_experts: Number of experts.
        num_experts_per_tok: Experts routed per token (Text-01: 2).
        embed_dim: Model width.
        mlp_dim: Per-expert hidden width.
    """

    def __init__(self, num_experts, num_experts_per_tok, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.experts = MiniMaxExperts(num_experts, embed_dim, mlp_dim, name="experts")

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
        top_vals = top_vals / ops.sum(top_vals, axis=-1, keepdims=True)
        one_hot = ops.one_hot(top_idx, self.num_experts)
        routing = ops.cast(ops.sum(one_hot * top_vals[..., None], axis=1), x.dtype)
        out = self.experts(x, routing)
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
class MiniMaxAttention(layers.Layer):
    """Full softmax grouped-query attention used on ``full_attention`` layers.

    Bias-free ``query`` / ``key`` / ``value`` / ``output_proj`` with rotary
    embeddings applied to the first ``cos.shape[-1]`` head channels (the
    released checkpoints rotate the full ``head_dim``; see the model's
    ``partial_rotary_factor``).

    Args:
        embed_dim: Model width.
        num_heads: Query heads.
        num_kv_heads: Key/value heads.
        head_dim: Per-head dim.

    Call args:
        hidden_states, cos, sin, attention_mask, past_key_value, use_cache: as
            in the other kerasformers attentions.
    """

    def __init__(self, embed_dim, num_heads, num_kv_heads, head_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim**-0.5
        self.query = layers.Dense(num_heads * head_dim, use_bias=False, name="query")
        self.key = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="key")
        self.value = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="value")
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")

    def project_qkv(self, hidden_states):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        q = ops.transpose(
            ops.reshape(
                self.query(hidden_states), (b, s, self.num_heads, self.head_dim)
            ),
            (0, 2, 1, 3),
        )
        k = ops.transpose(
            ops.reshape(
                self.key(hidden_states), (b, s, self.num_kv_heads, self.head_dim)
            ),
            (0, 2, 1, 3),
        )
        v = ops.transpose(
            ops.reshape(
                self.value(hidden_states), (b, s, self.num_kv_heads, self.head_dim)
            ),
            (0, 2, 1, 3),
        )
        return q, k, v

    def call(self, hidden_states, cos, sin, attention_mask=None, use_cache=False):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        q, k, v = self.project_qkv(hidden_states)
        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        q = apply_partial_rope(q, cos, sin)
        k = apply_partial_rope(k, cos, sin)
        new_kv = (k, v) if use_cache else None
        if self.num_kv_groups > 1:
            k = ops.repeat(k, self.num_kv_groups, axis=1)
            v = ops.repeat(v, self.num_kv_groups, axis=1)
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scaling
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), q.dtype)
        out = ops.matmul(attn, v)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, s, self.num_heads * self.head_dim)
        )
        out = self.output_proj(out)
        return (out, new_kv) if use_cache else out

    def decode_step(
        self, hidden_states, cos, sin, cache_k, cache_v, write_pos, key_mask
    ):
        b = ops.shape(hidden_states)[0]
        q, k, v = self.project_qkv(hidden_states)
        cos = ops.expand_dims(cos, axis=1)
        sin = ops.expand_dims(sin, axis=1)
        q = apply_partial_rope(q, cos, sin)
        k = apply_partial_rope(k, cos, sin)
        # rope runs in float32; match the cache dtype before writing
        k = ops.cast(k, cache_k.dtype)
        v = ops.cast(v, cache_v.dtype)
        cache_k = ops.slice_update(cache_k, (0, 0, write_pos, 0), k)
        cache_v = ops.slice_update(cache_v, (0, 0, write_pos, 0), v)
        kk, vv = cache_k, cache_v
        if self.num_kv_groups > 1:
            kk = ops.repeat(kk, self.num_kv_groups, axis=1)
            vv = ops.repeat(vv, self.num_kv_groups, axis=1)
        attn = ops.matmul(q, ops.transpose(kk, (0, 1, 3, 2))) * self.scaling
        attn = attn + key_mask
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), q.dtype)
        out = ops.matmul(attn, vv)
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
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class MiniMaxLightningAttention(layers.Layer):
    """MiniMax lightning (linear) attention used on ``linear_attention`` layers.

    SiLU is applied to the fused ``qkv`` projection, then attention is the
    block-decomposed linear recurrence over a per-head running ``K^T V`` state
    with per-head exponential decay (ALiBi-style slopes scaled by depth):
    within a block ``(Q K^T * diag_decay) V`` (intra) plus
    ``(Q * q_decay) state`` (inter), then
    ``state <- state * block_decay + (K * k_decay)^T V``. The head outputs are
    RMS-normed, gated by ``sigmoid(output_gate(x))``, and projected out. No
    rotary embedding and no softmax; padding is handled by zeroing the padded
    value rows at prefill.

    Args:
        embed_dim: Model width.
        num_heads: Attention heads (queries == keys == values).
        head_dim: Per-head dim.
        num_layers: Total decoder layers (the decay slope shrinks with depth).
        layer_idx: This layer's index.
        block_size: Block length of the chunked prefill scan.
        norm_eps: Epsilon of the output RMSNorm.

    Call args:
        hidden_states: ``(batch, seq, embed_dim)``.
        attention_mask: optional ``(batch, seq)`` 1/0 padding mask.
        use_cache: when ``True``, also return the final ``(batch, heads,
            head_dim, head_dim)`` KV state.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        head_dim,
        num_layers,
        layer_idx,
        block_size=256,
        norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.layer_idx = layer_idx
        self.block_size = block_size
        self.norm_eps = norm_eps

        base = 1.0 / (2.0 ** (8.0 / num_heads))
        factor = 1.0 - layer_idx / (num_layers - 1 + 1e-5) + 1e-5
        self.slope = (
            base ** np.arange(1, num_heads + 1, dtype="float32") * factor
        ).reshape(num_heads, 1, 1)

        self.qkv = layers.Dense(num_heads * head_dim * 3, use_bias=False, name="qkv")
        self.output_gate = layers.Dense(
            num_heads * head_dim, use_bias=False, name="output_gate"
        )
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")
        # HF hardcodes this norm's eps at the MixtralRMSNorm default (1e-6),
        # independent of config.rms_norm_eps.
        self.norm = MiniMaxRMSNorm(eps=1e-6, name="norm")

    def split_qkv(self, hidden_states):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        qkv = ops.silu(self.qkv(hidden_states))
        qkv = ops.reshape(qkv, (b, s, self.num_heads, 3 * self.head_dim))
        q, k, v = ops.split(qkv, 3, axis=-1)
        return (
            ops.transpose(q, (0, 2, 1, 3)),
            ops.transpose(k, (0, 2, 1, 3)),
            ops.transpose(v, (0, 2, 1, 3)),
        )

    def finalize(self, attn_output, hidden_states):
        b = ops.shape(attn_output)[0]
        s = ops.shape(attn_output)[2]
        out = ops.reshape(
            ops.transpose(attn_output, (0, 2, 1, 3)),
            (b, s, self.num_heads * self.head_dim),
        )
        out = self.norm(out)
        out = ops.sigmoid(self.output_gate(hidden_states)) * out
        return self.output_proj(out)

    def call(self, hidden_states, attention_mask=None, use_cache=False):
        b = ops.shape(hidden_states)[0]
        seq = int(hidden_states.shape[1])
        q, k, v = self.split_qkv(hidden_states)
        if attention_mask is not None:
            keep = ops.cast(ops.convert_to_tensor(attention_mask), v.dtype)
            v = v * keep[:, None, :, None]

        slope = ops.convert_to_tensor(self.slope)  # (H, 1, 1)
        kv_state = ops.zeros(
            (b, self.num_heads, self.head_dim, self.head_dim), dtype=v.dtype
        )
        num_blocks = -(-seq // self.block_size)
        outputs = []
        for i in range(num_blocks):
            start = i * self.block_size
            end = min(start + self.block_size, seq)
            cur = end - start
            qi = q[:, :, start:end]
            ki = k[:, :, start:end]
            vi = v[:, :, start:end]

            r = ops.cast(ops.arange(1, cur + 1), "float32")[:, None]  # (cur, 1)
            q_decay = ops.exp(-slope * r)  # (H, cur, 1)
            k_decay = ops.exp(-slope * (cur - r))
            diag = r - ops.transpose(r)  # (cur, cur), i - j
            diag = slope[None] * diag[None, None]
            diag = ops.where(diag >= 0.0, -diag, -np.inf)
            diag = ops.exp(diag)  # (1, H, cur, cur)
            block_decay = ops.exp(-slope * float(cur))  # (H, 1, 1)

            attn_intra = ops.matmul(qi, ops.transpose(ki, (0, 1, 3, 2)))
            out_intra = ops.matmul(attn_intra * ops.cast(diag, qi.dtype), vi)
            out_inter = ops.matmul(qi * ops.cast(q_decay, qi.dtype), kv_state)
            outputs.append(out_inter + out_intra)

            next_state = ops.matmul(
                ops.transpose(ki * ops.cast(k_decay, ki.dtype), (0, 1, 3, 2)), vi
            )
            kv_state = kv_state * ops.cast(block_decay, kv_state.dtype) + next_state

        attn_output = ops.concatenate(outputs, axis=-2)
        out = self.finalize(attn_output, hidden_states)
        return (out, kv_state) if use_cache else out

    def decode_step(self, hidden_states, kv_state):
        # Single-token linear-attention recurrence:
        # state <- exp(-slope) * state + k^T v ; out = q @ state.
        q, k, v = self.split_qkv(hidden_states)
        ratio = ops.cast(ops.exp(-ops.convert_to_tensor(self.slope)), kv_state.dtype)
        kv_state = ratio * kv_state + ops.matmul(ops.transpose(k, (0, 1, 3, 2)), v)
        out = ops.matmul(q, kv_state)
        return self.finalize(out, hidden_states), kv_state

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "num_layers": self.num_layers,
                "layer_idx": self.layer_idx,
                "block_size": self.block_size,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class MiniMaxDecoderLayer(layers.Layer):
    """One MiniMax block with the norm-first weighted-residual scheme.

    Unlike pre-norm transformers, the residual branch is the *normed* input:
    ``h = norm(x); h = h * alpha + attn(h) * beta`` and again for the MoE.
    The attention is full softmax or lightning depending on ``layer_type``.

    Args:
        embed_dim / mlp_dim / num_heads / num_kv_heads / head_dim: Dims.
        num_experts / num_experts_per_tok: MoE shape.
        layer_type: ``"full_attention"`` or ``"linear_attention"``.
        attn_alpha / attn_beta / mlp_alpha / mlp_beta: Residual weights.
        num_layers / layer_idx / block_size: Lightning-attention decay inputs.
        norm_eps: RMSNorm epsilon.
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        num_experts,
        num_experts_per_tok,
        layer_type,
        attn_alpha,
        attn_beta,
        mlp_alpha,
        mlp_beta,
        num_layers,
        layer_idx,
        block_size=256,
        norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.layer_type = layer_type
        self.attn_alpha = attn_alpha
        self.attn_beta = attn_beta
        self.mlp_alpha = mlp_alpha
        self.mlp_beta = mlp_beta
        self.num_layers = num_layers
        self.layer_idx = layer_idx
        self.block_size = block_size
        self.norm_eps = norm_eps

        self.attention_norm = MiniMaxRMSNorm(eps=norm_eps, name="attention_norm")
        if layer_type == "linear_attention":
            self.attention = MiniMaxLightningAttention(
                embed_dim,
                num_heads,
                head_dim,
                num_layers,
                layer_idx,
                block_size,
                norm_eps,
                name="attention",
            )
        else:
            self.attention = MiniMaxAttention(
                embed_dim, num_heads, num_kv_heads, head_dim, name="attention"
            )
        self.mlp_norm = MiniMaxRMSNorm(eps=norm_eps, name="mlp_norm")
        self.mlp = MiniMaxMoE(
            num_experts, num_experts_per_tok, embed_dim, mlp_dim, name="mlp"
        )

    def run_mlp(self, hidden_states):
        h = self.mlp_norm(hidden_states)
        return h * self.mlp_alpha + self.mlp(h) * self.mlp_beta

    def call(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        padding_mask=None,
        use_cache=False,
    ):
        h = self.attention_norm(hidden_states)
        cache_piece = None
        if self.layer_type == "linear_attention":
            attn_out = self.attention(
                h, attention_mask=padding_mask, use_cache=use_cache
            )
        else:
            attn_out = self.attention(
                h, cos, sin, attention_mask=attention_mask, use_cache=use_cache
            )
        if use_cache:
            attn_out, cache_piece = attn_out
        h = h * self.attn_alpha + attn_out * self.attn_beta
        h = self.run_mlp(h)
        return (h, cache_piece) if use_cache else h

    def decode_step(self, hidden_states, cos, sin, cache, write_pos, key_mask):
        h = self.attention_norm(hidden_states)
        if self.layer_type == "linear_attention":
            attn_out, new_cache = self.attention.decode_step(h, cache)
        else:
            attn_out, ck, cv = self.attention.decode_step(
                h, cos, sin, cache[:, 0], cache[:, 1], write_pos, key_mask
            )
            new_cache = ops.stack([ck, cv], axis=1)
        h = h * self.attn_alpha + attn_out * self.attn_beta
        h = self.run_mlp(h)
        return h, new_cache

    def compute_output_spec(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        padding_mask=None,
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
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "layer_type": self.layer_type,
                "attn_alpha": self.attn_alpha,
                "attn_beta": self.attn_beta,
                "mlp_alpha": self.mlp_alpha,
                "mlp_beta": self.mlp_beta,
                "num_layers": self.num_layers,
                "layer_idx": self.layer_idx,
                "block_size": self.block_size,
                "norm_eps": self.norm_eps,
            }
        )
        return config
