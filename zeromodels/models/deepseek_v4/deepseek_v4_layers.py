import keras
import numpy as np
from keras import layers, ops

MASK_NEG = -1e9


def rotate_half_interleaved(x):
    # out[2i] = -x[2i+1], out[2i+1] = x[2i] (GLM-style, layout preserved).
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    stacked = ops.stack([-x_odd, x_even], axis=-1)
    return ops.reshape(stacked, ops.shape(x))


def apply_v4_rope(x, cos, sin):
    """V4 interleaved rope on the *trailing* rope slice of ``x``.

    ``cos`` / ``sin`` carry one angle per pair ``(..., rd // 2)`` and are
    expanded with repeat-interleave; the last ``rd`` channels of ``x`` rotate
    in fp32 (pairs ``(2i, 2i+1)``, layout preserved), the leading nope
    channels pass through. ``x`` is ``(B, H, S, D)``; ``cos``/``sin`` are
    ``(B, S, rd // 2)``. Pass ``-sin`` for the conjugate (rope-undo) rotation.
    """
    cos = ops.repeat(cos, 2, axis=-1)[:, None]
    sin = ops.repeat(sin, 2, axis=-1)[:, None]
    rd = cos.shape[-1]
    nope = x[..., :-rd]
    rope = ops.cast(x[..., -rd:], "float32")
    rotated = rope * ops.cast(cos, "float32") + rotate_half_interleaved(
        rope
    ) * ops.cast(sin, "float32")
    return ops.concatenate([nope, ops.cast(rotated, x.dtype)], axis=-1)


def unweighted_rms_norm(x, eps=1e-6):
    scale = ops.rsqrt(
        ops.mean(ops.square(ops.cast(x, "float32")), axis=-1, keepdims=True) + eps
    )
    return x * ops.cast(scale, x.dtype)


def clamped_swiglu(gate, up, limit):
    # silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)
    gate = ops.minimum(gate, limit)
    up = ops.clip(up, -limit, limit)
    return ops.silu(gate) * up


@keras.saving.register_keras_serializable(package="zeromodels")
class DeepseekV4RMSNorm(layers.Layer):
    """Root-mean-square norm (plain learned weight, ones init).

    Args:
        eps: Variance epsilon (1e-6).
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
class DeepseekV4MLP(layers.Layer):
    """Clamped SwiGLU MLP: ``down(silu(gate.clamp) * up.clamp)``.

    Args:
        embed_dim / mlp_dim: Widths.
        swiglu_limit: Clamp bound (10.0).
    """

    def __init__(self, embed_dim, mlp_dim, swiglu_limit=10.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.swiglu_limit = swiglu_limit
        self.gate = layers.Dense(mlp_dim, use_bias=False, name="gate")
        self.up = layers.Dense(mlp_dim, use_bias=False, name="up")
        self.down = layers.Dense(embed_dim, use_bias=False, name="down")

    def build(self, input_shape):
        self.gate.build(input_shape)
        self.up.build(input_shape)
        self.down.build(tuple(input_shape[:-1]) + (self.mlp_dim,))
        self.built = True

    def call(self, x):
        return self.down(clamped_swiglu(self.gate(x), self.up(x), self.swiglu_limit))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "swiglu_limit": self.swiglu_limit,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DeepseekV4Experts(layers.Layer):
    """Dense bank of clamped-SwiGLU experts via einsum (fused HF layout).

    Args:
        num_experts / embed_dim / mlp_dim: Bank shape.
        swiglu_limit: Clamp bound.
    """

    def __init__(self, num_experts, embed_dim, mlp_dim, swiglu_limit=10.0, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.swiglu_limit = swiglu_limit

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
        act = clamped_swiglu(gate, up, self.swiglu_limit)
        expert_out = ops.einsum("tei,ehi->teh", act, self.down_proj)
        return ops.einsum("te,teh->th", routing_weights, expert_out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "swiglu_limit": self.swiglu_limit,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DeepseekV4MoE(layers.Layer):
    """V4 MoE block: sqrt-softplus router (learned or hash) + shared expert.

    The learned router takes a top-k of ``score + e_score_correction_bias``
    and keeps the unbiased ``sqrt(softplus(logits))`` scores, renormalized
    (+1e-20) and scaled by ``routed_scaling_factor``. ``hash_moe`` layers
    instead select experts from the frozen ``tid2eid[input_ids]`` lookup:
    only the weighting comes from the learned gate. A clamped-SwiGLU shared
    expert of the block input is added.

    Args:
        num_experts / num_experts_per_tok / embed_dim / mlp_dim: MoE shape.
        routed_scaling_factor: Routed-weight multiplier (1.5 / 2.5).
        is_hash: Use the ``tid2eid`` hash selection.
        vocab_size: Hash-table rows (hash layers only).
        swiglu_limit: Activation clamp.
    """

    def __init__(
        self,
        num_experts,
        num_experts_per_tok,
        embed_dim,
        mlp_dim,
        routed_scaling_factor=1.5,
        is_hash=False,
        vocab_size=129280,
        swiglu_limit=10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.routed_scaling_factor = routed_scaling_factor
        self.is_hash = is_hash
        self.vocab_size = vocab_size
        self.swiglu_limit = swiglu_limit
        self.experts = DeepseekV4Experts(
            num_experts, embed_dim, mlp_dim, swiglu_limit, name="experts"
        )
        self.shared_experts = DeepseekV4MLP(
            embed_dim, mlp_dim, swiglu_limit, name="shared_experts"
        )

    def build(self, input_shape):
        self.gate_weight = self.add_weight(
            name="gate_weight",
            shape=(self.num_experts, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        if self.is_hash:
            self.tid2eid = self.add_weight(
                name="tid2eid",
                shape=(self.vocab_size, self.num_experts_per_tok),
                initializer="zeros",
                trainable=False,
                dtype="int32",
            )
        else:
            self.e_score_correction_bias = self.add_weight(
                name="e_score_correction_bias",
                shape=(self.num_experts,),
                initializer="zeros",
                trainable=True,
            )
        self.built = True

    def call(self, hidden_states, input_ids=None):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        x = ops.reshape(hidden_states, (-1, self.embed_dim))
        shared_out = self.shared_experts(x)
        logits = ops.matmul(x, ops.transpose(self.gate_weight))
        scores = ops.sqrt(ops.softplus(logits))
        if self.is_hash:
            top_idx = ops.cast(
                ops.take(self.tid2eid, ops.reshape(input_ids, (-1,)), axis=0), "int32"
            )
            top_vals = ops.take_along_axis(scores, top_idx, axis=-1)
        else:
            biased = scores + ops.cast(self.e_score_correction_bias, scores.dtype)[None]
            _, top_idx = ops.top_k(biased, self.num_experts_per_tok)
            top_vals = ops.take_along_axis(scores, top_idx, axis=-1)
        top_vals = top_vals / (ops.sum(top_vals, axis=-1, keepdims=True) + 1e-20)
        top_vals = top_vals * self.routed_scaling_factor
        one_hot = ops.one_hot(top_idx, self.num_experts, dtype=top_vals.dtype)
        routing = ops.cast(ops.sum(one_hot * top_vals[..., None], axis=1), x.dtype)
        out = self.experts(x, routing) + shared_out
        return ops.reshape(out, (b, s, self.embed_dim))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "routed_scaling_factor": self.routed_scaling_factor,
                "is_hash": self.is_hash,
                "vocab_size": self.vocab_size,
                "swiglu_limit": self.swiglu_limit,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DeepseekV4Compressor(layers.Layer):
    """Windowed KV compressor shared by HCA, CSA, and the CSA indexer.

    Projects each source token to ``(kv, gate)`` (width ``head_dim`` for the
    single-series HCA flavor, ``2 * head_dim`` for the two-series CSA/indexer
    flavor), adds the learned per-window-slot ``position_bias``, and emits
    one compressed entry per ``compress_rate`` tokens as the softmax-gated
    sum over the window slots, RMS-normed and rotated by the compress-rope at
    the window's first absolute position. The two-series flavor lays window
    ``w``'s entry over ``2 * rate`` slots: the previous window's Ca slice
    (zeros / -inf gate for window 0) then the current window's Cb slice.

    Args:
        embed_dim: Model width (projection input).
        head_dim: Compressed-entry width.
        compress_rate: Tokens per window.
        two_series: CSA/indexer overlap layout.
        norm_eps: RMSNorm epsilon.
    """

    def __init__(
        self, embed_dim, head_dim, compress_rate, two_series, norm_eps=1e-6, **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.compress_rate = compress_rate
        self.two_series = two_series
        self.norm_eps = norm_eps
        width = 2 * head_dim if two_series else head_dim
        self.kv_proj = layers.Dense(width, use_bias=False, name="kv_proj")
        self.gate_proj = layers.Dense(width, use_bias=False, name="gate_proj")
        self.kv_norm = DeepseekV4RMSNorm(eps=norm_eps, name="kv_norm")

    def build(self, input_shape):
        width = 2 * self.head_dim if self.two_series else self.head_dim
        self.position_bias = self.add_weight(
            name="position_bias",
            shape=(self.compress_rate, width),
            initializer="zeros",
            trainable=True,
        )
        # kv_norm is only used inside compress_chunk (a plain method call), so
        # build it here under the compressor's own name scope: otherwise its
        # weight path collides with the attention's kv_norm.
        self.kv_norm.build((None, self.head_dim))
        self.built = True

    def compress_chunk(self, chunk_kv, chunk_gate, prior_kv=None, prior_gate=None):
        """Compress complete windows.

        ``chunk_kv`` / ``chunk_gate``: ``(B, n_windows, rate, width)`` already
        including ``position_bias`` on the gate. For the two-series flavor,
        ``prior_kv`` / ``prior_gate`` ``(B, rate, head_dim)`` optionally fill
        window 0's Ca half (decode path). Returns ``(B, n_windows, head_dim)``
        un-roped entries.
        """
        if not self.two_series:
            weights = ops.softmax(ops.cast(chunk_gate, "float32"), axis=2)
            entry = ops.sum(chunk_kv * ops.cast(weights, chunk_kv.dtype), axis=2)
            return self.kv_norm(entry)
        b = ops.shape(chunk_kv)[0]
        n = ops.shape(chunk_kv)[1]
        rate = self.compress_rate
        hd = self.head_dim
        cb_kv = chunk_kv[..., hd:]
        cb_gate = chunk_gate[..., hd:]
        ca_kv = chunk_kv[..., :hd]
        ca_gate = chunk_gate[..., :hd]
        # previous window's Ca slice; window 0 falls back to prior (or -inf).
        if prior_kv is None:
            prior_kv = ops.zeros((b, 1, rate, hd), dtype=chunk_kv.dtype)
            prior_gate = ops.full((b, 1, rate, hd), MASK_NEG, dtype=chunk_gate.dtype)
        else:
            prior_kv = prior_kv[:, None]
            prior_gate = prior_gate[:, None]
        shifted_kv = ops.concatenate([prior_kv, ca_kv[:, :-1]], axis=1)
        shifted_gate = ops.concatenate([prior_gate, ca_gate[:, :-1]], axis=1)
        new_kv = ops.concatenate([shifted_kv, cb_kv], axis=2)  # (B, n, 2r, hd)
        new_gate = ops.concatenate([shifted_gate, cb_gate], axis=2)
        weights = ops.softmax(ops.cast(new_gate, "float32"), axis=2)
        entry = ops.sum(new_kv * ops.cast(weights, new_kv.dtype), axis=2)
        del n
        return self.kv_norm(entry)

    def call(self, hidden_states):
        """Project the full sequence; returns raw ``(kv, gate)`` (no bias)."""
        return self.kv_proj(hidden_states), self.gate_proj(hidden_states)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "head_dim": self.head_dim,
                "compress_rate": self.compress_rate,
                "two_series": self.two_series,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DeepseekV4Attention(layers.Layer):
    """V4 attention: shared-KV MQA + sinks + sliding window + compressors.

    Queries take a low-rank bottleneck (``query_a`` -> RMS -> ``query_b`` ->
    unweighted per-head RMS); a single shared KV head (``kv`` -> RMS) is read
    as both key and value. Interleaved partial rope rotates the trailing
    ``qk_rope_head_dim`` channels of q and kv (sliding layers use the "main"
    tables, CSA/HCA layers the yarn "compress" tables), the conjugate
    rotation is applied to the attention output (undoing the rope the values
    carried), and the heads are mixed through the grouped low-rank output
    projection (``output_a`` block-diagonal over ``o_groups`` groups, then
    ``output_b``). CSA layers concatenate Lightning-Indexer-selected
    compressed entries onto the KV axis; HCA layers concatenate all causal
    compressed entries; every layer also applies the sliding-window mask to
    the raw KV.

    Args:
        embed_dim / num_heads / head_dim: Geometry (single KV head).
        q_lora_rank: Query bottleneck width.
        qk_rope_head_dim: Rotated trailing channels (64).
        o_groups / o_lora_rank: Grouped output projection shape.
        layer_type: ``"sliding_attention"`` / ``"compressed_sparse_attention"``
            / ``"heavily_compressed_attention"``.
        sliding_window: Raw-KV attention window (128).
        compress_rate: This layer's compressor window (4 CSA / 128 HCA).
        index_n_heads / index_head_dim / index_topk: Indexer geometry (CSA).
        norm_eps: RMSNorm epsilon.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        head_dim,
        q_lora_rank,
        qk_rope_head_dim,
        o_groups,
        o_lora_rank,
        layer_type="sliding_attention",
        sliding_window=128,
        compress_rate=4,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        compress_inv_freq=None,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.layer_type = layer_type
        self.sliding_window = sliding_window
        self.compress_rate = compress_rate
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.compress_inv_freq = (
            None
            if compress_inv_freq is None
            else np.asarray(compress_inv_freq, dtype="float32")
        )
        self.norm_eps = norm_eps
        self.scaling = head_dim**-0.5
        self.group_in = num_heads * head_dim // o_groups

        self.query_a = layers.Dense(q_lora_rank, use_bias=False, name="query_a")
        self.query_a_norm = DeepseekV4RMSNorm(eps=norm_eps, name="query_a_norm")
        self.query_b = layers.Dense(
            num_heads * head_dim, use_bias=False, name="query_b"
        )
        self.kv = layers.Dense(head_dim, use_bias=False, name="kv")
        self.kv_norm = DeepseekV4RMSNorm(eps=norm_eps, name="kv_norm")
        self.output_b = layers.Dense(embed_dim, use_bias=False, name="output_b")
        if layer_type != "sliding_attention":
            self.compressor = DeepseekV4Compressor(
                embed_dim,
                head_dim,
                compress_rate,
                two_series=layer_type == "compressed_sparse_attention",
                norm_eps=norm_eps,
                name="compressor",
            )
        if layer_type == "compressed_sparse_attention":
            self.index_compressor = DeepseekV4Compressor(
                embed_dim,
                index_head_dim,
                compress_rate,
                two_series=True,
                norm_eps=norm_eps,
                name="index_compressor",
            )
            self.index_query = layers.Dense(
                index_n_heads * index_head_dim, use_bias=False, name="index_query"
            )
            self.index_weights = layers.Dense(
                index_n_heads, use_bias=False, name="index_weights"
            )

    def build(self, input_shape):
        # This layer is only ever invoked through its helper methods (never
        # __call__), so the decoder layer's build() must call this explicitly
        # (creating the weights here keeps their paths under the proper scope).
        self.sinks = self.add_weight(
            name="sinks", shape=(self.num_heads,), initializer="zeros", trainable=True
        )
        self.output_a = self.add_weight(
            name="output_a",
            shape=(self.o_groups * self.o_lora_rank, self.group_in),
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def project_q(self, hidden_states, cos, sin):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        q_residual = self.query_a_norm(self.query_a(hidden_states))
        q = ops.transpose(
            ops.reshape(
                self.query_b(q_residual), (b, s, self.num_heads, self.head_dim)
            ),
            (0, 2, 1, 3),
        )
        q = unweighted_rms_norm(q, self.norm_eps)
        return apply_v4_rope(q, cos, sin), q_residual

    def project_kv(self, hidden_states, cos, sin):
        kv = self.kv_norm(self.kv(hidden_states))[:, None]  # (B, 1, S, D)
        return apply_v4_rope(kv, cos, sin)

    def rope_entries(self, entries, entry_cos, entry_sin):
        # Rotate compressed entries (B, n, hd) at their window-start positions.
        return apply_v4_rope(entries[:, None], entry_cos, entry_sin)[:, 0]

    def sink_attention(self, q, kv_all, mask):
        b = ops.shape(q)[0]
        s = ops.shape(q)[2]
        k = ops.broadcast_to(
            kv_all, (b, self.num_heads, ops.shape(kv_all)[2], self.head_dim)
        )
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scaling
        attn = attn + ops.cast(mask, attn.dtype)
        sinks = ops.broadcast_to(
            ops.reshape(self.sinks, (1, self.num_heads, 1, 1)),
            (b, self.num_heads, s, 1),
        )
        combined = ops.concatenate(
            [ops.cast(attn, "float32"), ops.cast(sinks, "float32")], axis=-1
        )
        combined = combined - ops.max(combined, axis=-1, keepdims=True)
        probs = ops.softmax(combined, axis=-1)[..., :-1]
        out = ops.matmul(ops.cast(probs, k.dtype), k)
        return out  # (B, H, S, D)

    def project_out(self, attn_output, cos, sin):
        # Conjugate rope on the output's rope slice, then grouped projection.
        b = ops.shape(attn_output)[0]
        s = ops.shape(attn_output)[2]
        out = apply_v4_rope(attn_output, cos, -sin)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, s, self.o_groups, self.group_in)
        )
        w = ops.reshape(self.output_a, (self.o_groups, self.o_lora_rank, self.group_in))
        grouped = ops.einsum("bsgi,goi->bsgo", out, w)
        grouped = ops.reshape(grouped, (b, s, self.o_groups * self.o_lora_rank))
        return self.output_b(grouped)

    def entry_rope_tables(self, positions):
        inv_freq = ops.convert_to_tensor(self.compress_inv_freq)
        freqs = ops.cast(positions, "float32")[..., None] * inv_freq
        return (
            ops.cast(ops.cos(freqs), self.compute_dtype),
            ops.cast(ops.sin(freqs), self.compute_dtype),
        )

    def index_block_bias(self, x, q_residual, idx_entries, visible, cos_comp, sin_comp):
        # Lightning-Indexer top-k selection over the compressed entries.
        b = ops.shape(x)[0]
        s = ops.shape(x)[1]
        hi = self.index_n_heads
        di = self.index_head_dim
        idx_q = ops.reshape(self.index_query(q_residual), (b, s, hi, di))
        idx_q = ops.transpose(idx_q, (0, 2, 1, 3))
        idx_q = apply_v4_rope(idx_q, cos_comp, sin_comp)
        idx_q = ops.transpose(idx_q, (0, 2, 1, 3))  # (B, S, Hi, Di)
        scores = ops.matmul(
            ops.cast(idx_q, "float32"),
            ops.transpose(ops.cast(idx_entries, "float32"), (0, 2, 1))[:, None],
        )  # (B, S, Hi, T)
        scores = ops.relu(scores) * (di**-0.5)
        weights = ops.cast(self.index_weights(x), "float32") * (hi**-0.5)
        scores = ops.sum(scores * weights[..., None], axis=2)  # (B, S, T)
        scores = ops.where(visible, scores, -float("inf"))
        n_entries = int(idx_entries.shape[1])
        top_k = min(self.index_topk, n_entries)
        top_scores, top_idx = ops.top_k(scores, top_k)
        valid = ops.cast(top_scores > -float("inf"), "float32")
        keep = ops.max(
            ops.one_hot(top_idx, n_entries, dtype="float32") * valid[..., None], axis=2
        )
        return ops.where(keep > 0, 0.0, MASK_NEG)[:, None]  # (B, 1, S, T)

    def call(
        self,
        hidden_states,
        cos,
        sin,
        cos_comp,
        sin_comp,
        position_ids=None,
        sliding_mask=None,
        use_cache=False,
    ):
        # ``hidden_states`` is already RMS-normed by the decoder layer.
        x = hidden_states
        q, q_residual = self.project_q(x, cos, sin)
        kv = self.project_kv(x, cos, sin)
        b = ops.shape(x)[0]
        seq = int(x.shape[1])
        rate = self.compress_rate
        pieces = {"kv": kv}
        kv_all = kv
        mask = sliding_mask

        if self.layer_type != "sliding_attention":
            kv_raw, gate_raw = self.compressor(x)
            n = seq // rate
            pieces["comp_kv"] = kv_raw
            pieces["comp_gate"] = gate_raw
            if self.layer_type == "compressed_sparse_attention":
                idx_kv_raw, idx_gate_raw = self.index_compressor(x)
                pieces["idx_kv"] = idx_kv_raw
                pieces["idx_gate"] = idx_gate_raw
            if n > 0:
                width = kv_raw.shape[-1]
                chunk_kv = ops.reshape(kv_raw[:, : n * rate], (b, n, rate, width))
                chunk_gate = (
                    ops.reshape(gate_raw[:, : n * rate], (b, n, rate, width))
                    + self.compressor.position_bias
                )
                entries = self.compressor.compress_chunk(chunk_kv, chunk_gate)
                positions = ops.broadcast_to(
                    ops.arange(n, dtype="int32")[None] * rate, (b, n)
                )
                ecos, esin = self.entry_rope_tables(positions)
                entries = self.rope_entries(entries, ecos, esin)
                pieces["comp_entries"] = entries
                threshold = (ops.cast(position_ids, "int32") + 1) // rate  # (B, S)
                entry_idx = ops.arange(n, dtype="int32")
                visible = entry_idx[None, None, :] < threshold[:, :, None]
                if self.layer_type == "heavily_compressed_attention":
                    block_bias = ops.where(visible, 0.0, MASK_NEG)[:, None]
                else:
                    iwidth = idx_kv_raw.shape[-1]
                    ichunk_kv = ops.reshape(
                        idx_kv_raw[:, : n * rate], (b, n, rate, iwidth)
                    )
                    ichunk_gate = (
                        ops.reshape(idx_gate_raw[:, : n * rate], (b, n, rate, iwidth))
                        + self.index_compressor.position_bias
                    )
                    idx_entries = self.index_compressor.compress_chunk(
                        ichunk_kv, ichunk_gate
                    )
                    idx_entries = self.rope_entries(idx_entries, ecos, esin)
                    pieces["idx_entries"] = idx_entries
                    block_bias = self.index_block_bias(
                        x, q_residual, idx_entries, visible, cos_comp, sin_comp
                    )
                kv_all = ops.concatenate([kv_all, entries[:, None]], axis=2)
                mask = ops.concatenate(
                    [mask, ops.cast(block_bias, mask.dtype)], axis=-1
                )

        attn = self.sink_attention(q, kv_all, mask)
        out = self.project_out(attn, cos, sin)
        return (out, pieces) if use_cache else out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "q_lora_rank": self.q_lora_rank,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "o_groups": self.o_groups,
                "o_lora_rank": self.o_lora_rank,
                "layer_type": self.layer_type,
                "sliding_window": self.sliding_window,
                "compress_rate": self.compress_rate,
                "index_n_heads": self.index_n_heads,
                "index_head_dim": self.index_head_dim,
                "index_topk": self.index_topk,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DeepseekV4HyperConnection(layers.Layer):
    """Manifold-Constrained Hyper-Connection (mHC) site.

    Turns the ``(B, S, hc_mult, H)`` residual streams into ``pre`` (stream
    collapse weights, sigmoid + eps), ``post`` (sublayer-output placement,
    2*sigmoid), and ``comb`` (an hc x hc stream mixer, softmax projected onto
    the doubly-stochastic manifold by Sinkhorn-Knopp). All math runs in
    float32.

    Args:
        hc_mult: Number of parallel residual streams (4).
        embed_dim: Model width.
        sinkhorn_iters: Sinkhorn-Knopp iterations (20).
        eps: Stabilizer added to sigmoid/softmax outputs.
    """

    def __init__(self, hc_mult, embed_dim, sinkhorn_iters=20, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.hc_mult = hc_mult
        self.embed_dim = embed_dim
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps

    def build(self, input_shape):
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = self.add_weight(
            name="fn",
            shape=(mix, self.hc_mult * self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.base = self.add_weight(
            name="base", shape=(mix,), initializer="zeros", trainable=True
        )
        self.scale = self.add_weight(
            name="scale", shape=(3,), initializer="ones", trainable=True
        )
        self.built = True

    def call(self, hidden_streams):
        hc = self.hc_mult
        b = ops.shape(hidden_streams)[0]
        s = ops.shape(hidden_streams)[1]
        flat = ops.reshape(
            ops.cast(hidden_streams, "float32"), (b, s, hc * self.embed_dim)
        )
        flat = unweighted_rms_norm(flat, self.eps)
        mixes = ops.matmul(flat, ops.transpose(ops.cast(self.fn, "float32")))
        pre_w = mixes[..., :hc]
        post_w = mixes[..., hc : 2 * hc]
        comb_w = mixes[..., 2 * hc :]
        base = ops.cast(self.base, "float32")
        scale = ops.cast(self.scale, "float32")
        pre = ops.sigmoid(pre_w * scale[0] + base[:hc]) + self.eps
        post = 2.0 * ops.sigmoid(post_w * scale[1] + base[hc : 2 * hc])
        comb_logits = ops.reshape(comb_w, (b, s, hc, hc)) * scale[2] + ops.reshape(
            base[2 * hc :], (hc, hc)
        )
        comb = ops.softmax(comb_logits, axis=-1) + self.eps
        comb = comb / (ops.sum(comb, axis=-2, keepdims=True) + self.eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (ops.sum(comb, axis=-1, keepdims=True) + self.eps)
            comb = comb / (ops.sum(comb, axis=-2, keepdims=True) + self.eps)
        collapsed = ops.cast(
            ops.sum(pre[..., None] * ops.cast(hidden_streams, "float32"), axis=2),
            hidden_streams.dtype,
        )
        return post, comb, collapsed

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hc_mult": self.hc_mult,
                "embed_dim": self.embed_dim,
                "sinkhorn_iters": self.sinkhorn_iters,
                "eps": self.eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DeepseekV4HyperHead(layers.Layer):
    """Final mHC stream collapse before the shared output RMSNorm.

    Args:
        hc_mult / embed_dim: Stream geometry.
        eps: Stabilizer.
    """

    def __init__(self, hc_mult, embed_dim, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.hc_mult = hc_mult
        self.embed_dim = embed_dim
        self.eps = eps

    def build(self, input_shape):
        self.hc_fn = self.add_weight(
            name="hc_fn",
            shape=(self.hc_mult, self.hc_mult * self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.hc_base = self.add_weight(
            name="hc_base", shape=(self.hc_mult,), initializer="zeros", trainable=True
        )
        self.hc_scale = self.add_weight(
            name="hc_scale", shape=(1,), initializer="ones", trainable=True
        )
        self.built = True

    def call(self, x):
        b = ops.shape(x)[0]
        s = ops.shape(x)[1]
        flat = ops.reshape(
            ops.cast(x, "float32"), (b, s, self.hc_mult * self.embed_dim)
        )
        flat = unweighted_rms_norm(flat, self.eps)
        mixes = ops.matmul(flat, ops.transpose(ops.cast(self.hc_fn, "float32")))
        pre = (
            ops.sigmoid(
                mixes * ops.cast(self.hc_scale, "float32")
                + ops.cast(self.hc_base, "float32")
            )
            + self.eps
        )
        return ops.cast(
            ops.sum(pre[..., None] * ops.cast(x, "float32"), axis=2), x.dtype
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {"hc_mult": self.hc_mult, "embed_dim": self.embed_dim, "eps": self.eps}
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class DeepseekV4DecoderLayer(layers.Layer):
    """One DeepSeek-V4 block with mHC parallel residual streams.

    The residual is ``hc_mult`` parallel streams ``(B, S, hc, H)``; each
    sublayer site collapses them with its hyper-connection's ``pre`` weights,
    runs the (pre-norm) sublayer on the collapsed sequence, and re-mixes:
    ``post (.) sublayer_out + comb^T @ streams``. The attention site runs the
    full V4 attention (sliding window + optional CSA/HCA compressor branch);
    the MLP site runs the V4 MoE (hash-routed on ``hash_moe`` layers).

    Args:
        embed_dim / num_heads / head_dim / q_lora_rank / qk_rope_head_dim /
        o_groups / o_lora_rank / sliding_window: Attention geometry.
        layer_type: Attention flavor.
        compress_rate: Window length for this layer's compressor.
        index_n_heads / index_head_dim / index_topk: Indexer geometry.
        num_experts / num_experts_per_tok / moe_mlp_dim /
        routed_scaling_factor / is_hash / vocab_size / swiglu_limit: MoE.
        hc_mult / hc_sinkhorn_iters / hc_eps: Hyper-connection geometry.
        norm_eps: RMSNorm epsilon.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        head_dim,
        q_lora_rank,
        qk_rope_head_dim,
        o_groups,
        o_lora_rank,
        layer_type,
        sliding_window,
        compress_rate,
        index_n_heads,
        index_head_dim,
        index_topk,
        num_experts,
        num_experts_per_tok,
        moe_mlp_dim,
        routed_scaling_factor,
        is_hash,
        vocab_size,
        swiglu_limit,
        hc_mult,
        hc_sinkhorn_iters,
        hc_eps,
        compress_inv_freq=None,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.layer_type = layer_type
        self.sliding_window = sliding_window
        self.compress_rate = compress_rate
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_mlp_dim = moe_mlp_dim
        self.routed_scaling_factor = routed_scaling_factor
        self.is_hash = is_hash
        self.vocab_size = vocab_size
        self.swiglu_limit = swiglu_limit
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.norm_eps = norm_eps

        self.attention_norm = DeepseekV4RMSNorm(eps=norm_eps, name="attention_norm")
        self.attention = DeepseekV4Attention(
            embed_dim,
            num_heads,
            head_dim,
            q_lora_rank,
            qk_rope_head_dim,
            o_groups,
            o_lora_rank,
            layer_type,
            sliding_window,
            compress_rate,
            index_n_heads,
            index_head_dim,
            index_topk,
            compress_inv_freq,
            norm_eps,
            name="attention",
        )
        self.mlp_norm = DeepseekV4RMSNorm(eps=norm_eps, name="mlp_norm")
        self.mlp = DeepseekV4MoE(
            num_experts,
            num_experts_per_tok,
            embed_dim,
            moe_mlp_dim,
            routed_scaling_factor,
            is_hash,
            vocab_size,
            swiglu_limit,
            name="mlp",
        )
        self.attn_hc = DeepseekV4HyperConnection(
            hc_mult, embed_dim, hc_sinkhorn_iters, hc_eps, name="attn_hc"
        )
        self.ffn_hc = DeepseekV4HyperConnection(
            hc_mult, embed_dim, hc_sinkhorn_iters, hc_eps, name="ffn_hc"
        )

    def build(self, input_shape):
        collapsed_shape = tuple(input_shape[:-2]) + (self.embed_dim,)
        self.attn_hc.build(input_shape)
        self.ffn_hc.build(input_shape)
        self.attention_norm.build(collapsed_shape)
        self.attention.build(collapsed_shape)
        self.mlp_norm.build(collapsed_shape)
        self.mlp.build(collapsed_shape)
        self.built = True

    def mix(self, streams, post, comb, sublayer_out):
        dtype = streams.dtype
        post = ops.cast(post, dtype)
        comb = ops.cast(comb, dtype)
        mixed = ops.einsum("bsjk,bsjd->bskd", comb, streams)
        return post[..., None] * sublayer_out[:, :, None, :] + mixed

    def call(
        self,
        hidden_streams,
        cos,
        sin,
        cos_comp,
        sin_comp,
        position_ids=None,
        sliding_mask=None,
        input_ids=None,
        use_cache=False,
    ):
        post, comb, collapsed = self.attn_hc(hidden_streams)
        attn_out = self.attention(
            self.attention_norm(collapsed),
            cos,
            sin,
            cos_comp,
            sin_comp,
            position_ids,
            sliding_mask,
            use_cache=use_cache,
        )
        pieces = None
        if use_cache:
            attn_out, pieces = attn_out
        hidden_streams = self.mix(hidden_streams, post, comb, attn_out)

        post, comb, collapsed = self.ffn_hc(hidden_streams)
        mlp_out = self.mlp(self.mlp_norm(collapsed), input_ids=input_ids)
        hidden_streams = self.mix(hidden_streams, post, comb, mlp_out)
        return (hidden_streams, pieces) if use_cache else hidden_streams

    def compute_output_spec(
        self,
        hidden_streams,
        cos,
        sin,
        cos_comp,
        sin_comp,
        position_ids=None,
        sliding_mask=None,
        input_ids=None,
        use_cache=False,
    ):
        return keras.KerasTensor(hidden_streams.shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "q_lora_rank": self.q_lora_rank,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "o_groups": self.o_groups,
                "o_lora_rank": self.o_lora_rank,
                "layer_type": self.layer_type,
                "sliding_window": self.sliding_window,
                "compress_rate": self.compress_rate,
                "index_n_heads": self.index_n_heads,
                "index_head_dim": self.index_head_dim,
                "index_topk": self.index_topk,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "moe_mlp_dim": self.moe_mlp_dim,
                "routed_scaling_factor": self.routed_scaling_factor,
                "is_hash": self.is_hash,
                "vocab_size": self.vocab_size,
                "swiglu_limit": self.swiglu_limit,
                "hc_mult": self.hc_mult,
                "hc_sinkhorn_iters": self.hc_sinkhorn_iters,
                "hc_eps": self.hc_eps,
                "norm_eps": self.norm_eps,
            }
        )
        return config
