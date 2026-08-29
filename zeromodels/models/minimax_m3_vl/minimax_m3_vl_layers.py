import keras
from keras import layers, ops

MASK_NEG = -1e9


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


def swiglu_oai(gate, up, alpha, limit):
    # GPT-OSS-style clamped SwiGLU: out = (up + 1) * gate * sigmoid(gate * alpha).
    gate = ops.minimum(gate, limit)
    up = ops.clip(up, -limit, limit)
    glu = gate * ops.sigmoid(gate * alpha)
    return (up + 1.0) * glu


@keras.saving.register_keras_serializable(package="zeromodels")
class MiniMaxM3VLRMSNorm(layers.Layer):
    """Gemma-style RMSNorm: fp32 normalize, scale by ``(1 + weight)`` (zeros init).

    Args:
        eps: Variance epsilon (1e-6).
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
        variance = ops.mean(ops.square(x), axis=-1, keepdims=True)
        x = x * ops.rsqrt(variance + self.eps)
        return ops.cast(x * (1.0 + ops.cast(self.weight, "float32")), dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MiniMaxM3VLDenseMLP(layers.Layer):
    """Dense SwiGLU-OAI MLP (clamped gate/up, ``(up + 1) * glu``).

    Used both as the dense per-layer MLP and as the MoE shared expert.

    Args:
        embed_dim: Model width.
        mlp_dim: Hidden width.
        swiglu_alpha: Sigmoid gain (1.702).
        swiglu_limit: Clamp bound (7.0).
    """

    def __init__(
        self, embed_dim, mlp_dim, swiglu_alpha=1.702, swiglu_limit=7.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit
        self.gate = layers.Dense(mlp_dim, use_bias=False, name="gate")
        self.up = layers.Dense(mlp_dim, use_bias=False, name="up")
        self.down = layers.Dense(embed_dim, use_bias=False, name="down")

    def call(self, x):
        return self.down(
            swiglu_oai(self.gate(x), self.up(x), self.swiglu_alpha, self.swiglu_limit)
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "swiglu_alpha": self.swiglu_alpha,
                "swiglu_limit": self.swiglu_limit,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MiniMaxM3VLExperts(layers.Layer):
    """Dense bank of SwiGLU-OAI experts evaluated via einsum.

    HF layout: fused ``gate_up_proj`` ``(E, 2I, H)`` (gate then up, not
    interleaved) and ``down_proj`` ``(E, H, I)``.

    Args:
        num_experts / embed_dim / mlp_dim: Bank shape.
        swiglu_alpha / swiglu_limit: Activation constants.
    """

    def __init__(
        self,
        num_experts,
        embed_dim,
        mlp_dim,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.swiglu_alpha = swiglu_alpha
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
        act = swiglu_oai(gate, up, self.swiglu_alpha, self.swiglu_limit)
        expert_out = ops.einsum("tei,ehi->teh", act, self.down_proj)
        return ops.einsum("te,teh->th", routing_weights, expert_out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "swiglu_alpha": self.swiglu_alpha,
                "swiglu_limit": self.swiglu_limit,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MiniMaxM3VLMoE(layers.Layer):
    """M3 sparse MoE: sigmoid+bias top-k routing, routed scaling, shared expert.

    Routing matches M2 (sigmoid scores; selection adds the learned
    ``e_score_correction_bias``; gathered weights stay unbiased and are
    renormalized). The routed output is multiplied by
    ``routed_scaling_factor`` and a SwiGLU-OAI shared expert is added.

    Args:
        num_experts / num_experts_per_tok: Routing shape (128 / 4).
        embed_dim / mlp_dim: Expert dims.
        shared_mlp_dim: Shared-expert hidden width.
        routed_scaling_factor: Routed-output multiplier (2.0).
        swiglu_alpha / swiglu_limit: Activation constants.
    """

    def __init__(
        self,
        num_experts,
        num_experts_per_tok,
        embed_dim,
        mlp_dim,
        shared_mlp_dim,
        routed_scaling_factor=2.0,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.shared_mlp_dim = shared_mlp_dim
        self.routed_scaling_factor = routed_scaling_factor
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit
        self.experts = MiniMaxM3VLExperts(
            num_experts, embed_dim, mlp_dim, swiglu_alpha, swiglu_limit, name="experts"
        )
        self.shared_experts = MiniMaxM3VLDenseMLP(
            embed_dim, shared_mlp_dim, swiglu_alpha, swiglu_limit, name="shared_experts"
        )

    def build(self, input_shape):
        self.gate_weight = self.add_weight(
            name="gate_weight",
            shape=(self.num_experts, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.e_score_correction_bias = self.add_weight(
            name="e_score_correction_bias",
            shape=(self.num_experts,),
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def call(self, hidden_states):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        x = ops.reshape(hidden_states, (-1, self.embed_dim))
        shared_out = self.shared_experts(x)
        logits = ops.matmul(x, ops.transpose(self.gate_weight))
        scores = ops.sigmoid(ops.cast(logits, "float32"))
        biased = scores + ops.cast(self.e_score_correction_bias, "float32")[None]
        _, top_idx = ops.top_k(biased, self.num_experts_per_tok)
        top_vals = ops.take_along_axis(scores, top_idx, axis=-1)
        top_vals = top_vals / ops.sum(top_vals, axis=-1, keepdims=True)
        one_hot = ops.one_hot(top_idx, self.num_experts)
        routing = ops.cast(ops.sum(one_hot * top_vals[..., None], axis=1), x.dtype)
        routed = self.experts(x, routing) * self.routed_scaling_factor
        out = routed + shared_out
        return ops.reshape(out, (b, s, self.embed_dim))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "shared_mlp_dim": self.shared_mlp_dim,
                "routed_scaling_factor": self.routed_scaling_factor,
                "swiglu_alpha": self.swiglu_alpha,
                "swiglu_limit": self.swiglu_limit,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MiniMaxM3VLAttention(layers.Layer):
    """M3 grouped-query attention: per-head Gemma QK-norm + partial RoPE.

    ``query_norm`` / ``key_norm`` are per-head ``(head_dim,)`` Gemma RMSNorms
    applied before the head transpose; rotary embeddings rotate the first
    ``cos.shape[-1]`` channels (rotary_dim 64 of head_dim 128). Layers marked
    sparse carry a :class:`MiniMaxM3VLIndexer` whose block mask replaces the
    plain causal mask.

    Args:
        embed_dim / num_heads / num_kv_heads / head_dim: Geometry.
        use_indexer: Whether this layer is a sparse-attention layer.
        index_n_heads / index_head_dim / index_block_size /
        index_topk_blocks / index_local_blocks: Indexer geometry.
        norm_eps: Norm epsilon.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        use_indexer=False,
        index_n_heads=4,
        index_head_dim=128,
        index_block_size=128,
        index_topk_blocks=16,
        index_local_blocks=1,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_indexer = use_indexer
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_block_size = index_block_size
        self.index_topk_blocks = index_topk_blocks
        self.index_local_blocks = index_local_blocks
        self.norm_eps = norm_eps
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim**-0.5
        self.query = layers.Dense(num_heads * head_dim, use_bias=False, name="query")
        self.key = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="key")
        self.value = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="value")
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")
        self.query_norm = MiniMaxM3VLRMSNorm(eps=norm_eps, name="query_norm")
        self.key_norm = MiniMaxM3VLRMSNorm(eps=norm_eps, name="key_norm")
        if use_indexer:
            self.index_query = layers.Dense(
                index_n_heads * index_head_dim, use_bias=False, name="index_query"
            )
            self.index_key = layers.Dense(
                index_head_dim, use_bias=False, name="index_key"
            )
            self.index_query_norm = MiniMaxM3VLRMSNorm(
                eps=norm_eps, name="index_query_norm"
            )
            self.index_key_norm = MiniMaxM3VLRMSNorm(
                eps=norm_eps, name="index_key_norm"
            )

    def project_index_q(self, hidden_states, cos, sin):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        q = ops.reshape(
            self.index_query(hidden_states),
            (b, s, self.index_n_heads, self.index_head_dim),
        )
        q = ops.transpose(self.index_query_norm(q), (0, 2, 1, 3))
        return apply_partial_rope(q, ops.expand_dims(cos, 1), ops.expand_dims(sin, 1))

    def project_index_k(self, hidden_states, cos, sin):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        k = ops.reshape(self.index_key(hidden_states), (b, s, 1, self.index_head_dim))
        k = ops.transpose(self.index_key_norm(k), (0, 2, 1, 3))
        return apply_partial_rope(k, ops.expand_dims(cos, 1), ops.expand_dims(sin, 1))

    def block_keep_mask(self, idx_q, idx_k, position_ids, key_positions):
        # Per-(query, key) keep verdict of the Lightning-Indexer block top-k:
        # max-pool per-key scores into blocks, always boost the local blocks,
        # keep the top-k blocks per query.
        b = ops.shape(idx_q)[0]
        q_len = ops.shape(idx_q)[2]
        k_len = int(idx_k.shape[2])
        num_blocks = -(-k_len // self.index_block_size)
        pad = num_blocks * self.index_block_size - k_len

        scores = ops.matmul(
            ops.cast(idx_q, "float32"),
            ops.transpose(ops.cast(idx_k, "float32"), (0, 1, 3, 2)),
        )
        future = (
            ops.cast(key_positions, "int32")[None, None, None, :]
            > ops.cast(position_ids, "int32")[:, None, :, None]
        )
        scores = ops.where(future, float("-inf"), scores)
        if pad:
            scores = ops.pad(
                scores,
                ((0, 0), (0, 0), (0, 0), (0, pad)),
                constant_values=float("-inf"),
            )
        scores = ops.reshape(
            scores, (b, self.index_n_heads, q_len, num_blocks, self.index_block_size)
        )
        block_scores = ops.max(ops.max(scores, axis=-1), axis=1)

        q_block = ops.cast(position_ids, "int32") // self.index_block_size
        if self.index_local_blocks > 0:
            local = ops.arange(self.index_local_blocks, dtype="int32")
            local_idx = ops.maximum(q_block[..., None] - local[None, None, :], 0)
            local_hot = ops.max(
                ops.one_hot(local_idx, num_blocks, dtype="float32"), axis=2
            )
            block_scores = ops.where(local_hot > 0, float("inf"), block_scores)

        topk = min(self.index_topk_blocks, num_blocks)
        top_scores, top_idx = ops.top_k(block_scores, topk)
        valid = ops.cast(top_scores > float("-inf"), "float32")
        keep_blocks = ops.max(
            ops.one_hot(top_idx, num_blocks, dtype="float32") * valid[..., None],
            axis=2,
        )
        keep = ops.repeat(keep_blocks, self.index_block_size, axis=-1)[..., :k_len]
        return keep[:, None, :, :] > 0

    def build_index_mask(self, hidden_states, cos, sin, position_ids, base_mask):
        idx_q = self.project_index_q(hidden_states, cos, sin)
        idx_k = self.project_index_k(hidden_states, cos, sin)
        seq = int(idx_k.shape[2])
        keep = self.block_keep_mask(idx_q, idx_k, position_ids, ops.arange(seq))
        keep = ops.logical_and(keep, base_mask > MASK_NEG / 2)
        return ops.where(keep, 0.0, MASK_NEG)

    def project_qkv(self, hidden_states):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        q = ops.reshape(
            self.query(hidden_states), (b, s, self.num_heads, self.head_dim)
        )
        q = ops.transpose(self.query_norm(q), (0, 2, 1, 3))
        k = ops.reshape(
            self.key(hidden_states), (b, s, self.num_kv_heads, self.head_dim)
        )
        k = ops.transpose(self.key_norm(k), (0, 2, 1, 3))
        v = ops.transpose(
            ops.reshape(
                self.value(hidden_states), (b, s, self.num_kv_heads, self.head_dim)
            ),
            (0, 2, 1, 3),
        )
        return q, k, v

    def call(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
    ):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        q, k, v = self.project_qkv(hidden_states)
        cos_e = ops.expand_dims(cos, axis=1)
        sin_e = ops.expand_dims(sin, axis=1)
        q = apply_partial_rope(q, cos_e, sin_e)
        k = apply_partial_rope(k, cos_e, sin_e)
        if self.use_indexer:
            attention_mask = self.build_index_mask(
                hidden_states, cos, sin, position_ids, attention_mask
            )
        new_kv = None
        if use_cache:
            if self.use_indexer:
                new_kv = (k, v, self.project_index_k(hidden_states, cos, sin))
            else:
                new_kv = (k, v)
        kk, vv = k, v
        if self.num_kv_groups > 1:
            kk = ops.repeat(kk, self.num_kv_groups, axis=1)
            vv = ops.repeat(vv, self.num_kv_groups, axis=1)
        attn = ops.matmul(q, ops.transpose(kk, (0, 1, 3, 2))) * self.scaling
        if attention_mask is not None:
            attn = attn + ops.cast(attention_mask, attn.dtype)
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), q.dtype)
        out = ops.matmul(attn, vv)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, s, self.num_heads * self.head_dim)
        )
        out = self.output_proj(out)
        return (out, new_kv) if use_cache else out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "use_indexer": self.use_indexer,
                "index_n_heads": self.index_n_heads,
                "index_head_dim": self.index_head_dim,
                "index_block_size": self.index_block_size,
                "index_topk_blocks": self.index_topk_blocks,
                "index_local_blocks": self.index_local_blocks,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MiniMaxM3VLDecoderLayer(layers.Layer):
    """One M3 block: pre-norm attention (dense or block-sparse), pre-norm MLP
    (dense SwiGLU-OAI or sigmoid-routed MoE with a shared expert).

    Args:
        embed_dim / num_heads / num_kv_heads / head_dim: Attention geometry.
        mlp_type: ``"sparse"`` (MoE) or ``"dense"``.
        mlp_dim: Expert hidden width (sparse), see ``dense_mlp_dim``.
        dense_mlp_dim: Dense-MLP hidden width.
        shared_mlp_dim: Shared-expert hidden width.
        num_experts / num_experts_per_tok / routed_scaling_factor: MoE shape.
        use_indexer + index_*: Sparse-attention geometry.
        swiglu_alpha / swiglu_limit: Activation constants.
        norm_eps: RMSNorm epsilon.
    """

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        dense_mlp_dim,
        shared_mlp_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        num_experts,
        num_experts_per_tok,
        mlp_type="sparse",
        routed_scaling_factor=2.0,
        use_indexer=False,
        index_n_heads=4,
        index_head_dim=128,
        index_block_size=128,
        index_topk_blocks=16,
        index_local_blocks=1,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.dense_mlp_dim = dense_mlp_dim
        self.shared_mlp_dim = shared_mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.mlp_type = mlp_type
        self.routed_scaling_factor = routed_scaling_factor
        self.use_indexer = use_indexer
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_block_size = index_block_size
        self.index_topk_blocks = index_topk_blocks
        self.index_local_blocks = index_local_blocks
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit
        self.norm_eps = norm_eps

        self.attention_norm = MiniMaxM3VLRMSNorm(eps=norm_eps, name="attention_norm")
        self.attention = MiniMaxM3VLAttention(
            embed_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            use_indexer,
            index_n_heads,
            index_head_dim,
            index_block_size,
            index_topk_blocks,
            index_local_blocks,
            norm_eps,
            name="attention",
        )
        self.mlp_norm = MiniMaxM3VLRMSNorm(eps=norm_eps, name="mlp_norm")
        if mlp_type == "sparse":
            self.mlp = MiniMaxM3VLMoE(
                num_experts,
                num_experts_per_tok,
                embed_dim,
                mlp_dim,
                shared_mlp_dim,
                routed_scaling_factor,
                swiglu_alpha,
                swiglu_limit,
                name="mlp",
            )
        else:
            self.mlp = MiniMaxM3VLDenseMLP(
                embed_dim, dense_mlp_dim, swiglu_alpha, swiglu_limit, name="mlp"
            )

    def call(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
    ):
        residual = hidden_states
        attn_out = self.attention(
            self.attention_norm(hidden_states),
            cos,
            sin,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )
        new_kv = None
        if use_cache:
            attn_out, new_kv = attn_out
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = residual + self.mlp(self.mlp_norm(hidden_states))
        return (hidden_states, new_kv) if use_cache else hidden_states

    def compute_output_spec(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
    ):
        # The residual stream keeps its shape; an explicit spec stops the
        # functional builder from tracing the sparse-indexer / dynamic-reshape
        # ops in attention (they resolve only at run time).
        return keras.KerasTensor(hidden_states.shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "dense_mlp_dim": self.dense_mlp_dim,
                "shared_mlp_dim": self.shared_mlp_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "mlp_type": self.mlp_type,
                "routed_scaling_factor": self.routed_scaling_factor,
                "use_indexer": self.use_indexer,
                "index_n_heads": self.index_n_heads,
                "index_head_dim": self.index_head_dim,
                "index_block_size": self.index_block_size,
                "index_topk_blocks": self.index_topk_blocks,
                "index_local_blocks": self.index_local_blocks,
                "swiglu_alpha": self.swiglu_alpha,
                "swiglu_limit": self.swiglu_limit,
                "norm_eps": self.norm_eps,
            }
        )
        return config


def vision_rope_3d(grid_thw, head_dim, theta, spatial_merge_size):
    """3D rotary tables for the vision tower (host-side grid iteration).

    ``2 * (head_dim // 2)`` rotary dims split evenly across T/H/W (each axis
    rounded down to a multiple of 2); coordinates follow the merge-window
    patch order. Returns ``(cos, sin)`` of shape ``(num_patches, 3 * axis_dim)``.
    """
    rope_dims = 2 * (head_dim // 2)
    axis_dim = 2 * ((rope_dims // 3) // 2)
    m = spatial_merge_size
    coords = []
    for t, h, w in grid_thw:
        hi = ops.tile(ops.reshape(ops.arange(h), (h, 1)), (1, w))
        hi = ops.reshape(
            ops.transpose(ops.reshape(hi, (h // m, m, w // m, m)), (0, 2, 1, 3)), (-1,)
        )
        wi = ops.tile(ops.reshape(ops.arange(w), (1, w)), (h, 1))
        wi = ops.reshape(
            ops.transpose(ops.reshape(wi, (h // m, m, w // m, m)), (0, 2, 1, 3)), (-1,)
        )
        ti = ops.repeat(ops.arange(t), h * w)
        coords.append(ops.stack([ti, ops.tile(hi, (t,)), ops.tile(wi, (t,))], axis=-1))
    coords = ops.cast(ops.concatenate(coords, axis=0), "float32")  # (N, 3)
    exponent = ops.cast(ops.arange(0, axis_dim, 2), "float32") / axis_dim
    inv_freq = ops.expand_dims(
        1.0 / ops.power(ops.convert_to_tensor(theta, "float32"), exponent), 0
    )
    freqs = ops.concatenate(
        [coords[:, i : i + 1] * inv_freq for i in range(3)], axis=-1
    )
    emb = ops.concatenate([freqs, freqs], axis=-1)
    return ops.cos(emb), ops.sin(emb)


@keras.saving.register_keras_serializable(package="zeromodels")
class MiniMaxM3VLVisionAttention(layers.Layer):
    """CLIP-style biased full attention with 3D rotary embeddings.

    The packed patch sequence (all images concatenated) attends globally;
    queries and keys are rotated by the per-patch ``(T, H, W)`` rope before
    the head transpose.

    Args:
        embed_dim: Vision width.
        num_heads: Attention heads.
    """

    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.query = layers.Dense(embed_dim, name="query")
        self.key = layers.Dense(embed_dim, name="key")
        self.value = layers.Dense(embed_dim, name="value")
        self.output_proj = layers.Dense(embed_dim, name="output_proj")

    def build(self, input_shape):
        self.query.build(input_shape)
        self.key.build(input_shape)
        self.value.build(input_shape)
        self.output_proj.build(input_shape)
        self.built = True

    def call(self, hidden_states, cos, sin):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        q = ops.reshape(
            self.query(hidden_states), (b, s, self.num_heads, self.head_dim)
        )
        k = ops.reshape(self.key(hidden_states), (b, s, self.num_heads, self.head_dim))
        v = ops.reshape(
            self.value(hidden_states), (b, s, self.num_heads, self.head_dim)
        )
        cos_e = ops.cast(cos, q.dtype)[None, :, None, :]
        sin_e = ops.cast(sin, q.dtype)[None, :, None, :]
        q = apply_partial_rope(q, cos_e, sin_e)
        k = apply_partial_rope(k, cos_e, sin_e)
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scaling
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), q.dtype)
        out = ops.matmul(attn, v)
        out = ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (b, s, self.embed_dim))
        return self.output_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MiniMaxM3VLVisionLayer(layers.Layer):
    """One CLIP-style vision block: pre-LN attention + pre-LN exact-gelu MLP.

    Args:
        embed_dim / mlp_dim / num_heads: Block dims.
        norm_eps: LayerNorm epsilon (1e-5).
    """

    def __init__(self, embed_dim, mlp_dim, num_heads, norm_eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.norm_eps = norm_eps
        self.layer_norm1 = layers.LayerNormalization(
            epsilon=norm_eps, name="layer_norm1"
        )
        self.attention = MiniMaxM3VLVisionAttention(
            embed_dim, num_heads, name="attention"
        )
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=norm_eps, name="layer_norm2"
        )
        self.fc1 = layers.Dense(mlp_dim, name="fc1")
        self.fc2 = layers.Dense(embed_dim, name="fc2")

    def build(self, input_shape):
        self.layer_norm1.build(input_shape)
        self.attention.build(input_shape)
        self.layer_norm2.build(input_shape)
        self.fc1.build(input_shape)
        self.fc2.build(tuple(input_shape[:-1]) + (self.mlp_dim,))
        self.built = True

    def call(self, hidden_states, cos, sin):
        hidden_states = hidden_states + self.attention(
            self.layer_norm1(hidden_states), cos, sin
        )
        x = self.layer_norm2(hidden_states)
        x = self.fc2(ops.gelu(self.fc1(x), approximate=False))
        return hidden_states + x

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
