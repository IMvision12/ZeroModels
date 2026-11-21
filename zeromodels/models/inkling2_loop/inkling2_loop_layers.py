import math

import keras
from keras import layers, ops

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2LoopRMSNorm(layers.Layer):
    """RMSNorm computing variance in float32 with a learned scale ``weight``."""

    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=(self.hidden_size,),
            initializer="ones",
            trainable=True,
        )
        self.built = True

    def call(self, hidden_states):
        input_dtype = hidden_states.dtype
        x = ops.cast(hidden_states, "float32")
        var = ops.mean(ops.square(x), axis=-1, keepdims=True)
        x = x * ops.rsqrt(var + self.eps)
        w = ops.cast(self.weight, "float32")
        return ops.cast(w * x, input_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size, "eps": self.eps})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2LoopUnweightedRMSNorm(layers.Layer):
    """Parameter-free RMSNorm (value norm, AttnRes, MessageGate)."""

    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, hidden_states):
        input_dtype = hidden_states.dtype
        x = ops.cast(hidden_states, "float32")
        x = x * ops.rsqrt(ops.mean(ops.square(x), axis=-1, keepdims=True) + self.eps)
        return ops.cast(x, input_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2LoopSoftcap(layers.Layer):
    """Applies ``tanh(x / softcap) * softcap``."""

    def __init__(self, softcap, **kwargs):
        super().__init__(**kwargs)
        self.softcap = float(softcap)

    def call(self, hidden_states):
        return ops.tanh(hidden_states / self.softcap) * self.softcap

    def get_config(self):
        config = super().get_config()
        config.update({"softcap": self.softcap})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2LoopSoftcappedSwiGLU(layers.Layer):
    """SwiGLU where sigmoid sees the raw gate and softcaps bound gate_lin and up."""

    def __init__(self, mlp_gate_softcap=4.0, mlp_up_softcap=25.0, **kwargs):
        super().__init__(**kwargs)
        self.mlp_gate_softcap = float(mlp_gate_softcap)
        self.mlp_up_softcap = float(mlp_up_softcap)

    def call(self, gate, up):
        input_dtype = gate.dtype
        g = ops.cast(gate, "float32")
        u = ops.cast(up, "float32")
        g_cap = ops.tanh(g / self.mlp_gate_softcap) * self.mlp_gate_softcap
        u_cap = ops.tanh(u / self.mlp_up_softcap) * self.mlp_up_softcap
        return ops.cast(g_cap * ops.sigmoid(g) * u_cap, input_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mlp_gate_softcap": self.mlp_gate_softcap,
                "mlp_up_softcap": self.mlp_up_softcap,
            }
        )
        return config


def l2_clip_fn(hidden_states, threshold):
    input_dtype = hidden_states.dtype
    x = ops.cast(hidden_states, "float32")
    norm = ops.sqrt(ops.sum(ops.square(x), axis=-1, keepdims=True))
    factor = ops.minimum(threshold / (norm + 1e-6), 1.0)
    return ops.cast(x * factor, input_dtype)


def log_scale_fn(hidden_states, position_ids, alpha, n_floor):
    input_dtype = hidden_states.dtype
    pos = ops.cast(position_ids + 1, "float32")
    scale = 1.0 + alpha * ops.log(ops.maximum(pos / float(n_floor), 1.0))
    return ops.cast(ops.cast(hidden_states, "float32") * scale[..., None, None], input_dtype)


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2LoopScaledL2Clip(layers.Layer):
    """Learned per-(head, dim) scale followed by L2 norm clipping and optional log-scale."""

    def __init__(
        self,
        num_heads,
        head_dim,
        threshold,
        log_scaling=False,
        log_scaling_alpha=0.1,
        log_scaling_n_floor=8192,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.threshold = float(threshold)
        self.log_scaling = bool(log_scaling)
        self.log_scaling_alpha = float(log_scaling_alpha)
        self.log_scaling_n_floor = int(log_scaling_n_floor)

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(self.num_heads, self.head_dim),
            initializer="ones",
            trainable=True,
        )
        self.built = True

    def call(self, hidden_states, position_ids=None):
        input_dtype = hidden_states.dtype
        x = ops.cast(hidden_states, "float32") * ops.cast(self.scale, "float32")
        x = ops.cast(l2_clip_fn(x, self.threshold), input_dtype)
        if self.log_scaling:
            x = log_scale_fn(
                x, position_ids, self.log_scaling_alpha, self.log_scaling_n_floor
            )
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "threshold": self.threshold,
                "log_scaling": self.log_scaling,
                "log_scaling_alpha": self.log_scaling_alpha,
                "log_scaling_n_floor": self.log_scaling_n_floor,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2LoopRelativePositionBias(layers.Layer):
    """Hidden-states conditioned relative position bias."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        d_rel=16,
        rel_extent=1024,
        rel_l2_clip_threshold=8.0,
        log_scaling=False,
        log_scaling_alpha=0.1,
        log_scaling_n_floor=8192,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.rel_l2_clip_threshold = float(rel_l2_clip_threshold)
        self.log_scaling = bool(log_scaling)
        self.log_scaling_alpha = float(log_scaling_alpha)
        self.log_scaling_n_floor = int(log_scaling_n_floor)

        self.linear = layers.Dense(num_heads * d_rel, use_bias=False, name="linear")
        self.norm = Inkling2LoopScaledL2Clip(
            num_heads=num_heads,
            head_dim=d_rel,
            threshold=rel_l2_clip_threshold,
            log_scaling=log_scaling,
            log_scaling_alpha=log_scaling_alpha,
            log_scaling_n_floor=log_scaling_n_floor,
            name="norm",
        )

    def build(self, input_shape):
        self.linear.build(input_shape)
        self.norm.build((None, None, self.num_heads, self.d_rel))
        self.proj = self.add_weight(
            name="proj",
            shape=(self.d_rel, self.rel_extent),
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def call(self, hidden_states, relative_distance, position_ids):
        b = ops.shape(hidden_states)[0]
        q_len = ops.shape(hidden_states)[1]
        kv_len = ops.shape(relative_distance)[1]
        rel = ops.reshape(self.linear(hidden_states), (b, q_len, self.num_heads, self.d_rel))
        rel = self.norm(rel, position_ids)
        rel_logits = ops.matmul(rel, ops.cast(self.proj, rel.dtype))
        rel_logits = ops.transpose(rel_logits, (0, 2, 1, 3))  # (B, H, Q, rel_extent)

        clamped = ops.clip(relative_distance, 0, self.rel_extent - 1)
        idx = ops.broadcast_to(
            clamped[None, None, :, :], (b, self.num_heads, q_len, kv_len)
        )
        bias = ops.take_along_axis(rel_logits, idx, axis=-1)
        valid = ops.logical_and(
            relative_distance >= 0, relative_distance < self.rel_extent
        )
        return ops.where(valid[None, None, :, :], bias, ops.cast(0.0, bias.dtype))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "d_rel": self.d_rel,
                "rel_extent": self.rel_extent,
                "rel_l2_clip_threshold": self.rel_l2_clip_threshold,
                "log_scaling": self.log_scaling,
                "log_scaling_alpha": self.log_scaling_alpha,
                "log_scaling_n_floor": self.log_scaling_n_floor,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2LoopShortConvolution(layers.Layer):
    """Static depthwise causal 1D short convolution with residual and ``1/sqrt(K+1)`` scaling.

    Weight layout ``conv_weight`` is ``(hidden_size, K)`` in chronological tap order
    (``conv_weight[:, K-1]`` weights the current token, matching PyTorch ``nn.Conv1d``).
    Cache state ``conv_state`` has fixed shape ``(B, hidden_size, K - 1)`` holding
    the last ``K - 1`` chronological masked inputs.
    """

    def __init__(self, hidden_size, conv_kernel_size=4, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.conv_kernel_size = conv_kernel_size
        self.out_scale = 1.0 / math.sqrt(conv_kernel_size + 1)

    def build(self, input_shape):
        self.conv_weight = self.add_weight(
            name="conv_weight",
            shape=(self.hidden_size, self.conv_kernel_size),
            initializer="zeros",
            dtype="float32",
            trainable=True,
        )
        self.built = True

    def call(self, hidden_states, conv_mask=None, conv_state=None, return_state=False):
        input_dtype = hidden_states.dtype
        x = ops.cast(hidden_states, "float32")
        residual = x
        if conv_mask is not None:
            x = x * ops.cast(conv_mask, "float32")[:, :, None]

        # (B, D, S)
        x_ch = ops.transpose(x, (0, 2, 1))
        k = self.conv_kernel_size
        w = ops.cast(self.conv_weight, "float32")  # (D, K)

        if conv_state is not None:
            # Decode / continuation path: prepend (B, D, K-1)
            combined = ops.concatenate([ops.cast(conv_state, "float32"), x_ch], axis=-1)
            new_state = combined[:, :, -(k - 1) :]
            # Compute causal conv over last S positions
            seq_len = ops.shape(x_ch)[2]
            if seq_len == 1:
                # Fast decode path: combined is (B, D, K)
                conv_out = ops.sum(combined * w[None, :, :], axis=-1, keepdims=True)
            else:
                out_steps = []
                for j in range(k):
                    end = seq_len + j
                    out_steps.append(
                        combined[:, :, j:end] * w[None, :, j : j + 1]
                    )
                conv_out = sum(out_steps)
        else:
            # Prefill path: left-pad with K-1 zeros
            b = ops.shape(x_ch)[0]
            pad = ops.zeros((b, self.hidden_size, k - 1), dtype="float32")
            padded = ops.concatenate([pad, x_ch], axis=-1)
            new_state = padded[:, :, -(k - 1) :]
            seq_len = ops.shape(x_ch)[2]
            out_steps = []
            for j in range(k):
                out_steps.append(
                    padded[:, :, j : j + seq_len] * w[None, :, j : j + 1]
                )
            conv_out = sum(out_steps)

        out = (ops.transpose(conv_out, (0, 2, 1)) + residual) * self.out_scale
        out = ops.cast(out, input_dtype)
        if return_state:
            return out, new_state
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "conv_kernel_size": self.conv_kernel_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2LoopDynamicShortConvolution(layers.Layer):
    """Dynamic causal short convolution (used for K/V in full-attention layers).

    ``static_weight`` has shape ``(K, hidden_size)`` where tap index 0 is the current
    token (lag 0) and tap index ``lag`` weights ``lag`` steps back. ``weight_proj``
    projects ``dynamic_input`` to per-head dynamic taps ``(B, S, num_heads, 1, K)``.
    Output is ``(conv + residual) / sqrt(2*K + 1)``.
    """

    def __init__(self, hidden_size, input_size, num_heads, conv_kernel_size=4, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.conv_kernel_size = conv_kernel_size
        self.out_scale = 1.0 / math.sqrt(2 * conv_kernel_size + 1)
        self.weight_proj = layers.Dense(
            num_heads * conv_kernel_size, use_bias=False, name="weight_proj"
        )

    def build(self, input_shape):
        self.static_weight = self.add_weight(
            name="static_weight",
            shape=(self.conv_kernel_size, self.hidden_size),
            initializer="zeros",
            trainable=True,
        )
        self.weight_proj.build((None, None, self.input_size))
        self.built = True

    def call(
        self,
        hidden_states,
        dynamic_input,
        conv_mask=None,
        conv_state=None,
        return_state=False,
    ):
        input_dtype = hidden_states.dtype
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        k = self.conv_kernel_size

        dyn = ops.cast(self.weight_proj(dynamic_input), "float32")
        dyn = ops.reshape(dyn, (b, s, self.num_heads, 1, k))
        static = ops.reshape(
            ops.transpose(ops.cast(self.static_weight, "float32")),
            (1, 1, self.num_heads, self.head_dim, k),
        )
        weights = static + dyn  # (B, S, num_heads, head_dim, K), lag 0 = current

        x = ops.cast(hidden_states, "float32")
        residual = x
        if conv_mask is not None:
            x = x * ops.cast(conv_mask, "float32")[:, :, None]

        x_ch = ops.transpose(x, (0, 2, 1))  # (B, D, S)
        if conv_state is not None:
            padded = ops.concatenate([ops.cast(conv_state, "float32"), x_ch], axis=-1)
        else:
            pad = ops.zeros((b, self.hidden_size, k - 1), dtype="float32")
            padded = ops.concatenate([pad, x_ch], axis=-1)
        new_state = padded[:, :, -(k - 1) :]

        # Build lag-ordered taps: lag 0 is padded[:, :, k-1 : k-1+s], lag j is padded[:, :, k-1-j : k-1-j+s]
        lag_slices = [
            ops.transpose(padded[:, :, (k - 1 - lag) : (k - 1 - lag) + s], (0, 2, 1))
            for lag in range(k)
        ]
        # (B, S, D, K) -> (B, S, num_heads, head_dim, K)
        windows = ops.stack(lag_slices, axis=-1)
        windows = ops.reshape(windows, (b, s, self.num_heads, self.head_dim, k))

        conv_out = ops.reshape(ops.sum(windows * weights, axis=-1), (b, s, self.hidden_size))
        out = ops.cast((conv_out + residual) * self.out_scale, input_dtype)
        if return_state:
            return out, new_state
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "input_size": self.input_size,
                "num_heads": self.num_heads,
                "conv_kernel_size": self.conv_kernel_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2LoopMessageGate(layers.Layer):
    """Low-rank message gate: ``(1 - tanh(forget)) * h + tanh(msg) * m`` in fp32."""

    def __init__(self, embed_dim, gate_rank=64, norm_eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.gate_rank = gate_rank
        self.norm_eps = norm_eps
        self.norm = Inkling2LoopUnweightedRMSNorm(eps=norm_eps, name="norm")
        self.down_proj = layers.Dense(gate_rank, use_bias=False, name="down_proj")
        self.forget_gate_proj = layers.Dense(
            embed_dim, use_bias=False, dtype="float32", name="forget_gate_proj"
        )
        self.message_gate_proj = layers.Dense(
            embed_dim, use_bias=False, dtype="float32", name="message_gate_proj"
        )

    def build(self, input_shape):
        self.down_proj.build((None, None, 2 * self.embed_dim))
        self.forget_gate_proj.build((None, None, self.gate_rank))
        self.message_gate_proj.build((None, None, self.gate_rank))
        self.built = True

    def call(self, hidden_states, message):
        h = ops.cast(hidden_states, "float32")
        m = ops.cast(message, "float32")
        g = ops.concatenate([self.norm(h), self.norm(m)], axis=-1)
        g = ops.cast(self.down_proj(ops.cast(g, self.down_proj.compute_dtype)), "float32")
        forget_gate = ops.tanh(self.forget_gate_proj(g))
        message_gate = ops.tanh(self.message_gate_proj(g))
        return (1.0 - forget_gate) * h + message_gate * m

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "gate_rank": self.gate_rank,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2LoopBlockAttentionResidual(layers.Layer):
    """Attends over accumulated block states + partial block state using learned fp32 queries."""

    def __init__(self, embed_dim, num_queries=1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.norm = Inkling2LoopUnweightedRMSNorm(eps=1e-6, name="norm")
        self.scaling = embed_dim**-0.5

    def build(self, input_shape):
        shape = (self.embed_dim,) if self.num_queries == 1 else (self.num_queries, self.embed_dim)
        self.query = self.add_weight(
            name="query",
            shape=shape,
            initializer="zeros",
            dtype="float32",
            trainable=True,
        )
        self.built = True

    def call(self, partial_block_states, block_states=None):
        partial = ops.cast(partial_block_states, "float32")[:, :, None, :]
        if block_states is None:
            stacked = partial
        else:
            stacked = ops.concatenate([ops.cast(block_states, "float32"), partial], axis=-2)
        stacked = self.norm(stacked)  # (B, S, N, H)
        q = ops.reshape(ops.cast(self.query, "float32"), (self.num_queries, self.embed_dim))
        scores = ops.einsum("qh,bsnh->bsqn", q, stacked) * self.scaling
        weights = ops.softmax(scores, axis=-1)
        out = ops.einsum("bsqn,bsnh->bsqh", weights, stacked)
        return tuple(out[:, :, i, :] for i in range(self.num_queries))

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_queries": self.num_queries})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2LoopMessageAttentionResidual(layers.Layer):
    """Retrieves 4 cross-pass messages (Q, K, V, MLP) from previous loop pass block states."""

    def __init__(self, num_loop_layers, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_loop_layers = num_loop_layers
        self.embed_dim = embed_dim
        self.norm = Inkling2LoopUnweightedRMSNorm(eps=1e-6, name="norm")
        self.scaling = embed_dim**-0.5

    def build(self, input_shape):
        self.query = self.add_weight(
            name="query",
            shape=(self.num_loop_layers, 4, self.embed_dim),
            initializer="zeros",
            dtype="float32",
            trainable=True,
        )
        self.built = True

    def call(self, partial_block_states, block_states=None, loop_layer_idx=0):
        partial = ops.cast(partial_block_states, "float32")[:, :, None, :]
        if block_states is None:
            stacked = partial
        else:
            stacked = ops.concatenate([ops.cast(block_states, "float32"), partial], axis=-2)
        stacked = self.norm(stacked)  # (B, S, N, H)
        q = ops.cast(self.query[loop_layer_idx], "float32")  # (4, H)
        scores = ops.einsum("qh,bsnh->bsqn", q, stacked) * self.scaling
        weights = ops.softmax(scores, axis=-1)
        return ops.einsum("bsqn,bsnh->bsqh", weights, stacked)  # (B, S, 4, H)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"num_loop_layers": self.num_loop_layers, "embed_dim": self.embed_dim}
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2LoopAttention(layers.Layer):
    """Hybrid sliding-window or full attention with sinks, relative bias, sconv, and gated output."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        is_sliding=True,
        sliding_window_size=1024,
        d_rel=16,
        rel_extent=1024,
        conv_kernel_size=4,
        norm_eps=1e-6,
        qk_l2_clip_threshold=22.627416997969522,
        rel_l2_clip_threshold=8.0,
        log_scaling_alpha=0.1,
        log_scaling_n_floor=8192,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.is_sliding = bool(is_sliding)
        self.sliding_window_size = sliding_window_size
        self.d_rel = d_rel
        self.rel_extent = sliding_window_size if is_sliding else rel_extent
        self.conv_kernel_size = conv_kernel_size
        self.norm_eps = norm_eps
        self.qk_l2_clip_threshold = float(qk_l2_clip_threshold)
        self.rel_l2_clip_threshold = float(rel_l2_clip_threshold)
        self.log_scaling_alpha = float(log_scaling_alpha)
        self.log_scaling_n_floor = int(log_scaling_n_floor)
        self.scaling = head_dim**-0.5

        log_scaling = not self.is_sliding
        kv_dim = num_kv_heads * head_dim

        self.q_linear = layers.Dense(num_heads * head_dim, use_bias=False, name="q_linear")
        self.q_norm = Inkling2LoopScaledL2Clip(
            num_heads=num_heads,
            head_dim=head_dim,
            threshold=qk_l2_clip_threshold,
            log_scaling=log_scaling,
            log_scaling_alpha=log_scaling_alpha,
            log_scaling_n_floor=log_scaling_n_floor,
            name="q_norm",
        )
        self.k_linear = layers.Dense(kv_dim, use_bias=False, name="k_linear")
        self.v_linear = layers.Dense(kv_dim, use_bias=False, name="v_linear")
        if self.is_sliding:
            self.k_sconv = Inkling2LoopShortConvolution(
                kv_dim, conv_kernel_size, name="k_sconv"
            )
            self.v_sconv = Inkling2LoopShortConvolution(
                kv_dim, conv_kernel_size, name="v_sconv"
            )
        else:
            self.k_sconv = Inkling2LoopDynamicShortConvolution(
                kv_dim, embed_dim, num_kv_heads, conv_kernel_size, name="k_sconv"
            )
            self.v_sconv = Inkling2LoopDynamicShortConvolution(
                kv_dim, embed_dim, num_kv_heads, conv_kernel_size, name="v_sconv"
            )
        self.v_norm = Inkling2LoopUnweightedRMSNorm(eps=norm_eps, name="v_norm")
        self.gate_proj = layers.Dense(num_heads * head_dim, use_bias=False, name="gate_proj")
        self.o_linear = layers.Dense(embed_dim, use_bias=False, name="o_linear")
        self.sink_proj = layers.Dense(num_heads, use_bias=False, name="sink_proj")
        self.relative_position_bias = Inkling2LoopRelativePositionBias(
            embed_dim=embed_dim,
            num_heads=num_heads,
            d_rel=d_rel,
            rel_extent=self.rel_extent,
            rel_l2_clip_threshold=rel_l2_clip_threshold,
            log_scaling=log_scaling,
            log_scaling_alpha=log_scaling_alpha,
            log_scaling_n_floor=log_scaling_n_floor,
            name="relative_position_bias",
        )

    def build(self, input_shape):
        shape = (None, None, self.embed_dim)
        kv_shape = (None, None, self.num_kv_heads * self.head_dim)
        self.q_linear.build(shape)
        self.q_norm.build((None, None, self.num_heads, self.head_dim))
        self.k_linear.build(shape)
        self.v_linear.build(shape)
        self.k_sconv.build(kv_shape)
        self.v_sconv.build(kv_shape)
        self.gate_proj.build(shape)
        self.o_linear.build((None, None, self.num_heads * self.head_dim))
        self.sink_proj.build(shape)
        self.relative_position_bias.build(shape)
        self.built = True

    def _project_qkv(
        self,
        hidden_states,
        key_hidden_states,
        value_hidden_states,
        position_ids,
        conv_mask=None,
        k_conv_state=None,
        v_conv_state=None,
        return_conv_states=False,
    ):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]

        q = ops.reshape(self.q_linear(hidden_states), (b, s, self.num_heads, self.head_dim))
        q = ops.transpose(self.q_norm(q, position_ids), (0, 2, 1, 3))  # (B, H, S, D)

        k_raw = self.k_linear(key_hidden_states)
        v_raw = self.v_linear(value_hidden_states)
        if self.is_sliding:
            k_res = self.k_sconv(
                k_raw, conv_mask=conv_mask, conv_state=k_conv_state, return_state=return_conv_states
            )
            v_res = self.v_sconv(
                v_raw, conv_mask=conv_mask, conv_state=v_conv_state, return_state=return_conv_states
            )
        else:
            k_res = self.k_sconv(
                k_raw,
                key_hidden_states,
                conv_mask=conv_mask,
                conv_state=k_conv_state,
                return_state=return_conv_states,
            )
            v_res = self.v_sconv(
                v_raw,
                value_hidden_states,
                conv_mask=conv_mask,
                conv_state=v_conv_state,
                return_state=return_conv_states,
            )
        if return_conv_states:
            k_raw, new_k_conv = k_res
            v_raw, new_v_conv = v_res
        else:
            k_raw, v_raw = k_res, v_res
            new_k_conv, new_v_conv = None, None

        k = ops.reshape(k_raw, (b, s, self.num_kv_heads, self.head_dim))
        k = ops.transpose(l2_clip_fn(k, self.qk_l2_clip_threshold), (0, 2, 1, 3))
        v = ops.reshape(v_raw, (b, s, self.num_kv_heads, self.head_dim))
        v = ops.transpose(self.v_norm(v), (0, 2, 1, 3))
        return q, k, v, new_k_conv, new_v_conv

    def _attend(self, q, k, v, hidden_states, attention_mask, relative_distance, position_ids):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]

        if self.num_kv_groups > 1:
            k = ops.repeat(k, self.num_kv_groups, axis=1)
            v = ops.repeat(v, self.num_kv_groups, axis=1)

        attn_weights = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scaling
        pos_bias = self.relative_position_bias(hidden_states, relative_distance, position_ids)
        attn_weights = attn_weights + ops.cast(pos_bias, attn_weights.dtype)
        if attention_mask is not None:
            attn_weights = attn_weights + ops.cast(attention_mask, attn_weights.dtype)

        sinks = ops.transpose(self.sink_proj(hidden_states), (0, 2, 1))[..., None]
        combined = ops.cast(
            ops.concatenate([attn_weights, ops.cast(sinks, attn_weights.dtype)], axis=-1),
            "float32",
        )
        attn_probs = ops.cast(ops.softmax(combined, axis=-1)[..., :-1], q.dtype)
        attn_out = ops.matmul(attn_probs, v)
        attn_out = ops.reshape(
            ops.transpose(attn_out, (0, 2, 1, 3)), (b, s, self.num_heads * self.head_dim)
        )
        gate = ops.sigmoid(ops.cast(self.gate_proj(hidden_states), "float32"))
        attn_out = ops.cast(ops.cast(attn_out, "float32") * gate, attn_out.dtype)
        return self.o_linear(attn_out)

    def call(
        self,
        hidden_states,
        attention_mask,
        relative_distance,
        position_ids,
        conv_mask=None,
        key_hidden_states=None,
        value_hidden_states=None,
        use_cache=False,
    ):
        k_in = hidden_states if key_hidden_states is None else key_hidden_states
        v_in = hidden_states if value_hidden_states is None else value_hidden_states
        q, k, v, k_conv, v_conv = self._project_qkv(
            hidden_states,
            k_in,
            v_in,
            position_ids,
            conv_mask=conv_mask,
            return_conv_states=use_cache,
        )
        out = self._attend(q, k, v, hidden_states, attention_mask, relative_distance, position_ids)
        if use_cache:
            return out, (k, v, k_conv, v_conv)
        return out

    def decode_step(
        self,
        hidden_states,
        key_hidden_states,
        value_hidden_states,
        kv_conv_state,
        cache_update_index,
        attention_mask,
        relative_distance,
        position_ids,
    ):
        k_cache, v_cache, k_conv_state, v_conv_state = kv_conv_state
        q, k_new, v_new, new_k_conv, new_v_conv = self._project_qkv(
            hidden_states,
            key_hidden_states,
            value_hidden_states,
            position_ids,
            conv_mask=None,
            k_conv_state=k_conv_state,
            v_conv_state=v_conv_state,
            return_conv_states=True,
        )
        k_cache = ops.slice_update(
            k_cache, (0, 0, cache_update_index, 0), ops.cast(k_new, k_cache.dtype)
        )
        v_cache = ops.slice_update(
            v_cache, (0, 0, cache_update_index, 0), ops.cast(v_new, v_cache.dtype)
        )
        out = self._attend(
            q,
            k_cache,
            v_cache,
            hidden_states,
            attention_mask,
            relative_distance,
            position_ids,
        )
        return out, (k_cache, v_cache, new_k_conv, new_v_conv)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "is_sliding": self.is_sliding,
                "sliding_window_size": self.sliding_window_size,
                "d_rel": self.d_rel,
                "rel_extent": self.rel_extent,
                "conv_kernel_size": self.conv_kernel_size,
                "norm_eps": self.norm_eps,
                "qk_l2_clip_threshold": self.qk_l2_clip_threshold,
                "rel_l2_clip_threshold": self.rel_l2_clip_threshold,
                "log_scaling_alpha": self.log_scaling_alpha,
                "log_scaling_n_floor": self.log_scaling_n_floor,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2LoopExperts(layers.Layer):
    """MoE expert bank with softcapped SwiGLU and adaptive sparse/dense dispatch.

    - ``gate_up_proj``: ``(E, 2I, H)`` (gate stacked over up)
    - ``down_proj``: ``(E, H, I)``
    - For small token counts (``T <= 8``, e.g. decode steps), gathers only the
      ``top_k`` experts via ``ops.take`` (~16x less HBM bandwidth on 128-expert decode).
    - For larger token counts (prefill), uses dense einsum over the expert bank to
      avoid materializing ``(T, k, 2I, H)`` weight slices.
    """

    def __init__(
        self,
        num_experts,
        embed_dim,
        mlp_dim,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.mlp_gate_softcap = float(mlp_gate_softcap)
        self.mlp_up_softcap = float(mlp_up_softcap)
        self.act_fn = Inkling2LoopSoftcappedSwiGLU(
            mlp_gate_softcap=mlp_gate_softcap,
            mlp_up_softcap=mlp_up_softcap,
            name="act_fn",
        )

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

    def call(
        self,
        hidden_states,
        routing_weights=None,
        top_k_indices=None,
        top_k_weights=None,
    ):
        # hidden_states: (T, H). Supports either sparse (top_k_indices, top_k_weights)
        # or dense routing_weights (T, E) for QuantizedExperts compatibility.
        input_dtype = hidden_states.dtype
        static_t = hidden_states.shape[0]
        use_sparse_gather = (
            top_k_indices is not None
            and top_k_weights is not None
            and static_t is not None
            and static_t <= 8
        )

        if use_sparse_gather:
            w_gu = ops.take(self.gate_up_proj, top_k_indices, axis=0)  # (T, K, 2I, H)
            gate_up = ops.einsum("th,tkoh->tko", hidden_states, w_gu)
            gate = gate_up[..., : self.mlp_dim]
            up = gate_up[..., self.mlp_dim :]
            act = self.act_fn(gate, up)  # (T, K, I)
            w_down = ops.take(self.down_proj, top_k_indices, axis=0)  # (T, K, H, I)
            expert_out = ops.cast(ops.einsum("tki,tkhi->tkh", act, w_down), "float32")
            out = ops.einsum("tk,tkh->th", ops.cast(top_k_weights, "float32"), expert_out)
            return ops.cast(out, input_dtype)

        # Prefill / dynamic-T / dense routing_weights path
        if routing_weights is None:
            one_hot = ops.one_hot(top_k_indices, self.num_experts, dtype="float32")
            dense_weights = ops.sum(
                one_hot * ops.cast(top_k_weights, "float32")[..., None], axis=1
            )
        else:
            dense_weights = ops.cast(routing_weights, "float32")
        gate_up = ops.einsum("th,eoh->teo", hidden_states, self.gate_up_proj)
        gate = gate_up[..., : self.mlp_dim]
        up = gate_up[..., self.mlp_dim :]
        act = self.act_fn(gate, up)
        expert_out = ops.cast(ops.einsum("tei,ehi->teh", act, self.down_proj), "float32")
        out = ops.einsum("te,teh->th", dense_weights, expert_out)
        return ops.cast(out, input_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "mlp_gate_softcap": self.mlp_gate_softcap,
                "mlp_up_softcap": self.mlp_up_softcap,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2LoopSparseMoeBlock(layers.Layer):
    """Top-k sigmoid router (fp32) with correction bias + routed expert bank."""

    def __init__(
        self,
        embed_dim,
        moe_mlp_dim,
        num_experts=128,
        num_experts_per_tok=8,
        route_scale=8.0,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.moe_mlp_dim = moe_mlp_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.route_scale = float(route_scale)
        self.mlp_gate_softcap = float(mlp_gate_softcap)
        self.mlp_up_softcap = float(mlp_up_softcap)
        self.experts = Inkling2LoopExperts(
            num_experts=num_experts,
            embed_dim=embed_dim,
            mlp_dim=moe_mlp_dim,
            mlp_gate_softcap=mlp_gate_softcap,
            mlp_up_softcap=mlp_up_softcap,
            name="experts",
        )

    def build(self, input_shape):
        self.gate_weight = self.add_weight(
            name="gate_weight",
            shape=(self.num_experts, self.embed_dim),
            initializer="zeros",
            dtype="float32",
            trainable=True,
        )
        self.e_score_correction_bias = self.add_weight(
            name="e_score_correction_bias",
            shape=(self.num_experts,),
            initializer="zeros",
            dtype="float32",
            trainable=False,
        )
        self.experts.build((None, self.embed_dim))
        self.built = True

    def call(self, hidden_states):
        b = ops.shape(hidden_states)[0]
        s = ops.shape(hidden_states)[1]
        flat = ops.reshape(hidden_states, (-1, self.embed_dim))
        flat_f32 = ops.cast(flat, "float32")
        logits = ops.matmul(flat_f32, ops.transpose(ops.cast(self.gate_weight, "float32")))
        scores = ops.sigmoid(logits)
        biased = scores + ops.cast(self.e_score_correction_bias, "float32")
        _, top_k_idx = ops.top_k(biased, k=self.num_experts_per_tok, sorted=False)
        weights = ops.take_along_axis(scores, top_k_idx, axis=-1)
        weights = weights / (ops.sum(weights, axis=-1, keepdims=True) + 1e-20)
        weights = weights * self.route_scale
        out = self.experts(flat, top_k_indices=top_k_idx, top_k_weights=weights)
        return ops.reshape(out, (b, s, self.embed_dim))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "moe_mlp_dim": self.moe_mlp_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "route_scale": self.route_scale,
                "mlp_gate_softcap": self.mlp_gate_softcap,
                "mlp_up_softcap": self.mlp_up_softcap,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2LoopDecoderLayer(layers.Layer):
    """Unified Inkling2Loop decoder layer (sliding/full × non-loop/loop)."""

    def __init__(
        self,
        layer_idx,
        embed_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        moe_mlp_dim,
        num_experts=128,
        num_experts_per_tok=8,
        route_scale=8.0,
        is_sliding=True,
        is_loop_layer=False,
        sliding_window_size=1024,
        d_rel=16,
        rel_extent=1024,
        conv_kernel_size=4,
        norm_eps=1e-6,
        qk_l2_clip_threshold=22.627416997969522,
        rel_l2_clip_threshold=8.0,
        log_scaling_alpha=0.1,
        log_scaling_n_floor=8192,
        n_layers_per_attn_res_block=6,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        loop_gate_rank=64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer_idx = layer_idx
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.moe_mlp_dim = moe_mlp_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.route_scale = float(route_scale)
        self.is_sliding = bool(is_sliding)
        self.is_loop_layer = bool(is_loop_layer)
        self.sliding_window_size = sliding_window_size
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.conv_kernel_size = conv_kernel_size
        self.norm_eps = norm_eps
        self.qk_l2_clip_threshold = float(qk_l2_clip_threshold)
        self.rel_l2_clip_threshold = float(rel_l2_clip_threshold)
        self.log_scaling_alpha = float(log_scaling_alpha)
        self.log_scaling_n_floor = int(log_scaling_n_floor)
        self.n_layers_per_attn_res_block = n_layers_per_attn_res_block
        self.mlp_gate_softcap = float(mlp_gate_softcap)
        self.mlp_up_softcap = float(mlp_up_softcap)
        self.loop_gate_rank = loop_gate_rank
        self.is_block_boundary = (layer_idx % n_layers_per_attn_res_block) == 0

        if layer_idx > 0:
            self.attention_input = Inkling2LoopBlockAttentionResidual(
                embed_dim, num_queries=1 if is_sliding else 3, name="attention_input"
            )
        else:
            self.attention_input = None
        self.mlp_input = Inkling2LoopBlockAttentionResidual(
            embed_dim, num_queries=1, name="mlp_input"
        )
        self.self_attn = Inkling2LoopAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            is_sliding=is_sliding,
            sliding_window_size=sliding_window_size,
            d_rel=d_rel,
            rel_extent=rel_extent,
            conv_kernel_size=conv_kernel_size,
            norm_eps=norm_eps,
            qk_l2_clip_threshold=qk_l2_clip_threshold,
            rel_l2_clip_threshold=rel_l2_clip_threshold,
            log_scaling_alpha=log_scaling_alpha,
            log_scaling_n_floor=log_scaling_n_floor,
            name="self_attn",
        )
        self.mlp = Inkling2LoopSparseMoeBlock(
            embed_dim=embed_dim,
            moe_mlp_dim=moe_mlp_dim,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            route_scale=route_scale,
            mlp_gate_softcap=mlp_gate_softcap,
            mlp_up_softcap=mlp_up_softcap,
            name="mlp",
        )
        if is_sliding:
            self.input_layernorm = Inkling2LoopRMSNorm(
                embed_dim, eps=norm_eps, name="input_layernorm"
            )
        else:
            self.q_input_layernorm = Inkling2LoopRMSNorm(
                embed_dim, eps=norm_eps, name="q_input_layernorm"
            )
            self.k_input_layernorm = Inkling2LoopRMSNorm(
                embed_dim, eps=norm_eps, name="k_input_layernorm"
            )
            self.v_input_layernorm = Inkling2LoopRMSNorm(
                embed_dim, eps=norm_eps, name="v_input_layernorm"
            )
        self.post_attention_layernorm = Inkling2LoopRMSNorm(
            embed_dim, eps=norm_eps, name="post_attention_layernorm"
        )
        self.post_attention_convolution = Inkling2LoopShortConvolution(
            embed_dim, conv_kernel_size, name="post_attention_convolution"
        )
        self.post_mlp_convolution = Inkling2LoopShortConvolution(
            embed_dim, conv_kernel_size, name="post_mlp_convolution"
        )
        if is_loop_layer:
            self.q_message_gate = Inkling2LoopMessageGate(
                embed_dim, loop_gate_rank, norm_eps, name="q_message_gate"
            )
            self.k_message_gate = Inkling2LoopMessageGate(
                embed_dim, loop_gate_rank, norm_eps, name="k_message_gate"
            )
            self.v_message_gate = Inkling2LoopMessageGate(
                embed_dim, loop_gate_rank, norm_eps, name="v_message_gate"
            )
            self.mlp_message_gate = Inkling2LoopMessageGate(
                embed_dim, loop_gate_rank, norm_eps, name="mlp_message_gate"
            )

    def build(self, input_shape):
        shape = (None, None, self.embed_dim)
        if self.attention_input is not None:
            self.attention_input.build(shape)
        self.mlp_input.build(shape)
        self.self_attn.build(shape)
        self.mlp.build(shape)
        if self.is_sliding:
            self.input_layernorm.build(shape)
        else:
            self.q_input_layernorm.build(shape)
            self.k_input_layernorm.build(shape)
            self.v_input_layernorm.build(shape)
        self.post_attention_layernorm.build(shape)
        self.post_attention_convolution.build(shape)
        self.post_mlp_convolution.build(shape)
        if self.is_loop_layer:
            self.q_message_gate.build(shape)
            self.k_message_gate.build(shape)
            self.v_message_gate.build(shape)
            self.mlp_message_gate.build(shape)
        self.built = True

    def _prepare_qkv_inputs(self, hidden_states, block_states, messages, dtype):
        if self.is_sliding:
            h = (
                self.attention_input(hidden_states, block_states=block_states)[0]
                if self.attention_input is not None
                else hidden_states
            )
            if messages is None:
                q_h = k_h = v_h = ops.cast(self.input_layernorm(h), dtype)
                mlp_msg = None
            else:
                q_msg, k_msg, v_msg, mlp_msg = (
                    messages[:, :, 0, :],
                    messages[:, :, 1, :],
                    messages[:, :, 2, :],
                    messages[:, :, 3, :],
                )
                q_h = ops.cast(self.input_layernorm(self.q_message_gate(h, q_msg)), dtype)
                k_h = ops.cast(self.input_layernorm(self.k_message_gate(h, k_msg)), dtype)
                v_h = ops.cast(self.input_layernorm(self.v_message_gate(h, v_msg)), dtype)
            return q_h, k_h, v_h, mlp_msg

        if self.attention_input is None:
            q_h = k_h = v_h = hidden_states
        else:
            q_h, k_h, v_h = self.attention_input(hidden_states, block_states=block_states)

        if messages is not None:
            q_msg, k_msg, v_msg, mlp_msg = (
                messages[:, :, 0, :],
                messages[:, :, 1, :],
                messages[:, :, 2, :],
                messages[:, :, 3, :],
            )
            q_h = self.q_message_gate(q_h, q_msg)
            k_h = self.k_message_gate(k_h, k_msg)
            v_h = self.v_message_gate(v_h, v_msg)
        else:
            mlp_msg = None

        q_h = ops.cast(self.q_input_layernorm(q_h), dtype)
        k_h = ops.cast(self.k_input_layernorm(k_h), dtype)
        v_h = ops.cast(self.v_input_layernorm(v_h), dtype)
        return q_h, k_h, v_h, mlp_msg

    def call(
        self,
        hidden_states,
        block_states,
        attention_mask,
        relative_distance,
        position_ids,
        messages=None,
        conv_mask=None,
        use_cache=False,
    ):
        dtype = hidden_states.dtype
        block_residual = hidden_states
        q_h, k_h, v_h, mlp_msg = self._prepare_qkv_inputs(
            hidden_states, block_states, messages, dtype
        )
        attn_res = self.self_attn(
            q_h,
            attention_mask=attention_mask,
            relative_distance=relative_distance,
            position_ids=position_ids,
            conv_mask=conv_mask,
            key_hidden_states=k_h,
            value_hidden_states=v_h,
            use_cache=use_cache,
        )
        if use_cache:
            h, (k_c, v_c, k_conv, v_conv) = attn_res
            h, attn_conv = self.post_attention_convolution(
                h, conv_mask=conv_mask, return_state=True
            )
        else:
            h = attn_res
            h = self.post_attention_convolution(h, conv_mask=conv_mask)

        residual = h = ops.cast(h, "float32")
        if self.is_block_boundary:
            if block_states is None:
                mlp_blocks = block_residual[:, :, None, :]
            else:
                mlp_blocks = ops.concatenate(
                    [block_states, block_residual[:, :, None, :]], axis=-2
                )
        else:
            residual = h = ops.cast(block_residual, "float32") + h
            mlp_blocks = block_states

        h = self.mlp_input(h, block_states=mlp_blocks)[0]
        if mlp_msg is not None:
            h = self.mlp_message_gate(h, mlp_msg)
        h = self.mlp(ops.cast(self.post_attention_layernorm(h), dtype))
        if use_cache:
            h, mlp_conv = self.post_mlp_convolution(
                h, conv_mask=conv_mask, return_state=True
            )
            out = ops.cast(residual + ops.cast(h, "float32"), dtype)
            return out, (k_c, v_c, k_conv, v_conv, attn_conv, mlp_conv)
        h = self.post_mlp_convolution(h, conv_mask=conv_mask)
        return ops.cast(residual + ops.cast(h, "float32"), dtype)

    def decode_step(
        self,
        hidden_states,
        block_states,
        layer_cache,
        cache_update_index,
        attention_mask,
        relative_distance,
        position_ids,
        messages=None,
    ):
        k_c, v_c, k_conv, v_conv, attn_conv, mlp_conv = layer_cache
        dtype = hidden_states.dtype
        block_residual = hidden_states
        q_h, k_h, v_h, mlp_msg = self._prepare_qkv_inputs(
            hidden_states, block_states, messages, dtype
        )
        h, (k_c, v_c, k_conv, v_conv) = self.self_attn.decode_step(
            q_h,
            k_h,
            v_h,
            (k_c, v_c, k_conv, v_conv),
            cache_update_index,
            attention_mask,
            relative_distance,
            position_ids,
        )
        h, attn_conv = self.post_attention_convolution(
            h, conv_state=attn_conv, return_state=True
        )
        residual = h = ops.cast(h, "float32")
        if self.is_block_boundary:
            if block_states is None:
                mlp_blocks = block_residual[:, :, None, :]
            else:
                mlp_blocks = ops.concatenate(
                    [block_states, block_residual[:, :, None, :]], axis=-2
                )
        else:
            residual = h = ops.cast(block_residual, "float32") + h
            mlp_blocks = block_states

        h = self.mlp_input(h, block_states=mlp_blocks)[0]
        if mlp_msg is not None:
            h = self.mlp_message_gate(h, mlp_msg)
        h = self.mlp(ops.cast(self.post_attention_layernorm(h), dtype))
        h, mlp_conv = self.post_mlp_convolution(
            h, conv_state=mlp_conv, return_state=True
        )
        out = ops.cast(residual + ops.cast(h, "float32"), dtype)
        return out, (k_c, v_c, k_conv, v_conv, attn_conv, mlp_conv)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layer_idx": self.layer_idx,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "moe_mlp_dim": self.moe_mlp_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "route_scale": self.route_scale,
                "is_sliding": self.is_sliding,
                "is_loop_layer": self.is_loop_layer,
                "sliding_window_size": self.sliding_window_size,
                "d_rel": self.d_rel,
                "rel_extent": self.rel_extent,
                "conv_kernel_size": self.conv_kernel_size,
                "norm_eps": self.norm_eps,
                "qk_l2_clip_threshold": self.qk_l2_clip_threshold,
                "rel_l2_clip_threshold": self.rel_l2_clip_threshold,
                "log_scaling_alpha": self.log_scaling_alpha,
                "log_scaling_n_floor": self.log_scaling_n_floor,
                "n_layers_per_attn_res_block": self.n_layers_per_attn_res_block,
                "mlp_gate_softcap": self.mlp_gate_softcap,
                "mlp_up_softcap": self.mlp_up_softcap,
                "loop_gate_rank": self.loop_gate_rank,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Vision tower layers (Inkling2Vision)
# ---------------------------------------------------------------------------


def rotate_pairwise(x):
    """Rotates each interleaved channel pair ``(x1, x2) -> (-x2, x1)``."""
    shape = ops.shape(x)
    x_pair = ops.reshape(x, (*shape[:-1], -1, 2))
    x1 = x_pair[..., 0]
    x2 = x_pair[..., 1]
    return ops.reshape(ops.stack([-x2, x1], axis=-1), shape)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    q_dtype, k_dtype = q.dtype, k.dtype
    qf, kf = ops.cast(q, "float32"), ops.cast(k, "float32")
    cos_f = ops.cast(cos, "float32")[:, None, :]
    sin_f = ops.cast(sin, "float32")[:, None, :]
    q_out = qf * cos_f + rotate_pairwise(qf) * sin_f
    k_out = kf * cos_f + rotate_pairwise(kf) * sin_f
    return ops.cast(q_out, q_dtype), ops.cast(k_out, k_dtype)


def vision_axial_rope_cos_sin(position_ids, head_dim, rope_theta=10000.0):
    """2D axial RoPE matching HF ``Inkling2VisionRotaryEmbedding``.

    ``position_ids`` has shape ``(N, 2)`` with ``(row, col)`` coords.
    Concatenates width then height frequencies, each repeated pairwise.
    """
    spatial_dim = head_dim // 2
    inv_freq = 1.0 / ops.power(
        float(rope_theta), ops.arange(0, spatial_dim, 2, dtype="float32") / spatial_dim
    )
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq  # (N, 2, spatial_dim//2)
    cos = ops.cos(freqs)
    sin = ops.sin(freqs)
    cos_cat = ops.repeat(ops.concatenate([cos[:, 1], cos[:, 0]], axis=-1), 2, axis=-1)
    sin_cat = ops.repeat(ops.concatenate([sin[:, 1], sin[:, 0]], axis=-1), 2, axis=-1)
    return cos_cat, sin_cat


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2VisionAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = layers.Dense(embed_dim, use_bias=True, name="q_proj")
        self.k_proj = layers.Dense(embed_dim, use_bias=True, name="k_proj")
        self.v_proj = layers.Dense(embed_dim, use_bias=True, name="v_proj")
        self.out_proj = layers.Dense(embed_dim, use_bias=True, name="out_proj")

    def build(self, input_shape):
        shape = (None, self.embed_dim)
        self.q_proj.build(shape)
        self.k_proj.build(shape)
        self.v_proj.build(shape)
        self.out_proj.build(shape)
        self.built = True

    def call(self, hidden_states, cos, sin, attention_mask=None):
        seq_len = ops.shape(hidden_states)[0]
        q = ops.reshape(self.q_proj(hidden_states), (seq_len, self.num_heads, self.head_dim))
        k = ops.reshape(self.k_proj(hidden_states), (seq_len, self.num_heads, self.head_dim))
        v = ops.reshape(self.v_proj(hidden_states), (seq_len, self.num_heads, self.head_dim))
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        q = ops.transpose(q, (1, 0, 2))[None]  # (1, H, N, D)
        k = ops.transpose(k, (1, 0, 2))[None]
        v = ops.transpose(v, (1, 0, 2))[None]
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scaling
        if attention_mask is not None:
            scores = scores + ops.cast(attention_mask, scores.dtype)
        probs = ops.cast(ops.softmax(ops.cast(scores, "float32"), axis=-1), q.dtype)
        out = ops.matmul(probs, v)[0]  # (H, N, D)
        out = ops.reshape(ops.transpose(out, (1, 0, 2)), (seq_len, self.embed_dim))
        return self.out_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2VisionMLP(layers.Layer):
    def __init__(self, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.fc1 = layers.Dense(mlp_dim, use_bias=True, name="fc1")
        self.fc2 = layers.Dense(embed_dim, use_bias=True, name="fc2")

    def build(self, input_shape):
        self.fc1.build((None, self.embed_dim))
        self.fc2.build((None, self.mlp_dim))
        self.built = True

    def call(self, hidden_states):
        return self.fc2(ops.gelu(self.fc1(hidden_states), approximate=False))

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "mlp_dim": self.mlp_dim})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2VisionEncoderLayer(layers.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_dim,
        norm_eps=1e-5,
        layer_scale_init_value=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.norm_eps = norm_eps
        self.layer_scale_init_value = float(layer_scale_init_value)
        self.layernorm_before = layers.LayerNormalization(
            epsilon=norm_eps, name="layernorm_before"
        )
        self.self_attn = Inkling2VisionAttention(
            embed_dim, num_heads, name="self_attn"
        )
        self.layernorm_after = layers.LayerNormalization(
            epsilon=norm_eps, name="layernorm_after"
        )
        self.mlp = Inkling2VisionMLP(embed_dim, mlp_dim, name="mlp")

    def build(self, input_shape):
        shape = (None, self.embed_dim)
        self.layernorm_before.build(shape)
        self.self_attn.build(shape)
        self.layernorm_after.build(shape)
        self.mlp.build(shape)
        init = keras.initializers.Constant(self.layer_scale_init_value)
        self.lambda_1 = self.add_weight(
            name="lambda_1", shape=(self.embed_dim,), initializer=init, trainable=True
        )
        self.lambda_2 = self.add_weight(
            name="lambda_2", shape=(self.embed_dim,), initializer=init, trainable=True
        )
        self.built = True

    def call(self, hidden_states, cos, sin, attention_mask=None):
        residual = hidden_states
        h = self.self_attn(
            self.layernorm_before(hidden_states), cos, sin, attention_mask=attention_mask
        )
        hidden_states = residual + ops.cast(self.lambda_1, h.dtype) * h
        residual = hidden_states
        h = self.mlp(self.layernorm_after(hidden_states))
        return residual + ops.cast(self.lambda_2, h.dtype) * h

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "norm_eps": self.norm_eps,
                "layer_scale_init_value": self.layer_scale_init_value,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2VisionMerger(layers.Layer):
    def __init__(
        self,
        embed_dim,
        spatial_merge_size,
        projection_hidden_dim,
        out_dim,
        projector_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.spatial_merge_size = spatial_merge_size
        self.projection_hidden_dim = projection_hidden_dim
        self.out_dim = out_dim
        self.projector_bias = bool(projector_bias)
        self.linear_1 = layers.Dense(
            projection_hidden_dim, use_bias=projector_bias, name="linear_1"
        )
        self.linear_2 = layers.Dense(
            out_dim, use_bias=projector_bias, name="linear_2"
        )

    def build(self, input_shape):
        in_dim = (self.spatial_merge_size**2) * self.embed_dim
        self.linear_1.build((None, in_dim))
        self.linear_2.build((None, self.projection_hidden_dim))
        self.built = True

    def call(self, image_features):
        h = self.linear_1(image_features)
        h = ops.gelu(h, approximate=False)
        return self.linear_2(h)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "spatial_merge_size": self.spatial_merge_size,
                "projection_hidden_dim": self.projection_hidden_dim,
                "out_dim": self.out_dim,
                "projector_bias": self.projector_bias,
            }
        )
        return config
