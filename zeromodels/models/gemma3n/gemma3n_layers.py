import math

import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3nRMSNorm(layers.Layer):
    """Gemma 3n RMSNorm: plain (non ``1 + w``) weight, epsilon inside the rsqrt.

    ``with_scale=False`` drops the learnable weight (the attention value norm)."""

    def __init__(self, eps=1e-6, with_scale=True, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.with_scale = with_scale

    def build(self, input_shape):
        if self.with_scale:
            self.weight = self.add_weight(
                name="weight", shape=(input_shape[-1],), initializer="ones"
            )
        super().build(input_shape)

    def call(self, x):
        dtype = x.dtype
        x = ops.cast(x, "float32")
        normed = x * ops.rsqrt(
            ops.mean(ops.square(x), axis=-1, keepdims=True) + self.eps
        )
        if self.with_scale:
            normed = normed * ops.cast(self.weight, "float32")
        return ops.cast(normed, dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps, "with_scale": self.with_scale})
        return config


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ops.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(x, cos, sin):
    # x: (batch, seq, heads, head_dim); cos/sin: (batch, seq, head_dim).
    cos = ops.expand_dims(cos, axis=2)
    sin = ops.expand_dims(sin, axis=2)
    return x * cos + rotate_half(x) * sin


def repeat_kv(hidden, n_rep):
    # (batch, num_kv, seq, head_dim) -> (batch, num_kv * n_rep, seq, head_dim).
    if n_rep == 1:
        return hidden
    b, nkv, s, hd = ops.shape(hidden)
    hidden = ops.expand_dims(hidden, axis=2)
    hidden = ops.broadcast_to(hidden, (b, nkv, n_rep, s, hd))
    return ops.reshape(hidden, (b, nkv * n_rep, s, hd))


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3nLaurelBlock(layers.Layer):
    """Learned Augmented Residual Layer: a low-rank residual, normed and added."""

    def __init__(self, laurel_rank, norm_eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.laurel_rank = laurel_rank
        self.norm_eps = norm_eps
        self.linear_left = layers.Dense(laurel_rank, use_bias=False, name="linear_left")
        self.linear_right = None  # built lazily to match hidden_size
        self.post_laurel_norm = Gemma3nRMSNorm(eps=norm_eps, name="post_laurel_norm")

    def build(self, input_shape):
        hidden = input_shape[-1]
        self.linear_right = layers.Dense(hidden, use_bias=False, name="linear_right")
        super().build(input_shape)

    def call(self, hidden_states):
        laurel = self.linear_right(self.linear_left(hidden_states))
        return hidden_states + self.post_laurel_norm(laurel)

    def get_config(self):
        config = super().get_config()
        config.update({"laurel_rank": self.laurel_rank, "norm_eps": self.norm_eps})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3nMLP(layers.Layer):
    """GeGLU MLP with optional Gaussian top-k activation sparsity on the gate."""

    def __init__(self, mlp_dim, activation_sparsity=0.0, **kwargs):
        super().__init__(**kwargs)
        self.mlp_dim = mlp_dim
        self.activation_sparsity = activation_sparsity
        self.gate = layers.Dense(mlp_dim, use_bias=False, name="gate")
        self.up = layers.Dense(mlp_dim, use_bias=False, name="up")
        self.down = None  # built lazily to match hidden_size
        # norm.ppf(sparsity): a per-layer constant used to place the top-k cutoff.
        if activation_sparsity > 0.0:
            self.std_multiplier = float(
                ops.convert_to_numpy(
                    ops.erfinv(ops.convert_to_tensor(2.0 * activation_sparsity - 1.0))
                )
                * math.sqrt(2.0)
            )
        else:
            self.std_multiplier = 0.0

    def build(self, input_shape):
        self.down = layers.Dense(input_shape[-1], use_bias=False, name="down")
        super().build(input_shape)

    def gaussian_topk(self, x):
        mean = ops.mean(x, axis=-1, keepdims=True)
        std = ops.sqrt(ops.mean(ops.square(x - mean), axis=-1, keepdims=True))
        cutoff = mean + std * ops.cast(self.std_multiplier, x.dtype)
        return ops.relu(x - cutoff)

    def call(self, x):
        gate = self.gate(x)
        if self.activation_sparsity > 0.0:
            gate = self.gaussian_topk(gate)
        return self.down(ops.gelu(gate, approximate=True) * self.up(x))

    def get_config(self):
        config = super().get_config()
        config.update(
            {"mlp_dim": self.mlp_dim, "activation_sparsity": self.activation_sparsity}
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3nAltUp(layers.Layer):
    """Alternating Updates: predict/correct over ``num_inputs`` parallel streams."""

    def __init__(
        self, hidden_size, num_inputs=4, active_idx=0, norm_eps=1e-6, **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_inputs = num_inputs
        self.active_idx = active_idx
        self.norm_eps = norm_eps
        self.correction_coefs = layers.Dense(
            num_inputs, use_bias=False, name="correction_coefs"
        )
        self.prediction_coefs = layers.Dense(
            num_inputs * num_inputs, use_bias=False, name="prediction_coefs"
        )
        self.modality_router = layers.Dense(
            num_inputs, use_bias=False, name="modality_router"
        )
        self.router_norm = Gemma3nRMSNorm(eps=norm_eps, name="router_norm")
        self.router_input_scale = float(hidden_size) ** -1.0

    def build(self, input_shape):
        # Driven through predict()/correct() (not __call__), so all sub-weights are
        # built here (triggered lazily from predict) to keep them under this layer's
        # scope (matching HF's ``layers.N.altup.*``).
        self.router_norm.build((None, self.hidden_size))
        self.modality_router.build((None, self.hidden_size))
        self.prediction_coefs.build((None, self.num_inputs))
        self.correction_coefs.build((None, self.num_inputs))
        self.correct_output_scale = self.add_weight(
            name="correct_output_scale", shape=(self.hidden_size,), initializer="zeros"
        )
        super().build(input_shape)

    def compute_router_modalities(self, x):
        routed = self.modality_router(self.router_norm(x) * self.router_input_scale)
        return ops.cast(ops.tanh(ops.cast(routed, "float32")), x.dtype)

    def predict(self, hidden):
        # hidden: (P, batch, seq, hidden).
        if not self.built:
            self.build(hidden.shape)
        p = self.num_inputs
        modalities = self.compute_router_modalities(hidden[self.active_idx])
        b, s = ops.shape(modalities)[0], ops.shape(modalities)[1]
        coefs = self.prediction_coefs(modalities)  # (b, s, P*P)
        coefs = ops.reshape(coefs, (b, s, p, p))
        coefs = ops.transpose(coefs, (0, 1, 3, 2))
        h_perm = ops.transpose(hidden, (1, 2, 3, 0))  # (b, s, hidden, P)
        predictions = ops.matmul(h_perm, coefs)  # (b, s, hidden, P)
        predictions = ops.transpose(predictions, (3, 0, 1, 2))  # (P, b, s, hidden)
        return ops.cast(predictions + hidden, hidden.dtype)

    def correct(self, predictions, activated):
        # predictions: (P, batch, seq, hidden); activated: (batch, seq, hidden).
        modalities = self.compute_router_modalities(activated)
        innovation = activated - predictions[self.active_idx]  # (b, s, hidden)
        innovation = ops.broadcast_to(
            ops.expand_dims(innovation, 0), ops.shape(predictions)
        )
        coefs = self.correction_coefs(modalities) + 1.0  # (b, s, P)
        coefs = ops.expand_dims(
            ops.transpose(coefs, (2, 0, 1)), axis=-1
        )  # (P, b, s, 1)
        corrected = innovation * coefs + predictions
        return ops.cast(corrected, activated.dtype)

    def scale_corrected_output(self, corrected):
        return corrected * ops.cast(self.correct_output_scale, corrected.dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_inputs": self.num_inputs,
                "active_idx": self.active_idx,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3nAttention(layers.Layer):
    """GQA attention with per-head q/k/v RMSNorm (value norm scaleless), full-head
    rope, unit scaling, and optional KV-sharing (shared layers carry no k/v/norm)."""

    def __init__(
        self,
        num_heads,
        num_kv_heads,
        head_dim,
        is_kv_shared=False,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.is_kv_shared = is_kv_shared
        self.norm_eps = norm_eps
        self.num_kv_groups = num_heads // num_kv_heads

        self.query = layers.Dense(num_heads * head_dim, use_bias=False, name="query")
        self.query_norm = Gemma3nRMSNorm(eps=norm_eps, name="query_norm")
        if not is_kv_shared:
            self.key = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="key")
            self.value = layers.Dense(
                num_kv_heads * head_dim, use_bias=False, name="value"
            )
            self.key_norm = Gemma3nRMSNorm(eps=norm_eps, name="key_norm")
            self.value_norm = Gemma3nRMSNorm(
                eps=norm_eps, with_scale=False, name="value_norm"
            )
        self.output_proj = layers.Dense(
            num_heads * head_dim, use_bias=False, name="output_proj"
        )

    def build(self, input_shape):
        prefix = tuple(input_shape[:-1])
        self.query.build(input_shape)
        self.query_norm.build(prefix + (self.num_heads, self.head_dim))
        if not self.is_kv_shared:
            self.key.build(input_shape)
            self.value.build(input_shape)
            self.key_norm.build(prefix + (self.num_kv_heads, self.head_dim))
            self.value_norm.build(prefix + (self.num_kv_heads, self.head_dim))
        self.output_proj.build(prefix + (self.num_heads * self.head_dim,))
        self.built = True

    def project_kv(self, hidden, cos, sin):
        b, s = int(hidden.shape[0]), int(hidden.shape[1])
        k = ops.reshape(self.key(hidden), (b, s, self.num_kv_heads, self.head_dim))
        k = apply_rotary_pos_emb(self.key_norm(k), cos, sin)
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.reshape(self.value(hidden), (b, s, self.num_kv_heads, self.head_dim))
        v = ops.transpose(self.value_norm(v), (0, 2, 1, 3))
        return k, v

    def call(self, hidden, cos, sin, attention_mask=None, shared_kv=None):
        b, s = int(hidden.shape[0]), int(hidden.shape[1])
        q = ops.reshape(self.query(hidden), (b, s, self.num_heads, self.head_dim))
        q = apply_rotary_pos_emb(self.query_norm(q), cos, sin)
        q = ops.transpose(q, (0, 2, 1, 3))  # (b, heads, s, hd)

        k, v = shared_kv if self.is_kv_shared else self.project_kv(hidden, cos, sin)
        kr = repeat_kv(k, self.num_kv_groups)
        vr = repeat_kv(v, self.num_kv_groups)

        scores = ops.matmul(q, ops.transpose(kr, (0, 1, 3, 2)))
        if attention_mask is not None:
            scores = scores + ops.cast(attention_mask, scores.dtype)
        weights = ops.cast(
            ops.softmax(ops.cast(scores, "float32"), axis=-1), hidden.dtype
        )
        out = ops.matmul(weights, vr)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, s, self.num_heads * self.head_dim)
        )
        return self.output_proj(out), (k, v)

    def decode_step(
        self, hidden, cos, sin, cache_k, cache_v, pos, mask, shared_kv=None
    ):
        # One-token decode. cache_k/cache_v: (b, num_kv, max_len, head_dim); for a
        # shared layer they are the storing layer's already-updated cache.
        b = int(hidden.shape[0])
        q = ops.reshape(self.query(hidden), (b, 1, self.num_heads, self.head_dim))
        q = apply_rotary_pos_emb(self.query_norm(q), cos, sin)
        q = ops.transpose(q, (0, 2, 1, 3))

        if self.is_kv_shared:
            k, v = cache_k, cache_v
        else:
            kn = ops.reshape(self.key(hidden), (b, 1, self.num_kv_heads, self.head_dim))
            kn = apply_rotary_pos_emb(self.key_norm(kn), cos, sin)
            kn = ops.transpose(kn, (0, 2, 1, 3))
            vn = ops.reshape(
                self.value(hidden), (b, 1, self.num_kv_heads, self.head_dim)
            )
            vn = ops.transpose(self.value_norm(vn), (0, 2, 1, 3))
            # rope runs in float32; match the cache dtype before writing
            kn = ops.cast(kn, cache_k.dtype)
            vn = ops.cast(vn, cache_v.dtype)
            k = ops.slice_update(cache_k, (0, 0, pos, 0), kn)
            v = ops.slice_update(cache_v, (0, 0, pos, 0), vn)

        kr = repeat_kv(k, self.num_kv_groups)
        vr = repeat_kv(v, self.num_kv_groups)
        scores = ops.matmul(q, ops.transpose(kr, (0, 1, 3, 2)))
        if mask is not None:
            scores = scores + ops.cast(mask, scores.dtype)
        weights = ops.cast(
            ops.softmax(ops.cast(scores, "float32"), axis=-1), hidden.dtype
        )
        out = ops.matmul(weights, vr)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)), (b, 1, self.num_heads * self.head_dim)
        )
        return self.output_proj(out), (k, v)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "is_kv_shared": self.is_kv_shared,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3nDecoderLayer(layers.Layer):
    """One Gemma 3n block: AltUp predict -> (LAuReL + attention) -> MLP -> AltUp
    correct, then the per-layer-input gate/projection folded into the streams."""

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        hidden_size_per_layer_input,
        laurel_rank,
        altup_num_inputs,
        altup_active_idx,
        altup_correct_scale,
        activation_sparsity=0.0,
        is_kv_shared=False,
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.altup_active_idx = altup_active_idx
        self.altup_correct_scale = altup_correct_scale
        self.hidden_size_per_layer_input = hidden_size_per_layer_input

        self.attention = Gemma3nAttention(
            num_heads, num_kv_heads, head_dim, is_kv_shared, norm_eps, name="attention"
        )
        self.mlp = Gemma3nMLP(mlp_dim, activation_sparsity, name="mlp")
        self.attention_norm = Gemma3nRMSNorm(eps=norm_eps, name="attention_norm")
        self.post_attention_norm = Gemma3nRMSNorm(
            eps=norm_eps, name="post_attention_norm"
        )
        self.pre_feedforward_norm = Gemma3nRMSNorm(
            eps=norm_eps, name="pre_feedforward_norm"
        )
        self.post_feedforward_norm = Gemma3nRMSNorm(
            eps=norm_eps, name="post_feedforward_norm"
        )
        self.altup = Gemma3nAltUp(
            embed_dim, altup_num_inputs, altup_active_idx, norm_eps, name="altup"
        )
        self.laurel = Gemma3nLaurelBlock(laurel_rank, norm_eps, name="laurel")
        self.per_layer_input_gate = layers.Dense(
            hidden_size_per_layer_input, use_bias=False, name="per_layer_input_gate"
        )
        self.per_layer_projection = layers.Dense(
            embed_dim, use_bias=False, name="per_layer_projection"
        )
        self.post_per_layer_input_norm = Gemma3nRMSNorm(
            eps=norm_eps, name="post_per_layer_input_norm"
        )

    def build(self, input_shape):
        # input_shape = (num_altup_inputs, batch, seq, embed_dim); the active
        # stream (and every norm/attention/mlp) runs on (batch, seq, embed_dim).
        active_shape = tuple(input_shape[1:])
        prefix = active_shape[:-1]
        self.altup.build(input_shape)
        self.attention_norm.build(active_shape)
        self.attention.build(active_shape)
        self.laurel.build(active_shape)
        self.post_attention_norm.build(active_shape)
        self.pre_feedforward_norm.build(active_shape)
        self.mlp.build(active_shape)
        self.post_feedforward_norm.build(active_shape)
        self.per_layer_input_gate.build(active_shape)
        self.per_layer_projection.build(prefix + (self.hidden_size_per_layer_input,))
        self.post_per_layer_input_norm.build(active_shape)
        self.built = True

    def finish(self, predictions, active, attn, laurel_output, per_layer_input):
        # Shared post-attention body: AltUp correct + per-layer-input fold-in.
        attn = self.post_attention_norm(attn)
        attn_gated = active + attn
        attn_laurel = (attn_gated + laurel_output) / math.sqrt(2.0)

        attn_ffw = self.mlp(self.pre_feedforward_norm(attn_laurel))
        attn_ffw_norm = self.post_feedforward_norm(attn_ffw)
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm

        corrected = self.altup.correct(predictions, attn_ffw_laurel_gated)
        first = corrected[self.altup_active_idx]
        if self.altup_correct_scale:
            first = self.altup.scale_corrected_output(first)
        first = ops.gelu(self.per_layer_input_gate(first), approximate=True)
        first = first * ops.cast(per_layer_input, first.dtype)
        first = self.post_per_layer_input_norm(self.per_layer_projection(first))

        # corrected[1:] += first (the active stream, index 0, is left as-is).
        head = corrected[:1]
        tail = corrected[1:] + ops.expand_dims(first, 0)
        return ops.concatenate([head, tail], axis=0)

    def call(
        self, hidden, cos, sin, per_layer_input, attention_mask=None, shared_kv=None
    ):
        predictions = self.altup.predict(hidden)  # (P, b, s, h)
        active = predictions[self.altup_active_idx]
        active_normed = self.attention_norm(active)
        laurel_output = self.laurel(active_normed)
        attn, kv = self.attention(
            active_normed, cos, sin, attention_mask=attention_mask, shared_kv=shared_kv
        )
        return self.finish(
            predictions, active, attn, laurel_output, per_layer_input
        ), kv

    def compute_output_spec(
        self, hidden, cos, sin, per_layer_input, attention_mask=None, shared_kv=None
    ):
        # Isolate the eager ``int(shape)`` / symbolic-dim comparisons in attention
        # from the functional-graph trace; the 4-stream state keeps its shape and
        # the layer emits its own (k, v) at (b, num_kv_heads, s, head_dim). k and v
        # MUST be distinct KerasTensors -- reusing one object aliases them in the
        # graph, silently mis-wiring KV-shared layers.
        b, s = hidden.shape[1], hidden.shape[2]
        kv_shape = (b, self.attention.num_kv_heads, s, self.attention.head_dim)
        k = keras.KerasTensor(kv_shape, dtype=self.compute_dtype)
        v = keras.KerasTensor(kv_shape, dtype=self.compute_dtype)
        return keras.KerasTensor(hidden.shape, dtype=hidden.dtype), (k, v)

    def decode_step(
        self,
        hidden,
        cos,
        sin,
        per_layer_input,
        cache_k,
        cache_v,
        pos,
        mask,
        shared_kv=None,
    ):
        predictions = self.altup.predict(hidden)
        active = predictions[self.altup_active_idx]
        active_normed = self.attention_norm(active)
        laurel_output = self.laurel(active_normed)
        attn, kv = self.attention.decode_step(
            active_normed, cos, sin, cache_k, cache_v, pos, mask, shared_kv=shared_kv
        )
        return self.finish(
            predictions, active, attn, laurel_output, per_layer_input
        ), kv


def clip_activations(x, g):
    return ops.clip(x, -g, g)


def glu(x, axis=-1):
    a, b = ops.split(x, 2, axis=axis)
    return a * ops.sigmoid(b)


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3nAudioCumulativeGroupNorm(layers.Layer):
    """Group norm (num_groups=1) accumulated over the time axis.

    Statistics at time ``t`` are computed over all feature dimensions of steps
    ``0..t`` (JAX ``cumulative=True``). Scale is per-channel (last axis)."""

    def __init__(self, num_channels, feature_dims, eps=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.feature_dims = tuple(feature_dims)
        self.eps = eps
        # Reduce over every axis except batch (0) and time (1).
        self.reduction_axes = tuple(range(2, 2 + len(self.feature_dims) + 1))

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight", shape=(self.num_channels,), initializer="ones"
        )
        super().build(input_shape)

    def call(self, x):
        dtype = x.dtype
        x = ops.cast(x, "float32")
        # Per-timestep sum then cumulative sum over time; count is the (constant)
        # number of elements per group, made cumulative the same way.
        sum_t = ops.sum(x, axis=self.reduction_axes, keepdims=True)
        cum_sum = ops.cumsum(sum_t, axis=1)
        n_per_t = ops.cast(math.prod(self.feature_dims) * self.num_channels, "float32")
        ones_t = ops.ones_like(sum_t) * n_per_t
        cum_count = ops.maximum(ops.cumsum(ones_t, axis=1), 1.0)
        cum_mean = cum_sum / cum_count

        sq_diff = ops.square(x - cum_mean)
        sum_sq_t = ops.sum(sq_diff, axis=self.reduction_axes, keepdims=True)
        cum_var = ops.cumsum(sum_sq_t, axis=1) / cum_count

        normed = (x - cum_mean) * ops.rsqrt(cum_var + self.eps)
        scale_shape = [1] * (len(x.shape) - 1) + [self.num_channels]
        normed = normed * ops.reshape(ops.cast(self.weight, "float32"), scale_shape)
        return ops.cast(normed, dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_channels": self.num_channels,
                "feature_dims": list(self.feature_dims),
                "eps": self.eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3nAudioSSCPConvBlock(layers.Layer):
    """Conv2d (reverse-causal in time, SAME-ish in freq) + cumulative group norm + ReLU."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride_size,
        manual_padding,
        input_freq_dim,
        norm_eps=1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tuple(kernel_size)
        self.stride_size = tuple(stride_size)
        self.manual_padding = tuple(
            manual_padding
        )  # (pad_f_l, pad_f_r, pad_t_top, pad_t_bot)
        self.input_freq_dim = input_freq_dim
        self.norm_eps = norm_eps

        self.conv = layers.Conv2D(
            out_channels,
            self.kernel_size,
            strides=self.stride_size,
            padding="valid",
            use_bias=False,
            name="conv",
        )
        f_in_padded = input_freq_dim + self.manual_padding[0] + self.manual_padding[1]
        f_out = (f_in_padded - self.kernel_size[1]) // self.stride_size[1] + 1
        self.f_out = f_out
        self.norm = Gemma3nAudioCumulativeGroupNorm(
            out_channels, (f_out,), eps=norm_eps, name="norm"
        )

    def call(self, x):
        # x: [B, T, F, C_in] (channels-last). Pad time (axis 1) reverse-causal and
        # frequency (axis 2) symmetrically.
        pad_f_l, pad_f_r, pad_t_top, pad_t_bot = self.manual_padding
        x = ops.pad(x, [[0, 0], [pad_t_top, pad_t_bot], [pad_f_l, pad_f_r], [0, 0]])
        x = self.conv(x)
        x = self.norm(x)
        return ops.relu(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "kernel_size": list(self.kernel_size),
                "stride_size": list(self.stride_size),
                "manual_padding": list(self.manual_padding),
                "input_freq_dim": self.input_freq_dim,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3nAudioSubSampleConvProjection(layers.Layer):
    """Two stride-2 SSCP conv blocks then a linear projection to hidden_size."""

    def __init__(
        self,
        input_feat_size,
        conv_channels,
        conv_kernel_size,
        conv_stride_size,
        hidden_size,
        norm_eps=1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_feat_size = input_feat_size
        self.conv_channels = list(conv_channels)
        self.conv_kernel_size = [list(k) for k in conv_kernel_size]
        self.conv_stride_size = [list(s) for s in conv_stride_size]
        self.hidden_size = hidden_size
        self.norm_eps = norm_eps

        cur_f = input_feat_size
        blocks = []
        for i in range(2):
            kh, kw = conv_kernel_size[i]
            sh, sw = conv_stride_size[i]
            manual_padding = (1, 1, 0, kh - 1)  # (f_left, f_right, t_top, t_bottom)
            in_ch = 1 if i == 0 else conv_channels[i - 1]
            block = Gemma3nAudioSSCPConvBlock(
                in_ch,
                conv_channels[i],
                (kh, kw),
                (sh, sw),
                manual_padding,
                cur_f,
                norm_eps=norm_eps,
                name=f"conv_{i}",
            )
            blocks.append(block)
            cur_f = block.f_out
        self.conv_0, self.conv_1 = blocks
        self.final_f_out = cur_f
        self.input_proj_linear = layers.Dense(
            hidden_size, use_bias=False, name="input_proj_linear"
        )

    def call(self, audio_mel):
        # audio_mel: [B, T, F] -> [B, T, F, 1]
        x = audio_mel[..., None]
        x = self.conv_0(x)
        x = self.conv_1(x)
        b = ops.shape(x)[0]
        t = ops.shape(x)[1]
        # [B, T, F_out, C_out] -> [B, T, F_out * C_out] (freq outer, channel inner)
        x = ops.reshape(x, (b, t, self.final_f_out * self.conv_channels[-1]))
        return self.input_proj_linear(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_feat_size": self.input_feat_size,
                "conv_channels": self.conv_channels,
                "conv_kernel_size": self.conv_kernel_size,
                "conv_stride_size": self.conv_stride_size,
                "hidden_size": self.hidden_size,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3nAudioRelativePositionEmbedding(layers.Layer):
    """Transformer-XL style relative-position bias for chunked local attention."""

    def __init__(self, hidden_size, num_heads, context_left, context_right, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_backward = max(0, context_left - 1)
        self.max_forward = context_right
        self.pos_proj = layers.Dense(
            num_heads * self.head_dim, use_bias=False, name="pos_proj"
        )
        num_timescales = hidden_size // 2
        log_increment = math.log(1.0e4) / max(num_timescales - 1, 1)
        inv = ops.convert_to_numpy(
            ops.exp(ops.arange(num_timescales, dtype="float32") * -log_increment)
        )
        self.inv_timescales = inv[None, None, :]  # [1, 1, num_timescales]
        pos = ops.convert_to_numpy(
            ops.arange(self.max_backward, -self.max_forward - 1, -1, dtype="float32")
        )
        self.pos_indices = pos[None]  # [1, F_span]
        self.max_span_plus_1 = self.pos_indices.shape[1]

    def timing_signal(self, dtype):
        position = ops.cast(ops.convert_to_tensor(self.pos_indices), "float32")[
            ..., None
        ]
        scaled = position * ops.convert_to_tensor(self.inv_timescales)
        signal = ops.concatenate([ops.sin(scaled), ops.cos(scaled)], axis=-1)
        return ops.cast(signal, dtype)  # [1, F_span, hidden]

    def relative_shift(self, term_bd, b, n, u, w, c):
        pad = (c + 1) - self.max_span_plus_1
        term_bd = ops.pad(term_bd, [[0, 0], [0, 0], [0, 0], [0, 0], [0, pad]])
        term_bd = ops.reshape(term_bd, (b, n, u, w * (c + 1)))
        term_bd = term_bd[:, :, :, : w * c]
        return ops.reshape(term_bd, (b, n, u, w, c))

    def call(self, queries, keys):
        # queries: [B, U, W, N, H]; keys: [B, U, C, N, H]
        b = ops.shape(queries)[0]
        u = ops.shape(queries)[1]
        w = int(queries.shape[2])
        c = int(keys.shape[2])
        n, h = self.num_heads, self.head_dim

        sin_emb = self.pos_proj(self.timing_signal(queries.dtype))
        sin_emb = ops.reshape(sin_emb, (self.max_span_plus_1, n, h))

        queries_p = ops.transpose(queries, (0, 3, 1, 2, 4))  # [B, N, U, W, H]
        keys_p_t = ops.transpose(keys, (0, 3, 1, 4, 2))  # [B, N, U, H, C]
        term_ac = ops.matmul(queries_p, keys_p_t)  # [B, N, U, W, C]

        s_perm = ops.transpose(sin_emb, (1, 2, 0))  # [N, H, F_span]
        q_reshaped = ops.reshape(queries_p, (b, n, u * w, h))
        term_bd = ops.matmul(q_reshaped, s_perm)  # [B, N, U*W, F_span]
        term_bd = ops.reshape(term_bd, (b, n, u, w, self.max_span_plus_1))
        term_bd = self.relative_shift(term_bd, b, n, u, w, c)
        return term_ac + term_bd

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "context_left": self.max_backward + 1,
                "context_right": self.max_forward,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3nAudioAttention(layers.Layer):
    """Chunked local self-attention with relative-position bias and logit soft-cap."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        chunk_size,
        context_left,
        context_right,
        logit_cap=50.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.chunk_size = chunk_size
        self.max_future_horizon = context_right
        self.max_past_horizon = max(0, context_left - 1)
        self.logit_cap = logit_cap
        self.context_size = chunk_size + self.max_past_horizon + self.max_future_horizon
        self.q_scale = (self.head_dim**-0.5) * (1.0 / math.log(2.0))

        units = num_heads * self.head_dim
        self.q_proj = layers.Dense(units, use_bias=False, name="q_proj")
        self.k_proj = layers.Dense(units, use_bias=False, name="k_proj")
        self.v_proj = layers.Dense(units, use_bias=False, name="v_proj")
        self.relative_position_embedding = Gemma3nAudioRelativePositionEmbedding(
            hidden_size,
            num_heads,
            context_left,
            context_right,
            name="relative_position_embedding",
        )

        # Static local causal validity mask [W, C].
        w, c = self.chunk_size, self.context_size
        lower = ops.transpose(ops.tril(ops.ones((c, w)), k=0))
        upper = ops.tril(
            ops.ones((w, c)),
            k=self.max_past_horizon + self.max_future_horizon,
        )
        self.local_causal_valid_mask = ops.convert_to_numpy(
            ops.logical_and(ops.cast(lower, "bool"), ops.cast(upper, "bool"))
        )
        # Block gather start indices; sliced per call by num_blocks.
        self._starts = ops.convert_to_numpy(ops.arange(4096))

    def build(self, input_shape):
        self.per_dim_scale = self.add_weight(
            name="per_dim_scale", shape=(self.head_dim,), initializer="zeros"
        )
        super().build(input_shape)

    def convert_to_block(self, x, num_blocks, seq_len):
        pad = num_blocks * self.chunk_size - seq_len
        pad_cfg = [[0, 0], [0, pad]] + [[0, 0]] * (len(x.shape) - 2)
        x = ops.pad(x, pad_cfg)
        tail = tuple(x.shape[2:])
        b = ops.shape(x)[0]
        return ops.reshape(x, (b, num_blocks, self.chunk_size) + tail)

    def extract_block_context(self, x, num_blocks, seq_len):
        pad_left = self.max_past_horizon
        pad_right = self.max_future_horizon + self.chunk_size - 1
        pad_cfg = [[0, 0], [pad_left, pad_right]] + [[0, 0]] * (len(x.shape) - 2)
        x = ops.pad(x, pad_cfg)
        idx = (
            self._starts[:num_blocks, None] * self.chunk_size
            + ops.convert_to_numpy(ops.arange(self.context_size))[None, :]
        )
        idx = ops.convert_to_tensor(idx.astype("int32"))
        return ops.take(x, idx, axis=1)  # [B, U, C, ...]

    def call(self, hidden_states, mask=None):
        b = ops.shape(hidden_states)[0]
        seq_len = int(hidden_states.shape[1])
        num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
        qkv_shape = (b, seq_len, self.num_heads, self.head_dim)

        q = ops.cast(ops.reshape(self.q_proj(hidden_states), qkv_shape), "float32")
        k = ops.cast(ops.reshape(self.k_proj(hidden_states), qkv_shape), "float32")
        v = ops.cast(ops.reshape(self.v_proj(hidden_states), qkv_shape), "float32")
        per_dim = ops.softplus(ops.cast(self.per_dim_scale, "float32"))
        q = q * self.q_scale * ops.reshape(per_dim, (1, 1, 1, self.head_dim))

        query_blocks = self.convert_to_block(q, num_blocks, seq_len)  # [B,U,W,N,H]
        key_blocks = self.extract_block_context(k, num_blocks, seq_len)  # [B,U,C,N,H]
        value_blocks = self.extract_block_context(v, num_blocks, seq_len)

        if mask is None:
            valid = ops.ones((b, seq_len), dtype="bool")
        else:
            valid = ops.logical_not(mask)
        valid_blocks = self.extract_block_context(valid, num_blocks, seq_len)  # [B,U,C]
        cond_valid = valid_blocks[:, None, :, None, :]  # [B,1,U,1,C]
        cond_causal = ops.convert_to_tensor(self.local_causal_valid_mask)[
            None, None, None, :, :
        ]  # [1,1,1,W,C]
        final_cond = ops.logical_and(cond_valid, cond_causal)

        logits = self.relative_position_embedding(query_blocks, key_blocks)
        logits = ops.tanh(logits / self.logit_cap) * self.logit_cap
        neg_inf = ops.cast(-3.4028234663852886e38, "float32")  # most negative float32
        logits = ops.where(final_cond, logits, neg_inf)
        probs = ops.softmax(logits, axis=-1)  # [B,N,U,W,C]

        # einsum("BNUWC,BUCNH->BUWNH")
        n = self.num_heads
        w = self.chunk_size
        c = self.context_size
        h = self.head_dim
        prob_flat = ops.reshape(ops.transpose(probs, (0, 2, 1, 3, 4)), (-1, w, c))
        v_flat = ops.reshape(ops.transpose(value_blocks, (0, 1, 3, 2, 4)), (-1, c, h))
        ctx = ops.matmul(prob_flat, v_flat)  # [B*U, W, H]... per (b,u,n)
        ctx = ops.reshape(ctx, (b, num_blocks, n, w, h))
        ctx = ops.transpose(ctx, (0, 1, 3, 2, 4))  # [B,U,W,N,H]
        ctx = ops.reshape(ctx, (b, num_blocks * w, n, h))
        ctx = ctx[:, :seq_len]
        return ops.cast(ctx, hidden_states.dtype)

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
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3nAudioConformerAttention(layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_heads,
        chunk_size,
        context_left,
        context_right,
        logit_cap=50.0,
        norm_eps=1e-6,
        gradient_clipping=1e10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.gradient_clipping = gradient_clipping
        self.pre_attn_norm = Gemma3nRMSNorm(eps=norm_eps, name="pre_attn_norm")
        self.attn = Gemma3nAudioAttention(
            hidden_size,
            num_heads,
            chunk_size,
            context_left,
            context_right,
            logit_cap,
            name="attn",
        )
        self.post = layers.Dense(hidden_size, use_bias=False, name="post")
        self.post_norm = Gemma3nRMSNorm(eps=norm_eps, name="post_norm")

    def call(self, x, mask=None):
        residual = x
        x = clip_activations(x, self.gradient_clipping)
        x = self.pre_attn_norm(x)
        x = self.attn(x, mask)  # [B, T, N, H]
        b = ops.shape(x)[0]
        t = ops.shape(x)[1]
        x = ops.reshape(x, (b, t, self.hidden_size))
        x = self.post(x)
        x = clip_activations(x, self.gradient_clipping)
        return residual + self.post_norm(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "gradient_clipping": self.gradient_clipping,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3nAudioConformerFeedForward(layers.Layer):
    def __init__(
        self,
        hidden_size,
        norm_eps=1e-6,
        residual_weight=0.5,
        gradient_clipping=1e10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.residual_weight = residual_weight
        self.gradient_clipping = gradient_clipping
        self.pre_layer_norm = Gemma3nRMSNorm(eps=norm_eps, name="pre_layer_norm")
        self.ffw_layer_1 = layers.Dense(
            hidden_size * 4, use_bias=False, name="ffw_layer_1"
        )
        self.ffw_layer_2 = layers.Dense(hidden_size, use_bias=False, name="ffw_layer_2")
        self.post_layer_norm = Gemma3nRMSNorm(eps=norm_eps, name="post_layer_norm")

    def call(self, x):
        residual = x
        x = clip_activations(x, self.gradient_clipping)
        x = self.pre_layer_norm(x)
        x = self.ffw_layer_1(x)
        x = keras.activations.silu(x)
        x = self.ffw_layer_2(x)
        x = clip_activations(x, self.gradient_clipping)
        x = self.post_layer_norm(x)
        return residual + x * self.residual_weight

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "residual_weight": self.residual_weight,
                "gradient_clipping": self.gradient_clipping,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3nAudioConformerLightConv1d(layers.Layer):
    def __init__(
        self,
        hidden_size,
        conv_kernel_size=5,
        norm_eps=1e-6,
        gradient_clipping=1e10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.conv_kernel_size = conv_kernel_size
        self.gradient_clipping = gradient_clipping
        self.causal_padding = conv_kernel_size - 1
        self.pre_layer_norm = Gemma3nRMSNorm(eps=norm_eps, name="pre_layer_norm")
        self.linear_start = layers.Dense(
            hidden_size * 2, use_bias=False, name="linear_start"
        )
        self.conv_norm = Gemma3nRMSNorm(eps=norm_eps, name="conv_norm")
        self.linear_end = layers.Dense(hidden_size, use_bias=False, name="linear_end")

    def build(self, input_shape):
        self.depthwise_kernel = self.add_weight(
            name="depthwise_conv1d_kernel",
            shape=(self.conv_kernel_size, self.hidden_size, 1),
            initializer="glorot_uniform",
        )
        super().build(input_shape)

    def call(self, x):
        residual = x
        x = self.pre_layer_norm(x)
        x = self.linear_start(x)
        x = glu(x, axis=-1)
        x = ops.pad(x, [[0, 0], [self.causal_padding, 0], [0, 0]])
        x = ops.depthwise_conv(
            x, ops.cast(self.depthwise_kernel, x.dtype), strides=1, padding="valid"
        )
        x = clip_activations(x, self.gradient_clipping)
        x = self.conv_norm(x)
        x = keras.activations.silu(x)
        x = self.linear_end(x)
        return x + residual

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "conv_kernel_size": self.conv_kernel_size,
                "gradient_clipping": self.gradient_clipping,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3nAudioConformerBlock(layers.Layer):
    """USM conformer block: FF, chunked attention, light conv, FF, output norm."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        chunk_size,
        context_left,
        context_right,
        conv_kernel_size=5,
        logit_cap=50.0,
        norm_eps=1e-6,
        residual_weight=0.5,
        gradient_clipping=1e10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gradient_clipping = gradient_clipping
        self.ffw_layer_start = Gemma3nAudioConformerFeedForward(
            hidden_size,
            norm_eps,
            residual_weight,
            gradient_clipping,
            name="ffw_layer_start",
        )
        self.attention = Gemma3nAudioConformerAttention(
            hidden_size,
            num_heads,
            chunk_size,
            context_left,
            context_right,
            logit_cap,
            norm_eps,
            gradient_clipping,
            name="attention",
        )
        self.lconv1d = Gemma3nAudioConformerLightConv1d(
            hidden_size, conv_kernel_size, norm_eps, gradient_clipping, name="lconv1d"
        )
        self.ffw_layer_end = Gemma3nAudioConformerFeedForward(
            hidden_size,
            norm_eps,
            residual_weight,
            gradient_clipping,
            name="ffw_layer_end",
        )
        self.norm = Gemma3nRMSNorm(eps=norm_eps, name="norm")

    def call(self, x, mask=None):
        x = self.ffw_layer_start(x)
        x = self.attention(x, mask)
        if mask is not None:
            valid = ops.cast(ops.logical_not(mask), x.dtype)[..., None]
            x = x * valid
        x = self.lconv1d(x)
        x = self.ffw_layer_end(x)
        x = clip_activations(x, self.gradient_clipping)
        return self.norm(x)

    def get_config(self):
        config = super().get_config()
        config.update({"gradient_clipping": self.gradient_clipping})
        return config


# MobileNet-V5 (``mobilenetv5_300m_enc``) architecture, decoded from timm's
# ``arch_def`` string DSL into explicit per-block specs. Block ``t``: er =
# EdgeResidual (inc/mid/out channels, k kernel, s stride, skip, eb/pb conv bias);
# uir = UniversalInvertedResidual (dws_/dwm_ = start/mid depthwise kernel + stride,
# ls layer-scale); mqa = MobileAttention (dim/out, heads, kd/vd key/value dim, kvs
# K/V stride, dwk down-conv kernel). msfa = MultiScaleFusionAdapter.
# fmt: off
MNV5_ARCH = {
    "stem": {"out": 64, "k": 3, "s": 2, "b": True},
    "stages": [
        [  # stage 0
            {"t": "er", "inc": 64, "mid": 256, "out": 128, "k": 3, "s": 2, "skip": False, "eb": False, "pb": False},
            {"t": "er", "inc": 128, "mid": 512, "out": 128, "k": 3, "s": 1, "skip": True, "eb": False, "pb": False},
            {"t": "er", "inc": 128, "mid": 512, "out": 128, "k": 3, "s": 1, "skip": True, "eb": False, "pb": False},
        ],
        [  # stage 1
            {"t": "uir", "inc": 128, "mid": 768, "out": 256, "dws_k": 3, "dws_s": 1, "dwm_k": 5, "dwm_s": 2, "skip": False, "ls": True},
            {"t": "uir", "inc": 256, "mid": 1024, "out": 256, "dws_k": 5, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "uir", "inc": 256, "mid": 1024, "out": 256, "dws_k": 3, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "uir", "inc": 256, "mid": 1024, "out": 256, "dws_k": 5, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "uir", "inc": 256, "mid": 1024, "out": 256, "dws_k": 3, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
        ],
        [  # stage 2
            {"t": "uir", "inc": 256, "mid": 1536, "out": 640, "dws_k": 5, "dws_s": 1, "dwm_k": 5, "dwm_s": 2, "skip": False, "ls": True},
            {"t": "uir", "inc": 640, "mid": 2560, "out": 640, "dws_k": 5, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "uir", "inc": 640, "mid": 2560, "out": 640, "dws_k": 5, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "uir", "inc": 640, "mid": 2560, "out": 640, "dws_k": 5, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "uir", "inc": 640, "mid": 2560, "out": 640, "dws_k": 5, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "uir", "inc": 640, "mid": 2560, "out": 640, "dws_k": 5, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "uir", "inc": 640, "mid": 2560, "out": 640, "dws_k": 5, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "uir", "inc": 640, "mid": 2560, "out": 640, "dws_k": 5, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "uir", "inc": 640, "mid": 640, "out": 640, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 640, "out": 640, "heads": 12, "kd": 64, "vd": 64, "kvs": 2, "dwk": 3, "skip": True},
            {"t": "uir", "inc": 640, "mid": 1280, "out": 640, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 640, "out": 640, "heads": 12, "kd": 64, "vd": 64, "kvs": 2, "dwk": 3, "skip": True},
            {"t": "uir", "inc": 640, "mid": 1280, "out": 640, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 640, "out": 640, "heads": 12, "kd": 64, "vd": 64, "kvs": 2, "dwk": 3, "skip": True},
            {"t": "uir", "inc": 640, "mid": 1280, "out": 640, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 640, "out": 640, "heads": 12, "kd": 64, "vd": 64, "kvs": 2, "dwk": 3, "skip": True},
            {"t": "uir", "inc": 640, "mid": 1280, "out": 640, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 640, "out": 640, "heads": 12, "kd": 64, "vd": 64, "kvs": 2, "dwk": 3, "skip": True},
            {"t": "uir", "inc": 640, "mid": 1280, "out": 640, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 640, "out": 640, "heads": 12, "kd": 64, "vd": 64, "kvs": 2, "dwk": 3, "skip": True},
            {"t": "uir", "inc": 640, "mid": 1280, "out": 640, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 640, "out": 640, "heads": 12, "kd": 64, "vd": 64, "kvs": 2, "dwk": 3, "skip": True},
            {"t": "uir", "inc": 640, "mid": 1280, "out": 640, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 640, "out": 640, "heads": 12, "kd": 64, "vd": 64, "kvs": 2, "dwk": 3, "skip": True},
            {"t": "uir", "inc": 640, "mid": 1280, "out": 640, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 640, "out": 640, "heads": 12, "kd": 64, "vd": 64, "kvs": 2, "dwk": 3, "skip": True},
            {"t": "uir", "inc": 640, "mid": 1280, "out": 640, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 640, "out": 640, "heads": 12, "kd": 64, "vd": 64, "kvs": 2, "dwk": 3, "skip": True},
            {"t": "uir", "inc": 640, "mid": 1280, "out": 640, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 640, "out": 640, "heads": 12, "kd": 64, "vd": 64, "kvs": 2, "dwk": 3, "skip": True},
            {"t": "uir", "inc": 640, "mid": 1280, "out": 640, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 640, "out": 640, "heads": 12, "kd": 64, "vd": 64, "kvs": 2, "dwk": 3, "skip": True},
            {"t": "uir", "inc": 640, "mid": 1280, "out": 640, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 640, "out": 640, "heads": 12, "kd": 64, "vd": 64, "kvs": 2, "dwk": 3, "skip": True},
            {"t": "uir", "inc": 640, "mid": 1280, "out": 640, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 640, "out": 640, "heads": 12, "kd": 64, "vd": 64, "kvs": 2, "dwk": 3, "skip": True},
            {"t": "uir", "inc": 640, "mid": 1280, "out": 640, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
        ],
        [  # stage 3
            {"t": "uir", "inc": 640, "mid": 3840, "out": 1280, "dws_k": 5, "dws_s": 1, "dwm_k": 5, "dwm_s": 2, "skip": False, "ls": True},
            {"t": "mqa", "dim": 1280, "out": 1280, "heads": 16, "kd": 96, "vd": 96, "kvs": 1, "dwk": 0, "skip": True},
            {"t": "uir", "inc": 1280, "mid": 2560, "out": 1280, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 1280, "out": 1280, "heads": 16, "kd": 96, "vd": 96, "kvs": 1, "dwk": 0, "skip": True},
            {"t": "uir", "inc": 1280, "mid": 2560, "out": 1280, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 1280, "out": 1280, "heads": 16, "kd": 96, "vd": 96, "kvs": 1, "dwk": 0, "skip": True},
            {"t": "uir", "inc": 1280, "mid": 2560, "out": 1280, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 1280, "out": 1280, "heads": 16, "kd": 96, "vd": 96, "kvs": 1, "dwk": 0, "skip": True},
            {"t": "uir", "inc": 1280, "mid": 2560, "out": 1280, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 1280, "out": 1280, "heads": 16, "kd": 96, "vd": 96, "kvs": 1, "dwk": 0, "skip": True},
            {"t": "uir", "inc": 1280, "mid": 2560, "out": 1280, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 1280, "out": 1280, "heads": 16, "kd": 96, "vd": 96, "kvs": 1, "dwk": 0, "skip": True},
            {"t": "uir", "inc": 1280, "mid": 2560, "out": 1280, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 1280, "out": 1280, "heads": 16, "kd": 96, "vd": 96, "kvs": 1, "dwk": 0, "skip": True},
            {"t": "uir", "inc": 1280, "mid": 2560, "out": 1280, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 1280, "out": 1280, "heads": 16, "kd": 96, "vd": 96, "kvs": 1, "dwk": 0, "skip": True},
            {"t": "uir", "inc": 1280, "mid": 2560, "out": 1280, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 1280, "out": 1280, "heads": 16, "kd": 96, "vd": 96, "kvs": 1, "dwk": 0, "skip": True},
            {"t": "uir", "inc": 1280, "mid": 2560, "out": 1280, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 1280, "out": 1280, "heads": 16, "kd": 96, "vd": 96, "kvs": 1, "dwk": 0, "skip": True},
            {"t": "uir", "inc": 1280, "mid": 2560, "out": 1280, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 1280, "out": 1280, "heads": 16, "kd": 96, "vd": 96, "kvs": 1, "dwk": 0, "skip": True},
            {"t": "uir", "inc": 1280, "mid": 2560, "out": 1280, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 1280, "out": 1280, "heads": 16, "kd": 96, "vd": 96, "kvs": 1, "dwk": 0, "skip": True},
            {"t": "uir", "inc": 1280, "mid": 2560, "out": 1280, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 1280, "out": 1280, "heads": 16, "kd": 96, "vd": 96, "kvs": 1, "dwk": 0, "skip": True},
            {"t": "uir", "inc": 1280, "mid": 2560, "out": 1280, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 1280, "out": 1280, "heads": 16, "kd": 96, "vd": 96, "kvs": 1, "dwk": 0, "skip": True},
            {"t": "uir", "inc": 1280, "mid": 2560, "out": 1280, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 1280, "out": 1280, "heads": 16, "kd": 96, "vd": 96, "kvs": 1, "dwk": 0, "skip": True},
            {"t": "uir", "inc": 1280, "mid": 2560, "out": 1280, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 1280, "out": 1280, "heads": 16, "kd": 96, "vd": 96, "kvs": 1, "dwk": 0, "skip": True},
            {"t": "uir", "inc": 1280, "mid": 2560, "out": 1280, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 1280, "out": 1280, "heads": 16, "kd": 96, "vd": 96, "kvs": 1, "dwk": 0, "skip": True},
            {"t": "uir", "inc": 1280, "mid": 2560, "out": 1280, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 1280, "out": 1280, "heads": 16, "kd": 96, "vd": 96, "kvs": 1, "dwk": 0, "skip": True},
            {"t": "uir", "inc": 1280, "mid": 2560, "out": 1280, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
            {"t": "mqa", "dim": 1280, "out": 1280, "heads": 16, "kd": 96, "vd": 96, "kvs": 1, "dwk": 0, "skip": True},
            {"t": "uir", "inc": 1280, "mid": 2560, "out": 1280, "dws_k": 0, "dws_s": 1, "dwm_k": 0, "dwm_s": 1, "skip": True, "ls": True},
        ],
    ],
    "msfa": {"inc": 1920, "out": 2048, "res": [16, 16], "exp": 2.0, "indices": [3, 4]},
}
# fmt: on


def gelu(x):
    # timm MobileNet-V5 uses tanh-approximate GELU.
    return ops.gelu(x, approximate=True)


def make_conv(out_channels, kernel, stride, depthwise, bias, name):
    if depthwise:
        return layers.DepthwiseConv2D(
            kernel, strides=stride, padding="same", use_bias=bias, name=name
        )
    return layers.Conv2D(
        out_channels, kernel, strides=stride, padding="same", use_bias=bias, name=name
    )


@keras.saving.register_keras_serializable(package="zeromodels")
class MnvRmsNorm(layers.Layer):
    """timm RmsNorm2d / RmsNormAct2d over the channel axis (channels-last), with an
    optional GELU (``apply_act``)."""

    def __init__(self, apply_act=False, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.apply_act = apply_act
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight", shape=(input_shape[-1],), initializer="ones"
        )
        super().build(input_shape)

    def call(self, x):
        dtype = x.dtype
        x = ops.cast(x, "float32")
        x = x * ops.rsqrt(ops.mean(ops.square(x), axis=-1, keepdims=True) + self.eps)
        x = x * ops.cast(self.weight, "float32")
        x = ops.cast(x, dtype)
        if self.apply_act:
            x = gelu(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"apply_act": self.apply_act, "eps": self.eps})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class LayerScale2d(layers.Layer):
    def __init__(self, init_value=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=(input_shape[-1],),
            initializer=keras.initializers.Constant(self.init_value),
        )
        super().build(input_shape)

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update({"init_value": self.init_value})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class ConvNormAct(layers.Layer):
    """timm ConvNormAct: conv (``conv``) + RmsNorm(Act) (``bn``)."""

    def __init__(
        self,
        out_channels,
        kernel,
        stride,
        depthwise,
        apply_act,
        bias=False,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.depthwise = depthwise
        self.apply_act = apply_act
        self.bias = bias
        self.conv = make_conv(out_channels, kernel, stride, depthwise, bias, "conv")
        self.bn = MnvRmsNorm(apply_act=apply_act, eps=eps, name="bn")

    def call(self, x):
        return self.bn(self.conv(x))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_channels": self.out_channels,
                "kernel": self.kernel,
                "stride": self.stride,
                "depthwise": self.depthwise,
                "apply_act": self.apply_act,
                "bias": self.bias,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class EdgeResidual(layers.Layer):
    def __init__(self, spec, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.spec = spec
        self.conv_exp = make_conv(
            spec["mid"], spec["k"], spec["s"], False, spec["eb"], "conv_exp"
        )
        self.bn1 = MnvRmsNorm(apply_act=True, eps=eps, name="bn1")
        self.conv_pwl = make_conv(spec["out"], 1, 1, False, spec["pb"], "conv_pwl")
        self.bn2 = MnvRmsNorm(apply_act=False, eps=eps, name="bn2")

    def call(self, x):
        shortcut = x
        x = self.bn1(self.conv_exp(x))
        x = self.bn2(self.conv_pwl(x))
        if self.spec["skip"]:
            x = x + shortcut
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"spec": self.spec})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class UniversalInvertedResidual(layers.Layer):
    def __init__(self, spec, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.spec = spec
        self.dw_start = (
            ConvNormAct(
                spec["inc"],
                spec["dws_k"],
                spec["dws_s"],
                True,
                False,
                eps=eps,
                name="dw_start",
            )
            if spec["dws_k"]
            else None
        )
        self.pw_exp = ConvNormAct(
            spec["mid"], 1, 1, False, True, eps=eps, name="pw_exp"
        )
        self.dw_mid = (
            ConvNormAct(
                spec["mid"],
                spec["dwm_k"],
                spec["dwm_s"],
                True,
                True,
                eps=eps,
                name="dw_mid",
            )
            if spec["dwm_k"]
            else None
        )
        self.pw_proj = ConvNormAct(
            spec["out"], 1, 1, False, False, eps=eps, name="pw_proj"
        )
        self.layer_scale = LayerScale2d(name="layer_scale") if spec["ls"] else None

    def call(self, x):
        shortcut = x
        if self.dw_start is not None:
            x = self.dw_start(x)
        x = self.pw_exp(x)
        if self.dw_mid is not None:
            x = self.dw_mid(x)
        x = self.pw_proj(x)
        if self.layer_scale is not None:
            x = self.layer_scale(x)
        if self.spec["skip"]:
            x = x + shortcut
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"spec": self.spec})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MultiQueryAttention2d(layers.Layer):
    """timm MultiQueryAttention2d (multi-query, optional K/V spatial downsample),
    channels-last."""

    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        key_dim,
        value_dim,
        kv_stride,
        dw_kernel,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.kv_stride = kv_stride
        self.dw_kernel = dw_kernel
        self.scale = key_dim**-0.5

        self.query_proj = layers.Conv2D(
            num_heads * key_dim, 1, use_bias=False, name="query_proj"
        )
        self.key_down = None
        self.key_norm = None
        self.value_down = None
        self.value_norm = None
        if kv_stride > 1:
            self.key_down = layers.DepthwiseConv2D(
                dw_kernel,
                strides=kv_stride,
                padding="same",
                use_bias=False,
                name="key_down_conv",
            )
            self.key_norm = MnvRmsNorm(apply_act=False, eps=eps, name="key_norm")
            self.value_down = layers.DepthwiseConv2D(
                dw_kernel,
                strides=kv_stride,
                padding="same",
                use_bias=False,
                name="value_down_conv",
            )
            self.value_norm = MnvRmsNorm(apply_act=False, eps=eps, name="value_norm")
        self.key_proj = layers.Conv2D(key_dim, 1, use_bias=False, name="key_proj")
        self.value_proj = layers.Conv2D(value_dim, 1, use_bias=False, name="value_proj")
        self.output_proj = layers.Conv2D(dim_out, 1, use_bias=False, name="output_proj")

    def call(self, x):
        b = ops.shape(x)[0]
        h, w = int(x.shape[1]), int(x.shape[2])
        nh, kd, vd = self.num_heads, self.key_dim, self.value_dim

        q = self.query_proj(x)  # [B, H, W, nh*kd]
        q = ops.reshape(q, (b, h * w, nh, kd))
        q = ops.transpose(q, (0, 2, 1, 3))  # [B, nh, HW, kd]

        kx = x
        if self.kv_stride > 1:
            kx = self.key_norm(self.key_down(x))
        k = self.key_proj(kx)  # [B, H', W', kd]
        kp = int(k.shape[1]) * int(k.shape[2])
        k = ops.reshape(k, (b, kp, kd))  # [B, P, kd]

        vx = x
        if self.kv_stride > 1:
            vx = self.value_norm(self.value_down(x))
        v = self.value_proj(vx)
        v = ops.reshape(v, (b, kp, vd))  # [B, P, vd]

        q = q * self.scale
        # attn [B, nh, HW, P]
        attn = ops.matmul(q, ops.transpose(k, (0, 2, 1))[:, None])
        attn = ops.softmax(attn, axis=-1)
        o = ops.matmul(attn, v[:, None])  # [B, nh, HW, vd]
        o = ops.transpose(o, (0, 2, 1, 3))  # [B, HW, nh, vd]
        o = ops.reshape(o, (b, h, w, nh * vd))
        return self.output_proj(o)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "dim_out": self.dim_out,
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "value_dim": self.value_dim,
                "kv_stride": self.kv_stride,
                "dw_kernel": self.dw_kernel,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class MobileAttention(layers.Layer):
    def __init__(self, spec, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.spec = spec
        self.norm = MnvRmsNorm(apply_act=False, eps=eps, name="norm")
        self.attn = MultiQueryAttention2d(
            spec["dim"],
            spec["out"],
            spec["heads"],
            spec["kd"],
            spec["vd"],
            spec["kvs"],
            spec["dwk"],
            eps=eps,
            name="attn",
        )
        self.layer_scale = LayerScale2d(name="layer_scale")

    def call(self, x):
        shortcut = x
        x = self.layer_scale(self.attn(self.norm(x)))
        if self.spec["skip"]:
            x = x + shortcut
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"spec": self.spec})
        return config


def build_block(spec, eps, name):
    if spec["t"] == "er":
        return EdgeResidual(spec, eps=eps, name=name)
    if spec["t"] == "uir":
        return UniversalInvertedResidual(spec, eps=eps, name=name)
    return MobileAttention(spec, eps=eps, name=name)


@keras.saving.register_keras_serializable(package="zeromodels")
class MobileNetV5MSFA(layers.Layer):
    """Multi-scale fusion adapter: upsample + concat the last two feature maps, a
    UIR FFN, pool to ``output_resolution``, and RmsNorm."""

    def __init__(self, spec, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.spec = spec
        self.out_res = tuple(spec["res"])
        # The MSFA fuser UIB has no layer scale (timm layer_scale_init_value=None).
        ffn_spec = {
            "inc": spec["inc"],
            "mid": int(spec["inc"] * spec["exp"]),
            "out": spec["out"],
            "dws_k": 0,
            "dws_s": 1,
            "dwm_k": 0,
            "dwm_s": 1,
            "skip": False,
            "ls": False,
        }
        self.ffn = UniversalInvertedResidual(ffn_spec, eps=eps, name="ffn")
        self.norm = MnvRmsNorm(apply_act=False, eps=eps, name="norm")

    def call(self, inputs):
        high = (int(inputs[0].shape[1]), int(inputs[0].shape[2]))
        resized = []
        for img in inputs:
            ih, iw = int(img.shape[1]), int(img.shape[2])
            if ih < high[0] or iw < high[1]:
                if high[0] % ih == 0 and high[1] % iw == 0:
                    # torch `nearest` upsample for an integer factor is an exact
                    # pixel repeat (no half-pixel centering).
                    img = ops.repeat(img, high[0] // ih, axis=1)
                    img = ops.repeat(img, high[1] // iw, axis=2)
                else:
                    img = ops.image.resize(
                        img,
                        high,
                        interpolation="nearest",
                        data_format="channels_last",
                    )
            resized.append(img)
        x = ops.concatenate(resized, axis=-1)
        x = self.ffn(x)
        if high != self.out_res:
            if high[0] % self.out_res[0] == 0 and high[1] % self.out_res[1] == 0:
                sh, sw = high[0] // self.out_res[0], high[1] // self.out_res[1]
                x = ops.nn.average_pool(
                    x,
                    (sh, sw),
                    (sh, sw),
                    padding="valid",
                    data_format="channels_last",
                )
            else:
                x = ops.image.resize(
                    x,
                    self.out_res,
                    interpolation="bilinear",
                    data_format="channels_last",
                )
        return self.norm(x)

    def get_config(self):
        config = super().get_config()
        config.update({"spec": self.spec})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3nMultimodalEmbedder(layers.Layer):
    """Projects hard soft-token ids or soft feature vectors into text space.

    Hard ids (``input_ids``, in ``[vocab_offset, vocab_offset + vocab_size)``) hit a
    small embedding table then ``hard_embedding_norm``; soft features
    (``inputs_embeds`` from a modality encoder) hit ``soft_embedding_norm``. Both
    then go through a linear projection to ``text_hidden_size`` and a scaleless
    post-projection norm."""

    def __init__(
        self,
        multimodal_hidden_size,
        text_hidden_size,
        vocab_size,
        vocab_offset,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.multimodal_hidden_size = multimodal_hidden_size
        self.text_hidden_size = text_hidden_size
        self.vocab_size = vocab_size
        self.vocab_offset = vocab_offset
        self.eps = eps
        self.embedding = layers.Embedding(
            vocab_size, multimodal_hidden_size, name="embedding"
        )
        self.hard_embedding_norm = Gemma3nRMSNorm(eps=eps, name="hard_embedding_norm")
        self.soft_embedding_norm = Gemma3nRMSNorm(eps=eps, name="soft_embedding_norm")
        self.embedding_projection = layers.Dense(
            text_hidden_size, use_bias=False, name="embedding_projection"
        )
        self.embedding_post_projection_norm = Gemma3nRMSNorm(
            eps=eps, with_scale=False, name="embedding_post_projection_norm"
        )

    def build(self, input_shape=None):
        # The two call paths (input_ids / inputs_embeds) take different inputs, so
        # Keras cannot infer the sub-layer shapes; build them explicitly (their
        # widths are fixed) to create the weights and avoid an "unbuilt state"
        # warning during weight transfer.
        mm = (None, self.multimodal_hidden_size)
        self.embedding.build((None, None))
        self.hard_embedding_norm.build(mm)
        self.soft_embedding_norm.build(mm)
        self.embedding_projection.build(mm)
        self.embedding_post_projection_norm.build((None, self.text_hidden_size))
        super().build(input_shape)

    def call(self, input_ids=None, inputs_embeds=None):
        if inputs_embeds is not None:
            emb_norm = self.soft_embedding_norm(inputs_embeds)
        else:
            hard = self.embedding(input_ids - self.vocab_offset)
            emb_norm = self.hard_embedding_norm(hard)
        proj = self.embedding_projection(emb_norm)
        return self.embedding_post_projection_norm(proj)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "multimodal_hidden_size": self.multimodal_hidden_size,
                "text_hidden_size": self.text_hidden_size,
                "vocab_size": self.vocab_size,
                "vocab_offset": self.vocab_offset,
                "eps": self.eps,
            }
        )
        return config
