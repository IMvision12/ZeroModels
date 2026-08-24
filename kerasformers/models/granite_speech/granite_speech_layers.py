import math

import keras
from keras import layers, ops

MASK_NEG = -1e9


def rotate_half(x):
    half = ops.shape(x)[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return ops.concatenate([-x2, x1], axis=-1)


def apply_rope(q, k, cos, sin):
    # cos/sin are built in float32, so multiplying a bf16 q/k by them silently
    # promoted the result to float32 and the bf16 KV cache then refused the
    # write ("Index put requires the source and destination dtypes match").
    # Rotate in float32 for precision, then hand back the dtype we were given.
    dtype = q.dtype
    cos = ops.expand_dims(cos, axis=1)
    sin = ops.expand_dims(sin, axis=1)
    q32 = ops.cast(q, "float32")
    k32 = ops.cast(k, "float32")
    q_embed = ops.cast(q32 * cos + rotate_half(q32) * sin, dtype)
    k_embed = ops.cast(k32 * cos + rotate_half(k32) * sin, dtype)
    return q_embed, k_embed


@keras.saving.register_keras_serializable(package="kerasformers")
class GraniteSpeechRMSNorm(layers.Layer):
    """RMSNorm (Llama/Granite style): normalize by RMS in float32, then scale.

    Args:
        eps: Variance epsilon. Defaults to ``1e-6``.
    """

    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
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
        return self.weight * ops.cast(x, dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class GraniteSpeechLoRADense(layers.Layer):
    """Bias-free Dense with an optional, conditionally-applied LoRA branch.

    Granite Speech's text decoder carries a LoRA adapter on the query and value
    projections that is only meant to be active when audio is present (HF toggles
    ``enable_adapters`` / ``disable_adapters`` in ``generate``). Here the base
    ``kernel`` is always applied; the low-rank ``lora_A`` / ``lora_B`` delta is
    added only when ``apply_lora`` is passed ``True`` at call time, scaled by
    ``alpha / rank``.

    Args:
        units: Output width.
        rank: LoRA rank (``0`` disables the adapter entirely).
        alpha: LoRA alpha; the delta is scaled by ``alpha / rank``.
    """

    def __init__(self, units, rank=0, alpha=1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.rank = rank
        self.alpha = alpha
        self.scaling = (alpha / rank) if rank else 0.0

    def build(self, input_shape):
        in_dim = input_shape[-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(in_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        if self.rank:
            self.lora_a = self.add_weight(
                name="lora_a",
                shape=(in_dim, self.rank),
                initializer="glorot_uniform",
                trainable=True,
            )
            self.lora_b = self.add_weight(
                name="lora_b",
                shape=(self.rank, self.units),
                initializer="zeros",
                trainable=True,
            )
        self.built = True

    def call(self, x, apply_lora=False):
        out = ops.matmul(x, self.kernel)
        if self.rank and apply_lora:
            out = out + self.scaling * ops.matmul(
                ops.matmul(x, self.lora_a), self.lora_b
            )
        return out

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "rank": self.rank, "alpha": self.alpha})
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class GraniteSpeechMLP(layers.Layer):
    """SwiGLU MLP (bias-free): ``down(silu(gate(x)) * up(x))``."""

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
class GraniteSpeechAttention(layers.Layer):
    """Grouped-query causal self-attention with RoPE and Granite's attention
    multiplier (a fixed ``scaling`` from ``config.attention_multiplier`` instead
    of ``head_dim ** -0.5``).

    Query/value projections carry an optional LoRA branch (active only when
    ``apply_lora`` is set); key/output projections are plain bias-free Denses. A
    KV cache can be threaded through ``past_key_value`` for incremental decoding.

    Call args:
        hidden_states: ``(batch, seq, hidden)``.
        cos, sin: rotary tables ``(batch, seq, head_dim)``.
        attention_mask: additive mask broadcastable to ``(batch, 1, q_len, kv_len)``.
        past_key_value: optional ``(past_k, past_v)``.
        use_cache: when ``True``, also return the updated ``(k, v)``.
        apply_lora: when ``True``, add the q/v LoRA deltas.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        attention_multiplier,
        lora_rank=0,
        lora_alpha=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.attention_multiplier = attention_multiplier
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = attention_multiplier

        self.query = GraniteSpeechLoRADense(
            num_heads * head_dim, rank=lora_rank, alpha=lora_alpha, name="query"
        )
        self.key = layers.Dense(num_kv_heads * head_dim, use_bias=False, name="key")
        self.value = GraniteSpeechLoRADense(
            num_kv_heads * head_dim, rank=lora_rank, alpha=lora_alpha, name="value"
        )
        self.output_proj = layers.Dense(embed_dim, use_bias=False, name="output_proj")

    def split_heads(self, x, num_heads):
        b = ops.shape(x)[0]
        s = ops.shape(x)[1]
        x = ops.reshape(x, (b, s, num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def call(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
        apply_lora=False,
    ):
        b = ops.shape(hidden_states)[0]
        q_len = ops.shape(hidden_states)[1]

        query = self.split_heads(
            self.query(hidden_states, apply_lora=apply_lora), self.num_heads
        )
        key = self.split_heads(self.key(hidden_states), self.num_kv_heads)
        value = self.split_heads(
            self.value(hidden_states, apply_lora=apply_lora), self.num_kv_heads
        )

        query, key = apply_rope(query, key, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            key = ops.concatenate([past_k, key], axis=2)
            value = ops.concatenate([past_v, value], axis=2)
        new_key_value = (key, value) if use_cache else None

        if self.num_kv_groups > 1:
            key = ops.repeat(key, self.num_kv_groups, axis=1)
            value = ops.repeat(value, self.num_kv_groups, axis=1)

        attn = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2))) * self.scaling
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), query.dtype)
        out = ops.matmul(attn, value)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, q_len, self.num_heads * self.head_dim))
        out = self.output_proj(out)
        return (out, new_key_value) if use_cache else out

    def decode_step(
        self,
        hidden_states,
        cos,
        sin,
        cache_k,
        cache_v,
        write_pos,
        key_mask,
        apply_lora=False,
    ):
        b = ops.shape(hidden_states)[0]
        query = self.split_heads(
            self.query(hidden_states, apply_lora=apply_lora), self.num_heads
        )
        key = self.split_heads(self.key(hidden_states), self.num_kv_heads)
        value = self.split_heads(
            self.value(hidden_states, apply_lora=apply_lora), self.num_kv_heads
        )
        query, key = apply_rope(query, key, cos, sin)
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
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, 1, self.num_heads * self.head_dim))
        return self.output_proj(out), cache_k, cache_v

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "attention_multiplier": self.attention_multiplier,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class GraniteSpeechDecoderLayer(layers.Layer):
    """One Granite decoder block: pre-norm GQA attention then pre-norm SwiGLU,
    each residual scaled by ``residual_multiplier``."""

    def __init__(
        self,
        embed_dim,
        mlp_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        norm_eps,
        attention_multiplier,
        residual_multiplier,
        lora_rank=0,
        lora_alpha=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.attention_multiplier = attention_multiplier
        self.residual_multiplier = residual_multiplier
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        self.input_layernorm = GraniteSpeechRMSNorm(
            eps=norm_eps, name="input_layernorm"
        )
        self.attention = GraniteSpeechAttention(
            embed_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            attention_multiplier,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            name="attention",
        )
        self.post_attention_layernorm = GraniteSpeechRMSNorm(
            eps=norm_eps, name="post_attention_layernorm"
        )
        self.mlp = GraniteSpeechMLP(embed_dim, mlp_dim, name="mlp")

    def call(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
        apply_lora=False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.attention(
            hidden_states,
            cos,
            sin,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            apply_lora=apply_lora,
        )
        if use_cache:
            attn_out, new_key_value = attn_out
        else:
            new_key_value = None
        hidden_states = residual + attn_out * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states) * self.residual_multiplier
        return (hidden_states, new_key_value) if use_cache else hidden_states

    def compute_output_spec(
        self,
        hidden_states,
        cos,
        sin,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
        apply_lora=False,
    ):
        return keras.KerasTensor(hidden_states.shape, dtype=self.compute_dtype)

    def decode_step(
        self,
        hidden_states,
        cos,
        sin,
        cache_k,
        cache_v,
        write_pos,
        key_mask,
        apply_lora=False,
    ):
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        attn_out, cache_k, cache_v = self.attention.decode_step(
            x, cos, sin, cache_k, cache_v, write_pos, key_mask, apply_lora=apply_lora
        )
        hidden_states = residual + attn_out * self.residual_multiplier
        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(x) * self.residual_multiplier
        return hidden_states, cache_k, cache_v

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "norm_eps": self.norm_eps,
                "attention_multiplier": self.attention_multiplier,
                "residual_multiplier": self.residual_multiplier,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class GraniteSpeechConformerFeedForward(layers.Layer):
    """Conformer feed-forward: ``down(silu(up(layernorm(x))))`` (biased Denses)."""

    def __init__(self, hidden_dim, feedforward_mult, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.feedforward_mult = feedforward_mult
        self.pre_norm = layers.LayerNormalization(epsilon=1e-5, name="pre_norm")
        self.up_proj = layers.Dense(hidden_dim * feedforward_mult, name="up_proj")
        self.down_proj = layers.Dense(hidden_dim, name="down_proj")

    def call(self, x):
        x = self.pre_norm(x)
        x = ops.silu(self.up_proj(x))
        return self.down_proj(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"hidden_dim": self.hidden_dim, "feedforward_mult": self.feedforward_mult}
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class GraniteSpeechConformerAttention(layers.Layer):
    """Block-local conformer attention with Shaw relative positional embeddings.

    The feature sequence is split into fixed ``context_size`` blocks; within each
    block, full attention is computed with a learned relative-position bias added
    to the scores (``pos_attn = einsum(q, rel_pos_emb) * scale``). A right-pad
    fills the final partial block and is masked out. ``attention_dists`` (the
    clamped relative-position index matrix) is supplied by the encoder.

    Call args:
        hidden_states: ``(batch, num_features, hidden_dim)``.
        rel_pos_emb: gathered relative-position table ``(context, context, dim_head)``.
    """

    def __init__(
        self, hidden_dim, num_heads, dim_head, context_size, max_pos_emb, **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.context_size = context_size
        self.max_pos_emb = max_pos_emb
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.pre_norm = layers.LayerNormalization(epsilon=1e-5, name="pre_norm")
        self.to_q = layers.Dense(inner_dim, use_bias=False, name="to_q")
        self.to_kv = layers.Dense(inner_dim * 2, use_bias=False, name="to_kv")
        self.to_out = layers.Dense(hidden_dim, name="to_out")

    def build(self, input_shape):
        self.rel_pos_emb = self.add_weight(
            name="rel_pos_emb",
            shape=(2 * self.max_pos_emb + 1, self.dim_head),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.built = True

    def call(self, hidden_states):
        c = self.context_size
        seq = ops.arange(c)
        dists = ops.cast(
            ops.clip(seq[:, None] - seq[None, :], -c, c) + self.max_pos_emb, "int32"
        )
        rel_pos_emb = ops.take(self.rel_pos_emb, dists, axis=0)
        hidden_states = self.pre_norm(hidden_states)
        bsz = ops.shape(hidden_states)[0]
        num_features = ops.shape(hidden_states)[1]
        c = self.context_size

        num_blocks = ops.cast(ops.ceil(num_features / c), "int32")
        padded = num_blocks * c
        pad = padded - num_features
        hidden_states = ops.pad(hidden_states, [[0, 0], [0, pad], [0, 0]])

        query = self.to_q(hidden_states)
        kv = self.to_kv(hidden_states)
        key, value = ops.split(kv, 2, axis=-1)

        def to_blocks(t):
            t = ops.reshape(t, (bsz, num_blocks, c, self.num_heads, self.dim_head))
            return ops.transpose(t, (0, 1, 3, 2, 4))

        query = to_blocks(query)
        key = to_blocks(key)
        value = to_blocks(value)

        pos_attn = ops.einsum("bmhid,ijd->bmhij", query, rel_pos_emb) * self.scale

        attn = ops.matmul(query, ops.transpose(key, (0, 1, 2, 4, 3))) * self.scale
        attn = attn + pos_attn

        valid_key = ops.arange(c) < (c - pad)
        is_last = ops.arange(num_blocks) == (num_blocks - 1)
        block_key_mask = ops.where(
            ops.logical_and(is_last[:, None], ops.logical_not(valid_key)[None, :]),
            MASK_NEG,
            0.0,
        )
        attn = attn + ops.cast(block_key_mask, attn.dtype)[None, :, None, None, :]

        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), query.dtype)
        out = ops.matmul(attn, value)
        out = ops.transpose(out, (0, 1, 3, 2, 4))
        out = ops.reshape(out, (bsz, padded, self.num_heads * self.dim_head))
        out = out[:, :num_features, :]
        return self.to_out(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "dim_head": self.dim_head,
                "context_size": self.context_size,
                "max_pos_emb": self.max_pos_emb,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class GraniteSpeechConformerConvModule(layers.Layer):
    """Conformer conv module: LN -> pointwise up-conv -> GLU -> depthwise conv ->
    batch-norm + SiLU -> pointwise down-conv.

    All convolutions operate over the time axis; pointwise (1x1) convs are
    Denses over the channel axis, the depthwise conv is a grouped 1D conv with
    one kernel per channel (kept as a manual ``conv1d`` so the weight maps
    directly from PyTorch's ``(channels, 1, kernel)`` layout).
    """

    def __init__(self, hidden_dim, conv_expansion_factor, conv_kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.conv_expansion_factor = conv_expansion_factor
        self.conv_kernel_size = conv_kernel_size
        self.inner_dim = hidden_dim * conv_expansion_factor
        pad = conv_kernel_size // 2
        pad_offset = (conv_kernel_size + 1) % 2
        self.left_pad = pad
        self.right_pad = pad - pad_offset

        self.norm = layers.LayerNormalization(epsilon=1e-5, name="norm")
        self.up_conv = layers.Dense(self.inner_dim * 2, name="up_conv")
        self.down_conv = layers.Dense(hidden_dim, name="down_conv")
        self.batch_norm = layers.BatchNormalization(
            axis=-1, momentum=0.9, epsilon=1e-5, name="batch_norm"
        )

    def build(self, input_shape):
        self.depth_kernel = self.add_weight(
            name="depth_kernel",
            shape=(self.conv_kernel_size, self.inner_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.built = True

    def call(self, x, training=False):
        x = self.norm(x)
        x = self.up_conv(x)
        a, b = ops.split(x, 2, axis=-1)
        x = a * ops.sigmoid(b)

        x = ops.pad(x, [[0, 0], [self.left_pad, self.right_pad], [0, 0]])
        kernel = self.depth_kernel[:, :, None]
        x = ops.depthwise_conv(
            x, kernel, strides=1, padding="valid", data_format="channels_last"
        )

        x = self.batch_norm(x, training=training)
        x = ops.silu(x)
        x = self.down_conv(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "conv_expansion_factor": self.conv_expansion_factor,
                "conv_kernel_size": self.conv_kernel_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class GraniteSpeechConformerBlock(layers.Layer):
    """One conformer block: ``0.5*ff1 -> attn -> conv -> 0.5*ff2 -> post-norm``,
    each sub-module a residual add (FFs are half-step macaron residuals)."""

    def __init__(
        self,
        hidden_dim,
        feedforward_mult,
        num_heads,
        dim_head,
        context_size,
        max_pos_emb,
        conv_expansion_factor,
        conv_kernel_size,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.feedforward_mult = feedforward_mult
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.context_size = context_size
        self.max_pos_emb = max_pos_emb
        self.conv_expansion_factor = conv_expansion_factor
        self.conv_kernel_size = conv_kernel_size

        self.ff1 = GraniteSpeechConformerFeedForward(
            hidden_dim, feedforward_mult, name="ff1"
        )
        self.attn = GraniteSpeechConformerAttention(
            hidden_dim, num_heads, dim_head, context_size, max_pos_emb, name="attn"
        )
        self.conv = GraniteSpeechConformerConvModule(
            hidden_dim, conv_expansion_factor, conv_kernel_size, name="conv"
        )
        self.ff2 = GraniteSpeechConformerFeedForward(
            hidden_dim, feedforward_mult, name="ff2"
        )
        self.post_norm = layers.LayerNormalization(epsilon=1e-5, name="post_norm")

    def call(self, x, training=False):
        x = 0.5 * self.ff1(x) + x
        x = self.attn(x) + x
        x = self.conv(x, training=training) + x
        x = 0.5 * self.ff2(x) + x
        return self.post_norm(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "feedforward_mult": self.feedforward_mult,
                "num_heads": self.num_heads,
                "dim_head": self.dim_head,
                "context_size": self.context_size,
                "max_pos_emb": self.max_pos_emb,
                "conv_expansion_factor": self.conv_expansion_factor,
                "conv_kernel_size": self.conv_kernel_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class GraniteSpeechCTCEncoder(layers.Layer):
    """Conformer CTC audio encoder.

    ``input_linear -> num_layers x conformer block -> (mid CTC injection)``. At the
    midpoint layer, the running hidden state gets a softmax-CTC projection added
    back through ``out`` / ``out_mid`` (this is how Granite Speech threads CTC
    supervision into the encoder). When ``cat_hidden_layers`` is set
    (GraniteSpeechPlus), the listed intermediate layer outputs are concatenated
    with the final output along the feature axis before being returned.

    Call args:
        input_features: ``(batch, num_features, input_dim)`` stacked mel frames.

    Returns:
        ``(batch, num_features, hidden_dim)`` (or ``hidden_dim * (len(cat)+1)``
        for the Plus concatenation).
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        feedforward_mult,
        num_heads,
        dim_head,
        output_dim,
        context_size,
        max_pos_emb,
        conv_expansion_factor,
        conv_kernel_size,
        cat_hidden_layers=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.feedforward_mult = feedforward_mult
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.output_dim = output_dim
        self.context_size = context_size
        self.max_pos_emb = max_pos_emb
        self.conv_expansion_factor = conv_expansion_factor
        self.conv_kernel_size = conv_kernel_size
        self.cat_hidden_layers = list(cat_hidden_layers) if cat_hidden_layers else None

        self.input_linear = layers.Dense(hidden_dim, name="input_linear")
        self.conformer_layers = [
            GraniteSpeechConformerBlock(
                hidden_dim,
                feedforward_mult,
                num_heads,
                dim_head,
                context_size,
                max_pos_emb,
                conv_expansion_factor,
                conv_kernel_size,
                name=f"conformer_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.out = layers.Dense(output_dim, name="out")
        self.out_mid = layers.Dense(hidden_dim, name="out_mid")

    def call(self, input_features, training=False):
        hidden = self.input_linear(input_features)

        cat_layers = set(self.cat_hidden_layers or [])
        exported = []
        if 0 in cat_layers:
            exported.append(hidden)

        for idx, layer in enumerate(self.conformer_layers, start=1):
            hidden = layer(hidden, training=training)
            if idx in cat_layers:
                exported.append(hidden)
            if idx == self.num_layers // 2:
                mid = self.out(hidden)
                hidden = hidden + self.out_mid(ops.softmax(mid, axis=-1))

        if exported:
            hidden = ops.concatenate([*exported, hidden], axis=-1)
        return hidden

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "feedforward_mult": self.feedforward_mult,
                "num_heads": self.num_heads,
                "dim_head": self.dim_head,
                "output_dim": self.output_dim,
                "context_size": self.context_size,
                "max_pos_emb": self.max_pos_emb,
                "conv_expansion_factor": self.conv_expansion_factor,
                "conv_kernel_size": self.conv_kernel_size,
                "cat_hidden_layers": self.cat_hidden_layers,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class GraniteSpeechQFormerMultiHeadAttention(layers.Layer):
    """BERT-style multi-head attention used inside the Q-Former. Self-attention
    when ``encoder_hidden_states`` is ``None``, else cross-attention onto the
    audio features (keys/values projected from ``encoder_hidden_size``)."""

    def __init__(self, hidden_size, num_heads, encoder_hidden_size=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.encoder_hidden_size = encoder_hidden_size
        self.head_size = hidden_size // num_heads
        kv_in = encoder_hidden_size if encoder_hidden_size is not None else hidden_size
        self.kv_in = kv_in
        self.query = layers.Dense(hidden_size, name="query")
        self.key = layers.Dense(hidden_size, name="key")
        self.value = layers.Dense(hidden_size, name="value")

    def transpose_for_scores(self, x):
        b = ops.shape(x)[0]
        s = ops.shape(x)[1]
        x = ops.reshape(x, (b, s, self.num_heads, self.head_size))
        return ops.transpose(x, (0, 2, 1, 3))

    def call(self, hidden_states, encoder_hidden_states=None):
        context = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(context))
        value = self.transpose_for_scores(self.value(context))

        attn = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2)))
        attn = attn / math.sqrt(self.head_size)
        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), query.dtype)
        out = ops.matmul(attn, value)
        out = ops.transpose(out, (0, 2, 1, 3))
        b = ops.shape(out)[0]
        s = ops.shape(out)[1]
        return ops.reshape(out, (b, s, self.hidden_size))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "encoder_hidden_size": self.encoder_hidden_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class GraniteSpeechQFormerAttention(layers.Layer):
    """Q-Former attention sublayer: MHA + a residual ``dense -> LayerNorm`` output."""

    def __init__(
        self, hidden_size, num_heads, layer_norm_eps, encoder_hidden_size=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layer_norm_eps = layer_norm_eps
        self.encoder_hidden_size = encoder_hidden_size
        self.attention = GraniteSpeechQFormerMultiHeadAttention(
            hidden_size,
            num_heads,
            encoder_hidden_size=encoder_hidden_size,
            name="attention",
        )
        self.dense = layers.Dense(hidden_size, name="dense")
        self.layer_norm = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="LayerNorm"
        )

    def call(self, hidden_states, encoder_hidden_states=None):
        attn = self.attention(
            hidden_states, encoder_hidden_states=encoder_hidden_states
        )
        attn = self.dense(attn)
        return self.layer_norm(attn + hidden_states)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "layer_norm_eps": self.layer_norm_eps,
                "encoder_hidden_size": self.encoder_hidden_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class GraniteSpeechQFormerLayer(layers.Layer):
    """One Q-Former layer: query self-attention, (optional) cross-attention onto
    the audio features, then a gated feed-forward, all over the query tokens.

    The Granite Speech projector runs the Q-Former with query tokens only (no
    text), so only the ``*_query`` feed-forward path is used.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        encoder_hidden_size,
        layer_norm_eps,
        has_cross_attention,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.encoder_hidden_size = encoder_hidden_size
        self.layer_norm_eps = layer_norm_eps
        self.has_cross_attention = has_cross_attention

        self.attention = GraniteSpeechQFormerAttention(
            hidden_size, num_heads, layer_norm_eps, name="attention"
        )
        if has_cross_attention:
            self.crossattention = GraniteSpeechQFormerAttention(
                hidden_size,
                num_heads,
                layer_norm_eps,
                encoder_hidden_size=encoder_hidden_size,
                name="crossattention",
            )
        self.intermediate_query = layers.Dense(
            intermediate_size, name="intermediate_query"
        )
        self.output_query_dense = layers.Dense(hidden_size, name="output_query_dense")
        self.output_query_norm = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="output_query_LayerNorm"
        )

    def call(self, hidden_states, encoder_hidden_states):
        attn = self.attention(hidden_states)
        if self.has_cross_attention:
            attn = self.crossattention(
                attn, encoder_hidden_states=encoder_hidden_states
            )
        intermediate = ops.gelu(self.intermediate_query(attn), approximate=False)
        out = self.output_query_dense(intermediate)
        return self.output_query_norm(out + attn)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "encoder_hidden_size": self.encoder_hidden_size,
                "layer_norm_eps": self.layer_norm_eps,
                "has_cross_attention": self.has_cross_attention,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class GraniteSpeechEncoderProjector(layers.Layer):
    """Windowed BLIP-2 Q-Former projector: audio features -> LLM-width embeddings.

    The encoder output is chunked into ``window_size`` blocks; a fixed set of
    ``num_queries = window_size // downsample_rate`` learned query tokens cross-
    attend (per block) to each window via a small Q-Former, and the pooled query
    outputs are linearly projected to the text decoder's hidden size. Output
    length is ``nblocks * num_queries``.

    Call args:
        hidden_states: ``(batch, seq, encoder_hidden_size)`` encoder output.

    Returns:
        ``(batch, nblocks * num_queries, text_hidden_size)``.
    """

    def __init__(
        self,
        hidden_size,
        text_hidden_size,
        encoder_hidden_size,
        num_layers,
        num_heads,
        intermediate_size,
        cross_attention_frequency,
        layer_norm_eps,
        window_size,
        downsample_rate,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.text_hidden_size = text_hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.cross_attention_frequency = cross_attention_frequency
        self.layer_norm_eps = layer_norm_eps
        self.window_size = window_size
        self.downsample_rate = downsample_rate
        self.num_queries = window_size // downsample_rate

        self.layernorm = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="layernorm"
        )
        self.qformer_layers = [
            GraniteSpeechQFormerLayer(
                hidden_size,
                num_heads,
                intermediate_size,
                encoder_hidden_size,
                layer_norm_eps,
                has_cross_attention=(i % cross_attention_frequency == 0),
                name=f"qformer_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.linear = layers.Dense(text_hidden_size, name="linear")

    def build(self, input_shape):
        self.query = self.add_weight(
            name="query",
            shape=(1, self.num_queries, self.hidden_size),
            initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            trainable=True,
        )
        self.built = True

    def call(self, hidden_states):
        batch = ops.shape(hidden_states)[0]
        seq = ops.shape(hidden_states)[1]
        dim = ops.shape(hidden_states)[2]
        w = self.window_size

        nblocks = ops.cast(ops.ceil(seq / w), "int32")
        pad = nblocks * w - seq
        hidden_states = ops.pad(hidden_states, [[0, 0], [0, pad], [0, 0]])
        hidden_states = ops.reshape(hidden_states, (batch * nblocks, w, dim))

        query = ops.broadcast_to(
            self.query, (batch * nblocks, self.num_queries, self.hidden_size)
        )
        query = self.layernorm(query)
        for layer in self.qformer_layers:
            query = layer(query, hidden_states)

        out_len = nblocks * w // self.downsample_rate
        query = ops.reshape(query, (batch, out_len, self.hidden_size))
        return self.linear(query)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "text_hidden_size": self.text_hidden_size,
                "encoder_hidden_size": self.encoder_hidden_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "cross_attention_frequency": self.cross_attention_frequency,
                "layer_norm_eps": self.layer_norm_eps,
                "window_size": self.window_size,
                "downsample_rate": self.downsample_rate,
            }
        )
        return config
