import keras
from keras import layers, ops

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="zeromodels")
class DownsampleMask(layers.Layer):
    """Halve a ``(batch, time)`` padding mask: a half-rate frame is valid iff both
    source frames are valid (a trailing odd frame is dropped, mirroring the pooled
    residual of the subsampling block). Wrapped as a layer so the dynamic
    ``time // 2`` reshape is traced at call time rather than during graph build."""

    def call(self, attention_mask):
        length = ops.shape(attention_mask)[1]
        half = length // 2
        trimmed = attention_mask[:, : 2 * half]
        pairs = ops.reshape(trimmed, (ops.shape(attention_mask)[0], half, 2))
        return ops.cast(ops.all(ops.cast(pairs, "bool"), axis=2), attention_mask.dtype)

    def compute_output_shape(self, input_shape):
        # Some backends (TF) call this with input_shape=None during the symbolic
        # pass; the halved mask is dynamic either way, so (None, None) is correct.
        if input_shape is None:
            return (None, None)
        batch, time = input_shape
        return (batch, None if time is None else time // 2)


@keras.saving.register_keras_serializable(package="zeromodels")
class GraniteSpeech5FeedForward(layers.Layer):
    """Conformer feed-forward: ``linear2(silu(linear1(x)))`` (both biased). The
    pre-norm lives in the block, not here."""

    def __init__(self, hidden_size, intermediate_size, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.use_bias = use_bias
        self.linear1 = layers.Dense(
            intermediate_size, use_bias=use_bias, name="linear1"
        )
        self.linear2 = layers.Dense(hidden_size, use_bias=use_bias, name="linear2")

    def build(self, input_shape):
        self.linear1.build(input_shape)
        self.linear2.build((*input_shape[:-1], self.intermediate_size))
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        return self.linear2(ops.silu(self.linear1(x)))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "use_bias": self.use_bias,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GraniteSpeech5Attention(layers.Layer):
    """Block-wise conformer self-attention with Shaw's relative positional bias.

    The time axis is right-padded to a whole number of ``context_size`` blocks and
    folded so every block attends independently. Within a block the scores are
    ``(q @ k^T + q @ rel) * scaling``, where ``rel`` gathers a learned relative-
    position table by the clamped distance matrix. Keys that fall on padded frames
    (per the ``attention_mask``, block padding included) are masked out.

    Call args:
        hidden_states: ``(batch, time, hidden_size)`` (already pre-normed).
        attention_mask: ``(batch, time)`` int/bool padding mask, or ``None``.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        head_dim,
        context_size,
        max_position_embeddings,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.context_size = context_size
        self.max_position_embeddings = max_position_embeddings
        self.scaling = head_dim**-0.5
        inner = num_heads * head_dim
        self.q_proj = layers.Dense(inner, use_bias=False, name="q_proj")
        self.k_proj = layers.Dense(inner, use_bias=False, name="k_proj")
        self.v_proj = layers.Dense(inner, use_bias=False, name="v_proj")
        self.o_proj = layers.Dense(hidden_size, use_bias=True, name="o_proj")

    def build(self, input_shape):
        self.q_proj.build(input_shape)
        self.k_proj.build(input_shape)
        self.v_proj.build(input_shape)
        self.o_proj.build((*input_shape[:-1], self.num_heads * self.head_dim))
        self.rel_pos_emb = self.add_weight(
            name="rel_pos_emb",
            shape=(2 * self.max_position_embeddings + 1, self.head_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, hidden_states, attention_mask=None):
        c = self.context_size
        h = self.num_heads
        d = self.head_dim
        batch = ops.shape(hidden_states)[0]
        seq = ops.shape(hidden_states)[1]

        num_blocks = ops.cast(ops.ceil(seq / c), "int32")
        padded = num_blocks * c
        pad = padded - seq
        hidden_states = ops.pad(hidden_states, [[0, 0], [0, pad], [0, 0]])

        # clamped relative-distance index matrix -> gathered (c, c, head_dim) table
        rng = ops.arange(c)
        dists = ops.cast(
            ops.clip(rng[:, None] - rng[None, :], -c, c) + self.max_position_embeddings,
            "int32",
        )
        rel = ops.take(self.rel_pos_emb, dists, axis=0)

        def to_blocks(x):
            x = ops.reshape(x, (batch, num_blocks, c, h, d))
            return ops.transpose(x, (0, 1, 3, 2, 4))  # (b, nblk, h, c, d)

        query = to_blocks(self.q_proj(hidden_states))
        key = to_blocks(self.k_proj(hidden_states))
        value = to_blocks(self.v_proj(hidden_states))

        content = ops.matmul(query, ops.transpose(key, (0, 1, 2, 4, 3)))
        pos = ops.einsum("bnhqd,qkd->bnhqk", query, rel)
        attn = (content + pos) * self.scaling

        # mask keys on padded frames (real padding + the trailing block pad)
        if attention_mask is not None:
            key_mask = ops.cast(attention_mask, hidden_states.dtype)
        else:
            key_mask = ops.ones((batch, seq), dtype=hidden_states.dtype)
        key_mask = ops.pad(key_mask, [[0, 0], [0, pad]])
        key_mask = ops.reshape(key_mask, (batch, num_blocks, c))
        add_mask = (1.0 - key_mask) * MASK_NEG
        attn = attn + add_mask[:, :, None, None, :]

        attn = ops.cast(ops.softmax(ops.cast(attn, "float32"), axis=-1), query.dtype)
        out = ops.matmul(attn, value)  # (b, nblk, h, c, d)
        out = ops.transpose(out, (0, 1, 3, 2, 4))  # (b, nblk, c, h, d)
        out = ops.reshape(out, (batch, padded, h * d))
        out = out[:, :seq, :]
        return self.o_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "context_size": self.context_size,
                "max_position_embeddings": self.max_position_embeddings,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GraniteSpeech5ConvModule(layers.Layer):
    """Conformer convolution module: pointwise up-proj -> GLU -> (mask) -> depthwise
    conv (optionally stride-2) -> BatchNorm + SiLU -> pointwise down-proj. The
    pre-norm lives in the block. All convolutions run over the time axis; the
    depthwise conv is a grouped 1-D conv (one kernel per channel)."""

    def __init__(
        self, hidden_size, conv_expansion_factor, conv_kernel_size, stride=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.conv_expansion_factor = conv_expansion_factor
        self.conv_kernel_size = conv_kernel_size
        self.stride = stride
        self.inner_dim = hidden_size * conv_expansion_factor
        self.pad = (conv_kernel_size - 1) // 2

        self.pointwise_lin1 = layers.Dense(self.inner_dim * 2, name="pointwise_lin1")
        self.pointwise_lin2 = layers.Dense(hidden_size, name="pointwise_lin2")
        # Kept in float32 even when the model is built at bf16 (matching HF's
        # `_keep_in_fp32_modules_strict = ["conv.norm"]`): BatchNorm's running stats
        # lose too much precision in bf16.
        self.norm = layers.BatchNormalization(
            axis=-1, epsilon=1e-5, dtype="float32", name="norm"
        )

    def build(self, input_shape):
        self.pointwise_lin1.build(input_shape)
        inner_shape = (*input_shape[:-1], self.inner_dim)
        self.norm.build(inner_shape)
        self.pointwise_lin2.build(inner_shape)
        self.depthwise_conv = self.add_weight(
            name="depthwise_conv",
            shape=(self.conv_kernel_size, self.inner_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.built = True

    def compute_output_shape(self, input_shape):
        return (input_shape[0], None, self.hidden_size)

    def call(self, x, attention_mask=None, training=False):
        x = self.pointwise_lin1(x)
        a, b = ops.split(x, 2, axis=-1)
        x = a * ops.sigmoid(b)  # GLU

        if attention_mask is not None:
            keep = ops.cast(attention_mask, x.dtype)[:, :, None]
            x = x * keep

        x = ops.pad(x, [[0, 0], [self.pad, self.pad], [0, 0]])
        kernel = self.depthwise_conv[:, :, None]
        x = ops.depthwise_conv(
            x, kernel, strides=self.stride, padding="valid", data_format="channels_last"
        )
        x = self.norm(x, training=training)
        x = ops.silu(x)
        x = self.pointwise_lin2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "conv_expansion_factor": self.conv_expansion_factor,
                "conv_kernel_size": self.conv_kernel_size,
                "stride": self.stride,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GraniteSpeech5Block(layers.Layer):
    """One conformer block (macaron): ``0.5*ff1 -> attn -> conv -> 0.5*ff2 ->
    post-norm``, each sub-module pre-normed by its own LayerNorm. When
    ``subsample`` is set, the conv strides by 2 and the block's residual is
    mean-pooled by 2 so the block output is at half the time resolution."""

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_heads,
        head_dim,
        context_size,
        max_position_embeddings,
        conv_expansion_factor,
        conv_kernel_size,
        attention_bias=True,
        subsample=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.context_size = context_size
        self.max_position_embeddings = max_position_embeddings
        self.conv_expansion_factor = conv_expansion_factor
        self.conv_kernel_size = conv_kernel_size
        self.attention_bias = attention_bias
        self.subsample = subsample

        self.feed_forward1 = GraniteSpeech5FeedForward(
            hidden_size,
            intermediate_size,
            use_bias=attention_bias,
            name="feed_forward1",
        )
        self.self_attn = GraniteSpeech5Attention(
            hidden_size,
            num_heads,
            head_dim,
            context_size,
            max_position_embeddings,
            name="self_attn",
        )
        self.conv = GraniteSpeech5ConvModule(
            hidden_size,
            conv_expansion_factor,
            conv_kernel_size,
            stride=2 if subsample else 1,
            name="conv",
        )
        self.feed_forward2 = GraniteSpeech5FeedForward(
            hidden_size,
            intermediate_size,
            use_bias=attention_bias,
            name="feed_forward2",
        )
        self.norm_feed_forward1 = layers.LayerNormalization(
            epsilon=1e-5, name="norm_feed_forward1"
        )
        self.norm_self_att = layers.LayerNormalization(
            epsilon=1e-5, name="norm_self_att"
        )
        self.norm_conv = layers.LayerNormalization(epsilon=1e-5, name="norm_conv")
        self.norm_feed_forward2 = layers.LayerNormalization(
            epsilon=1e-5, name="norm_feed_forward2"
        )
        self.norm_out = layers.LayerNormalization(epsilon=1e-5, name="norm_out")

    def build(self, input_shape):
        for sublayer in (
            self.norm_feed_forward1,
            self.feed_forward1,
            self.norm_self_att,
            self.self_attn,
            self.norm_conv,
            self.conv,
            self.norm_feed_forward2,
            self.feed_forward2,
            self.norm_out,
        ):
            sublayer.build(input_shape)
        self.built = True

    def compute_output_shape(self, input_shape):
        batch, time = input_shape[0], input_shape[1]
        if self.subsample:
            time = None if time is None else time // 2
        return (batch, time, self.hidden_size)

    def call(self, hidden_states, attention_mask=None, training=False):
        hidden_states = hidden_states + 0.5 * self.feed_forward1(
            self.norm_feed_forward1(hidden_states)
        )
        hidden_states = hidden_states + self.self_attn(
            self.norm_self_att(hidden_states), attention_mask=attention_mask
        )

        conv_out = self.conv(
            self.norm_conv(hidden_states),
            attention_mask=attention_mask,
            training=training,
        )
        if self.subsample:
            length = ops.shape(hidden_states)[1]
            half = length // 2
            pooled = ops.mean(
                ops.reshape(
                    hidden_states[:, : 2 * half],
                    (ops.shape(hidden_states)[0], half, 2, self.hidden_size),
                ),
                axis=2,
            )
            hidden_states = pooled + conv_out[:, : ops.shape(pooled)[1]]
        else:
            hidden_states = hidden_states + conv_out

        hidden_states = hidden_states + 0.5 * self.feed_forward2(
            self.norm_feed_forward2(hidden_states)
        )
        return self.norm_out(hidden_states)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "context_size": self.context_size,
                "max_position_embeddings": self.max_position_embeddings,
                "conv_expansion_factor": self.conv_expansion_factor,
                "conv_kernel_size": self.conv_kernel_size,
                "attention_bias": self.attention_bias,
                "subsample": self.subsample,
            }
        )
        return config
