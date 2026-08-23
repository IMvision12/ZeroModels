import math

import keras
import numpy as np
from keras import layers, ops

from kerasformers.base.base_attention import fused_attention


@keras.saving.register_keras_serializable(package="kerasformers")
class Tipsv2RegisterTokens(layers.Layer):
    """Insert learnable register tokens between the CLS token and the patch tokens.

    Produces ``[CLS, register_tokens, patches]``; the register tokens receive no
    position embedding (position embeddings are added before this layer).
    """

    def __init__(self, num_tokens=1, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens

    def build(self, input_shape):
        self.register_tokens = self.add_weight(
            name="register_tokens",
            shape=(1, self.num_tokens, input_shape[-1]),
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        embed_dim = inputs.shape[-1]
        reg = ops.broadcast_to(
            self.register_tokens, [batch_size, self.num_tokens, embed_dim]
        )
        return ops.concatenate([inputs[:, :1], reg, inputs[:, 1:]], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"num_tokens": self.num_tokens})
        return config


def sinusoidal_table(num_positions, embed_dim):
    """tensor2tensor-style sinusoidal position table, matching TIPSv2's text tower."""
    half_dim = embed_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim, dtype="float32") * -emb)
    emb = np.arange(num_positions, dtype="float32")[:, None] * emb[None, :]
    table = np.concatenate([np.sin(emb), np.cos(emb)], axis=1)
    if embed_dim % 2 == 1:
        table = np.concatenate([table, np.zeros((num_positions, 1), "float32")], axis=1)
    return table.astype("float32")


@keras.saving.register_keras_serializable(package="kerasformers")
class Tipsv2TextEmbedding(layers.Layer):
    """Token embedding scaled by ``sqrt(hidden_dim)`` plus fixed sinusoidal positions.

    The sinusoidal table is a non-trainable, computed buffer (not stored in TIPSv2
    checkpoints), so the weight converter skips it.
    """

    def __init__(self, max_seq_len, embed_dim, scale_sqrt_depth=True, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.scale_sqrt_depth = scale_sqrt_depth
        self.embed_scale = math.sqrt(embed_dim) if scale_sqrt_depth else 1.0

    def build(self, input_shape):
        table = sinusoidal_table(self.max_seq_len, self.embed_dim)
        self.pos_table = self.add_weight(
            name="pos_table",
            shape=(self.max_seq_len, self.embed_dim),
            initializer=lambda shape, dtype=None: ops.convert_to_tensor(table),
            trainable=False,
        )
        self.built = True

    def call(self, token_embeddings):
        seq_len = token_embeddings.shape[1]
        pos = ops.cast(self.pos_table[:seq_len], token_embeddings.dtype)
        return token_embeddings * self.embed_scale + pos[None]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_seq_len": self.max_seq_len,
                "embed_dim": self.embed_dim,
                "scale_sqrt_depth": self.scale_sqrt_depth,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Tipsv2TextAttention(layers.Layer):
    """Bidirectional multi-head attention with separate q/k/v/out projections.

    Applies an additive padding mask via :func:`fused_attention` (softmax in float32).
    """

    def __init__(self, hidden_dim, num_heads, block_prefix, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.block_prefix = block_prefix
        prefix = f"{block_prefix}_"
        self.q_proj = layers.Dense(hidden_dim, name=prefix + "q_proj")
        self.k_proj = layers.Dense(hidden_dim, name=prefix + "k_proj")
        self.v_proj = layers.Dense(hidden_dim, name=prefix + "v_proj")
        self.out_proj = layers.Dense(hidden_dim, name=prefix + "out_proj")

    def call(self, hidden_states, attention_mask=None):
        batch_size = ops.shape(hidden_states)[0]
        seq_len = hidden_states.shape[1]
        shape = (batch_size, seq_len, self.num_heads, self.head_dim)

        q = ops.transpose(ops.reshape(self.q_proj(hidden_states), shape), (0, 2, 1, 3))
        k = ops.transpose(ops.reshape(self.k_proj(hidden_states), shape), (0, 2, 1, 3))
        v = ops.transpose(ops.reshape(self.v_proj(hidden_states), shape), (0, 2, 1, 3))

        out = fused_attention(q, k, v, self.scale, attention_mask=attention_mask)
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)),
            (batch_size, seq_len, self.hidden_dim),
        )
        return self.out_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "block_prefix": self.block_prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class Tipsv2PaddingMask(layers.Layer):
    """Turn a ``(B, L)`` padding mask (1 = keep) into an additive ``(B, 1, 1, L)`` mask."""

    def call(self, padding_mask):
        mask = ops.cast(padding_mask, "float32")
        additive = (1.0 - mask) * -1e9
        return additive[:, None, None, :]


@keras.saving.register_keras_serializable(package="kerasformers")
class Tipsv2MaskedMeanPool(layers.Layer):
    """Masked mean over the sequence axis: ``sum(h * mask) / (sum(mask) + eps)``."""

    def __init__(self, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, hidden_states, padding_mask):
        mask = ops.cast(padding_mask, hidden_states.dtype)[..., None]
        summed = ops.sum(hidden_states * mask, axis=1)
        counts = ops.sum(mask, axis=1) + self.epsilon
        return summed / counts

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


def tipsv2_text_encoder_layer(
    hidden_states,
    attention_mask,
    hidden_dim,
    num_heads,
    mlp_dim,
    hidden_act,
    layer_norm_eps,
    name,
):
    """One pre-LN TIPSv2 text block: LN -> masked attn -> res, LN -> MLP -> res."""
    residual = hidden_states
    x = layers.LayerNormalization(epsilon=layer_norm_eps, name=f"{name}_layer_norm1")(
        hidden_states
    )
    x = Tipsv2TextAttention(hidden_dim, num_heads, block_prefix=f"{name}_self_attn")(
        x, attention_mask=attention_mask
    )
    x = layers.Add(name=f"{name}_add_1")([residual, x])

    residual = x
    y = layers.LayerNormalization(epsilon=layer_norm_eps, name=f"{name}_layer_norm2")(x)
    y = layers.Dense(mlp_dim, name=f"{name}_mlp_fc1")(y)
    y = layers.Activation(hidden_act, name=f"{name}_mlp_act")(y)
    y = layers.Dense(hidden_dim, name=f"{name}_mlp_fc2")(y)
    return layers.Add(name=f"{name}_add_2")([residual, y])
