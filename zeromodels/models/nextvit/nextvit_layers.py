import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


@keras.saving.register_keras_serializable(package="zeromodels")
class NextViTEfficientAttention(layers.Layer):
    """Efficient Multi-Head Self-Attention (E-MHSA) with optional spatial reduction.

    Operates on sequence tensors of shape ``(B, N, C)``. When ``sr_ratio > 1``,
    keys and values are spatially reduced via average pooling followed by batch
    normalization before computing attention, reducing the quadratic cost from
    ``O(N^2)`` to ``O(N * N/sr_ratio^2)``.

    Reference:
    - [Next-ViT: Next Generation Vision Transformer for Efficient Deployment
      in Realistic Industrial Scenarios](https://arxiv.org/abs/2207.05501)

    Args:
        dim: int, input and output feature dimension.
        head_dim: int, dimension per attention head. The number of heads is
            ``dim // head_dim``. Defaults to ``32``.
        sr_ratio: int, spatial reduction ratio for keys and values.
            When ``1``, no reduction is applied. Defaults to ``1``.
        attn_drop: float, dropout rate for attention weights.
            Defaults to ``0.0``.
        proj_drop: float, dropout rate for the output projection.
            Defaults to ``0.0``.
        prefix: string, name prefix for all sub-layers.
            Defaults to ``""``.
        **kwargs: Additional keyword arguments passed to the ``Layer`` class.
    """

    def __init__(
        self,
        dim,
        head_dim=32,
        sr_ratio=1,
        attn_drop=0.0,
        proj_drop=0.0,
        prefix="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.prefix = prefix
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5
        self.sr_ratio = sr_ratio
        self.n_ratio = sr_ratio**2

        self.q = layers.Dense(dim, use_bias=True, name=prefix + "e_mhsa_q")
        self.k = layers.Dense(dim, use_bias=True, name=prefix + "e_mhsa_k")
        self.v = layers.Dense(dim, use_bias=True, name=prefix + "e_mhsa_v")
        self.proj = layers.Dense(dim, use_bias=True, name=prefix + "e_mhsa_proj")
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj_drop = layers.Dropout(proj_drop)

        self.use_sr = sr_ratio > 1
        if self.use_sr:
            # BN on transposed (B, C, N_reduced): axis=1 is always the
            # channel dimension in this internal representation
            self.norm = layers.BatchNormalization(
                axis=1,
                epsilon=1e-5,
                momentum=0.9,
                name=prefix + "e_mhsa_norm",
            )

    def call(self, x, training=None):
        input_shape = ops.shape(x)
        B = input_shape[0]
        N = input_shape[1]

        q = self.q(x)
        q = ops.reshape(q, [B, N, self.num_heads, self.head_dim])
        q = ops.transpose(q, [0, 2, 1, 3])

        if self.use_sr:
            x_sr = ops.transpose(x, [0, 2, 1])
            N_reduced = N // self.n_ratio
            x_sr = ops.reshape(x_sr, [B, self.dim, N_reduced, self.n_ratio])
            x_sr = ops.mean(x_sr, axis=-1)
            x_sr = self.norm(x_sr, training=training)
            x_sr = ops.transpose(x_sr, [0, 2, 1])
        else:
            x_sr = x

        k = self.k(x_sr)
        v = self.v(x_sr)
        N_kv = ops.shape(k)[1]
        k = ops.reshape(k, [B, N_kv, self.num_heads, self.head_dim])
        k = ops.transpose(k, [0, 2, 1, 3])
        v = ops.reshape(v, [B, N_kv, self.num_heads, self.head_dim])
        v = ops.transpose(v, [0, 2, 1, 3])

        out = fused_attention(
            q, k, v, self.scale, dropout=self.attn_drop, training=training
        )
        out = ops.transpose(out, [0, 2, 1, 3])
        out = ops.reshape(out, input_shape)

        out = self.proj(out)
        out = self.proj_drop(out, training=training)
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "head_dim": self.head_dim,
                "sr_ratio": self.sr_ratio,
                "attn_drop": self.attn_drop.rate,
                "proj_drop": self.proj_drop.rate,
                "prefix": self.prefix,
            }
        )
        return config
