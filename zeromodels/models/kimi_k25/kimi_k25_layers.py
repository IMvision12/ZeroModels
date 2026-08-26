import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="zeromodels")
class KimiK25MultimodalProjection(layers.Layer):
    """Patch-merger projector: norm over the ViT width, then a two-layer GELU MLP.

    The vision tower hands over ``(num_merged, kh * kw, vision_embed_dim)`` with
    the merge axis still separate, because ``pre_norm`` normalizes over
    ``vision_embed_dim`` alone. Only afterwards do the ``kh * kw`` neighbours
    flatten into ``mm_dim`` and project into the text width. Exact (erf) GELU,
    not the tanh approximation the vision MLP uses.

    Args:
        mm_dim: Flattened merged width (``kh * kw * vision_embed_dim``).
        embed_dim: Text-tower width.
        norm_eps: ``pre_norm`` epsilon.
    """

    def __init__(self, mm_dim, embed_dim, norm_eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.mm_dim = mm_dim
        self.embed_dim = embed_dim
        self.norm_eps = norm_eps
        self.pre_norm = layers.LayerNormalization(epsilon=norm_eps, name="pre_norm")
        self.in_proj = layers.Dense(mm_dim, use_bias=True, name="in_proj")
        self.out_proj = layers.Dense(embed_dim, use_bias=True, name="out_proj")

    def call(self, x):
        x = ops.reshape(self.pre_norm(x), (-1, self.mm_dim))
        return self.out_proj(ops.gelu(self.in_proj(x), approximate=False))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mm_dim": self.mm_dim,
                "embed_dim": self.embed_dim,
                "norm_eps": self.norm_eps,
            }
        )
        return config
