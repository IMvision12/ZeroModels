import keras
from keras import layers, ops

from zeromodels.models.gemma4.gemma4_layers import Gemma4MultimodalEmbedder


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma4UnifiedVisionEmbedder(layers.Layer):
    """Encoder-free vision embedder for the Gemma 4 unified checkpoints.

    Replaces the entire NaViT / SigLIP vision tower with a projection: raw merged
    pixel patches (``model_patch_size**2 * 3`` channels, where ``model_patch_size
    = patch_size * pooling_kernel_size``) go through ``LayerNorm -> Dense ->
    LayerNorm``, gain a factorized 2D position embedding (a ``(mm_posemb_size, 2,
    mm_embed_dim)`` table looked up once per axis, padding patches masked out),
    another ``LayerNorm``, and the shared soft-token projector
    (:class:`Gemma4MultimodalEmbedder`: weightless RMSNorm then Dense into the
    text embedding space). The plain ``LayerNorm`` sublayers use torch's ``1e-5``
    default, distinct from the model's ``rms_norm_eps``.

    Args:
        patch_dim: Raw channels per merged patch (``model_patch_size**2 * 3``).
        mm_embed_dim: Width of the patch Dense projection and position table.
        mm_posemb_size: Length of the factorized 2D position table.
        text_hidden_size: Output width (text residual-stream dim).
        eps: RMSNorm epsilon of the final soft-token projector.
    """

    def __init__(
        self,
        patch_dim,
        mm_embed_dim,
        mm_posemb_size,
        text_hidden_size,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_dim = patch_dim
        self.mm_embed_dim = mm_embed_dim
        self.mm_posemb_size = mm_posemb_size
        self.text_hidden_size = text_hidden_size
        self.eps = eps

        self.patch_ln1 = layers.LayerNormalization(epsilon=1e-5, name="patch_ln1")
        self.patch_dense = layers.Dense(mm_embed_dim, name="patch_dense")
        self.patch_ln2 = layers.LayerNormalization(epsilon=1e-5, name="patch_ln2")
        self.pos_norm = layers.LayerNormalization(epsilon=1e-5, name="pos_norm")
        self.multimodal_embedder = Gemma4MultimodalEmbedder(
            text_hidden_size, eps=eps, name="multimodal_embedder"
        )

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            shape=(self.mm_posemb_size, 2, self.mm_embed_dim),
            initializer="zeros",
            name="pos_embedding",
        )
        super().build(input_shape)

    def call(self, pixel_values, image_position_ids):
        # pixel_values: (b, num_patches, patch_dim). image_position_ids:
        # (b, num_patches, 2) integer XY, with (-1, -1) marking padding patches.
        hidden = self.patch_ln1(pixel_values)
        hidden = self.patch_dense(hidden)
        hidden = self.patch_ln2(hidden)

        positions = ops.cast(image_position_ids, "int32")
        clamped = ops.maximum(positions, 0)
        valid_x = ops.cast(positions[..., 0] != -1, hidden.dtype)[..., None]
        valid_y = ops.cast(positions[..., 1] != -1, hidden.dtype)[..., None]
        pos_x = ops.take(self.pos_embedding[:, 0, :], clamped[..., 0], axis=0)
        pos_y = ops.take(self.pos_embedding[:, 1, :], clamped[..., 1], axis=0)
        hidden = hidden + ops.cast(pos_x, hidden.dtype) * valid_x
        hidden = hidden + ops.cast(pos_y, hidden.dtype) * valid_y
        hidden = self.pos_norm(hidden)

        return self.multimodal_embedder(hidden)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_dim": self.patch_dim,
                "mm_embed_dim": self.mm_embed_dim,
                "mm_posemb_size": self.mm_posemb_size,
                "text_hidden_size": self.text_hidden_size,
                "eps": self.eps,
            }
        )
        return config
