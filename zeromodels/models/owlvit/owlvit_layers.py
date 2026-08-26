import math

import keras
import numpy as np
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


def quick_gelu(x):
    return x * ops.sigmoid(1.702 * x)


@keras.saving.register_keras_serializable(package="zeromodels")
class OwlViTPositionEmbedding(layers.Embedding):
    """Position-embedding lookup with on-load grid interpolation.

    Subclasses :class:`keras.layers.Embedding` so the saved kernel of
    shape ``(num_positions, hidden_dim)`` is resized to the layer's
    current ``input_dim`` whenever a checkpoint trained at a different
    image resolution is loaded. The first row stays as the CLS-position;
    the remaining ``num_positions - 1`` rows are treated as a square
    grid, bilinearly resized with antialiasing, and flattened back.
    """

    def load_own_variables(self, store):
        source = np.asarray(store["0"])
        target_shape = tuple(self.embeddings.shape)
        if tuple(source.shape) == target_shape:
            self.embeddings.assign(source)
            return
        target_num_positions, hidden_dim = target_shape
        source_num_positions = source.shape[0]
        source_grid = int(round(math.sqrt(source_num_positions - 1)))
        target_grid = int(round(math.sqrt(target_num_positions - 1)))
        source = ops.cast(source, dtype="float32")
        cls = source[:1]
        spatial = source[1:]
        spatial = ops.reshape(spatial, [1, source_grid, source_grid, hidden_dim])
        spatial = ops.image.resize(
            spatial,
            size=[target_grid, target_grid],
            interpolation="bilinear",
            antialias=True,
            data_format="channels_last",
        )
        spatial = ops.reshape(spatial, [target_grid * target_grid, hidden_dim])
        new_kernel = ops.concatenate([cls, spatial], axis=0)
        self.embeddings.assign(new_kernel)


@keras.saving.register_keras_serializable(package="zeromodels")
class OwlViTVisionEmbeddings(layers.Layer):
    """Patch embedding + class token + learned position embedding for OWL-ViT.

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Args:
        hidden_dim: Integer, embedding dimension of each patch token.
        image_size: Integer, square image edge in pixels.
        patch_size: Integer, square patch edge in pixels. Must divide
            ``image_size``.
        num_channels: Integer, number of input image channels. Defaults
            to ``3``.
        **kwargs: Additional keyword arguments passed to ``Layer``.

    Input Shape:
        4D tensor: ``(batch_size, image_size, image_size, num_channels)``.

    Output Shape:
        3D tensor: ``(batch_size, num_patches + 1, hidden_dim)``.
    """

    def __init__(
        self,
        hidden_dim,
        image_size,
        patch_size,
        num_channels=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = (image_size // patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self._data_format = keras.config.image_data_format()

        self.patch_embedding = layers.Conv2D(
            filters=hidden_dim,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=False,
            data_format=self._data_format,
            name="patch_embedding",
        )
        self.position_embedding = OwlViTPositionEmbedding(
            self.num_positions,
            hidden_dim,
            name="position_embedding",
        )

    def build(self, input_shape):
        self.class_embedding = self.add_weight(
            name="class_embedding",
            shape=(self.hidden_dim,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, pixel_values):
        patch_embeds = self.patch_embedding(pixel_values)
        if self._data_format == "channels_first":
            patch_embeds = ops.transpose(patch_embeds, (0, 2, 3, 1))
        b = ops.shape(patch_embeds)[0]
        patch_embeds = ops.reshape(patch_embeds, (b, self.num_patches, self.hidden_dim))
        cls = ops.broadcast_to(
            ops.reshape(self.class_embedding, (1, 1, self.hidden_dim)),
            (b, 1, self.hidden_dim),
        )
        embeddings = ops.concatenate([cls, patch_embeds], axis=1)
        position_ids = ops.arange(0, self.num_positions, dtype="int32")
        pos_embed = self.position_embedding(position_ids)
        return embeddings + ops.expand_dims(pos_embed, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "num_channels": self.num_channels,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class OwlViTTextEmbeddings(layers.Layer):
    """Token embedding + learned position embedding for the OWL-ViT text tower.

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Args:
        vocab_size: Integer, text vocabulary size.
        hidden_dim: Integer, hidden size of the text tower.
        max_seq_len: Integer, maximum text sequence length.
        **kwargs: Additional keyword arguments passed to ``Layer``.

    Input Shape:
        2D integer tensor: ``(batch_size, sequence_length)``.

    Output Shape:
        3D tensor: ``(batch_size, sequence_length, hidden_dim)``.
    """

    def __init__(
        self,
        vocab_size,
        hidden_dim,
        max_seq_len,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = layers.Embedding(
            vocab_size, hidden_dim, name="token_embedding"
        )
        self.position_embedding = layers.Embedding(
            max_seq_len, hidden_dim, name="position_embedding"
        )

    def call(self, input_ids):
        token_embeds = self.token_embedding(input_ids)
        position_ids = ops.arange(0, self.max_seq_len, dtype="int32")
        position_embeds = self.position_embedding(position_ids)
        return token_embeds + ops.expand_dims(position_embeds, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_dim": self.hidden_dim,
                "max_seq_len": self.max_seq_len,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class OwlViTAttention(layers.Layer):
    """Multi-head self-attention with separate q/k/v/out projections.

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Args:
        hidden_dim: Integer, total model dimension. Must be divisible
            by ``num_heads``.
        num_heads: Integer, number of parallel attention heads.
        **kwargs: Additional keyword arguments passed to ``Layer``.

    Input Shape:
        3D tensor: ``(batch_size, seq_len, hidden_dim)``. An optional
        additive ``attention_mask`` broadcastable to
        ``(batch_size, num_heads, seq_len, seq_len)`` may be provided.

    Output Shape:
        3D tensor: ``(batch_size, seq_len, hidden_dim)``.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads "
                f"({num_heads})."
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.k_proj = layers.Dense(hidden_dim, name="k_proj")
        self.v_proj = layers.Dense(hidden_dim, name="v_proj")
        self.q_proj = layers.Dense(hidden_dim, name="q_proj")
        self.out_proj = layers.Dense(hidden_dim, name="out_proj")

    def split_heads(self, x):
        b = ops.shape(x)[0]
        s = ops.shape(x)[1]
        x = ops.reshape(x, (b, s, self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def call(self, hidden_states, attention_mask=None):
        q = self.split_heads(self.q_proj(hidden_states))
        k = self.split_heads(self.k_proj(hidden_states))
        v = self.split_heads(self.v_proj(hidden_states))

        out = fused_attention(q, k, v, self.scale, attention_mask)
        out = ops.transpose(out, (0, 2, 1, 3))
        b = ops.shape(out)[0]
        s = ops.shape(out)[1]
        out = ops.reshape(out, (b, s, self.hidden_dim))
        return self.out_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class OwlViTSplitBatchQueries(layers.Layer):
    """Split a flat ``(B*Q, ...)`` tensor into a per-image ``(B, Q, ...)`` tensor.

    OwlViT receives text queries flattened across the batch
    (``B * Q`` rows). This layer reshapes that flat tensor back into
    ``(B, Q, ...)`` using a separate batch-reference input to recover
    ``B`` at runtime, so it works correctly under the Keras Functional
    API where the leading ``B*Q`` dim is unknown at trace time.

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Args:
        **kwargs: Additional keyword arguments passed to ``Layer``.

    Input Shape:
        Two tensors:
        - ``flat``: ``(B*Q, ..., last_dim)``
        - ``batch_ref``: any tensor whose first dim is ``B``.

    Output Shape:
        ``(B, Q, ..., last_dim)``.
    """

    def call(self, flat, batch_ref):
        b = ops.shape(batch_ref)[0]
        last = flat.shape[-1]
        return ops.reshape(flat, (b, -1, last))


def compute_box_bias(num_patches_height, num_patches_width):
    """Constant log-space box bias added to raw ``box_head`` outputs.

    Each patch's default predicted box is biased toward its grid
    location with a one-patch size.

    Reference:
    - [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)

    Args:
        num_patches_height: Integer, vertical patch count.
        num_patches_width: Integer, horizontal patch count.

    Returns:
        Tensor of shape ``(num_patches_height * num_patches_width, 4)``
        containing the log-odds bias appended to the box head output.
    """
    nh = num_patches_height
    nw = num_patches_width
    x = ops.arange(1, nw + 1, dtype="float32")
    y = ops.arange(1, nh + 1, dtype="float32")
    xx, yy = ops.meshgrid(x, y, indexing="xy")
    box_coords = ops.stack([xx / float(nw), yy / float(nh)], axis=-1)
    box_coords = ops.reshape(box_coords, (nh * nw, 2))
    box_coords = ops.clip(box_coords, 0.0, 1.0)

    box_coord_bias = ops.log(box_coords + 1e-4) - ops.log1p(-box_coords + 1e-4)

    box_size = ops.stack(
        [
            ops.ones((nh * nw,), dtype="float32") / float(nw),
            ops.ones((nh * nw,), dtype="float32") / float(nh),
        ],
        axis=-1,
    )
    box_size_bias = ops.log(box_size + 1e-4) - ops.log1p(-box_size + 1e-4)

    return ops.concatenate([box_coord_bias, box_size_bias], axis=-1)
