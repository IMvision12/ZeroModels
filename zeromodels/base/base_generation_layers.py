import keras
from keras import layers, ops

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="zeromodels")
class CausalMask(layers.Layer):
    """Additive causal (+ padding, + optional sliding-window) attention mask.

    Produces ``(batch, 1, seq, seq)`` holding ``0`` where a query may attend and
    a large negative where it may not: a lower-triangular causal block (optionally
    also bounded below by ``sliding_window``) plus the per-key padding taken from
    ``attention_mask``. Isolating the dynamic ``arange`` inside a layer with an
    explicit output spec keeps the functional graph shape-inferable; the KV-cache
    decode path builds its own concrete-length mask instead.
    """

    def __init__(self, sliding_window=None, **kwargs):
        super().__init__(**kwargs)
        self.sliding_window = sliding_window

    def call(self, input_ids, attention_mask=None):
        seq = ops.shape(input_ids)[1]
        idx = ops.arange(seq)
        allowed = idx[None, :] <= idx[:, None]
        if self.sliding_window is not None:
            allowed = ops.logical_and(
                allowed, idx[None, :] > idx[:, None] - self.sliding_window
            )
        mask = ops.cast(ops.where(allowed, 0.0, MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(attention_mask, "float32")
            mask = mask + (1.0 - am)[:, None, None, :] * MASK_NEG
        return mask

    def compute_output_spec(self, input_ids, attention_mask=None):
        seq = input_ids.shape[1]
        return keras.KerasTensor((input_ids.shape[0], 1, seq, seq), dtype="float32")

    def get_config(self):
        config = super().get_config()
        config.update({"sliding_window": self.sliding_window})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class TiedHead(layers.Layer):
    """Weightless language-model head that projects with a tied token embedding.

    Computes ``hidden @ embedding.embeddings.T`` reading the live embedding weight
    inside ``call`` (never baking a build-time copy into the functional graph). It
    holds the embedding by a non-tracked reference so the embedding keeps its
    single weight and its place in the saved-weights layout: the head adds no
    variable, so the functional checkpoint is identical to the plain backbone's
    and the two load the same file. The embedding is re-supplied when the model is
    rebuilt from config, so this reference is never serialized.
    """

    def __init__(self, embedding, **kwargs):
        super().__init__(**kwargs)
        # object.__setattr__ bypasses Keras layer tracking so the embedding is not
        # re-registered as a sublayer here (which would duplicate it in the saved
        # structure and break loading against the flat backbone checkpoint).
        object.__setattr__(self, "embedding", embedding)

    def call(self, hidden):
        kernel = ops.transpose(ops.cast(self.embedding.embeddings, hidden.dtype))
        return ops.matmul(hidden, kernel)

    def compute_output_spec(self, hidden):
        shape = list(hidden.shape)
        shape[-1] = self.embedding.input_dim
        return keras.KerasTensor(shape, dtype=hidden.dtype)


def merge_media(hidden, input_ids, features, token_id, embed_dim):
    """Scatter ``features`` into ``hidden`` wherever ``input_ids == token_id``.

    A running count of the placeholder mask assigns each media token its feature
    row (a gather, which every backend differentiates and compiles). A no-op when
    the token is absent (empty mask). Runs only with concrete shapes -- inside
    :class:`MediaMerge` (functional graph) or a KV-cache prefill (eager).
    """
    flat_ids = ops.reshape(input_ids, (-1,))
    flat_hidden = ops.reshape(hidden, (-1, embed_dim))
    mask = ops.equal(flat_ids, token_id)
    idx = ops.cumsum(ops.cast(mask, "int32")) - 1
    gathered = ops.take(features, ops.maximum(idx, 0), axis=0)
    flat_hidden = ops.where(
        mask[:, None], ops.cast(gathered, flat_hidden.dtype), flat_hidden
    )
    return ops.reshape(flat_hidden, ops.shape(hidden))


@keras.saving.register_keras_serializable(package="zeromodels")
class MediaMerge(layers.Layer):
    """Weightless in-graph media scatter for VLM backbones (see :func:`merge_media`).

    Wraps the cumsum-gather-where scatter of vision/audio features into the text
    hidden states at placeholder-token slots. An explicit output spec keeps the
    dynamic flatten/reshape out of the symbolic build (it would otherwise bake a
    ``None`` into a Reshape); the merge runs at (eager) runtime with concrete
    shapes. Add one instance per media stream (image, video, ...).
    """

    def __init__(self, token_id, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_id = token_id
        self.embed_dim = embed_dim

    def call(self, hidden, input_ids, features):
        return merge_media(hidden, input_ids, features, self.token_id, self.embed_dim)

    def compute_output_spec(self, hidden, input_ids, features):
        return keras.KerasTensor(hidden.shape, dtype=hidden.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"token_id": self.token_id, "embed_dim": self.embed_dim})
        return config
