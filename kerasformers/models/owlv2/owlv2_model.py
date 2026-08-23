import keras
from keras import layers, ops

from kerasformers.base import BaseModel
from kerasformers.utils import standardize_input_shape

from .owlv2_config import Owlv2Config
from .owlv2_layers import (
    Owlv2Attention,
    Owlv2SplitBatchQueries,
    Owlv2TextEmbeddings,
    Owlv2VisionEmbeddings,
    compute_box_bias,
    quick_gelu,
)


def owlv2_mlp(x, hidden_dim, mlp_dim, block_prefix):
    """Two-layer feed-forward MLP block (``fc1`` → quick_gelu → ``fc2``).

    The CLIP-style MLP used in both the vision and text towers: a Dense
    expanding to ``mlp_dim``, ``quick_gelu`` activation, and
    a Dense projecting back to ``hidden_dim``.

    Reference:
        - `Scaling Open-Vocabulary Object Detection
          <https://arxiv.org/abs/2306.09683>`_

    Args:
        x: Input tensor of shape ``(B, seq_len, hidden_dim)``.
        hidden_dim: Tower hidden dimension (also the output dim).
        mlp_dim: FFN intermediate dimension.
        block_prefix: Layer name prefix.

    Returns:
        Tensor of shape ``(B, seq_len, hidden_dim)``.
    """
    x = layers.Dense(mlp_dim, name=f"{block_prefix}_fc1")(x)
    x = quick_gelu(x)
    x = layers.Dense(hidden_dim, name=f"{block_prefix}_fc2")(x)
    return x


def owlv2_transformer_block(
    x,
    attention_mask,
    num_layers,
    hidden_dim,
    num_heads,
    mlp_dim,
    block_prefix,
):
    """Stack of pre-norm transformer blocks shared by the vision and text towers.

    Each layer applies pre-norm self-attention (LayerNorm →
    :class:`Owlv2Attention` → residual) followed by pre-norm MLP
    (LayerNorm → :func:`owlv2_mlp` → residual). The vision tower
    passes ``attention_mask=None`` for bidirectional attention; the
    text tower passes a causal mask for autoregressive-style attention.

    Args:
        x: Input tensor of shape ``(B, seq_len, hidden_dim)``.
        attention_mask: Optional additive attention mask broadcastable
            to ``(B, num_heads, seq_len, seq_len)``. Use ``None`` for
            full attention.
        num_layers: Number of stacked transformer layers.
        hidden_dim: Model dimension.
        num_heads: Number of attention heads.
        mlp_dim: FFN intermediate dimension.
        block_prefix: Layer name prefix shared across all layers in
            the stack.

    Returns:
        Tensor of shape ``(B, seq_len, hidden_dim)``.
    """
    for i in range(num_layers):
        prefix = f"{block_prefix}_layers_{i}"
        residual = x
        x = layers.LayerNormalization(epsilon=1e-5, name=f"{prefix}_layer_norm1")(x)
        x = Owlv2Attention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            name=f"{prefix}_self_attn",
        )(x, attention_mask=attention_mask)
        x = layers.Add(name=f"{prefix}_sa_residual")([residual, x])

        residual = x
        x = layers.LayerNormalization(epsilon=1e-5, name=f"{prefix}_layer_norm2")(x)
        x = owlv2_mlp(
            x,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            block_prefix=f"{prefix}_mlp",
        )
        x = layers.Add(name=f"{prefix}_ff_residual")([residual, x])
    return x


def owlv2_vision_transformer(
    pixel_values,
    hidden_dim,
    image_size,
    patch_size,
    num_hidden_layers,
    num_heads,
    mlp_dim,
    block_prefix,
):
    """OWLv2 vision tower: patch embeddings → pre-LN → encoder → post-LN.

    Builds the CLIP-style vision encoder: a learned patch + position +
    CLS embedding via :class:`Owlv2VisionEmbeddings`, a pre-encoder
    LayerNorm, ``num_hidden_layers`` pre-norm transformer blocks, and
    a post-encoder LayerNorm. The output keeps the CLS token at
    index 0; detection heads in :class:`Owlv2Detect` use the CLS
    token to modulate the patch tokens before the box / class /
    objectness predictors.

    Args:
        pixel_values: Image input tensor of shape ``(B, H, W, 3)`` (or
            ``(B, 3, H, W)`` for channels_first).
        hidden_dim: Vision tower hidden dimension.
        image_size: Image side length in pixels (square images).
        patch_size: ViT patch size.
        num_hidden_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
        mlp_dim: FFN intermediate dimension.
        block_prefix: Layer name prefix.

    Returns:
        ``(B, num_patches + 1, hidden_dim)`` vision encoder output
        (CLS token at index 0 followed by patch tokens).
    """
    x = Owlv2VisionEmbeddings(
        hidden_dim=hidden_dim,
        image_size=image_size,
        patch_size=patch_size,
        name=f"{block_prefix}_embeddings",
    )(pixel_values)
    x = layers.LayerNormalization(epsilon=1e-5, name=f"{block_prefix}_pre_layernorm")(x)
    x = owlv2_transformer_block(
        x,
        attention_mask=None,
        num_layers=num_hidden_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        block_prefix=block_prefix,
    )
    x = layers.LayerNormalization(epsilon=1e-5, name=f"{block_prefix}_post_layernorm")(
        x
    )
    return x


def owlv2_text_transformer(
    input_ids,
    vocab_size,
    hidden_dim,
    max_seq_len,
    num_hidden_layers,
    num_heads,
    mlp_dim,
    block_prefix,
):
    """OWLv2 text tower: token embeddings → causal encoder → LN → EOS pool.

    Builds a CLIP-style text encoder: token + position embeddings via
    :class:`Owlv2TextEmbeddings`, ``num_hidden_layers`` pre-norm
    transformer blocks with a causal attention mask, a final
    LayerNorm, and an EOS-token pooling step. The EOS token is taken
    as the highest-id token in each sequence, so ``argmax(input_ids)``
    gives the pool index per sample.

    Args:
        input_ids: Integer token IDs of shape
            ``(B, max_seq_len)``.
        vocab_size: Text tokenizer vocabulary size.
        hidden_dim: Text tower hidden dimension.
        max_seq_len: Maximum sequence length.
        num_hidden_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
        mlp_dim: FFN intermediate dimension.
        block_prefix: Layer name prefix.

    Returns:
        last_hidden_state: ``(B, seq_len, hidden_dim)`` post-LN encoder
            output (full token sequence).
        pooled: ``(B, hidden_dim)`` EOS-pooled text feature.
    """
    x = Owlv2TextEmbeddings(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        name=f"{block_prefix}_embeddings",
    )(input_ids)

    seq_len = max_seq_len
    i = ops.arange(seq_len)[:, None]
    j = ops.arange(seq_len)[None, :]
    causal = ops.where(j > i, ops.cast(-1e9, "float32"), ops.cast(0.0, "float32"))
    causal = ops.reshape(causal, (1, 1, seq_len, seq_len))

    x = owlv2_transformer_block(
        x,
        attention_mask=causal,
        num_layers=num_hidden_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        block_prefix=block_prefix,
    )
    last_hidden_state = layers.LayerNormalization(
        epsilon=1e-5, name=f"{block_prefix}_final_layer_norm"
    )(x)

    pool_indices = ops.cast(ops.argmax(input_ids, axis=-1), "int32")
    gather = ops.expand_dims(ops.expand_dims(pool_indices, -1), -1)
    pooled = ops.take_along_axis(last_hidden_state, gather, axis=1)
    pooled = ops.squeeze(pooled, axis=1)
    return last_hidden_state, pooled


def owlv2_box_predictor(image_features, hidden_dim, out_dim, block_prefix):
    """3-layer MLP predicting per-patch box-style outputs.

    Applies ``Dense → GELU → Dense → GELU → Dense(out_dim)``. With
    ``out_dim=4`` this is the standard box predictor (delta in logit
    space, paired with :func:`compute_box_bias` and a sigmoid in
    :func:`owlv2_detection_head`). With ``out_dim=1`` it is reused as
    the OWLv2 objectness head producing a per-patch objectness logit.

    Args:
        image_features: Per-patch image features of shape
            ``(B, num_patches, hidden_dim)``.
        hidden_dim: MLP hidden dimension.
        out_dim: Output dimension (``4`` for box, ``1`` for objectness).
        block_prefix: Layer name prefix.

    Returns:
        ``(B, num_patches, out_dim)`` prediction tensor.
    """
    x = layers.Dense(hidden_dim, name=f"{block_prefix}_dense0")(image_features)
    x = ops.gelu(x, approximate=False)
    x = layers.Dense(hidden_dim, name=f"{block_prefix}_dense1")(x)
    x = ops.gelu(x, approximate=False)
    x = layers.Dense(out_dim, name=f"{block_prefix}_dense2")(x)
    return x


def owlv2_class_predictor(
    image_embeds, query_embeds, query_mask, out_dim, block_prefix
):
    """Text-conditional class predictor with L2-normalized cosine similarity.

    Projects image patch embeddings to ``out_dim``, L2-normalizes both
    the image projections and the per-sample text query embeddings,
    then computes cosine similarity. Two image-conditional scalars
    (a logit shift and a logit scale) are predicted to allow per-image
    temperature calibration. Padding queries
    (``query_mask[..., 0] == 0``) are masked to a very negative logit
    so they don't contribute after softmax.

    Args:
        image_embeds: Image patch embeddings
            ``(B, num_patches, hidden_dim)``.
        query_embeds: Per-sample text query embeddings
            ``(B, num_queries, projection_dim)``.
        query_mask: Optional boolean mask of valid queries with shape
            ``(B, num_queries)``. ``True`` keeps the query, ``False``
            masks it out.
        out_dim: Output dimension of the image projection (matches
            ``query_embeds`` last-dim modulo cosine similarity).
        block_prefix: Layer name prefix.

    Returns:
        pred_logits: Cosine-similarity-based class logits
            ``(B, num_patches, num_queries)``.
        image_class_embeds: L2-normalized image projections
            ``(B, num_patches, out_dim)``: matches the reference's
            returned ``class_embeds`` so callers can use them directly
            for cosine-similarity scoring against custom text queries
            (e.g. one-shot / image-guided detection).
    """
    image_class_embeds = layers.Dense(out_dim, name=f"{block_prefix}_dense0")(
        image_embeds
    )
    image_norm = (
        ops.sqrt(
            ops.sum(image_class_embeds * image_class_embeds, axis=-1, keepdims=True)
            + 1e-12
        )
        + 1e-6
    )
    image_class_embeds_n = image_class_embeds / image_norm

    query_norm = (
        ops.sqrt(ops.sum(query_embeds * query_embeds, axis=-1, keepdims=True) + 1e-12)
        + 1e-6
    )
    query_embeds_n = query_embeds / query_norm

    pred_logits = ops.matmul(
        image_class_embeds_n, ops.transpose(query_embeds_n, (0, 2, 1))
    )

    logit_shift = layers.Dense(1, name=f"{block_prefix}_logit_shift")(image_embeds)
    logit_scale_pred = layers.Dense(1, name=f"{block_prefix}_logit_scale")(image_embeds)
    logit_scale_pred = ops.elu(logit_scale_pred) + 1.0

    pred_logits = (pred_logits + logit_shift) * logit_scale_pred

    if query_mask is not None:
        mask = ops.expand_dims(ops.cast(query_mask, "bool"), axis=-2)
        very_neg = ops.cast(ops.full_like(pred_logits, -1e30), pred_logits.dtype)
        pred_logits = ops.where(mask, pred_logits, very_neg)

    return pred_logits, image_class_embeds_n


def owlv2_detection_head(
    image_embeds_raw,
    text_embeds,
    input_ids,
    vision_hidden_dim,
    text_hidden_dim,
    num_patches_h,
    num_patches_w,
):
    """Build OWLv2 detection heads on top of the dual-tower embeddings.

    Mixes the vision CLS token into the patch tokens (elementwise
    multiplication, then LayerNorm), broadcasts the per-batch text
    queries across image samples via :class:`Owlv2SplitBatchQueries`,
    and applies the per-patch box predictor, objectness predictor, and
    text-conditional class predictor. The box predictor produces raw
    box deltas; an explicit grid bias from :func:`compute_box_bias` is
    added before sigmoid so final boxes lie in normalized ``[0, 1]``
    coordinates.

    The objectness predictor is the OWLv2 delta vs OWL-ViT: a separate
    3-layer MLP outputting a scalar objectness logit per patch, used at
    inference to filter low-objectness patches before applying the
    text-conditional class scores.

    Args:
        image_embeds_raw: Raw vision encoder output
            ``(B, num_patches + 1, vision_hidden_dim)``, including the
            CLS token at index 0.
        text_embeds: L2-normalized text projection
            ``(B, projection_dim)`` from the text tower.
        input_ids: Original text token IDs, used to build the query
            mask (queries whose first token is 0 are treated as padding).
        vision_hidden_dim: Vision tower hidden dimension.
        text_hidden_dim: Text tower hidden dimension (also the
            class predictor output dim).
        num_patches_h: Number of patches along height.
        num_patches_w: Number of patches along width.

    Returns:
        Dict with keys:
        - ``logits``: ``(B, num_patches, num_queries)`` class logits.
        - ``objectness_logits``: ``(B, num_patches)`` per-patch
          objectness logits: multiply with sigmoid(logits) at
          inference to score detection proposals.
        - ``pred_boxes``: ``(B, num_patches, 4)`` sigmoid-bounded
          boxes in ``(cx, cy, w, h)``.
        - ``text_embeds``: Per-image broadcast query embeddings.
        - ``image_embeds``: Patch features reshaped to a 2D feature
          map ``(B, num_patches_h, num_patches_w, vision_hidden_dim)``.
        - ``class_embeds``: Pre-normalization image class projections.
    """
    num_patches = num_patches_h * num_patches_w

    cls_token = image_embeds_raw[:, :1, :]
    patch_embeds = image_embeds_raw[:, 1:, :] * cls_token
    patch_embeds = layers.LayerNormalization(epsilon=1e-5, name="layer_norm")(
        patch_embeds
    )

    feature_map = ops.reshape(
        patch_embeds,
        (-1, num_patches_h, num_patches_w, vision_hidden_dim),
    )
    image_feats = ops.reshape(patch_embeds, (-1, num_patches, vision_hidden_dim))

    query_embeds = Owlv2SplitBatchQueries(name="split_text_embeds")(
        text_embeds, patch_embeds
    )
    input_ids_b = Owlv2SplitBatchQueries(name="split_input_ids")(
        input_ids, patch_embeds
    )
    query_mask = input_ids_b[..., 0] > 0

    pred_logits, class_embeds = owlv2_class_predictor(
        image_feats,
        query_embeds,
        query_mask,
        out_dim=text_hidden_dim,
        block_prefix="class_head",
    )

    box_bias = ops.cast(compute_box_bias(num_patches_h, num_patches_w), "float32")
    pred_boxes = owlv2_box_predictor(
        image_feats,
        hidden_dim=vision_hidden_dim,
        out_dim=4,
        block_prefix="box_head",
    )
    pred_boxes = pred_boxes + ops.cast(box_bias, pred_boxes.dtype)
    pred_boxes = ops.sigmoid(pred_boxes)

    objectness_logits = owlv2_box_predictor(
        image_feats,
        hidden_dim=vision_hidden_dim,
        out_dim=1,
        block_prefix="objectness_head",
    )
    objectness_logits = ops.squeeze(objectness_logits, axis=-1)

    return {
        "logits": pred_logits,
        "objectness_logits": objectness_logits,
        "pred_boxes": pred_boxes,
        "text_embeds": query_embeds,
        "image_embeds": feature_map,
        "class_embeds": class_embeds,
    }


@keras.saving.register_keras_serializable(package="kerasformers")
class Owlv2VisionModel(BaseModel):
    """OWLv2 vision tower as a standalone model.

    Patch + position + CLS
    embeddings, pre-LN, transformer encoder, and post-LN. Use this when
    you only need image features and don't want to instantiate the text
    tower. Composed by :class:`Owlv2Model`.

    Output dict:

    .. code-block:: python

        out = model(pixel_values)
        out["last_hidden_state"]   # (B, num_patches + 1, vision_hidden_dim)
        out["pooler_output"]       # (B, vision_hidden_dim): CLS token

    Reference:
        - `Scaling Open-Vocabulary Object Detection
          <https://arxiv.org/abs/2306.09683>`_
    """

    BASE_MODEL_CONFIG = None
    HF_MODEL_TYPE = "owlv2"

    @classmethod
    def config_from_hf(cls, hf_config):
        return Owlv2Model.config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_owlv2_hf_to_keras import transfer_owlv2_encoder_weights

        transfer_owlv2_encoder_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        vision_image_size,
        vision_patch_size,
        vision_hidden_dim,
        vision_intermediate_size,
        vision_num_layers,
        vision_num_heads,
        image_size=None,
        input_tensor=None,
        name="Owlv2VisionModel",
        **kwargs,
    ):
        for k in (
            "text_hidden_dim",
            "text_intermediate_size",
            "text_num_heads",
            "text_num_layers",
            "text_max_position_embeddings",
            "text_vocab_size",
            "projection_dim",
            "text_input_shape",
        ):
            kwargs.pop(k, None)

        if image_size is None:
            image_size = vision_image_size
        data_format = keras.config.image_data_format()
        image_size = standardize_input_shape(image_size, data_format)
        # The patch grid (and thus the position-embedding size) is driven by
        # the actual input resolution, so users set the resolution purely via
        # ``image_size``; ``vision_image_size`` only supplies the
        # native default. A non-native size interpolates the pretrained
        # position embeddings on load (see Owlv2PositionEmbedding).
        image_edge = image_size[1] if data_format == "channels_first" else image_size[0]

        if input_tensor is None:
            pixel_values = layers.Input(shape=image_size, name="pixel_values")
        else:
            pixel_values = input_tensor

        last_hidden_state = owlv2_vision_transformer(
            pixel_values,
            hidden_dim=vision_hidden_dim,
            image_size=image_edge,
            patch_size=vision_patch_size,
            num_hidden_layers=vision_num_layers,
            num_heads=vision_num_heads,
            mlp_dim=vision_intermediate_size,
            block_prefix="vision_model",
        )
        pooler_output = layers.Lambda(lambda t: t[:, 0, :], name="vision_pooler")(
            last_hidden_state
        )

        super().__init__(
            inputs=pixel_values,
            outputs={
                "last_hidden_state": last_hidden_state,
                "pooler_output": pooler_output,
            },
            name=name,
            **kwargs,
        )

        self.vision_image_size = vision_image_size
        self.vision_patch_size = vision_patch_size
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_intermediate_size = vision_intermediate_size
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.image_size = image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_image_size": self.vision_image_size,
                "vision_patch_size": self.vision_patch_size,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_intermediate_size": self.vision_intermediate_size,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "image_size": self.image_size,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class Owlv2TextModel(BaseModel):
    """OWLv2 text tower as a standalone model.

    Token + position
    embeddings, causal-masked transformer encoder, final LayerNorm, and
    EOS-token pooling. Use this when you only need text features and
    don't want to instantiate the vision tower. Composed by
    :class:`Owlv2Model`. The ``text_projection`` Dense lives in
    :class:`Owlv2Model`, not here.

    Output dict:

    .. code-block:: python

        out = model(input_ids)
        out["last_hidden_state"]   # (B, seq_len, text_hidden_dim)
        out["pooler_output"]       # (B, text_hidden_dim): EOS token

    Reference:
        - `Scaling Open-Vocabulary Object Detection
          <https://arxiv.org/abs/2306.09683>`_
    """

    BASE_MODEL_CONFIG = None
    HF_MODEL_TYPE = "owlv2"

    @classmethod
    def config_from_hf(cls, hf_config):
        return Owlv2Model.config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_owlv2_hf_to_keras import transfer_owlv2_encoder_weights

        transfer_owlv2_encoder_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        text_hidden_dim,
        text_intermediate_size,
        text_num_heads,
        text_num_layers=12,
        text_max_position_embeddings=16,
        text_vocab_size=49408,
        text_input_shape=None,
        input_tensor=None,
        name="Owlv2TextModel",
        **kwargs,
    ):
        for k in (
            "vision_image_size",
            "vision_patch_size",
            "vision_hidden_dim",
            "vision_intermediate_size",
            "vision_num_layers",
            "vision_num_heads",
            "projection_dim",
            "image_size",
        ):
            kwargs.pop(k, None)

        if text_input_shape is None:
            text_input_shape = (text_max_position_embeddings,)

        if input_tensor is None:
            input_ids = layers.Input(
                shape=text_input_shape, dtype="int32", name="input_ids"
            )
        else:
            input_ids = input_tensor

        last_hidden_state, pooler_output = owlv2_text_transformer(
            input_ids,
            vocab_size=text_vocab_size,
            hidden_dim=text_hidden_dim,
            max_seq_len=text_max_position_embeddings,
            num_hidden_layers=text_num_layers,
            num_heads=text_num_heads,
            mlp_dim=text_intermediate_size,
            block_prefix="text_model",
        )

        super().__init__(
            inputs=input_ids,
            outputs={
                "last_hidden_state": last_hidden_state,
                "pooler_output": pooler_output,
            },
            name=name,
            **kwargs,
        )

        self.text_hidden_dim = text_hidden_dim
        self.text_intermediate_size = text_intermediate_size
        self.text_num_heads = text_num_heads
        self.text_num_layers = text_num_layers
        self.text_max_position_embeddings = text_max_position_embeddings
        self.text_vocab_size = text_vocab_size
        self.input_tensor = input_tensor
        self._text_input_shape_arg = text_input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "text_hidden_dim": self.text_hidden_dim,
                "text_intermediate_size": self.text_intermediate_size,
                "text_num_heads": self.text_num_heads,
                "text_num_layers": self.text_num_layers,
                "text_max_position_embeddings": self.text_max_position_embeddings,
                "text_vocab_size": self.text_vocab_size,
                "text_input_shape": self._text_input_shape_arg,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class Owlv2Model(BaseModel):
    """OWLv2 vision + text encoder (no detection heads).

    Composes
    :class:`Owlv2VisionModel` and :class:`Owlv2TextModel` (exposed as
    ``model.vision_model`` / ``model.text_model``), and applies the
    ``text_projection`` + L2-normalization on the text side. Returns the
    raw vision encoder output and the L2-normalized text projection:
    suitable for zero-shot similarity scoring or as a backbone for
    custom heads. For full detection (with the objectness predictor),
    use :class:`Owlv2Detect`.

    Reference:
    - [Scaling Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.09683)
    """

    BASE_MODEL_CONFIG = None
    HF_MODEL_TYPE = "owlv2"

    def __init__(
        self,
        vision_image_size,
        vision_patch_size,
        vision_hidden_dim,
        vision_intermediate_size,
        vision_num_layers,
        vision_num_heads,
        text_hidden_dim,
        text_intermediate_size,
        text_num_heads,
        projection_dim,
        text_num_layers=12,
        text_max_position_embeddings=16,
        text_vocab_size=49408,
        image_size=None,
        text_input_shape=None,
        name="Owlv2Model",
        **kwargs,
    ):
        if image_size is None:
            image_size = vision_image_size
        data_format = keras.config.image_data_format()
        image_size = standardize_input_shape(image_size, data_format)
        if text_input_shape is None:
            text_input_shape = (text_max_position_embeddings,)

        pixel_values = layers.Input(shape=image_size, name="pixel_values")
        input_ids = layers.Input(
            shape=text_input_shape, dtype="int32", name="input_ids"
        )

        vision_model = Owlv2VisionModel(
            vision_image_size=vision_image_size,
            vision_patch_size=vision_patch_size,
            vision_hidden_dim=vision_hidden_dim,
            vision_intermediate_size=vision_intermediate_size,
            vision_num_layers=vision_num_layers,
            vision_num_heads=vision_num_heads,
            image_size=image_size,
            input_tensor=pixel_values,
            name=f"{name}_vision_tower",
        )
        text_model = Owlv2TextModel(
            text_hidden_dim=text_hidden_dim,
            text_intermediate_size=text_intermediate_size,
            text_num_heads=text_num_heads,
            text_num_layers=text_num_layers,
            text_max_position_embeddings=text_max_position_embeddings,
            text_vocab_size=text_vocab_size,
            text_input_shape=text_input_shape,
            input_tensor=input_ids,
            name=f"{name}_text_tower",
        )

        image_embeds = vision_model.output["last_hidden_state"]
        text_pooled = text_model.output["pooler_output"]
        text_embeds = layers.Dense(
            projection_dim, use_bias=False, name="text_projection"
        )(text_pooled)
        norm = ops.sqrt(
            ops.sum(text_embeds * text_embeds, axis=-1, keepdims=True) + 1e-12
        )
        text_embeds = text_embeds / norm

        outputs = {
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
        }
        inputs = {"pixel_values": pixel_values, "input_ids": input_ids}
        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        self.vision_model = vision_model
        self.text_model = text_model
        self.vision_image_size = vision_image_size
        self.vision_patch_size = vision_patch_size
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_intermediate_size = vision_intermediate_size
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.text_hidden_dim = text_hidden_dim
        self.text_intermediate_size = text_intermediate_size
        self.text_num_heads = text_num_heads
        self.text_num_layers = text_num_layers
        self.text_max_position_embeddings = text_max_position_embeddings
        self.text_vocab_size = text_vocab_size
        self.projection_dim = projection_dim
        self.image_size = image_size
        self._text_input_shape_arg = text_input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_image_size": self.vision_image_size,
                "vision_patch_size": self.vision_patch_size,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_intermediate_size": self.vision_intermediate_size,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "text_hidden_dim": self.text_hidden_dim,
                "text_intermediate_size": self.text_intermediate_size,
                "text_num_heads": self.text_num_heads,
                "text_num_layers": self.text_num_layers,
                "text_max_position_embeddings": self.text_max_position_embeddings,
                "text_vocab_size": self.text_vocab_size,
                "projection_dim": self.projection_dim,
                "image_size": self.image_size,
                "text_input_shape": self._text_input_shape_arg,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def config_from_hf(cls, hf_config):
        vc = hf_config["vision_config"]
        tc = hf_config["text_config"]
        return {
            "vision_image_size": vc["image_size"],
            "vision_patch_size": vc["patch_size"],
            "vision_hidden_dim": vc["hidden_size"],
            "vision_intermediate_size": vc["intermediate_size"],
            "vision_num_layers": vc["num_hidden_layers"],
            "vision_num_heads": vc["num_attention_heads"],
            "text_hidden_dim": tc["hidden_size"],
            "text_intermediate_size": tc["intermediate_size"],
            "text_num_heads": tc["num_attention_heads"],
            "text_num_layers": tc["num_hidden_layers"],
            "text_max_position_embeddings": tc["max_position_embeddings"],
            "text_vocab_size": tc["vocab_size"],
            "projection_dim": hf_config["projection_dim"],
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_owlv2_hf_to_keras import transfer_owlv2_encoder_weights

        transfer_owlv2_encoder_weights(keras_model, hf_state_dict)


@keras.saving.register_keras_serializable(package="kerasformers")
class Owlv2Detect(BaseModel):
    """OWLv2 object detection model (encoder + class/box/objectness heads).

    Produces
    per-patch boxes, per-patch objectness logits, and text-conditional
    class similarity logits. The set of detection classes is the set of
    text queries provided at inference time. The objectness head (new
    in OWLv2) lets the model rank patches by general "is-an-object"
    score independent of the text queries.

    Reference:
    - [Scaling Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.09683)
    """

    BASE_MODEL_CONFIG = None
    config_class = Owlv2Config
    HF_MODEL_TYPE = "owlv2"

    def __init__(
        self,
        vision_image_size,
        vision_patch_size,
        vision_hidden_dim,
        vision_intermediate_size,
        vision_num_layers,
        vision_num_heads,
        text_hidden_dim,
        text_intermediate_size,
        text_num_heads,
        projection_dim,
        text_num_layers=12,
        text_max_position_embeddings=16,
        text_vocab_size=49408,
        image_size=None,
        text_input_shape=None,
        name="Owlv2Detect",
        **kwargs,
    ):
        # Resolution is driven by ``image_size``; ``vision_image_size``
        # is only the native default. The patch grid (and detection-head box
        # bias) must match the actual input resolution.
        if image_size is None:
            image_size = vision_image_size
        _data_format = keras.config.image_data_format()
        _resolved = standardize_input_shape(image_size, _data_format)
        image_edge = _resolved[1] if _data_format == "channels_first" else _resolved[0]
        num_patches_h = image_edge // vision_patch_size
        num_patches_w = image_edge // vision_patch_size

        base = Owlv2Model(
            vision_image_size=vision_image_size,
            vision_patch_size=vision_patch_size,
            vision_hidden_dim=vision_hidden_dim,
            vision_intermediate_size=vision_intermediate_size,
            vision_num_layers=vision_num_layers,
            vision_num_heads=vision_num_heads,
            text_hidden_dim=text_hidden_dim,
            text_intermediate_size=text_intermediate_size,
            text_num_heads=text_num_heads,
            text_num_layers=text_num_layers,
            text_max_position_embeddings=text_max_position_embeddings,
            text_vocab_size=text_vocab_size,
            projection_dim=projection_dim,
            image_size=image_size,
            text_input_shape=text_input_shape,
            name=f"{name}_model",
        )
        image_embeds_raw = base.output["image_embeds"]
        text_embeds = base.output["text_embeds"]
        input_ids = base.input["input_ids"]

        outputs = owlv2_detection_head(
            image_embeds_raw,
            text_embeds,
            input_ids,
            vision_hidden_dim=vision_hidden_dim,
            text_hidden_dim=text_hidden_dim,
            num_patches_h=num_patches_h,
            num_patches_w=num_patches_w,
        )
        super().__init__(inputs=base.input, outputs=outputs, name=name, **kwargs)

        self.vision_image_size = vision_image_size
        self.vision_patch_size = vision_patch_size
        self.vision_hidden_dim = vision_hidden_dim
        self.vision_intermediate_size = vision_intermediate_size
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.text_hidden_dim = text_hidden_dim
        self.text_intermediate_size = text_intermediate_size
        self.text_num_heads = text_num_heads
        self.text_num_layers = text_num_layers
        self.text_max_position_embeddings = text_max_position_embeddings
        self.text_vocab_size = text_vocab_size
        self.projection_dim = projection_dim
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.image_size = base.image_size
        self._text_input_shape_arg = text_input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_image_size": self.vision_image_size,
                "vision_patch_size": self.vision_patch_size,
                "vision_hidden_dim": self.vision_hidden_dim,
                "vision_intermediate_size": self.vision_intermediate_size,
                "vision_num_layers": self.vision_num_layers,
                "vision_num_heads": self.vision_num_heads,
                "text_hidden_dim": self.text_hidden_dim,
                "text_intermediate_size": self.text_intermediate_size,
                "text_num_heads": self.text_num_heads,
                "text_num_layers": self.text_num_layers,
                "text_max_position_embeddings": self.text_max_position_embeddings,
                "text_vocab_size": self.text_vocab_size,
                "projection_dim": self.projection_dim,
                "image_size": self.image_size,
                "text_input_shape": self._text_input_shape_arg,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def config_from_hf(cls, hf_config):
        return Owlv2Model.config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_owlv2_hf_to_keras import transfer_owlv2_detection_weights

        transfer_owlv2_detection_weights(keras_model, hf_state_dict)
