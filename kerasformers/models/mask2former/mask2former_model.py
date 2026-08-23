import keras
from keras import layers, ops

from kerasformers.base import BaseModel
from kerasformers.utils import standardize_input_shape

from .mask2former_config import Mask2FormerConfig
from .mask2former_layers import (
    Mask2FormerCrossAttention,
    Mask2FormerDeformableAttention,
    Mask2FormerLearnedEmbedding,
    Mask2FormerReferencePoints,
    Mask2FormerSelfAttention,
    Mask2FormerSinePositionEmbedding,
)
from .mask2former_swin_layers import Mask2FormerSwinBackbone


def mask2former_input_projection(
    feature,
    hidden_dim,
    data_format,
    channels_axis,
    block_prefix="pixel_decoder_input_projections_0",
):
    """1×1 conv + GroupNorm to project a backbone feature to ``hidden_dim``."""
    x = layers.Conv2D(
        hidden_dim,
        1,
        padding="valid",
        use_bias=True,
        data_format=data_format,
        name=f"{block_prefix}_conv",
    )(feature)
    x = layers.GroupNormalization(
        groups=32, axis=channels_axis, epsilon=1e-5, name=f"{block_prefix}_norm"
    )(x)
    return x


def mask2former_msda_encoder_layer(
    hidden_states,
    pos_embed,
    reference_points,
    spatial_shapes,
    hidden_dim,
    n_heads,
    n_levels,
    n_points,
    mlp_dim,
    block_prefix,
):
    """One MSDeformAttn encoder layer (post-LN)."""
    residual = hidden_states
    query = layers.Add(name=f"{block_prefix}_with_pos")([hidden_states, pos_embed])
    attn_out = Mask2FormerDeformableAttention(
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_levels=n_levels,
        n_points=n_points,
        spatial_shapes=spatial_shapes,
        name=f"{block_prefix}_self_attn",
    )(query, reference_points, hidden_states)
    hidden_states = layers.Add(name=f"{block_prefix}_attn_residual")(
        [residual, attn_out]
    )
    hidden_states = layers.LayerNormalization(
        epsilon=1e-5, name=f"{block_prefix}_self_attn_layer_norm"
    )(hidden_states)

    residual = hidden_states
    y = layers.Dense(mlp_dim, name=f"{block_prefix}_fc1")(hidden_states)
    y = layers.Activation("relu", name=f"{block_prefix}_fc1_relu")(y)
    y = layers.Dense(hidden_dim, name=f"{block_prefix}_fc2")(y)
    hidden_states = layers.Add(name=f"{block_prefix}_ffn_residual")([residual, y])
    hidden_states = layers.LayerNormalization(
        epsilon=1e-5, name=f"{block_prefix}_final_layer_norm"
    )(hidden_states)
    return hidden_states


def mask2former_pixel_decoder(
    backbone_features,
    hidden_dim,
    mask_feature_size,
    encoder_num_layers,
    encoder_ffn_dim,
    n_heads,
    n_points,
    data_format,
    channels_axis,
):
    """MSDeformAttn pixel decoder.

    Takes the 3 coarsest backbone features (stages 2, 3, 4) and runs them
    through a 6-layer MSDeformAttn encoder. The finest stage (stage 1) is
    used in an FPN-style fusion step to produce the high-resolution mask
    features.

    Returns:
        mask_features: ``(B, H/4, W/4, mask_feature_size)`` for mask prediction.
        multi_scale_features: 3 feature maps at strides 32, 16, 8, fed to
            the transformer decoder as cross-attention memory (one per scale).
        spatial_shapes_msda: list of (h, w) per level (coarsest first).
    """
    msda_features = []
    spatial_shapes = []  # (h, w) per level, coarsest first
    for proj_idx, stage_idx in enumerate([3, 2, 1]):
        feat = backbone_features[stage_idx]
        proj = mask2former_input_projection(
            feat,
            hidden_dim,
            data_format,
            channels_axis,
            block_prefix=f"pixel_decoder_input_projections_{proj_idx}",
        )
        msda_features.append(proj)
        spatial_shapes.append(
            (proj.shape[2], proj.shape[3])
            if data_format == "channels_first"
            else (proj.shape[1], proj.shape[2])
        )

    n_levels = len(msda_features)

    flat_features = []
    flat_pos = []
    for i, feat in enumerate(msda_features):
        pos = Mask2FormerSinePositionEmbedding(
            hidden_dim=hidden_dim,
            data_format=data_format,
            name=f"pixel_decoder_position_embedding_{i}",
        )(feat)
        h_l, w_l = (
            (feat.shape[2], feat.shape[3])
            if data_format == "channels_first"
            else (feat.shape[1], feat.shape[2])
        )
        n = h_l * w_l
        flat_features.append(
            layers.Reshape((n, hidden_dim), name=f"pixel_decoder_flatten_src_{i}")(
                ops.transpose(feat, (0, 2, 3, 1))
                if data_format == "channels_first"
                else feat
            )
        )
        flat_pos.append(
            layers.Reshape((n, hidden_dim), name=f"pixel_decoder_flatten_pos_{i}")(
                ops.transpose(pos, (0, 2, 3, 1))
                if data_format == "channels_first"
                else pos
            )
        )

    level_embed = Mask2FormerLearnedEmbedding(
        num_embeddings=n_levels,
        hidden_dim=hidden_dim,
        name="pixel_decoder_level_embed",
    )(backbone_features[0])
    flat_pos_with_level = []
    for i, fp in enumerate(flat_pos):
        le = level_embed[:, i : i + 1, :]
        flat_pos_with_level.append(fp + le)

    src = layers.Concatenate(axis=1, name="pixel_decoder_concat_src")(flat_features)
    pos_embed = layers.Concatenate(axis=1, name="pixel_decoder_concat_pos")(
        flat_pos_with_level
    )

    reference_points = Mask2FormerReferencePoints(
        spatial_shapes=spatial_shapes,
        name="pixel_decoder_reference_points",
    )(backbone_features[0])

    hidden_states = src
    for i in range(encoder_num_layers):
        hidden_states = mask2former_msda_encoder_layer(
            hidden_states,
            pos_embed,
            reference_points,
            spatial_shapes,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_levels=n_levels,
            n_points=n_points,
            mlp_dim=encoder_ffn_dim,
            block_prefix=f"pixel_decoder_encoder_layers_{i}",
        )

    multi_scale_features = []
    start = 0
    for i, (h, w) in enumerate(spatial_shapes):
        n = h * w
        feat_flat = hidden_states[:, start : start + n, :]
        start += n
        feat_2d = layers.Reshape(
            (h, w, hidden_dim), name=f"pixel_decoder_unflatten_{i}"
        )(feat_flat)
        multi_scale_features.append(
            ops.transpose(feat_2d, (0, 3, 1, 2))
            if data_format == "channels_first"
            else feat_2d
        )

    finest_msda = multi_scale_features[-1]
    adapter = layers.Conv2D(
        hidden_dim,
        1,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="pixel_decoder_adapter_1_conv",
    )(backbone_features[0])
    adapter = layers.GroupNormalization(
        groups=32, axis=channels_axis, epsilon=1e-5, name="pixel_decoder_adapter_1_norm"
    )(adapter)
    upsampled = layers.UpSampling2D(
        size=(2, 2),
        interpolation="bilinear",
        data_format=data_format,
        name="pixel_decoder_fpn_upsample",
    )(finest_msda)
    fused = layers.Add(name="pixel_decoder_fpn_add")([upsampled, adapter])
    fused = layers.Conv2D(
        hidden_dim,
        3,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name="pixel_decoder_layer_1_conv",
    )(fused)
    fused = layers.GroupNormalization(
        groups=32, axis=channels_axis, epsilon=1e-5, name="pixel_decoder_layer_1_norm"
    )(fused)
    fused = layers.Activation("relu", name="pixel_decoder_layer_1_relu")(fused)

    mask_features = layers.Conv2D(
        mask_feature_size,
        1,
        padding="valid",
        use_bias=True,
        data_format=data_format,
        name="pixel_decoder_mask_projection",
    )(fused)

    return mask_features, multi_scale_features, spatial_shapes


def mask2former_decoder_layer(
    hidden_states,
    memory,
    memory_pos,
    query_pos,
    attn_mask,
    hidden_dim,
    num_heads,
    mlp_dim,
    block_prefix,
):
    """One Mask2Former decoder layer: masked cross-attn → self-attn → FFN.

    The
    cross-attention runs first (with the predicted-mask additive mask),
    then self-attention over the queries, then the FFN.
    """
    residual = hidden_states
    q = layers.Add(name=f"{block_prefix}_ca_q_add")([hidden_states, query_pos])
    k = layers.Add(name=f"{block_prefix}_ca_k_add")([memory, memory_pos])
    cross_out = Mask2FormerCrossAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        name=f"{block_prefix}_cross_attn",
    )(q, k, memory, attn_mask=attn_mask)
    hidden_states = layers.Add(name=f"{block_prefix}_ca_residual")(
        [residual, cross_out]
    )
    hidden_states = layers.LayerNormalization(
        epsilon=1e-5, name=f"{block_prefix}_cross_attn_layer_norm"
    )(hidden_states)

    residual = hidden_states
    sq = layers.Add(name=f"{block_prefix}_sa_q_add")([hidden_states, query_pos])
    sk = layers.Add(name=f"{block_prefix}_sa_k_add")([hidden_states, query_pos])
    self_out = Mask2FormerSelfAttention(
        hidden_dim=hidden_dim, num_heads=num_heads, name=f"{block_prefix}_self_attn"
    )(sq, sk, hidden_states)
    hidden_states = layers.Add(name=f"{block_prefix}_sa_residual")([residual, self_out])
    hidden_states = layers.LayerNormalization(
        epsilon=1e-5, name=f"{block_prefix}_self_attn_layer_norm"
    )(hidden_states)

    residual = hidden_states
    y = layers.Dense(mlp_dim, name=f"{block_prefix}_fc1")(hidden_states)
    y = layers.Activation("relu", name=f"{block_prefix}_fc1_relu")(y)
    y = layers.Dense(hidden_dim, name=f"{block_prefix}_fc2")(y)
    hidden_states = layers.Add(name=f"{block_prefix}_ffn_residual")([residual, y])
    hidden_states = layers.LayerNormalization(
        epsilon=1e-5, name=f"{block_prefix}_final_layer_norm"
    )(hidden_states)
    return hidden_states


def mask2former_functional(
    pixel_values,
    *,
    backbone_embed_dim,
    backbone_depths,
    backbone_num_heads,
    backbone_window_size,
    hidden_dim,
    mask_feature_size,
    encoder_num_layers,
    encoder_ffn_dim,
    decoder_num_layers,
    decoder_ffn_dim,
    num_heads,
    num_queries,
    num_classes,
    data_format,
    channels_axis,
    n_msda_points=4,
):
    """Build the full Mask2Former graph (functional)."""
    backbone = Mask2FormerSwinBackbone(
        embed_dim=backbone_embed_dim,
        depths=backbone_depths,
        num_heads=backbone_num_heads,
        window_size=backbone_window_size,
        data_format=data_format,
        name="backbone",
    )
    backbone_features = backbone(pixel_values)

    mask_features, multi_scale_features, msda_spatial_shapes = (
        mask2former_pixel_decoder(
            backbone_features,
            hidden_dim=hidden_dim,
            mask_feature_size=mask_feature_size,
            encoder_num_layers=encoder_num_layers,
            encoder_ffn_dim=encoder_ffn_dim,
            n_heads=num_heads,
            n_points=n_msda_points,
            data_format=data_format,
            channels_axis=channels_axis,
        )
    )

    queries_features_layer = Mask2FormerLearnedEmbedding(
        num_embeddings=num_queries,
        hidden_dim=hidden_dim,
        name="transformer_decoder_queries_features",
    )
    queries_embedder_layer = Mask2FormerLearnedEmbedding(
        num_embeddings=num_queries,
        hidden_dim=hidden_dim,
        name="transformer_decoder_queries_embedder",
    )
    hidden_states = queries_features_layer(backbone_features[0])
    query_pos = queries_embedder_layer(backbone_features[0])

    level_pos_embeds = []
    for i, feat in enumerate(multi_scale_features):
        pos = Mask2FormerSinePositionEmbedding(
            hidden_dim=hidden_dim,
            data_format=data_format,
            name=f"transformer_decoder_level_pos_{i}",
        )(feat)
        h_l, w_l = (
            (feat.shape[2], feat.shape[3])
            if data_format == "channels_first"
            else (feat.shape[1], feat.shape[2])
        )
        flat = layers.Reshape(
            (h_l * w_l, hidden_dim),
            name=f"transformer_decoder_flatten_level_pos_{i}",
        )(ops.transpose(pos, (0, 2, 3, 1)) if data_format == "channels_first" else pos)
        level_pos_embeds.append(flat)

    transformer_level_embed = Mask2FormerLearnedEmbedding(
        num_embeddings=len(multi_scale_features),
        hidden_dim=hidden_dim,
        name="transformer_decoder_level_embed",
    )(backbone_features[0])
    level_memories = []
    for i, feat in enumerate(multi_scale_features):
        h_l, w_l = (
            (feat.shape[2], feat.shape[3])
            if data_format == "channels_first"
            else (feat.shape[1], feat.shape[2])
        )
        flat = layers.Reshape(
            (h_l * w_l, hidden_dim),
            name=f"transformer_decoder_flatten_memory_{i}",
        )(
            ops.transpose(feat, (0, 2, 3, 1))
            if data_format == "channels_first"
            else feat
        )
        level_memories.append(flat + transformer_level_embed[:, i : i + 1, :])

    decoder_layernorm = layers.LayerNormalization(
        epsilon=1e-5, name="transformer_decoder_layernorm"
    )
    class_predictor = layers.Dense(num_classes + 1, name="class_predictor")
    mask_embedder_0 = layers.Dense(
        hidden_dim, name="transformer_decoder_mask_embedder_0"
    )
    mask_embedder_0_act = layers.Activation(
        "relu", name="transformer_decoder_mask_embedder_0_relu"
    )
    mask_embedder_1 = layers.Dense(
        hidden_dim, name="transformer_decoder_mask_embedder_1"
    )
    mask_embedder_1_act = layers.Activation(
        "relu", name="transformer_decoder_mask_embedder_1_relu"
    )
    mask_embedder_2 = layers.Dense(
        mask_feature_size, name="transformer_decoder_mask_embedder_2"
    )

    def compute_mask_embeddings(h):
        """Run the shared 3-layer query MLP, returning per-query mask embeddings."""
        y = mask_embedder_0(h)
        y = mask_embedder_0_act(y)
        y = mask_embedder_1(y)
        y = mask_embedder_1_act(y)
        return mask_embedder_2(y)

    def predict_masks(h):
        """Produce class logits and ``(B, Q, H, W)`` mask logits from decoder states.

        Applies the shared pre-prediction LayerNorm, the class predictor, and
        dots the per-query mask embeddings with the pixel-decoder mask features.
        """
        h_norm = decoder_layernorm(h)
        cls = class_predictor(h_norm)
        mask_emb = compute_mask_embeddings(h_norm)
        mask_eq = (
            "bqc,bhwc->bqhw" if data_format == "channels_last" else "bqc,bchw->bqhw"
        )
        mask_logits = ops.einsum(mask_eq, mask_emb, mask_features)
        return cls, mask_logits

    def downsample_mask_for_level(mask_logits, level_h, level_w):
        """Resize predicted masks into a per-level masked-attention bias.

        Bilinearly resizes the ``(B, Q, H, W)`` mask logits to the level's
        ``(level_h, level_w)`` grid and converts them to an additive
        cross-attention mask: ``0`` where attention is allowed and ``-1e9``
        where ``sigmoid(mask) < 0.5``, broadcast across the attention heads.

        Args:
            mask_logits: Predicted mask logits of shape ``(B, Q, H, W)``.
            level_h: Target height for this decoder level.
            level_w: Target width for this decoder level.

        Returns:
            Additive attention mask of shape
            ``(B, num_heads, Q, level_h * level_w)``.
        """

        mask_hwq = ops.transpose(mask_logits, (0, 2, 3, 1))  # (B, H, W, Q)
        mask_resized = ops.image.resize(
            mask_hwq,
            (level_h, level_w),
            interpolation="bilinear",
            data_format="channels_last",
        )
        mask_resized = ops.transpose(mask_resized, (0, 3, 1, 2))  # (B, Q, h, w)
        mask_flat = ops.reshape(mask_resized, (-1, num_queries, level_h * level_w))
        bool_mask = mask_flat < 0.0
        all_masked = ops.all(bool_mask, axis=-1, keepdims=True)
        bool_mask = ops.where(all_masked, ops.zeros_like(bool_mask), bool_mask)
        additive = ops.where(
            bool_mask,
            ops.cast(-1e9, "float32"),
            ops.cast(0.0, "float32"),
        )
        additive = ops.expand_dims(additive, axis=1)
        additive = ops.repeat(additive, num_heads, axis=1)
        return additive

    cls_init, mask_init = predict_masks(hidden_states)
    intermediate_logits = [cls_init]
    intermediate_masks = [mask_init]

    n_levels = len(multi_scale_features)
    for i in range(decoder_num_layers):
        level_idx = i % n_levels
        feat = multi_scale_features[level_idx]
        h_l, w_l = (
            (feat.shape[2], feat.shape[3])
            if data_format == "channels_first"
            else (feat.shape[1], feat.shape[2])
        )

        attn_mask = downsample_mask_for_level(intermediate_masks[-1], h_l, w_l)

        hidden_states = mask2former_decoder_layer(
            hidden_states,
            memory=level_memories[level_idx],
            memory_pos=level_pos_embeds[level_idx],
            query_pos=query_pos,
            attn_mask=attn_mask,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mlp_dim=decoder_ffn_dim,
            block_prefix=f"transformer_decoder_layers_{i}",
        )
        cls_i, mask_i = predict_masks(hidden_states)
        intermediate_logits.append(cls_i)
        intermediate_masks.append(mask_i)

    return {
        "class_queries_logits": intermediate_logits[-1],
        "masks_queries_logits": intermediate_masks[-1],
    }


@keras.saving.register_keras_serializable(package="kerasformers")
class Mask2FormerModel(BaseModel):
    """Mask2Former base (backbone + pixel decoder + transformer, with segment heads).

    Reference:
    - [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527)
    """

    BASE_MODEL_CONFIG = None
    HF_MODEL_TYPE = "mask2former"

    @classmethod
    def config_from_hf(cls, hf_config):
        """Map a Mask2Former config.json to constructor kwargs.

        Reads the Swin backbone sub-config and the top-level transformer /
        MSDeformAttn settings and returns the keyword arguments for building
        the equivalent Keras model.

        Args:
            hf_config: The ``Mask2FormerConfig`` as a dict.

        Returns:
            Dict of constructor keyword arguments for this model class.
        """
        backbone = hf_config.get("backbone_config", {})
        depths = backbone.get("depths", [2, 2, 6, 2])
        num_heads = backbone.get("num_heads", [3, 6, 12, 24])

        from kerasformers.base.base_model import hf_num_classes

        return {
            "backbone_embed_dim": backbone.get("embed_dim", 96),
            "backbone_depths": tuple(depths),
            "backbone_num_heads": tuple(num_heads),
            "backbone_window_size": backbone.get("window_size", 12),
            "hidden_dim": hf_config.get("hidden_dim", 256),
            "mask_feature_size": hf_config.get("mask_feature_size", 256),
            "encoder_num_layers": hf_config.get("encoder_layers", 6),
            "encoder_ffn_dim": hf_config.get("encoder_feedforward_dim", 1024),
            "decoder_num_layers": hf_config.get("decoder_layers", 10) - 1,
            "decoder_ffn_dim": hf_config.get("dim_feedforward", 2048),
            "num_heads": hf_config.get("num_attention_heads", 8),
            "num_queries": hf_config.get("num_queries", 100),
            "num_classes": hf_num_classes(hf_config),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        """Copy weights from a Mask2Former checkpoint into the model.

        Args:
            keras_model: The freshly-built Keras model to populate.
            hf_state_dict: The source model ``state_dict`` (numpy arrays).
        """
        from .convert_mask2former_hf_to_keras import transfer_mask2former_weights

        transfer_mask2former_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        backbone_embed_dim=96,
        backbone_depths=(2, 2, 6, 2),
        backbone_num_heads=(3, 6, 12, 24),
        backbone_window_size=12,
        hidden_dim=256,
        mask_feature_size=256,
        encoder_num_layers=6,
        encoder_ffn_dim=1024,
        decoder_num_layers=9,
        decoder_ffn_dim=2048,
        num_heads=8,
        num_queries=100,
        num_classes=80,
        image_size=384,
        name="Mask2FormerModel",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1
        image_size = standardize_input_shape(image_size, data_format)

        pixel_values = layers.Input(shape=image_size, name="pixel_values")

        outputs = mask2former_functional(
            pixel_values,
            backbone_embed_dim=backbone_embed_dim,
            backbone_depths=backbone_depths,
            backbone_num_heads=backbone_num_heads,
            backbone_window_size=backbone_window_size,
            hidden_dim=hidden_dim,
            mask_feature_size=mask_feature_size,
            encoder_num_layers=encoder_num_layers,
            encoder_ffn_dim=encoder_ffn_dim,
            decoder_num_layers=decoder_num_layers,
            decoder_ffn_dim=decoder_ffn_dim,
            num_heads=num_heads,
            num_queries=num_queries,
            num_classes=num_classes,
            data_format=data_format,
            channels_axis=channels_axis,
        )
        super().__init__(inputs=pixel_values, outputs=outputs, name=name, **kwargs)

        self.backbone_embed_dim = backbone_embed_dim
        self.backbone_depths = tuple(backbone_depths)
        self.backbone_num_heads = tuple(backbone_num_heads)
        self.backbone_window_size = backbone_window_size
        self.hidden_dim = hidden_dim
        self.mask_feature_size = mask_feature_size
        self.encoder_num_layers = encoder_num_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_num_layers = decoder_num_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.image_size = image_size

    def get_config(self):
        c = super().get_config()
        c.update(
            {
                "backbone_embed_dim": self.backbone_embed_dim,
                "backbone_depths": self.backbone_depths,
                "backbone_num_heads": self.backbone_num_heads,
                "backbone_window_size": self.backbone_window_size,
                "hidden_dim": self.hidden_dim,
                "mask_feature_size": self.mask_feature_size,
                "encoder_num_layers": self.encoder_num_layers,
                "encoder_ffn_dim": self.encoder_ffn_dim,
                "decoder_num_layers": self.decoder_num_layers,
                "decoder_ffn_dim": self.decoder_ffn_dim,
                "num_heads": self.num_heads,
                "num_queries": self.num_queries,
                "num_classes": self.num_classes,
                "image_size": self.image_size,
                "name": self.name,
            }
        )
        return c

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class Mask2FormerUniversalSegment(Mask2FormerModel):
    """Universal-segmentation alias for ``Mask2FormerModel``.

    Mask2Former has integrated heads: there's no "base vs segment" split
    like in MaskFormer. ``Mask2FormerUniversalSegment`` is provided as an alias for
    API symmetry with the other segmentation classes in kerasformers.
    """

    config_class = Mask2FormerConfig
    # Weights load by Hub repo id, e.g.
    # from_weights("kerasformers/mask2former-swin-tiny-coco-instance"), via
    # kf_config.json on the repo (no url table in the package).
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "mask2former"

    def __init__(self, name="Mask2FormerUniversalSegment", **kwargs):
        super().__init__(name=name, **kwargs)
