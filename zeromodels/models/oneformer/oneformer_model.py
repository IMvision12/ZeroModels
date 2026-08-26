import keras
from keras import layers, ops

from zeromodels.base import BaseModel
from zeromodels.utils import standardize_input_shape

from .oneformer_config import OneFormerConfig
from .oneformer_layers import (
    OneFormerCrossAttention,
    OneFormerDeformableAttention,
    OneFormerLearnedEmbedding,
    OneFormerReferencePoints,
    OneFormerSelfAttention,
    OneFormerSinePositionEmbedding,
)
from .oneformer_swin_layers import OneFormerSwinBackbone


def oneformer_input_projection(
    feature, hidden_dim, data_format, channels_axis, block_prefix
):
    """1x1 conv + GroupNorm to project a backbone feature to ``hidden_dim``."""
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


def oneformer_msda_encoder_layer(
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
    attn_out = OneFormerDeformableAttention(
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


def oneformer_pixel_decoder(
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
    """MSDeformAttn pixel decoder (identical recipe to Mask2Former's).

    Returns ``(mask_features, multi_scale_features, spatial_shapes)``.
    """
    msda_features = []
    spatial_shapes = []
    for proj_idx, stage_idx in enumerate([3, 2, 1]):
        feat = backbone_features[stage_idx]
        proj = oneformer_input_projection(
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
        pos = OneFormerSinePositionEmbedding(
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

    level_embed = OneFormerLearnedEmbedding(
        num_embeddings=n_levels,
        hidden_dim=hidden_dim,
        name="pixel_decoder_level_embed",
    )(backbone_features[0])
    flat_pos_with_level = []
    for i, fp in enumerate(flat_pos):
        flat_pos_with_level.append(fp + level_embed[:, i : i + 1, :])

    src = layers.Concatenate(axis=1, name="pixel_decoder_concat_src")(flat_features)
    pos_embed = layers.Concatenate(axis=1, name="pixel_decoder_concat_pos")(
        flat_pos_with_level
    )
    reference_points = OneFormerReferencePoints(
        spatial_shapes=spatial_shapes, name="pixel_decoder_reference_points"
    )(backbone_features[0])

    hidden_states = src
    for i in range(encoder_num_layers):
        hidden_states = oneformer_msda_encoder_layer(
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


def oneformer_query_transformer_layer(
    output, memory, memory_pos, query_pos, hidden_dim, num_heads, mlp_dim, block_prefix
):
    """One post-LN query-transformer layer: self-attn -> cross-attn -> FFN.

    The standard DETR decoder-layer order (the masked main decoder uses the
    opposite cross-first order).
    """
    residual = output
    sq = layers.Add(name=f"{block_prefix}_sa_q_add")([output, query_pos])
    self_out = OneFormerCrossAttention(
        hidden_dim=hidden_dim, num_heads=num_heads, name=f"{block_prefix}_self_attn"
    )(sq, sq, output)
    output = layers.Add(name=f"{block_prefix}_sa_residual")([residual, self_out])
    output = layers.LayerNormalization(epsilon=1e-5, name=f"{block_prefix}_norm1")(
        output
    )

    residual = output
    q = layers.Add(name=f"{block_prefix}_ca_q_add")([output, query_pos])
    k = layers.Add(name=f"{block_prefix}_ca_k_add")([memory, memory_pos])
    cross_out = OneFormerCrossAttention(
        hidden_dim=hidden_dim, num_heads=num_heads, name=f"{block_prefix}_cross_attn"
    )(q, k, memory)
    output = layers.Add(name=f"{block_prefix}_ca_residual")([residual, cross_out])
    output = layers.LayerNormalization(epsilon=1e-5, name=f"{block_prefix}_norm2")(
        output
    )

    residual = output
    y = layers.Dense(mlp_dim, name=f"{block_prefix}_linear1")(output)
    y = layers.Activation("relu", name=f"{block_prefix}_linear1_relu")(y)
    y = layers.Dense(hidden_dim, name=f"{block_prefix}_linear2")(y)
    output = layers.Add(name=f"{block_prefix}_ffn_residual")([residual, y])
    output = layers.LayerNormalization(epsilon=1e-5, name=f"{block_prefix}_norm3")(
        output
    )
    return output


def oneformer_decoder_layer(
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
    """One OneFormer main decoder layer: masked cross-attn -> self-attn -> FFN."""
    residual = hidden_states
    q = layers.Add(name=f"{block_prefix}_ca_q_add")([hidden_states, query_pos])
    k = layers.Add(name=f"{block_prefix}_ca_k_add")([memory, memory_pos])
    cross_out = OneFormerCrossAttention(
        hidden_dim=hidden_dim, num_heads=num_heads, name=f"{block_prefix}_cross_attn"
    )(q, k, memory, attn_mask=attn_mask)
    hidden_states = layers.Add(name=f"{block_prefix}_ca_residual")(
        [residual, cross_out]
    )
    hidden_states = layers.LayerNormalization(
        epsilon=1e-5, name=f"{block_prefix}_cross_attn_norm"
    )(hidden_states)

    residual = hidden_states
    sq = layers.Add(name=f"{block_prefix}_sa_q_add")([hidden_states, query_pos])
    sk = layers.Add(name=f"{block_prefix}_sa_k_add")([hidden_states, query_pos])
    self_out = OneFormerSelfAttention(
        hidden_dim=hidden_dim, num_heads=num_heads, name=f"{block_prefix}_self_attn"
    )(sq, sk, hidden_states)
    hidden_states = layers.Add(name=f"{block_prefix}_sa_residual")([residual, self_out])
    hidden_states = layers.LayerNormalization(
        epsilon=1e-5, name=f"{block_prefix}_self_attn_norm"
    )(hidden_states)

    residual = hidden_states
    y = layers.Dense(mlp_dim, name=f"{block_prefix}_linear1")(hidden_states)
    y = layers.Activation("relu", name=f"{block_prefix}_linear1_relu")(y)
    y = layers.Dense(hidden_dim, name=f"{block_prefix}_linear2")(y)
    hidden_states = layers.Add(name=f"{block_prefix}_ffn_residual")([residual, y])
    hidden_states = layers.LayerNormalization(
        epsilon=1e-5, name=f"{block_prefix}_ffn_norm"
    )(hidden_states)
    return hidden_states


def oneformer_functional(
    pixel_values,
    task_inputs,
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
    query_dec_layers,
    num_heads,
    num_queries,
    num_classes,
    task_seq_len,
    data_format,
    channels_axis,
    n_msda_points=4,
):
    """Build the full OneFormer graph (functional)."""
    backbone = OneFormerSwinBackbone(
        embed_dim=backbone_embed_dim,
        depths=backbone_depths,
        num_heads=backbone_num_heads,
        window_size=backbone_window_size,
        data_format=data_format,
        name="backbone",
    )
    backbone_features = backbone(pixel_values)

    mask_features, multi_scale_features, _ = oneformer_pixel_decoder(
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

    # --- task token: 2-layer MLP over the raw (task_seq_len,) token-id vector
    task = layers.Dense(hidden_dim, name="task_encoder_task_mlp_0")(task_inputs)
    task = layers.Activation("relu", name="task_encoder_task_mlp_0_relu")(task)
    task = layers.Dense(hidden_dim, name="task_encoder_task_mlp_1")(task)
    task_token = layers.Reshape((1, hidden_dim), name="task_token_reshape")(task)

    decoder_norm = layers.LayerNormalization(
        epsilon=1e-5, name="transformer_decoder_norm"
    )
    task_token = decoder_norm(task_token)  # use_task_norm

    # --- query transformer: init the object queries from the mask features
    queries_embedder_layer = OneFormerLearnedEmbedding(
        num_embeddings=num_queries,
        hidden_dim=hidden_dim,
        name="transformer_decoder_queries_embedder",
    )
    query_pos_all = queries_embedder_layer(backbone_features[0])  # (B, Q, C)
    query_pos_obj = query_pos_all[:, : num_queries - 1, :]

    mf_h, mf_w = (
        (mask_features.shape[2], mask_features.shape[3])
        if data_format == "channels_first"
        else (mask_features.shape[1], mask_features.shape[2])
    )
    # HF passes the sine positional map as the memory and the projected mask
    # features as the memory positions (the original OneFormer quirk).
    query_features_pos = OneFormerSinePositionEmbedding(
        hidden_dim=hidden_dim,
        data_format=data_format,
        name="transformer_decoder_query_features_pos",
    )(mask_features)
    qt_memory = layers.Reshape(
        (mf_h * mf_w, hidden_dim), name="transformer_decoder_qt_memory"
    )(
        ops.transpose(query_features_pos, (0, 2, 3, 1))
        if data_format == "channels_first"
        else query_features_pos
    )
    proj_mask_features = layers.Conv2D(
        hidden_dim,
        1,
        padding="valid",
        use_bias=True,
        data_format=data_format,
        name="transformer_decoder_query_input_projection",
    )(mask_features)
    qt_memory_pos = layers.Reshape(
        (mf_h * mf_w, hidden_dim), name="transformer_decoder_qt_memory_pos"
    )(
        ops.transpose(proj_mask_features, (0, 2, 3, 1))
        if data_format == "channels_first"
        else proj_mask_features
    )

    qt_output = ops.repeat(task_token, num_queries - 1, axis=1)
    for i in range(query_dec_layers):
        qt_output = oneformer_query_transformer_layer(
            qt_output,
            qt_memory,
            qt_memory_pos,
            query_pos_obj,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mlp_dim=decoder_ffn_dim,
            block_prefix=f"transformer_decoder_query_transformer_layers_{i}",
        )
    qt_output = layers.LayerNormalization(
        epsilon=1e-5, name="transformer_decoder_query_transformer_norm"
    )(qt_output)

    hidden_states = layers.Concatenate(axis=1, name="transformer_decoder_queries")(
        [qt_output, task_token]
    )

    # --- multi-scale memories for the masked main decoder
    level_pos_embeds = []
    level_memories = []
    transformer_level_embed = OneFormerLearnedEmbedding(
        num_embeddings=len(multi_scale_features),
        hidden_dim=hidden_dim,
        name="transformer_decoder_level_embed",
    )(backbone_features[0])
    spatial_hw = []
    for i, feat in enumerate(multi_scale_features):
        h_l, w_l = (
            (feat.shape[2], feat.shape[3])
            if data_format == "channels_first"
            else (feat.shape[1], feat.shape[2])
        )
        spatial_hw.append((h_l, w_l))
        pos = OneFormerSinePositionEmbedding(
            hidden_dim=hidden_dim,
            data_format=data_format,
            name=f"transformer_decoder_level_pos_{i}",
        )(feat)
        flat_pos = layers.Reshape(
            (h_l * w_l, hidden_dim), name=f"transformer_decoder_flatten_level_pos_{i}"
        )(ops.transpose(pos, (0, 2, 3, 1)) if data_format == "channels_first" else pos)
        level_pos_embeds.append(flat_pos)
        flat = layers.Reshape(
            (h_l * w_l, hidden_dim), name=f"transformer_decoder_flatten_memory_{i}"
        )(
            ops.transpose(feat, (0, 2, 3, 1))
            if data_format == "channels_first"
            else feat
        )
        level_memories.append(flat + transformer_level_embed[:, i : i + 1, :])

    class_embed = layers.Dense(num_classes + 1, name="transformer_decoder_class_embed")
    mask_embed_0 = layers.Dense(hidden_dim, name="transformer_decoder_mask_embed_0")
    mask_embed_0_act = layers.Activation(
        "relu", name="transformer_decoder_mask_embed_0_relu"
    )
    mask_embed_1 = layers.Dense(hidden_dim, name="transformer_decoder_mask_embed_1")
    mask_embed_1_act = layers.Activation(
        "relu", name="transformer_decoder_mask_embed_1_relu"
    )
    mask_embed_2 = layers.Dense(
        mask_feature_size, name="transformer_decoder_mask_embed_2"
    )

    def predict_masks(h):
        h_norm = decoder_norm(h)
        cls = class_embed(h_norm)
        emb = mask_embed_2(
            mask_embed_1_act(mask_embed_1(mask_embed_0_act(mask_embed_0(h_norm))))
        )
        mask_eq = (
            "bqc,bhwc->bqhw" if data_format == "channels_last" else "bqc,bchw->bqhw"
        )
        return cls, ops.einsum(mask_eq, emb, mask_features)

    def downsample_mask_for_level(mask_logits, level_h, level_w):
        mask_hwq = ops.transpose(mask_logits, (0, 2, 3, 1))
        mask_resized = ops.image.resize(
            mask_hwq,
            (level_h, level_w),
            interpolation="bilinear",
            data_format="channels_last",
        )
        mask_resized = ops.transpose(mask_resized, (0, 3, 1, 2))
        mask_flat = ops.reshape(mask_resized, (-1, num_queries, level_h * level_w))
        bool_mask = mask_flat < 0.0
        all_masked = ops.all(bool_mask, axis=-1, keepdims=True)
        bool_mask = ops.where(all_masked, ops.zeros_like(bool_mask), bool_mask)
        additive = ops.where(
            bool_mask, ops.cast(-1e9, "float32"), ops.cast(0.0, "float32")
        )
        additive = ops.expand_dims(additive, axis=1)
        return ops.repeat(additive, num_heads, axis=1)

    cls_i, mask_i = predict_masks(hidden_states)
    n_levels = len(multi_scale_features)
    for i in range(decoder_num_layers):
        level_idx = i % n_levels
        h_l, w_l = spatial_hw[level_idx]
        attn_mask = downsample_mask_for_level(mask_i, h_l, w_l)
        hidden_states = oneformer_decoder_layer(
            hidden_states,
            memory=level_memories[level_idx],
            memory_pos=level_pos_embeds[level_idx],
            query_pos=query_pos_all,
            attn_mask=attn_mask,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mlp_dim=decoder_ffn_dim,
            block_prefix=f"transformer_decoder_layers_{i}",
        )
        cls_i, mask_i = predict_masks(hidden_states)

    return {"class_queries_logits": cls_i, "masks_queries_logits": mask_i}


@keras.saving.register_keras_serializable(package="zeromodels")
class OneFormerModel(BaseModel):
    """OneFormer universal segmentation (Swin backbone, task-conditioned queries).

    One model performs semantic, instance, and panoptic segmentation: the
    task is selected at inference by the tokenized ``"the task is {task}"``
    prompt (``task_inputs``). A Swin backbone feeds an MSDeformAttn pixel
    decoder; the object queries are initialized by a small query transformer
    over the mask features (seeded with the task token, which is also
    appended as the final query), then refined by a masked-attention decoder
    identical in spirit to Mask2Former's. The training-only text encoder /
    text mapper (contrastive loss) is not ported: the released inference
    checkpoints (``is_training: false``) carry no weights for it.

    Inputs: ``{"pixel_values": (B, H, W, 3), "task_inputs": (B, 77)}`` (the
    tokenized task prompt as float token ids).
    Outputs: ``class_queries_logits`` ``(B, Q, num_classes + 1)`` and
    ``masks_queries_logits`` ``(B, Q, H/4, W/4)``.

    Reference:
    - [OneFormer: One Transformer to Rule Universal Image Segmentation](https://arxiv.org/abs/2211.06220)
    """

    BASE_MODEL_CONFIG = None
    HF_MODEL_TYPE = "oneformer"

    @classmethod
    def config_from_hf(cls, hf_config):
        backbone = hf_config.get("backbone_config", {})
        from zeromodels.base.base_model import hf_num_classes

        return {
            "backbone_embed_dim": backbone.get("embed_dim", 96),
            "backbone_depths": tuple(backbone.get("depths", [2, 2, 6, 2])),
            "backbone_num_heads": tuple(backbone.get("num_heads", [3, 6, 12, 24])),
            "backbone_window_size": backbone.get("window_size", 7),
            "hidden_dim": hf_config.get("hidden_dim", 256),
            "mask_feature_size": hf_config.get("mask_dim", 256),
            "encoder_num_layers": hf_config.get("encoder_layers", 6),
            "encoder_ffn_dim": hf_config.get("encoder_feedforward_dim", 1024),
            "decoder_num_layers": hf_config.get("decoder_layers", 10) - 1,
            "decoder_ffn_dim": hf_config.get("dim_feedforward", 2048),
            "query_dec_layers": hf_config.get("query_dec_layers", 2),
            "num_heads": hf_config.get("num_attention_heads", 8),
            "num_queries": hf_config.get("num_queries", 150),
            "num_classes": hf_num_classes(hf_config),
            "task_seq_len": hf_config.get("task_seq_len", 77),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_oneformer_hf_to_keras import transfer_oneformer_weights

        transfer_oneformer_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        backbone_embed_dim=96,
        backbone_depths=(2, 2, 6, 2),
        backbone_num_heads=(3, 6, 12, 24),
        backbone_window_size=7,
        hidden_dim=256,
        mask_feature_size=256,
        encoder_num_layers=6,
        encoder_ffn_dim=1024,
        decoder_num_layers=9,
        decoder_ffn_dim=2048,
        query_dec_layers=2,
        num_heads=8,
        num_queries=150,
        num_classes=150,
        task_seq_len=77,
        image_size=512,
        name="OneFormerModel",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1
        image_size = standardize_input_shape(image_size, data_format)

        pixel_values = layers.Input(shape=image_size, name="pixel_values")
        task_inputs = layers.Input(shape=(task_seq_len,), name="task_inputs")

        outputs = oneformer_functional(
            pixel_values,
            task_inputs,
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
            query_dec_layers=query_dec_layers,
            num_heads=num_heads,
            num_queries=num_queries,
            num_classes=num_classes,
            task_seq_len=task_seq_len,
            data_format=data_format,
            channels_axis=channels_axis,
        )
        super().__init__(
            inputs={"pixel_values": pixel_values, "task_inputs": task_inputs},
            outputs=outputs,
            name=name,
            **kwargs,
        )

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
        self.query_dec_layers = query_dec_layers
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.task_seq_len = task_seq_len
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
                "query_dec_layers": self.query_dec_layers,
                "num_heads": self.num_heads,
                "num_queries": self.num_queries,
                "num_classes": self.num_classes,
                "task_seq_len": self.task_seq_len,
                "image_size": self.image_size,
                "name": self.name,
            }
        )
        return c

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class OneFormerUniversalSegment(OneFormerModel):
    """Universal-segmentation alias for :class:`OneFormerModel`.

    OneFormer has integrated heads; this alias exists for API symmetry with
    the other segmentation classes in zeromodels.
    """

    config_class = OneFormerConfig
    # Weights load by Hub repo id, e.g.
    # from_weights("zeromodels/oneformer_ade20k_swin_tiny"), via zm_config.json
    # on the repo (no url table in the package).
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "oneformer"

    def __init__(self, name="OneFormerUniversalSegment", **kwargs):
        super().__init__(name=name, **kwargs)
