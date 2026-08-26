import keras
from keras import layers, ops, utils

from kerasformers.base import BaseModel
from kerasformers.base.base_model import hf_num_classes
from kerasformers.conversion import copy_weights_by_path_suffix
from kerasformers.utils import standardize_input_shape

from .rt_detr_v2_config import RTDetrV2Config
from .rt_detr_v2_layers import (
    RTDETRV2MultiHeadAttention,
    RTDETRV2MultiScaleDeformableAttention,
)


def rt_detr_sine_pos_embed(height, width, embed_dim, temperature=10000):
    """Compute 2D sinusoidal position embedding.

    Generates non-learnable sine/cosine positional encodings for a 2D
    spatial grid. The embedding dimension is split into four equal parts
    encoding height-sin, height-cos, width-sin, and width-cos. Used by
    the AIFI encoder to add spatial information to flattened feature
    tokens.

    Reference:
        - `RT-DETR <https://arxiv.org/abs/2304.08069>`_

    Args:
        height: Integer, spatial height of the feature map.
        width: Integer, spatial width of the feature map.
        embed_dim: Integer, total embedding dimension. Must be
            divisible by 4.
        temperature: Integer, temperature scaling factor for the
            sinusoidal frequencies. Defaults to ``10000``.

    Returns:
        Tensor of shape ``(1, height * width, embed_dim)``.
    """
    pos_dim = embed_dim // 4
    dim_t = ops.cast(ops.arange(pos_dim), "float32") / pos_dim
    dim_t = 1.0 / (temperature**dim_t)
    grid_w = ops.cast(ops.arange(width), "float32")
    grid_h = ops.cast(ops.arange(height), "float32")
    grid_h, grid_w = ops.meshgrid(grid_h, grid_w, indexing="ij")
    out_w = ops.reshape(grid_w, [-1, 1]) * ops.reshape(dim_t, [1, -1])
    out_h = ops.reshape(grid_h, [-1, 1]) * ops.reshape(dim_t, [1, -1])
    pos = ops.concatenate(
        [ops.sin(out_h), ops.cos(out_h), ops.sin(out_w), ops.cos(out_w)],
        axis=-1,
    )
    return ops.expand_dims(pos, axis=0)


def rt_detr_backbone(
    input_tensor,
    depths,
    hidden_sizes,
    embedding_size,
    layer_type="bottleneck",
    data_format=None,
    channels_axis=-1,
):
    """Build a ResNet-vd backbone for RT-DETR feature extraction.

    Constructs a ResNet backbone with a 3-convolution stem (ResNet-vd
    variant) and four residual stages. Supports both basic (ResNet-18/34)
    and bottleneck (ResNet-50/101) block types. Extracts multi-scale
    features from stages 1, 2, and 3 for the hybrid encoder.

    Reference:
        - `RT-DETR <https://arxiv.org/abs/2304.08069>`_
        - `Bag of Tricks for Image Classification
          <https://arxiv.org/abs/1812.01187>`_

    Args:
        input_tensor: Keras input tensor of shape
            ``(batch_size, height, width, 3)``.
        depths: List of integers, number of residual blocks
            per stage (e.g., ``[3, 4, 6, 3]`` for ResNet-50).
        hidden_sizes: List of integers, output channels per stage
            (e.g., ``[256, 512, 1024, 2048]``).
        embedding_size: Integer, stem output channels.
            Defaults to ``64``.
        layer_type: String, ``"bottleneck"`` for 3-layer blocks
            (ResNet-50/101) or ``"basic"`` for 2-layer blocks
            (ResNet-18/34). Defaults to ``"bottleneck"``.
        data_format: String, Keras data format.
        channels_axis: Integer, channel axis index.

    Returns:
        Tuple of three feature tensors from stages 1, 2, and 3.
    """
    x = input_tensor
    stem_cfgs = [
        (embedding_size // 2, 2),
        (embedding_size // 2, 1),
        (embedding_size, 1),
    ]
    for i, (out_ch, stride) in enumerate(stem_cfgs):
        x = layers.ZeroPadding2D(padding=1, data_format=data_format)(x)
        x = layers.Conv2D(
            out_ch,
            3,
            strides=stride,
            padding="valid",
            use_bias=False,
            data_format=data_format,
            name=f"backbone_embedder_{i}_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=1e-5,
            momentum=0.1,
            name=f"backbone_embedder_{i}_bn",
        )(x)
        x = layers.ReLU()(x)
    x = layers.ZeroPadding2D(padding=1, data_format=data_format)(x)
    x = layers.MaxPooling2D(
        pool_size=3, strides=2, padding="valid", data_format=data_format
    )(x)
    stage_outputs = []
    is_basic = layer_type == "basic"
    if is_basic:
        filters_list = list(hidden_sizes)
    else:
        filters_list = [hs // 4 for hs in hidden_sizes]
    for si, nb in enumerate(depths):
        filt = filters_list[si]
        out_ch = hidden_sizes[si]
        for bi in range(nb):
            pf = f"backbone_stage{si}_{bi}"
            st = 2 if bi == 0 and si > 0 else 1
            res = x
            in_ch = res.shape[channels_axis]
            if is_basic:
                if st > 1:
                    x = layers.ZeroPadding2D(padding=1, data_format=data_format)(x)
                    x = layers.Conv2D(
                        out_ch,
                        3,
                        strides=st,
                        padding="valid",
                        use_bias=False,
                        data_format=data_format,
                        name=f"{pf}_conv1",
                    )(x)
                else:
                    x = layers.Conv2D(
                        out_ch,
                        3,
                        padding="same",
                        use_bias=False,
                        data_format=data_format,
                        name=f"{pf}_conv1",
                    )(x)
                x = layers.BatchNormalization(
                    axis=channels_axis, epsilon=1e-5, momentum=0.1, name=f"{pf}_bn1"
                )(x)
                x = layers.ReLU()(x)
                x = layers.Conv2D(
                    out_ch,
                    3,
                    padding="same",
                    use_bias=False,
                    data_format=data_format,
                    name=f"{pf}_conv2",
                )(x)
                x = layers.BatchNormalization(
                    axis=channels_axis, epsilon=1e-5, momentum=0.1, name=f"{pf}_bn2"
                )(x)
            else:
                x = layers.Conv2D(
                    filt,
                    1,
                    padding="valid",
                    use_bias=False,
                    data_format=data_format,
                    name=f"{pf}_conv1",
                )(x)
                x = layers.BatchNormalization(
                    axis=channels_axis, epsilon=1e-5, momentum=0.1, name=f"{pf}_bn1"
                )(x)
                x = layers.ReLU()(x)
                if st > 1:
                    x = layers.ZeroPadding2D(padding=1, data_format=data_format)(x)
                    x = layers.Conv2D(
                        filt,
                        3,
                        strides=st,
                        padding="valid",
                        use_bias=False,
                        data_format=data_format,
                        name=f"{pf}_conv2",
                    )(x)
                else:
                    x = layers.Conv2D(
                        filt,
                        3,
                        padding="same",
                        use_bias=False,
                        data_format=data_format,
                        name=f"{pf}_conv2",
                    )(x)
                x = layers.BatchNormalization(
                    axis=channels_axis, epsilon=1e-5, momentum=0.1, name=f"{pf}_bn2"
                )(x)
                x = layers.ReLU()(x)
                x = layers.Conv2D(
                    out_ch,
                    1,
                    padding="valid",
                    use_bias=False,
                    data_format=data_format,
                    name=f"{pf}_conv3",
                )(x)
                x = layers.BatchNormalization(
                    axis=channels_axis, epsilon=1e-5, momentum=0.1, name=f"{pf}_bn3"
                )(x)
            needs_shortcut = in_ch != out_ch or st != 1 or (is_basic and bi == 0)
            if needs_shortcut:
                if si == 0 and bi == 0:
                    res = layers.Conv2D(
                        out_ch,
                        1,
                        padding="valid",
                        use_bias=False,
                        data_format=data_format,
                        name=f"{pf}_shortcut_conv",
                    )(res)
                    res = layers.BatchNormalization(
                        axis=channels_axis,
                        epsilon=1e-5,
                        momentum=0.1,
                        name=f"{pf}_shortcut_bn",
                    )(res)
                else:
                    if st > 1:
                        res = layers.AveragePooling2D(
                            pool_size=2,
                            strides=2,
                            padding="valid",
                            data_format=data_format,
                        )(res)
                    res = layers.Conv2D(
                        out_ch,
                        1,
                        padding="valid",
                        use_bias=False,
                        data_format=data_format,
                        name=f"{pf}_shortcut_conv",
                    )(res)
                    res = layers.BatchNormalization(
                        axis=channels_axis,
                        epsilon=1e-5,
                        momentum=0.1,
                        name=f"{pf}_shortcut_bn",
                    )(res)
            x = layers.Add()([x, res])
            x = layers.ReLU()(x)
        stage_outputs.append(x)
    return stage_outputs[1], stage_outputs[2], stage_outputs[3]


def rt_detr_conv_norm(
    x,
    out_ch,
    ks,
    stride,
    padding=None,
    activation=None,
    data_format=None,
    channels_axis=-1,
    name="",
):
    """Convolution + BatchNorm + optional activation block.

    Applies zero-padding, a ``Conv2D`` without bias, batch
    normalization, and an optional activation function. Used as the
    building block for CCFM lateral convolutions, downsample
    convolutions, and RepVGG/CSP blocks.

    Reference:
        - `RT-DETR <https://arxiv.org/abs/2304.08069>`_

    Args:
        x: Input tensor.
        out_ch: Integer, number of output channels.
        ks: Integer, kernel size.
        stride: Integer, convolution stride.
        padding: Integer or ``None``. If ``None``, defaults to
            ``(ks - 1) // 2``.
        activation: String or ``None``, activation name
            (e.g., ``"silu"``).
        data_format: String, Keras data format.
        channels_axis: Integer, channel axis index.
        name: String, layer name prefix.

    Returns:
        Output tensor with the same spatial layout convention as
        the input.
    """
    pad = (ks - 1) // 2 if padding is None else padding
    if pad > 0:
        x = layers.ZeroPadding2D(padding=pad, data_format=data_format)(x)
    x = layers.Conv2D(
        out_ch,
        ks,
        strides=stride,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis, epsilon=1e-5, momentum=0.1, name=f"{name}_norm"
    )(x)
    if activation is not None:
        x = layers.Activation(activation, name=f"{name}_act")(x)
    return x


def rt_detr_rep_vgg_block(
    x, ch, activation="silu", data_format=None, channels_axis=-1, name=""
):
    """RepVGG block with parallel 3x3 and 1x1 convolution branches.

    Applies two parallel convolution paths (3x3 and 1x1), sums their
    outputs, and applies an activation. Used within
    ``rt_detr_csp_rep_layer`` as the core feature refinement unit.

    Reference:
        - `RepVGG: Making VGG-style ConvNets Great Again
          <https://arxiv.org/abs/2101.03697>`_
        - `RT-DETR <https://arxiv.org/abs/2304.08069>`_

    Args:
        x: Input tensor.
        ch: Integer, number of channels for both branches.
        activation: String, activation name. Defaults to ``"silu"``.
        data_format: String, Keras data format.
        channels_axis: Integer, channel axis index.
        name: String, layer name prefix.

    Returns:
        Output tensor of the same shape as the input.
    """
    b1 = rt_detr_conv_norm(
        x,
        ch,
        3,
        1,
        padding=1,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv1",
    )
    b2 = rt_detr_conv_norm(
        x,
        ch,
        1,
        1,
        padding=0,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv2",
    )
    y = layers.Add(name=f"{name}_add")([b1, b2])
    return layers.Activation(activation, name=f"{name}_act")(y)


def rt_detr_csp_rep_layer(
    x,
    out_ch,
    expansion=1.0,
    depths=3,
    activation="silu",
    data_format=None,
    channels_axis=-1,
    name="",
):
    """Cross Stage Partial layer with RepVGG blocks.

    Splits the input into two paths: one passes through a sequence of
    ``rt_detr_rep_vgg_block`` bottlenecks, the other through a single
    1x1 convolution. Both paths are summed and optionally projected to
    the target output channels. Used in the FPN and PAN stages of the
    CCFM hybrid encoder.

    Reference:
        - `RT-DETR <https://arxiv.org/abs/2304.08069>`_

    Args:
        x: Input tensor (typically a concatenation with
            ``2 * out_ch`` channels).
        out_ch: Integer, output channel dimension.
        expansion: Float, hidden channel expansion ratio relative to
            ``out_ch``. Defaults to ``1.0``.
        depths: Integer, number of RepVGG bottleneck blocks.
            Defaults to ``3``.
        activation: String, activation name. Defaults to ``"silu"``.
        data_format: String, Keras data format.
        channels_axis: Integer, channel axis index.
        name: String, layer name prefix.

    Returns:
        Output tensor with ``out_ch`` channels.
    """
    hid = int(out_ch * expansion)
    p1 = rt_detr_conv_norm(
        x,
        hid,
        1,
        1,
        padding=0,
        activation=activation,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv1",
    )
    for i in range(depths):
        p1 = rt_detr_rep_vgg_block(
            p1,
            hid,
            activation=activation,
            data_format=data_format,
            channels_axis=channels_axis,
            name=f"{name}_bottlenecks_{i}",
        )
    p2 = rt_detr_conv_norm(
        x,
        hid,
        1,
        1,
        padding=0,
        activation=activation,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv2",
    )
    merged = layers.Add(name=f"{name}_merge")([p1, p2])
    if hid != out_ch:
        merged = rt_detr_conv_norm(
            merged,
            out_ch,
            1,
            1,
            padding=0,
            activation=activation,
            data_format=data_format,
            channels_axis=channels_axis,
            name=f"{name}_conv3",
        )
    return merged


def rt_detr_aifi_encoder_layer(
    x,
    pos_embed,
    hidden_dim,
    num_heads,
    mlp_dim,
    activation="gelu",
    name="aifi_0_layers_0",
):
    """Single AIFI transformer encoder layer.

    Applies self-attention followed by a feedforward network, each with
    a residual connection and post-norm layer normalization. Positional
    embeddings are added to the query and key inputs of self-attention
    but not to the values. Used for intra-scale feature interaction on
    the highest-level feature map.

    Reference:
        - `RT-DETR <https://arxiv.org/abs/2304.08069>`_

    Args:
        x: Input tensor of shape
            ``(batch_size, seq_len, hidden_dim)``.
        pos_embed: Positional embedding tensor of shape
            ``(1, seq_len, hidden_dim)``, added to the query and key
            inputs of self-attention.
        hidden_dim: Integer, model dimension.
        num_heads: Integer, number of attention heads.
        mlp_dim: Integer, intermediate dimension of the feedforward
            network.
        activation: String, FFN activation function name.
            Defaults to ``"gelu"``.
        name: String, name prefix for all sub-layers in this block.
            Defaults to ``"aifi_0_layers_0"``.

    Returns:
        Output tensor of shape ``(batch_size, seq_len, hidden_dim)``.
    """
    sa = RTDETRV2MultiHeadAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        block_prefix=f"{name}_self_attn",
        name=f"{name}_self_attn",
    )
    q = k = layers.Add(name=f"{name}_sa_qk_add")([x, pos_embed])
    residual = x
    attn = sa(q, k, x)
    x = layers.LayerNormalization(epsilon=1e-5, name=f"{name}_self_attn_layer_norm")(
        layers.Add(name=f"{name}_sa_res")([residual, attn])
    )
    residual = x
    ff = layers.Dense(mlp_dim, name=f"{name}_fc1")(x)
    ff = layers.Activation(activation, name=f"{name}_gelu")(ff)
    ff = layers.Dense(hidden_dim, name=f"{name}_fc2")(ff)
    x = layers.LayerNormalization(epsilon=1e-5, name=f"{name}_final_layer_norm")(
        layers.Add(name=f"{name}_ff_res")([residual, ff])
    )
    return x


def rt_detr_v2_decoder_layer(
    hidden_states,
    encoder_hidden_states,
    query_pos,
    reference_points,
    hidden_dim,
    num_heads,
    dim_feedforward,
    activation,
    n_levels,
    n_points,
    spatial_shapes,
    level_start_index,
    name="decoder_layers_0",
):
    """Single RT-DETRv2 decoder layer built with the functional API.

    Composes self-attention, multi-scale deformable cross-attention, and a
    feedforward network with post-norm residual connections. Mirrors the
    style of ``rt_detr_aifi_encoder_layer``.

    Reference:
        - `RT-DETR <https://arxiv.org/abs/2304.08069>`_

    Args:
        hidden_states: Query tensor of shape ``(B, num_queries, hidden_dim)``.
        encoder_hidden_states: Flattened multi-scale memory of shape
            ``(B, total_tokens, hidden_dim)``.
        query_pos: Positional embedding for queries, broadcast over
            self-attention Q/K and added to the deformable attention input.
        reference_points: Reference boxes of shape ``(B, num_queries, 1, 4)``.
        hidden_dim: Model hidden dimension.
        num_heads: Number of attention heads.
        dim_feedforward: FFN intermediate dimension.
        activation: FFN activation name.
        n_levels: Number of feature levels for deformable attention.
        n_points: Sampling points per level/head.
        spatial_shapes: List of ``(H, W)`` tuples per feature level.
        level_start_index: List of token start indices per level.
        name: String, layer name prefix shared by all sub-layers.

    Returns:
        Output tensor of shape ``(B, num_queries, hidden_dim)``.
    """
    sa = RTDETRV2MultiHeadAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        block_prefix=f"{name}_self_attn",
        name=f"{name}_self_attn",
    )
    q = k = layers.Add(name=f"{name}_sa_qk_add")([hidden_states, query_pos])
    residual = hidden_states
    attn_out = sa(q, k, hidden_states)
    hidden_states = layers.LayerNormalization(
        epsilon=1e-5, name=f"{name}_self_attn_layer_norm"
    )(layers.Add(name=f"{name}_sa_res")([residual, attn_out]))

    ea = RTDETRV2MultiScaleDeformableAttention(
        hidden_dim=hidden_dim,
        n_levels=n_levels,
        n_heads=num_heads,
        n_points=n_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        name=f"{name}_encoder_attn",
    )
    residual = hidden_states
    cross_out = ea(
        hidden_states,
        reference_points,
        encoder_hidden_states,
        position_embeddings=query_pos,
    )
    hidden_states = layers.LayerNormalization(
        epsilon=1e-5, name=f"{name}_encoder_attn_layer_norm"
    )(layers.Add(name=f"{name}_ca_res")([residual, cross_out]))

    residual = hidden_states
    ff = layers.Dense(dim_feedforward, activation=activation, name=f"{name}_fc1")(
        hidden_states
    )
    ff = layers.Dense(hidden_dim, name=f"{name}_fc2")(ff)
    hidden_states = layers.LayerNormalization(
        epsilon=1e-5, name=f"{name}_final_layer_norm"
    )(layers.Add(name=f"{name}_ff_res")([residual, ff]))

    return hidden_states


def rt_detr_inverse_sigmoid(t, e=1e-5):
    """Inverse sigmoid (logit) used by iterative bbox refinement.

    Args:
        t: Input tensor with values in ``(0, 1)``.
        e: Epsilon for clipping. Defaults to ``1e-5``.

    Returns:
        ``log(t / (1 - t))`` with ``t`` clipped to ``[e, 1 - e]``.
    """
    t = ops.clip(t, e, 1 - e)
    return ops.log(t / (1 - t))


def rt_detr_hybrid_encoder(
    bk_feats,
    encoder_hidden_dim,
    encoder_num_layers,
    encoder_ffn_dim,
    encoder_num_heads,
    encode_proj_layers,
    encoder_activation_function,
    activation_function,
    hidden_expansion,
    num_feature_levels,
    feat_strides,
    spatial_h,
    spatial_w,
):
    """Build RT-DETRv2's hybrid encoder: AIFI transformer + CCFM (FPN + PAN).

    Each backbone stage feature is first projected to
    ``encoder_hidden_dim`` via a 1x1 conv + batch norm. The AIFI
    transformer (one or more post-norm self-attention encoder layers
    with sine positional embeddings) is then applied on the feature
    levels listed in ``encode_proj_layers`` for intra-scale feature
    interaction. Finally, a CCFM cross-scale fusion runs a top-down
    FPN followed by a bottom-up PAN, each fusing adjacent levels via
    ``rt_detr_csp_rep_layer`` blocks.

    Args:
        bk_feats: List of backbone feature tensors, one per pyramid level.
        encoder_hidden_dim: Channel dim inside the hybrid encoder.
        encoder_num_layers: Number of AIFI transformer encoder layers
            applied at each ``encode_proj_layers`` level.
        encoder_ffn_dim: FFN dim inside each AIFI layer.
        encoder_num_heads: Attention heads inside each AIFI layer.
        encode_proj_layers: Feature-level indices on which to run AIFI.
        encoder_activation_function: Activation in the AIFI FFN.
        activation_function: Activation in CCFM convolutions.
        hidden_expansion: CSP hidden-channel expansion ratio in
            CCFM blocks.
        num_feature_levels: Number of multi-scale levels produced.
        feat_strides: Feature strides per level (e.g. ``(8, 16, 32)``).
        spatial_h: Input image height in pixels.
        spatial_w: Input image width in pixels.

    Returns:
        List of ``num_feature_levels`` post-PAN feature tensors,
        ordered from highest spatial resolution to lowest, each with
        ``encoder_hidden_dim`` channels.
    """
    data_format = keras.config.image_data_format()
    channels_axis = -1 if data_format == "channels_last" else 1

    proj_feats = []
    for i, feat in enumerate(bk_feats):
        p = layers.Conv2D(
            encoder_hidden_dim,
            1,
            padding="valid",
            use_bias=False,
            data_format=data_format,
            name=f"encoder_input_proj_{i}_conv",
        )(feat)
        p = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=1e-5,
            momentum=0.1,
            name=f"encoder_input_proj_{i}_bn",
        )(p)
        proj_feats.append(p)

    for ai, enc_lvl in enumerate(encode_proj_layers):
        feat = proj_feats[enc_lvl]
        if data_format == "channels_first":
            feat = layers.Permute((2, 3, 1), name=f"aifi_{ai}_to_cl")(feat)
        h = spatial_h // feat_strides[enc_lvl]
        w = spatial_w // feat_strides[enc_lvl]
        flat = layers.Reshape((h * w, encoder_hidden_dim), name=f"aifi_{ai}_flatten")(
            feat
        )
        pe = rt_detr_sine_pos_embed(h, w, encoder_hidden_dim, 10000)
        for li in range(encoder_num_layers):
            flat = rt_detr_aifi_encoder_layer(
                flat,
                pe,
                encoder_hidden_dim,
                encoder_num_heads,
                encoder_ffn_dim,
                activation=encoder_activation_function,
                name=f"aifi_{ai}_layers_{li}",
            )
        unflat = layers.Reshape(
            (h, w, encoder_hidden_dim), name=f"aifi_{ai}_unflatten"
        )(flat)
        if data_format == "channels_first":
            unflat = layers.Permute((3, 1, 2), name=f"aifi_{ai}_to_cf")(unflat)
        proj_feats[enc_lvl] = unflat

    num_fpn = num_feature_levels - 1
    fpn = [proj_feats[-1]]
    for idx in range(num_fpn):
        bk_feat = proj_feats[num_fpn - idx - 1]
        top = fpn[-1]
        top = rt_detr_conv_norm(
            top,
            encoder_hidden_dim,
            1,
            1,
            padding=0,
            activation=activation_function,
            data_format=data_format,
            channels_axis=channels_axis,
            name=f"lateral_convs_{idx}",
        )
        fpn[-1] = top
        top = layers.UpSampling2D(
            size=2,
            interpolation="nearest",
            data_format=data_format,
            name=f"fpn_up_{idx}",
        )(top)
        fused = layers.Concatenate(axis=channels_axis, name=f"fpn_cat_{idx}")(
            [top, bk_feat]
        )
        fpn.append(
            rt_detr_csp_rep_layer(
                fused,
                encoder_hidden_dim,
                expansion=hidden_expansion,
                activation=activation_function,
                data_format=data_format,
                channels_axis=channels_axis,
                name=f"fpn_blocks_{idx}",
            )
        )
    fpn.reverse()

    pan = [fpn[0]]
    for idx in range(num_fpn):
        top_pan = pan[-1]
        fpn_feat = fpn[idx + 1]
        down = rt_detr_conv_norm(
            top_pan,
            encoder_hidden_dim,
            3,
            2,
            padding=1,
            activation=activation_function,
            data_format=data_format,
            channels_axis=channels_axis,
            name=f"downsample_convs_{idx}",
        )
        fused = layers.Concatenate(axis=channels_axis, name=f"pan_cat_{idx}")(
            [down, fpn_feat]
        )
        pan.append(
            rt_detr_csp_rep_layer(
                fused,
                encoder_hidden_dim,
                expansion=hidden_expansion,
                activation=activation_function,
                data_format=data_format,
                channels_axis=channels_axis,
                name=f"pan_blocks_{idx}",
            )
        )

    return pan


def rt_detr_decoder_inputs(
    pan,
    hidden_dim,
    feat_strides,
    spatial_h,
    spatial_w,
):
    """Prepare flattened tokens and anchor proposals for the RT-DETRv2 decoder.

    Projects each post-PAN feature level to ``hidden_dim`` via a 1x1 conv +
    batch norm, flattens the multi-scale feature maps into a single
    ``(B, sum(H*W), hidden_dim)`` token sequence, and generates a regular
    anchor grid of ``(cx, cy, w, h)`` boxes per token. Anchors are
    converted to logits (sigmoid inverse) using numpy at graph build
    time; anchors near the image border are masked out and given a
    large sentinel logit.

    Args:
        pan: List of post-PAN feature tensors from
            :func:`rt_detr_hybrid_encoder`.
        hidden_dim: Decoder model dimension.
        feat_strides: Feature strides per level used to derive per-level
            spatial shapes.
        spatial_h: Input image height in pixels.
        spatial_w: Input image width in pixels.

    Returns:
        source_flat: ``(B, sum(H*W), hidden_dim)`` flattened decoder memory
            across all feature levels.
        spatial_shapes: List of ``(H, W)`` per feature level.
        level_start: List of token start indices per level (cumulative
            ``H*W`` offsets).
        anchors_t: Anchor proposals as sigmoid-inverse logits, shape
            ``(1, sum(H*W), 4)``.
        vmask_t: Validity mask (float, 1.0 for valid anchors), shape
            ``(1, sum(H*W), 1)``.
    """
    data_format = keras.config.image_data_format()
    channels_axis = -1 if data_format == "channels_last" else 1

    dec_sources = []
    for i, feat in enumerate(pan):
        p = layers.Conv2D(
            hidden_dim,
            1,
            padding="valid",
            use_bias=False,
            data_format=data_format,
            name=f"decoder_input_proj_{i}_conv",
        )(feat)
        p = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=1e-5,
            momentum=0.1,
            name=f"decoder_input_proj_{i}_bn",
        )(p)
        dec_sources.append(p)

    spatial_shapes = [(spatial_h // s, spatial_w // s) for s in feat_strides]
    flat_list = []
    for i, src in enumerate(dec_sources):
        hi, wi = spatial_shapes[i]
        if data_format == "channels_first":
            src = layers.Permute((2, 3, 1), name=f"dec_flat_{i}_to_cl")(src)
        flat_list.append(
            layers.Reshape((hi * wi, hidden_dim), name=f"dec_flat_{i}")(src)
        )
    source_flat = layers.Concatenate(axis=1, name="dec_src_cat")(flat_list)

    level_start = []
    cum = 0
    for hi, wi in spatial_shapes:
        level_start.append(cum)
        cum += hi * wi

    gs = 0.05
    anc_parts = []
    for lvl, (hi, wi) in enumerate(spatial_shapes):
        gy, gx = ops.meshgrid(
            ops.cast(ops.arange(hi), "float32"),
            ops.cast(ops.arange(wi), "float32"),
            indexing="ij",
        )
        xy = ops.reshape(ops.stack([gx, gy], axis=-1), [1, hi * wi, 2])
        xy = (xy + 0.5) / ops.convert_to_tensor(
            [[[float(wi), float(hi)]]], dtype="float32"
        )
        wh = ops.ones_like(xy) * gs * (2.0**lvl)
        anc_parts.append(ops.concatenate([xy, wh], axis=-1))
    anchors = ops.concatenate(anc_parts, axis=1)
    vmask = ops.cast(
        ops.all((anchors > 1e-2) & (anchors < 1 - 1e-2), axis=-1, keepdims=True),
        "float32",
    )
    anc_logit = ops.where(
        vmask > 0.5,
        ops.log(anchors / (1 - anchors)),
        ops.convert_to_tensor(3.4028235e38, dtype="float32"),
    )
    anchors_t = anc_logit
    vmask_t = vmask

    return source_flat, spatial_shapes, level_start, anchors_t, vmask_t


def rt_detr_two_stage_proposals(
    source_flat,
    anchors_t,
    vmask_t,
    hidden_dim,
    num_classes,
    num_queries,
):
    """Two-stage encoder query selection.

    Projects + normalizes the encoder memory, predicts class scores and
    bbox refinement deltas on every token, refines the anchor proposals
    by adding the bbox deltas (in sigmoid-inverse logit space), and
    selects the top-``num_queries`` highest-scoring tokens as initial
    decoder queries. Both the selected query features and the
    corresponding refined refpoint logits are detached from the
    encoder gradient path.

    Args:
        source_flat: ``(B, total_tokens, hidden_dim)`` flattened decoder
            memory from :func:`rt_detr_decoder_inputs`.
        anchors_t: Anchor proposals as sigmoid-inverse logits.
        vmask_t: Validity mask (float).
        hidden_dim: Decoder model dimension.
        num_classes: Number of classes for the first-stage scoring head.
        num_queries: Number of decoder queries / top-k selected tokens.

    Returns:
        target: ``(B, num_queries, hidden_dim)`` selected query features.
        ref_logit: ``(B, num_queries, 4)`` selected refined reference
            points in sigmoid-inverse logit space.
    """
    memory = source_flat * vmask_t
    enc_out = layers.Dense(hidden_dim, name="enc_output_linear")(memory)
    enc_out = layers.LayerNormalization(epsilon=1e-5, name="enc_output_layernorm")(
        enc_out
    )
    enc_scores = layers.Dense(num_classes, name="enc_score_head")(enc_out)
    enc_bb = layers.Dense(hidden_dim, activation="relu", name="enc_bbox_head_0")(
        enc_out
    )
    enc_bb = layers.Dense(hidden_dim, activation="relu", name="enc_bbox_head_1")(enc_bb)
    enc_bb = layers.Dense(4, name="enc_bbox_head_2")(enc_bb)
    enc_bb_logits = enc_bb + anchors_t

    max_sc = ops.max(enc_scores, axis=-1)
    _, topk_idx = ops.top_k(max_sc, k=num_queries)
    idx3 = ops.expand_dims(topk_idx, -1)
    target = ops.take_along_axis(enc_out, idx3, axis=1)
    target = ops.stop_gradient(target)
    idx4 = ops.repeat(idx3, 4, axis=-1)
    ref_logit = ops.take_along_axis(enc_bb_logits, idx4, axis=1)
    ref_logit = ops.stop_gradient(ref_logit)

    return target, ref_logit


def rt_detr_v2_decoder(
    target,
    ref_logit,
    source_flat,
    spatial_shapes,
    level_start,
    hidden_dim,
    decoder_num_layers,
    decoder_ffn_dim,
    decoder_num_heads,
    decoder_n_points,
    decoder_activation_function,
    num_feature_levels,
):
    """RT-DETRv2 deformable decoder with iterative bounding box refinement.

    Runs ``decoder_num_layers`` :func:`rt_detr_v2_decoder_layer` blocks
    (functional-API decoder layers using V2's multi-head attention and
    multi-scale deformable attention) on top of the flattened
    multi-scale encoder memory. Each layer takes the current query
    features, a query positional encoding derived from the current
    reference points, and the reference points themselves for
    deformable cross-attention. After each layer, per-layer bbox embed
    heads refine the refpoints via residual addition in sigmoid-inverse
    logit space; refpoints are detached from the gradient between
    layers.

    Args:
        target: Initial query features ``(B, num_queries, hidden_dim)``
            from :func:`rt_detr_two_stage_proposals`.
        ref_logit: Initial reference points in sigmoid-inverse logit
            space, shape ``(B, num_queries, 4)``.
        source_flat: Flattened multi-scale encoder memory
            ``(B, total_tokens, hidden_dim)``.
        spatial_shapes: List of ``(H, W)`` per feature level.
        level_start: List of token start indices per level.
        hidden_dim: Decoder model dimension.
        decoder_num_layers: Number of decoder layers.
        decoder_ffn_dim: FFN dim inside each decoder layer.
        decoder_num_heads: Attention heads in each decoder layer.
        decoder_n_points: Sampling points per level for deformable
            attention.
        decoder_activation_function: Activation in the decoder FFN.
        num_feature_levels: Number of multi-scale feature levels.

    Returns:
        hs_last: ``(B, num_queries, hidden_dim)`` decoder last hidden
            state after the final layer.
        last_boxes: ``(B, num_queries, 4)`` final refined boxes in
            ``(cx, cy, w, h)`` normalized coordinates.
    """
    qp_d0 = layers.Dense(hidden_dim * 2, activation="relu", name="query_pos_head_0")
    qp_d1 = layers.Dense(hidden_dim, name="query_pos_head_1")
    hs = target
    ref_pts = ops.sigmoid(ref_logit)
    last_boxes = None

    for di in range(decoder_num_layers):
        query_pos = qp_d1(qp_d0(ref_pts))
        rp_in = ops.expand_dims(ref_pts, axis=2)
        hs = rt_detr_v2_decoder_layer(
            hs,
            source_flat,
            query_pos,
            rp_in,
            hidden_dim=hidden_dim,
            num_heads=decoder_num_heads,
            dim_feedforward=decoder_ffn_dim,
            activation=decoder_activation_function,
            n_levels=num_feature_levels,
            n_points=decoder_n_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start,
            name=f"decoder_layers_{di}",
        )
        bb_i = layers.Dense(hidden_dim, activation="relu", name=f"bbox_embed_{di}_0")(
            hs
        )
        bb_i = layers.Dense(hidden_dim, activation="relu", name=f"bbox_embed_{di}_1")(
            bb_i
        )
        bb_i = layers.Dense(4, name=f"bbox_embed_{di}_2")(bb_i)

        new_ref = ops.sigmoid(bb_i + rt_detr_inverse_sigmoid(ref_pts))
        ref_pts = ops.stop_gradient(new_ref)
        last_boxes = new_ref

    return hs, last_boxes


def rt_detr_v2_functional(
    inputs,
    backbone_hidden_sizes,
    backbone_block_repeats,
    backbone_embedding_size,
    backbone_layer_type,
    encoder_hidden_dim,
    encoder_num_layers,
    encoder_ffn_dim,
    encoder_num_heads,
    encode_proj_layers,
    encoder_activation_function,
    activation_function,
    hidden_expansion,
    hidden_dim,
    decoder_num_layers,
    decoder_ffn_dim,
    decoder_num_heads,
    decoder_n_points,
    decoder_activation_function,
    num_feature_levels,
    feat_strides,
    num_queries,
    num_classes,
    input_shape,
):
    """Build the full RT-DETRv2 architecture from an input tensor (no class heads).

    Top-level orchestrator that wires the four architectural stages:

    1. :func:`rt_detr_backbone`, ResNet-vd backbone producing
       multi-scale features (typically at strides 8, 16, 32).
    2. :func:`rt_detr_hybrid_encoder`, AIFI transformer + CCFM
       (FPN + PAN) for cross-scale fusion.
    3. :func:`rt_detr_decoder_inputs` + :func:`rt_detr_two_stage_proposals`
       token preparation, anchor generation, and two-stage encoder
       query selection.
    4. :func:`rt_detr_v2_decoder`, deformable decoder with iterative
       bbox refinement, using V2's functional-API decoder layer.

    Per-layer class prediction heads are intentionally not built here:
    they are added by :class:`RTDETRV2Detect`, which composes
    :class:`RTDetrV2Model` around this graph.

    Args:
        inputs: Keras input tensor of shape ``(B, H, W, 3)`` (or
            ``(B, 3, H, W)`` for ``channels_first``).
        backbone_hidden_sizes: Output channels per backbone stage.
        backbone_block_repeats: Number of residual blocks per stage.
        backbone_embedding_size: Stem output channels.
        backbone_layer_type: ``"bottleneck"`` or ``"basic"``.
        encoder_hidden_dim: Channel dim inside the hybrid encoder.
        encoder_num_layers: Number of AIFI transformer encoder layers.
        encoder_ffn_dim: FFN dim in the AIFI encoder.
        encoder_num_heads: Attention heads in the AIFI encoder.
        encode_proj_layers: Feature-level indices where AIFI is applied.
        encoder_activation_function: Activation in the AIFI FFN.
        activation_function: Activation in CCFM (FPN/PAN) blocks.
        hidden_expansion: CSP hidden channel expansion ratio.
        hidden_dim: Decoder model dimension.
        decoder_num_layers: Number of decoder layers.
        decoder_ffn_dim: FFN dim in each decoder layer.
        decoder_num_heads: Attention heads in the decoder.
        decoder_n_points: Sampling points per level for deformable
            attention.
        decoder_activation_function: Activation in the decoder FFN.
        num_feature_levels: Number of multi-scale levels.
        feat_strides: Feature strides per level.
        num_queries: Number of decoder queries.
        num_classes: Number of classes for the first-stage scoring head.
        input_shape: ``(H, W, C)`` shape of ``inputs`` (or
            ``(C, H, W)`` for ``channels_first``).

    Returns:
        hs_last: ``(B, num_queries, hidden_dim)`` decoder last hidden state.
        last_boxes: ``(B, num_queries, 4)`` final refined boxes in
            ``(cx, cy, w, h)``.
    """
    data_format = keras.config.image_data_format()
    channels_axis = -1 if data_format == "channels_last" else 1

    if data_format == "channels_first":
        spatial_h, spatial_w = input_shape[1], input_shape[2]
    else:
        spatial_h, spatial_w = input_shape[0], input_shape[1]

    feat_s3, feat_s4, feat_s5 = rt_detr_backbone(
        inputs,
        list(backbone_block_repeats),
        list(backbone_hidden_sizes),
        backbone_embedding_size,
        layer_type=backbone_layer_type,
        data_format=data_format,
        channels_axis=channels_axis,
    )
    bk_feats = [feat_s3, feat_s4, feat_s5]

    pan = rt_detr_hybrid_encoder(
        bk_feats,
        encoder_hidden_dim=encoder_hidden_dim,
        encoder_num_layers=encoder_num_layers,
        encoder_ffn_dim=encoder_ffn_dim,
        encoder_num_heads=encoder_num_heads,
        encode_proj_layers=encode_proj_layers,
        encoder_activation_function=encoder_activation_function,
        activation_function=activation_function,
        hidden_expansion=hidden_expansion,
        num_feature_levels=num_feature_levels,
        feat_strides=feat_strides,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
    )

    source_flat, spatial_shapes, level_start, anchors_t, vmask_t = (
        rt_detr_decoder_inputs(
            pan,
            hidden_dim=hidden_dim,
            feat_strides=feat_strides,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
        )
    )

    target, ref_logit = rt_detr_two_stage_proposals(
        source_flat,
        anchors_t,
        vmask_t,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_queries=num_queries,
    )

    return rt_detr_v2_decoder(
        target,
        ref_logit,
        source_flat,
        spatial_shapes,
        level_start,
        hidden_dim=hidden_dim,
        decoder_num_layers=decoder_num_layers,
        decoder_ffn_dim=decoder_ffn_dim,
        decoder_num_heads=decoder_num_heads,
        decoder_n_points=decoder_n_points,
        decoder_activation_function=decoder_activation_function,
        num_feature_levels=num_feature_levels,
    )


@keras.saving.register_keras_serializable(package="kerasformers")
class RTDetrV2Model(BaseModel):
    """RT-DETR-V2 backbone + hybrid encoder + decoder (no class heads).

    Matches the reference ``RTDetrV2Model`` pattern: outputs the
    decoder ``last_hidden_state`` with shape ``(B, num_queries,
    hidden_dim)``. Iterative bbox refinement layers stay in the model
    (they feed back into the decoder); only per-layer class prediction
    heads are pruned from the output graph. Use ``RTDETRV2Detect`` for
    full detection outputs.

    Reference:
        - `RT-DETRv2: Improved Baseline with Bag-of-Freebies for
          Real-Time Detection Transformer
          <https://arxiv.org/abs/2407.17140>`_
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "rt_detr_v2"

    def __init__(
        self,
        backbone_hidden_sizes=(256, 512, 1024, 2048),
        backbone_block_repeats=(3, 4, 6, 3),
        backbone_embedding_size=64,
        backbone_layer_type="bottleneck",
        encoder_in_channels=(512, 1024, 2048),
        encoder_hidden_dim=256,
        encoder_num_layers=1,
        encoder_ffn_dim=1024,
        encoder_num_heads=8,
        encode_proj_layers=(2,),
        encoder_activation_function="gelu",
        activation_function="silu",
        hidden_expansion=1.0,
        hidden_dim=256,
        decoder_num_layers=6,
        decoder_ffn_dim=1024,
        decoder_num_heads=8,
        decoder_n_points=4,
        decoder_activation_function="relu",
        num_feature_levels=3,
        feat_strides=(8, 16, 32),
        num_queries=300,
        num_classes=80,
        image_size=640,
        input_tensor=None,
        name="RTDetrV2Model",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        image_size = standardize_input_shape(image_size, data_format)

        if input_tensor is None:
            img_input = layers.Input(shape=image_size)
        else:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=image_size)
            else:
                img_input = input_tensor

        hs_last, last_boxes = rt_detr_v2_functional(
            img_input,
            backbone_hidden_sizes=backbone_hidden_sizes,
            backbone_block_repeats=backbone_block_repeats,
            backbone_embedding_size=backbone_embedding_size,
            backbone_layer_type=backbone_layer_type,
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_num_layers=encoder_num_layers,
            encoder_ffn_dim=encoder_ffn_dim,
            encoder_num_heads=encoder_num_heads,
            encode_proj_layers=encode_proj_layers,
            encoder_activation_function=encoder_activation_function,
            activation_function=activation_function,
            hidden_expansion=hidden_expansion,
            hidden_dim=hidden_dim,
            decoder_num_layers=decoder_num_layers,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_num_heads=decoder_num_heads,
            decoder_n_points=decoder_n_points,
            decoder_activation_function=decoder_activation_function,
            num_feature_levels=num_feature_levels,
            feat_strides=feat_strides,
            num_queries=num_queries,
            num_classes=num_classes,
            input_shape=image_size,
        )

        outputs = {"last_hidden_state": hs_last, "last_boxes": last_boxes}
        super().__init__(inputs=img_input, outputs=outputs, name=name, **kwargs)

        self._backbone_hidden_sizes = list(backbone_hidden_sizes)
        self._backbone_block_repeats = list(backbone_block_repeats)
        self._backbone_embedding_size = backbone_embedding_size
        self._backbone_layer_type = backbone_layer_type
        self._encoder_in_channels = list(encoder_in_channels)
        self._encoder_hidden_dim = encoder_hidden_dim
        self._encoder_layers = encoder_num_layers
        self._encoder_ffn_dim = encoder_ffn_dim
        self._encoder_num_heads = encoder_num_heads
        self._encode_proj_layers = list(encode_proj_layers)
        self._encoder_activation_function = encoder_activation_function
        self._activation_function = activation_function
        self._hidden_expansion = hidden_expansion
        self._d_model = hidden_dim
        self._decoder_layers = decoder_num_layers
        self._decoder_ffn_dim = decoder_ffn_dim
        self._decoder_num_heads = decoder_num_heads
        self._decoder_n_points = decoder_n_points
        self._decoder_activation_function = decoder_activation_function
        self._num_feature_levels = num_feature_levels
        self._feat_strides = list(feat_strides)
        self._num_queries = num_queries
        self._num_classes = num_classes
        self.image_size = image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone_hidden_sizes": self._backbone_hidden_sizes,
                "backbone_block_repeats": self._backbone_block_repeats,
                "backbone_embedding_size": self._backbone_embedding_size,
                "backbone_layer_type": self._backbone_layer_type,
                "encoder_in_channels": self._encoder_in_channels,
                "encoder_hidden_dim": self._encoder_hidden_dim,
                "encoder_num_layers": self._encoder_layers,
                "encoder_ffn_dim": self._encoder_ffn_dim,
                "encoder_num_heads": self._encoder_num_heads,
                "encode_proj_layers": self._encode_proj_layers,
                "encoder_activation_function": self._encoder_activation_function,
                "activation_function": self._activation_function,
                "hidden_expansion": self._hidden_expansion,
                "hidden_dim": self._d_model,
                "decoder_num_layers": self._decoder_layers,
                "decoder_ffn_dim": self._decoder_ffn_dim,
                "decoder_num_heads": self._decoder_num_heads,
                "decoder_n_points": self._decoder_n_points,
                "decoder_activation_function": self._decoder_activation_function,
                "num_feature_levels": self._num_feature_levels,
                "feat_strides": self._feat_strides,
                "num_queries": self._num_queries,
                "num_classes": self._num_classes,
                "image_size": self.image_size,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def config_from_hf(cls, hf_config):
        return RTDETRV2Detect.config_from_hf(hf_config)

    @classmethod
    def from_hf(cls, hf_id, load_weights=True, skip_mismatch=False, **kwargs):
        model = super().from_hf(hf_id, load_weights=False, **kwargs)
        if load_weights:
            src = RTDETRV2Detect.from_hf(hf_id, skip_mismatch=skip_mismatch)
            unmatched = copy_weights_by_path_suffix(src, model)
            if unmatched and not skip_mismatch:
                raise ValueError(
                    f"{cls.__name__}.from_hf: {len(unmatched)} weight(s) not "
                    f"matched from the {type(src).__name__} checkpoint: "
                    f"{unmatched[:5]}"
                )
            del src
        return model


@keras.saving.register_keras_serializable(package="kerasformers")
class RTDETRV2Detect(BaseModel):
    """RT-DETR: Real-Time DEtection TRansformer.

    A real-time object detection model that combines a ResNet-vd backbone
    with a hybrid encoder (AIFI transformer + CCFM feature pyramid) and a
    deformable DETR decoder with iterative bounding box refinement. Uses
    two-stage query initialization from encoder proposals and per-layer
    prediction heads.

    Reference:
        - `RT-DETR: DETRs Beat YOLOs on Real-time Object Detection
          <https://arxiv.org/abs/2304.08069>`_

    Args:
        backbone_hidden_sizes: Output channels per backbone stage.
        backbone_block_repeats: Number of residual blocks per stage.
        backbone_embedding_size: Stem output channels.
        backbone_layer_type: ``"bottleneck"`` (ResNet-50/101) or
            ``"basic"`` (ResNet-18/34).
        encoder_in_channels: Backbone channels fed to the encoder.
        encoder_hidden_dim: Hidden dimension of the hybrid encoder.
        encoder_num_layers: Number of AIFI transformer encoder layers.
        encoder_ffn_dim: FFN dimension in the AIFI encoder.
        encoder_num_heads: Attention heads in the AIFI encoder.
        encode_proj_layers: Feature level indices to apply AIFI to.
        encoder_activation_function: Activation in the AIFI FFN.
        activation_function: Activation in CCFM (FPN/PAN) blocks.
        hidden_expansion: CSP hidden channel expansion ratio.
        hidden_dim: Decoder model dimension.
        decoder_num_layers: Number of decoder layers.
        decoder_ffn_dim: FFN dimension in the decoder.
        decoder_num_heads: Attention heads in the decoder.
        decoder_n_points: Sampling points per level in deformable
            attention.
        decoder_activation_function: Activation in the decoder FFN.
        num_feature_levels: Number of multi-scale feature levels.
        feat_strides: Feature strides from the backbone.
        num_queries: Number of object queries.
        num_classes: Number of object classes (COCO: 80).
        weights: Pre-trained weight identifier or file path.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `640`.
        input_tensor: Optional input Keras tensor.
        name: Model name.
    """

    BASE_MODEL_CONFIG = None
    config_class = RTDetrV2Config
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "rt_detr_v2"

    def __init__(
        self,
        backbone_hidden_sizes=(256, 512, 1024, 2048),
        backbone_block_repeats=(3, 4, 6, 3),
        backbone_embedding_size=64,
        backbone_layer_type="bottleneck",
        encoder_in_channels=(512, 1024, 2048),
        encoder_hidden_dim=256,
        encoder_num_layers=1,
        encoder_ffn_dim=1024,
        encoder_num_heads=8,
        encode_proj_layers=(2,),
        encoder_activation_function="gelu",
        activation_function="silu",
        hidden_expansion=1.0,
        hidden_dim=256,
        decoder_num_layers=6,
        decoder_ffn_dim=1024,
        decoder_num_heads=8,
        decoder_n_points=4,
        decoder_activation_function="relu",
        num_feature_levels=3,
        feat_strides=(8, 16, 32),
        num_queries=300,
        num_classes=80,
        image_size=640,
        input_tensor=None,
        name="RTDETRV2Detect",
        **kwargs,
    ):
        base = RTDetrV2Model(
            backbone_hidden_sizes=backbone_hidden_sizes,
            backbone_block_repeats=backbone_block_repeats,
            backbone_embedding_size=backbone_embedding_size,
            backbone_layer_type=backbone_layer_type,
            encoder_in_channels=encoder_in_channels,
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_num_layers=encoder_num_layers,
            encoder_ffn_dim=encoder_ffn_dim,
            encoder_num_heads=encoder_num_heads,
            encode_proj_layers=encode_proj_layers,
            encoder_activation_function=encoder_activation_function,
            activation_function=activation_function,
            hidden_expansion=hidden_expansion,
            hidden_dim=hidden_dim,
            decoder_num_layers=decoder_num_layers,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_num_heads=decoder_num_heads,
            decoder_n_points=decoder_n_points,
            decoder_activation_function=decoder_activation_function,
            num_feature_levels=num_feature_levels,
            feat_strides=feat_strides,
            num_queries=num_queries,
            num_classes=num_classes,
            image_size=image_size,
            input_tensor=input_tensor,
            name=f"{name}_model",
        )
        hs_last = base.output["last_hidden_state"]
        last_boxes = base.output["last_boxes"]

        logits = layers.Dense(
            num_classes, name=f"class_embed_{decoder_num_layers - 1}"
        )(hs_last)

        outputs = {"logits": logits, "pred_boxes": last_boxes}
        super().__init__(inputs=base.input, outputs=outputs, name=name, **kwargs)

        self._backbone_hidden_sizes = list(backbone_hidden_sizes)
        self._backbone_block_repeats = list(backbone_block_repeats)
        self._backbone_embedding_size = backbone_embedding_size
        self._backbone_layer_type = backbone_layer_type
        self._encoder_in_channels = list(encoder_in_channels)
        self._encoder_hidden_dim = encoder_hidden_dim
        self._encoder_layers = encoder_num_layers
        self._encoder_ffn_dim = encoder_ffn_dim
        self._encoder_num_heads = encoder_num_heads
        self._encode_proj_layers = list(encode_proj_layers)
        self._encoder_activation_function = encoder_activation_function
        self._activation_function = activation_function
        self._hidden_expansion = hidden_expansion
        self._d_model = hidden_dim
        self._decoder_layers = decoder_num_layers
        self._decoder_ffn_dim = decoder_ffn_dim
        self._decoder_num_heads = decoder_num_heads
        self._decoder_n_points = decoder_n_points
        self._decoder_activation_function = decoder_activation_function
        self._num_feature_levels = num_feature_levels
        self._feat_strides = list(feat_strides)
        self._num_queries = num_queries
        self._num_classes = num_classes
        self.image_size = base.image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone_hidden_sizes": self._backbone_hidden_sizes,
                "backbone_block_repeats": self._backbone_block_repeats,
                "backbone_embedding_size": self._backbone_embedding_size,
                "backbone_layer_type": self._backbone_layer_type,
                "encoder_in_channels": self._encoder_in_channels,
                "encoder_hidden_dim": self._encoder_hidden_dim,
                "encoder_num_layers": self._encoder_layers,
                "encoder_ffn_dim": self._encoder_ffn_dim,
                "encoder_num_heads": self._encoder_num_heads,
                "encode_proj_layers": self._encode_proj_layers,
                "encoder_activation_function": self._encoder_activation_function,
                "activation_function": self._activation_function,
                "hidden_expansion": self._hidden_expansion,
                "hidden_dim": self._d_model,
                "decoder_num_layers": self._decoder_layers,
                "decoder_ffn_dim": self._decoder_ffn_dim,
                "decoder_num_heads": self._decoder_num_heads,
                "decoder_n_points": self._decoder_n_points,
                "decoder_activation_function": self._decoder_activation_function,
                "num_feature_levels": self._num_feature_levels,
                "feat_strides": self._feat_strides,
                "num_queries": self._num_queries,
                "num_classes": self._num_classes,
                "image_size": self.image_size,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def config_from_hf(cls, hf_config):
        # Hub config.json files omit defaulted keys (e.g. rtdetr_v2_r18vd ships
        # without num_attention_heads); fall back to the RTDetrV2Config defaults.
        bb = hf_config["backbone_config"]
        return {
            "backbone_hidden_sizes": tuple(
                bb.get("hidden_sizes", (256, 512, 1024, 2048))
            ),
            "backbone_block_repeats": tuple(bb.get("depths", (3, 4, 6, 3))),
            "backbone_embedding_size": bb.get("embedding_size", 64),
            "backbone_layer_type": bb.get("layer_type", "bottleneck"),
            "encoder_in_channels": tuple(
                hf_config.get("encoder_in_channels", (512, 1024, 2048))
            ),
            "encoder_hidden_dim": hf_config.get("encoder_hidden_dim", 256),
            "encoder_num_layers": hf_config.get("encoder_layers", 1),
            "encoder_ffn_dim": hf_config.get("encoder_ffn_dim", 1024),
            "encoder_num_heads": hf_config.get("num_attention_heads", 8),
            "encode_proj_layers": tuple(hf_config.get("encode_proj_layers", (2,))),
            "encoder_activation_function": hf_config.get(
                "encoder_activation_function", "gelu"
            ),
            "activation_function": hf_config.get("activation_function", "silu"),
            "hidden_expansion": hf_config.get("hidden_expansion", 1.0),
            "hidden_dim": hf_config.get("d_model", 256),
            "decoder_num_layers": hf_config.get("decoder_layers", 6),
            "decoder_ffn_dim": hf_config.get("decoder_ffn_dim", 1024),
            "decoder_num_heads": hf_config.get("decoder_attention_heads", 8),
            "decoder_n_points": hf_config.get("decoder_n_points", 4),
            "decoder_activation_function": hf_config.get(
                "decoder_activation_function", "relu"
            ),
            "num_feature_levels": hf_config.get("num_feature_levels", 3),
            "feat_strides": tuple(hf_config.get("feat_strides", (8, 16, 32))),
            "num_queries": hf_config.get("num_queries", 300),
            "num_classes": hf_num_classes(hf_config),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_rt_detr_v2_hf_to_keras import transfer_rt_detr_v2_weights

        transfer_rt_detr_v2_weights(keras_model, hf_state_dict)
