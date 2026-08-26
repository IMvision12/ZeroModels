import keras
from keras import layers, ops, utils

from kerasformers.base import BaseModel
from kerasformers.base.base_model import hf_num_classes
from kerasformers.conversion import copy_weights_by_path_suffix
from kerasformers.utils import standardize_input_shape

from .dfine_config import DFineConfig
from .dfine_layers import (
    DFineDecoderLayer,
    DFineDecoderParams,
    DFineLearnableAffineBlock,
    DFineMultiHeadAttention,
)


def dfine_sine_pos_embed(height, width, embed_dim, temperature=10000):
    """Compute 2D sinusoidal position embedding.

    Generates non-learnable sine/cosine positional encodings for a 2D
    spatial grid. The embedding dimension is split into four equal parts
    encoding height-sin, height-cos, width-sin, and width-cos.

    Reference:
        - `D-FINE <https://arxiv.org/abs/2410.13842>`_

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


def dfine_conv_bn(
    x,
    out_ch,
    ks,
    stride,
    groups=1,
    padding=None,
    activation="relu",
    use_lab=False,
    data_format=None,
    channels_axis=-1,
    name="",
):
    """Conv + BatchNorm + optional activation + optional LAB block.

    Reproduces the HGNetV2 conv layer from the reference implementation.

    Args:
        x: Input tensor.
        out_ch: Integer, number of output channels.
        ks: Integer, kernel size.
        stride: Integer, convolution stride.
        groups: Integer, number of convolution groups. Defaults to ``1``.
        padding: Integer or ``None``. If ``None``, defaults to
            ``(ks - 1) // 2``.
        activation: String or ``None``, activation name.
            Defaults to ``"relu"``.
        use_lab: Boolean, whether to apply a Learnable Affine Block
            after activation. Defaults to ``False``.
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
        groups=groups,
        data_format=data_format,
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.1,
        name=f"{name}_bn",
    )(x)
    if activation is not None:
        x = layers.Activation(activation, name=f"{name}_act")(x)
        if use_lab:
            x = dfine_lab_layer(x, name=f"{name}_lab")
    return x


def dfine_lab_layer(x, name=""):
    """Learnable Affine Block: scale * x + bias with scalar parameters.

    Args:
        x: Input tensor.
        name: String, layer name prefix.

    Returns:
        Output tensor with learnable affine transformation applied.
    """
    lab = DFineLearnableAffineBlock(name=name)
    return lab(x)


def dfine_light_conv_block(
    x,
    out_ch,
    ks,
    use_lab=False,
    data_format=None,
    channels_axis=-1,
    name="",
):
    """Light convolution block: 1x1 conv + depthwise conv.

    Mirrors HGNetV2ConvLayerLight.

    Args:
        x: Input tensor.
        out_ch: Integer, number of output channels.
        ks: Integer, kernel size for the depthwise convolution.
        use_lab: Boolean, whether to apply a Learnable Affine Block.
            Defaults to ``False``.
        data_format: String, Keras data format.
        channels_axis: Integer, channel axis index.
        name: String, layer name prefix.

    Returns:
        Output tensor.
    """
    x = dfine_conv_bn(
        x,
        out_ch,
        1,
        1,
        activation=None,
        use_lab=False,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv1",
    )
    x = dfine_conv_bn(
        x,
        out_ch,
        ks,
        1,
        groups=out_ch,
        activation="relu",
        use_lab=use_lab,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv2",
    )
    return x


def dfine_basic_layer(
    x,
    mid_ch,
    out_ch,
    num_layers,
    kernel_size=3,
    residual=False,
    light_block=False,
    use_lab=False,
    data_format=None,
    channels_axis=-1,
    name="",
):
    """HGNetV2 basic layer with sequential convolutions and aggregation.

    All intermediate outputs are concatenated and aggregated through
    two 1x1 convolutions (squeeze + excitation).

    Args:
        x: Input tensor.
        mid_ch: Integer, intermediate channel dimension.
        out_ch: Integer, output channel dimension.
        num_layers: Integer, number of sequential convolutions.
        kernel_size: Integer, convolution kernel size. Defaults to ``3``.
        residual: Boolean, whether to add a residual connection.
            Defaults to ``False``.
        light_block: Boolean, whether to use light convolution blocks.
            Defaults to ``False``.
        use_lab: Boolean, whether to apply Learnable Affine Blocks.
            Defaults to ``False``.
        data_format: String, Keras data format.
        channels_axis: Integer, channel axis index.
        name: String, layer name prefix.

    Returns:
        Output tensor.
    """
    identity = x
    outputs = [x]
    for i in range(num_layers):
        if light_block:
            x = dfine_light_conv_block(
                x,
                mid_ch,
                kernel_size,
                use_lab=use_lab,
                data_format=data_format,
                channels_axis=channels_axis,
                name=f"{name}_layers_{i}",
            )
        else:
            x = dfine_conv_bn(
                x,
                mid_ch,
                kernel_size,
                1,
                activation="relu",
                use_lab=use_lab,
                data_format=data_format,
                channels_axis=channels_axis,
                name=f"{name}_layers_{i}",
            )
        outputs.append(x)
    x = layers.Concatenate(axis=channels_axis, name=f"{name}_cat")(outputs)
    x = dfine_conv_bn(
        x,
        out_ch // 2,
        1,
        1,
        activation="relu",
        use_lab=use_lab,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_agg_0",
    )
    x = dfine_conv_bn(
        x,
        out_ch,
        1,
        1,
        activation="relu",
        use_lab=use_lab,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_agg_1",
    )
    if residual:
        x = layers.Add(name=f"{name}_add")([x, identity])
    return x


def dfine_backbone(
    input_tensor,
    stem_channels,
    stage_in_channels,
    stage_mid_channels,
    stage_out_channels,
    stage_num_blocks,
    stage_downsample,
    stage_light_block,
    stage_kernel_size,
    stage_numb_of_layers,
    use_lab=False,
    out_stage_indices=None,
    data_format="channels_last",
    channels_axis=-1,
):
    """Build the HGNetV2 backbone for D-FINE.

    Args:
        input_tensor: Keras input tensor (B, H, W, 3).
        stem_channels: List [in_ch, stem_ch, stem_out_ch].
        stage_in_channels: Input channels per backbone stage.
        stage_mid_channels: Middle channels per backbone stage.
        stage_out_channels: Output channels per backbone stage.
        stage_num_blocks: Number of basic blocks per stage.
        stage_downsample: Whether to downsample per stage.
        stage_light_block: Whether to use light blocks per stage.
        stage_kernel_size: Kernel size per stage.
        stage_numb_of_layers: Conv layers per basic block per stage.
        use_lab: Whether to use learnable affine blocks.
        out_stage_indices: List of stage indices to return features from
            (e.g., [2, 3] for nano, [1, 2, 3] for others).
        data_format: String, Keras data format. Defaults to
            ``"channels_last"``.
        channels_axis: Integer, channel axis index. Defaults to ``-1``.

    Returns:
        List of feature tensors from the requested stages.
    """
    if out_stage_indices is None:
        out_stage_indices = [1, 2, 3]

    stem_ch = stem_channels[1]
    stem_out = stem_channels[2]

    x = dfine_conv_bn(
        input_tensor,
        stem_ch,
        3,
        2,
        activation="relu",
        use_lab=use_lab,
        data_format=data_format,
        channels_axis=channels_axis,
        name="backbone_stem1",
    )
    x_pad = layers.ZeroPadding2D(
        padding=((0, 1), (0, 1)),
        data_format=data_format,
    )(x)
    stem2a = layers.Conv2D(
        stem_ch // 2,
        2,
        strides=1,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="backbone_stem2a_conv",
    )(x_pad)
    stem2a = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.1,
        name="backbone_stem2a_bn",
    )(stem2a)
    stem2a = layers.Activation("relu", name="backbone_stem2a_act")(stem2a)
    if use_lab:
        stem2a = dfine_lab_layer(stem2a, name="backbone_stem2a_lab")

    stem2a_pad = layers.ZeroPadding2D(
        padding=((0, 1), (0, 1)),
        data_format=data_format,
    )(stem2a)
    stem2b = layers.Conv2D(
        stem_ch,
        2,
        strides=1,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="backbone_stem2b_conv",
    )(stem2a_pad)
    stem2b = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.1,
        name="backbone_stem2b_bn",
    )(stem2b)
    stem2b = layers.Activation("relu", name="backbone_stem2b_act")(stem2b)
    if use_lab:
        stem2b = dfine_lab_layer(stem2b, name="backbone_stem2b_lab")

    pooled = layers.ZeroPadding2D(
        padding=((0, 1), (0, 1)),
        data_format=data_format,
    )(x)
    pooled = layers.MaxPooling2D(
        pool_size=2,
        strides=1,
        padding="valid",
        data_format=data_format,
    )(pooled)

    x = layers.Concatenate(
        axis=channels_axis,
        name="backbone_stem_cat",
    )([pooled, stem2b])
    x = dfine_conv_bn(
        x,
        stem_ch,
        3,
        2,
        activation="relu",
        use_lab=use_lab,
        data_format=data_format,
        channels_axis=channels_axis,
        name="backbone_stem3",
    )
    x = dfine_conv_bn(
        x,
        stem_out,
        1,
        1,
        activation="relu",
        use_lab=use_lab,
        data_format=data_format,
        channels_axis=channels_axis,
        name="backbone_stem4",
    )

    stage_outputs = []
    for si in range(len(stage_num_blocks)):
        if stage_downsample[si]:
            in_ch_ds = stage_in_channels[si]
            x = dfine_conv_bn(
                x,
                in_ch_ds,
                3,
                2,
                groups=in_ch_ds,
                activation=None,
                use_lab=False,
                data_format=data_format,
                channels_axis=channels_axis,
                name=f"backbone_stage{si}_downsample",
            )

        nb = stage_num_blocks[si]
        for bi in range(nb):
            x = dfine_basic_layer(
                x,
                mid_ch=stage_mid_channels[si],
                out_ch=stage_out_channels[si],
                num_layers=stage_numb_of_layers[si],
                kernel_size=stage_kernel_size[si],
                residual=(bi != 0),
                light_block=stage_light_block[si],
                use_lab=use_lab,
                data_format=data_format,
                channels_axis=channels_axis,
                name=f"backbone_stage{si}_block{bi}",
            )
        stage_outputs.append(x)

    return [stage_outputs[i] for i in out_stage_indices]


def dfine_conv_norm(
    x,
    out_ch,
    ks,
    stride,
    groups=1,
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

    Args:
        x: Input tensor.
        out_ch: Integer, number of output channels.
        ks: Integer, kernel size.
        stride: Integer, convolution stride.
        groups: Integer, number of convolution groups. Defaults to ``1``.
        padding: Integer or ``None``. If ``None``, defaults to
            ``(ks - 1) // 2``.
        activation: String or ``None``, activation name.
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
        groups=groups,
        data_format=data_format,
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.1,
        name=f"{name}_norm",
    )(x)
    if activation is not None:
        x = layers.Activation(activation, name=f"{name}_act")(x)
    return x


def dfine_rep_vgg_block(
    x,
    ch,
    activation="silu",
    data_format=None,
    channels_axis=-1,
    name="",
):
    """RepVGG block: parallel 3x3 and 1x1 conv branches, summed.

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
    b1 = dfine_conv_norm(
        x,
        ch,
        3,
        1,
        padding=1,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv1",
    )
    b2 = dfine_conv_norm(
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


def dfine_csp_rep_layer(
    x,
    out_ch,
    expansion=1.0,
    depths=1,
    activation="silu",
    data_format=None,
    channels_axis=-1,
    name="",
):
    """CSP Rep Layer: conv1 path through RepVGG bottlenecks + conv2
    shortcut, summed, optionally projected via conv3.

    Args:
        x: Input tensor.
        out_ch: Integer, output channel dimension.
        expansion: Float, hidden channel expansion ratio relative to
            ``out_ch``. Defaults to ``1.0``.
        depths: Integer, number of RepVGG bottleneck blocks.
            Defaults to ``1``.
        activation: String, activation name. Defaults to ``"silu"``.
        data_format: String, Keras data format.
        channels_axis: Integer, channel axis index.
        name: String, layer name prefix.

    Returns:
        Output tensor with ``out_ch`` channels.
    """
    hid = int(out_ch * expansion)
    p1 = dfine_conv_norm(
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
        p1 = dfine_rep_vgg_block(
            p1,
            hid,
            activation=activation,
            data_format=data_format,
            channels_axis=channels_axis,
            name=f"{name}_bottlenecks_{i}",
        )
    p2 = dfine_conv_norm(
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
        merged = dfine_conv_norm(
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


def dfine_rep_ncspelan4(
    x,
    encoder_hidden_dim,
    hidden_expansion,
    activation="silu",
    depths=1,
    data_format=None,
    channels_axis=-1,
    name="",
):
    """RepNCSPELAN4 block used in D-FINE's FPN and PAN.

    Splits the conv1 output into two halves, processes through two
    CSPRepLayers with intermediate convolutions, then concatenates
    all branches and projects via conv4.

    Args:
        x: Input tensor.
        encoder_hidden_dim: Integer, encoder hidden dimension.
        hidden_expansion: Float, hidden channel expansion ratio.
        activation: String, activation name. Defaults to ``"silu"``.
        depths: Integer, number of RepVGG bottleneck blocks.
            Defaults to ``1``.
        data_format: String, Keras data format.
        channels_axis: Integer, channel axis index.
        name: String, layer name prefix.

    Returns:
        Output tensor with ``encoder_hidden_dim`` channels.
    """
    conv3_dim = encoder_hidden_dim * 2
    conv4_dim = round(hidden_expansion * encoder_hidden_dim // 2)
    conv_dim = conv3_dim // 2

    y = dfine_conv_norm(
        x,
        conv3_dim,
        1,
        1,
        padding=0,
        activation=activation,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv1",
    )
    split_a = (
        y[..., :conv_dim] if data_format == "channels_last" else y[:, :conv_dim, :, :]
    )
    split_b = (
        y[..., conv_dim:] if data_format == "channels_last" else y[:, conv_dim:, :, :]
    )

    branch1 = dfine_csp_rep_layer(
        split_b,
        conv4_dim,
        depths=depths,
        activation=activation,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_csp_rep1",
    )
    branch1 = dfine_conv_norm(
        branch1,
        conv4_dim,
        3,
        1,
        padding=1,
        activation=activation,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv2",
    )

    branch2 = dfine_csp_rep_layer(
        branch1,
        conv4_dim,
        depths=depths,
        activation=activation,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_csp_rep2",
    )
    branch2 = dfine_conv_norm(
        branch2,
        conv4_dim,
        3,
        1,
        padding=1,
        activation=activation,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv3",
    )

    merged = layers.Concatenate(axis=channels_axis, name=f"{name}_cat")(
        [split_a, split_b, branch1, branch2]
    )
    out = dfine_conv_norm(
        merged,
        encoder_hidden_dim,
        1,
        1,
        padding=0,
        activation=activation,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv4",
    )
    return out


def dfine_sc_down(
    x,
    encoder_hidden_dim,
    ks,
    stride,
    data_format=None,
    channels_axis=-1,
    name="",
):
    """SCDown: 1x1 conv + depthwise conv for downsampling in PAN.

    Args:
        x: Input tensor.
        encoder_hidden_dim: Integer, number of output channels.
        ks: Integer, kernel size for the depthwise convolution.
        stride: Integer, stride for the depthwise convolution.
        data_format: String, Keras data format.
        channels_axis: Integer, channel axis index.
        name: String, layer name prefix.

    Returns:
        Output tensor with ``encoder_hidden_dim`` channels.
    """
    x = dfine_conv_norm(
        x,
        encoder_hidden_dim,
        1,
        1,
        padding=0,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv1",
    )
    x = dfine_conv_norm(
        x,
        encoder_hidden_dim,
        ks,
        stride,
        groups=encoder_hidden_dim,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_conv2",
    )
    return x


def dfine_aifi_encoder_layer(
    x,
    pos_embed,
    hidden_dim,
    num_heads,
    mlp_dim,
    activation="gelu",
    name="aifi_0_layers_0",
):
    """Single AIFI transformer encoder layer for D-FINE.

    Applies self-attention followed by a feedforward network, each with
    a residual connection and post-norm layer normalization. Positional
    embeddings are added to the query and key inputs of self-attention
    but not to the values.

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
    sa = DFineMultiHeadAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        block_prefix=f"{name}_self_attn",
        name=f"{name}_self_attn",
    )
    q = k = layers.Add(name=f"{name}_sa_qk_add")([x, pos_embed])
    residual = x
    attn = sa(q, k, x)
    x = layers.LayerNormalization(
        epsilon=1e-5,
        name=f"{name}_self_attn_layer_norm",
    )(layers.Add(name=f"{name}_sa_res")([residual, attn]))
    residual = x
    ff = layers.Dense(mlp_dim, name=f"{name}_fc1")(x)
    ff = layers.Activation(activation, name=f"{name}_gelu")(ff)
    ff = layers.Dense(hidden_dim, name=f"{name}_fc2")(ff)
    x = layers.LayerNormalization(
        epsilon=1e-5,
        name=f"{name}_final_layer_norm",
    )(layers.Add(name=f"{name}_ff_res")([residual, ff]))
    return x


def dfine_integral(pred_corners, project, max_num_bins):
    """Apply DFine integral to convert distribution to distances.

    Args:
        pred_corners: Tensor of shape ``(B, Q, 4 * (max_num_bins + 1))``.
        project: Tensor of shape ``(max_num_bins + 1,)``.
        max_num_bins: Integer, number of bins.

    Returns:
        Tensor of shape ``(B, Q, 4)`` with distances.
    """
    nbins = max_num_bins + 1
    orig_shape = pred_corners.shape
    flat = ops.reshape(pred_corners, [-1, nbins])
    flat = ops.softmax(flat, axis=1)
    proj = ops.reshape(project, [nbins, 1])
    flat = ops.matmul(flat, proj)
    flat = ops.reshape(flat, [-1, 4])
    return ops.reshape(flat, [-1, orig_shape[1], 4])


def dfine_distance2bbox(points, distance, reg_scale):
    """Convert reference points + distances to cxcywh bounding boxes.

    Args:
        points: Tensor ``(B, Q, 4)`` as ``(cx, cy, w, h)`` in [0, 1].
        distance: Tensor ``(B, Q, 4)`` with distances.
        reg_scale: Scalar tensor, regression scale.

    Returns:
        Tensor ``(B, Q, 4)`` in ``(cx, cy, w, h)`` format.
    """
    rs = ops.abs(reg_scale)
    half_rs = ops.multiply(ops.convert_to_tensor(0.5, dtype=rs.dtype), rs)
    pw = ops.divide(points[..., 2], rs)
    ph = ops.divide(points[..., 3], rs)
    x1 = ops.subtract(
        points[..., 0], ops.multiply(ops.add(half_rs, distance[..., 0]), pw)
    )
    y1 = ops.subtract(
        points[..., 1], ops.multiply(ops.add(half_rs, distance[..., 1]), ph)
    )
    x2 = ops.add(points[..., 0], ops.multiply(ops.add(half_rs, distance[..., 2]), pw))
    y2 = ops.add(points[..., 1], ops.multiply(ops.add(half_rs, distance[..., 3]), ph))
    cx = ops.divide(ops.add(x1, x2), 2.0)
    cy = ops.divide(ops.add(y1, y2), 2.0)
    w = ops.subtract(x2, x1)
    h = ops.subtract(y2, y1)
    return ops.stack([cx, cy, w, h], axis=-1)


def dfine_inverse_sigmoid(t, eps=1e-5):
    """Numerically stable inverse sigmoid.

    Args:
        t: Input tensor with values in (0, 1).
        eps: Float, clamping epsilon. Defaults to ``1e-5``.

    Returns:
        Tensor with inverse sigmoid applied.
    """
    t = ops.clip(t, eps, 1 - eps)
    return ops.log(t / (1 - t))


def dfine_hybrid_encoder(
    bk_feats,
    encoder_hidden_dim,
    encoder_ffn_dim,
    encode_proj_layers,
    hidden_expansion,
    ccfm_num_blocks,
    num_feature_levels,
    feat_strides,
    spatial_h,
    spatial_w,
):
    """Build D-FINE's hybrid encoder: AIFI transformer + CCFM (FPN + PAN).

    Each backbone stage feature is projected to ``encoder_hidden_dim`` via
    a 1x1 conv + batch norm. The AIFI transformer (a single self-attention
    encoder layer with sine positional embeddings) is then applied on the
    feature levels listed in ``encode_proj_layers``. Finally, a CCFM
    cross-scale fusion runs a top-down FPN followed by a bottom-up PAN,
    each fusing adjacent levels through ``dfine_rep_ncspelan4`` blocks.

    Args:
        bk_feats: List of backbone feature tensors, one per pyramid
            level, with channel counts matching ``encoder_in_channels``.
        encoder_hidden_dim: Channel dim used inside the hybrid encoder
            (must match ``hidden_dim`` to skip the decoder input projection).
        encoder_ffn_dim: FFN dim inside the AIFI transformer layer.
        encode_proj_layers: Tuple of feature-level indices on which to
            apply the AIFI transformer (e.g. ``(2,)`` runs AIFI only on
            the highest-stride level).
        hidden_expansion: CSP hidden-channel expansion ratio in the
            CCFM ``RepNCSPELAN4`` blocks.
        ccfm_num_blocks: Number of RepVGG bottleneck blocks per CCFM stage.
        num_feature_levels: Number of multi-scale levels produced.
        feat_strides: Feature strides per level (e.g. ``(8, 16, 32)``).
        spatial_h: Input image height in pixels (used to derive per-level
            spatial shapes for the AIFI position embedding).
        spatial_w: Input image width in pixels.

    Returns:
        List of ``num_feature_levels`` post-PAN feature tensors, ordered
        from highest spatial resolution to lowest, each with
        ``encoder_hidden_dim`` channels.
    """
    encoder_num_layers = 1
    encoder_num_heads = 8

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
        pe = dfine_sine_pos_embed(h, w, encoder_hidden_dim, 10000)
        for li in range(encoder_num_layers):
            flat = dfine_aifi_encoder_layer(
                flat,
                pe,
                encoder_hidden_dim,
                encoder_num_heads,
                encoder_ffn_dim,
                activation="gelu",
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
        top = dfine_conv_norm(
            top,
            encoder_hidden_dim,
            1,
            1,
            padding=0,
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
            dfine_rep_ncspelan4(
                fused,
                encoder_hidden_dim,
                hidden_expansion,
                activation="silu",
                depths=ccfm_num_blocks,
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
        down = dfine_sc_down(
            top_pan,
            encoder_hidden_dim,
            3,
            2,
            data_format=data_format,
            channels_axis=channels_axis,
            name=f"downsample_convs_{idx}",
        )
        fused = layers.Concatenate(axis=channels_axis, name=f"pan_cat_{idx}")(
            [down, fpn_feat]
        )
        pan.append(
            dfine_rep_ncspelan4(
                fused,
                encoder_hidden_dim,
                hidden_expansion,
                activation="silu",
                depths=ccfm_num_blocks,
                data_format=data_format,
                channels_axis=channels_axis,
                name=f"pan_blocks_{idx}",
            )
        )

    return pan


def dfine_decoder_inputs(
    pan,
    hidden_dim,
    encoder_hidden_dim,
    feat_strides,
    spatial_h,
    spatial_w,
):
    """Prepare flattened tokens and anchor proposals for the D-FINE decoder.

    When ``hidden_dim`` differs from ``encoder_hidden_dim`` each PAN feature
    level is first re-projected to ``hidden_dim`` via a 1x1 conv + batch
    norm. Features are then flattened from ``(B, H, W, hidden_dim)`` per
    level into a single ``(B, sum(H*W), hidden_dim)`` token sequence. In
    parallel, a regular anchor grid of ``(cx, cy, w, h)`` boxes is
    generated for every token, converted to logits (sigmoid inverse),
    and a validity mask is computed (anchors near the image border are
    masked out and given a large sentinel logit).

    Args:
        pan: List of post-PAN feature tensors from
            :func:`dfine_hybrid_encoder`.
        hidden_dim: Decoder model dimension.
        encoder_hidden_dim: Channel dim of the PAN features (controls
            whether re-projection to ``hidden_dim`` is needed).
        feat_strides: Feature strides per level used to derive per-level
            spatial shapes.
        spatial_h: Input image height in pixels.
        spatial_w: Input image width in pixels.

    Returns:
        source_flat: ``(B, sum(H*W), hidden_dim)`` flat token sequence over
            all feature levels.
        spatial_shapes: List of ``(H, W)`` per feature level, in the
            order tokens appear in ``source_flat``.
        anchors_t: ``(1, sum(H*W), 4)`` anchor logits in
            ``inverse_sigmoid((cx, cy, w, h))`` space.
        vmask_t: ``(1, sum(H*W), 1)`` float mask, ``1.0`` for tokens
            whose anchor is fully inside the image, ``0.0`` otherwise.
    """
    data_format = keras.config.image_data_format()
    channels_axis = -1 if data_format == "channels_last" else 1

    dec_sources = []
    for i, feat in enumerate(pan):
        if hidden_dim != encoder_hidden_dim:
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
        else:
            dec_sources.append(feat)

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
    vmask_t = ops.cast(
        ops.all((anchors > 1e-2) & (anchors < 1 - 1e-2), axis=-1, keepdims=True),
        "float32",
    )
    anchors_t = ops.where(
        vmask_t > 0.5,
        ops.log(anchors / (1 - anchors)),
        ops.convert_to_tensor(3.4028235e38, dtype="float32"),
    )

    return source_flat, spatial_shapes, anchors_t, vmask_t


def dfine_two_stage_proposals(
    source_flat,
    anchors_t,
    vmask_t,
    hidden_dim,
    num_classes,
    num_queries,
):
    """Score every token and pick the top-K decoder queries (two-stage init).

    The flat token sequence is masked by ``vmask_t``, projected through
    a Linear + LayerNorm (``enc_output_*``), and then scored two ways:
    ``enc_score_head`` produces a per-token class logit and
    ``enc_bbox_head`` produces a 4-d delta added to ``anchors_t`` to get
    a candidate reference box per token. The ``num_queries`` tokens with
    the highest max-class score are picked: their post-projection
    embedding becomes the initial decoder query (``target``) and their
    candidate box becomes the initial reference (``ref_logit``). Both
    are wrapped in ``stop_gradient`` to detach the second stage from
    the encoder during backprop, matching the original D-FINE training
    recipe.

    Args:
        source_flat: Flat token sequence from :func:`dfine_decoder_inputs`.
        anchors_t: Anchor logits per token.
        vmask_t: Validity mask per token.
        hidden_dim: Decoder model dimension (output channel of the encoder
            projection).
        num_classes: Number of object classes (output channel of
            ``enc_score_head``).

    Returns:
        target: ``(B, num_queries, hidden_dim)`` initial decoder query
            embeddings, detached.
        ref_logit: ``(B, num_queries, 4)`` initial reference box logits
            in inverse-sigmoid space, detached.
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


def dfine_fdr_block(
    target,
    ref_logit,
    source_flat,
    spatial_shapes,
    hidden_dim,
    decoder_num_layers,
    decoder_ffn_dim,
    decoder_n_points,
    num_feature_levels,
):
    """Run the D-FINE iterative decoder with Fine-grained Distribution Refinement.

    Stacks ``decoder_num_layers`` ``DFineDecoderLayer`` blocks; each layer
    cross-attends queries to ``source_flat`` via multi-scale deformable
    attention guided by the current reference points. After the first
    decoder layer, a small ``pre_bbox_head`` MLP turns the initial
    queries into ``ref_points_initial``: the anchor used by FDR.

    Per layer the FDR head ``bbox_embed_{di}`` predicts a residual
    over a discretized distribution of ``4 * (max_num_bins + 1)``
    corner offsets, which is accumulated across layers. The
    distribution is integrated via ``dfine_integral`` into 4 corner
    distances, and ``dfine_distance2bbox`` reconstructs a refined box
    that replaces the reference points for the next iteration. Class
    heads and LQE quality heads are not built here: they live in
    ``DFineDetect``, which consumes ``last_pred_corners``.

    Args:
        target: Initial decoder query embeddings from
            :func:`dfine_two_stage_proposals`.
        ref_logit: Initial reference box logits.
        source_flat: Flattened multi-scale encoder tokens (the cross-
            attention "memory").
        spatial_shapes: Per-level ``(H, W)`` for deformable attention.
        hidden_dim: Decoder model dimension.
        decoder_num_layers: Number of stacked decoder layers.
        decoder_ffn_dim: FFN dimension inside each decoder layer.
        decoder_n_points: Sampling points per level for multi-scale
            deformable attention.
        num_feature_levels: Number of feature levels in ``source_flat``.

    Returns:
        hs: Last decoder hidden state, ``(B, num_queries, hidden_dim)``.
        last_boxes: Final refined boxes from the last decoder layer,
            ``(B, num_queries, 4)`` as ``(cx, cy, w, h)``.
        last_pred_corners: Accumulated FDR corner-distribution logits
            from the last layer, ``(B, num_queries, 4 * (max_num_bins + 1))``;
            consumed by ``DFineDetect`` for LQE quality scoring.
    """
    decoder_num_heads = 8
    max_num_bins = 32

    decoder_params = DFineDecoderParams(
        max_num_bins=max_num_bins, name="decoder_params"
    )

    qp_d0 = layers.Dense(hidden_dim * 2, activation="relu", name="query_pos_head_0")
    qp_d1 = layers.Dense(hidden_dim, name="query_pos_head_1")

    pre_bb_d0 = layers.Dense(hidden_dim, activation="relu", name="pre_bbox_head_0")
    pre_bb_d1 = layers.Dense(hidden_dim, activation="relu", name="pre_bbox_head_1")
    pre_bb_d2 = layers.Dense(4, name="pre_bbox_head_2")

    hs = target
    ref_pts = ops.sigmoid(ref_logit)
    ref_detach = ops.stop_gradient(ref_pts)

    output_detach = ops.zeros_like(hs)
    nbins_out = 4 * (max_num_bins + 1)
    pred_corners_accum = None

    ref_points_initial = None
    last_boxes = None
    last_pred_corners = None
    project = None
    rs_val = None

    for di in range(decoder_num_layers):
        rp_in = ops.expand_dims(ref_detach, axis=2)
        query_pos = qp_d1(qp_d0(ref_detach))
        query_pos = ops.clip(query_pos, -10.0, 10.0)

        dl = DFineDecoderLayer(
            hidden_dim=hidden_dim,
            num_heads=decoder_num_heads,
            dim_feedforward=decoder_ffn_dim,
            activation="relu",
            n_levels=num_feature_levels,
            num_points_list=decoder_n_points,
            offset_scale=0.5,
            spatial_shapes=spatial_shapes,
            block_prefix=f"decoder_layers_{di}",
            name=f"decoder_layers_{di}",
        )
        hs = dl(hs, source_flat, query_pos, rp_in)

        if di == 0:
            hs, project, rs_val = decoder_params(hs)
            pre_bb = pre_bb_d2(pre_bb_d1(pre_bb_d0(hs)))
            new_ref = ops.sigmoid(pre_bb + dfine_inverse_sigmoid(ref_detach))
            ref_points_initial = ops.stop_gradient(new_ref)

        bb_i = layers.Dense(hidden_dim, activation="relu", name=f"bbox_embed_{di}_0")(
            hs + output_detach
        )
        bb_i = layers.Dense(hidden_dim, activation="relu", name=f"bbox_embed_{di}_1")(
            bb_i
        )
        bb_i = layers.Dense(nbins_out, name=f"bbox_embed_{di}_2")(bb_i)
        pred_corners = bb_i if pred_corners_accum is None else bb_i + pred_corners_accum

        distances = dfine_integral(pred_corners, project, max_num_bins)
        inter_ref_bbox = dfine_distance2bbox(ref_points_initial, distances, rs_val)

        pred_corners_accum = pred_corners
        ref_detach = ops.stop_gradient(inter_ref_bbox)
        output_detach = ops.stop_gradient(hs)

        last_boxes = inter_ref_bbox
        last_pred_corners = pred_corners

    return hs, last_boxes, last_pred_corners


def dfine_decoder(
    pan,
    hidden_dim,
    encoder_hidden_dim,
    decoder_num_layers,
    decoder_ffn_dim,
    decoder_n_points,
    num_feature_levels,
    feat_strides,
    num_classes,
    num_queries,
    spatial_h,
    spatial_w,
):
    """Build the full D-FINE decoder stage on top of the hybrid encoder output.

    Orchestrator that wires the three decoder sub-stages:

    1. :func:`dfine_decoder_inputs`, re-project PAN features to
       ``hidden_dim``, flatten to a single token sequence, and generate
       anchor proposals with validity masks.
    2. :func:`dfine_two_stage_proposals`, score every token and select
       the top ``num_queries`` as initial decoder queries + reference
       boxes.
    3. :func:`dfine_fdr_block`, iterative deformable decoder with FDR
       (Fine-grained Distribution Refinement) bbox refinement.

    Args:
        pan: Multi-scale features from :func:`dfine_hybrid_encoder`.
        hidden_dim: Decoder model dimension.
        encoder_hidden_dim: Channel dim of ``pan`` (controls whether
            the input-projection conv is needed).
        decoder_num_layers: Number of stacked decoder layers.
        decoder_ffn_dim: FFN dimension inside each decoder layer.
        decoder_n_points: Sampling points per level for deformable
            attention.
        num_feature_levels: Number of multi-scale levels.
        feat_strides: Feature strides per level.
        num_classes: Number of object classes (used by the
            two-stage scoring head).
        spatial_h: Input image height in pixels.
        spatial_w: Input image width in pixels.

    Returns:
        Tuple ``(hs, last_boxes, last_pred_corners)``: see
        :func:`dfine_fdr_block` for shapes and semantics.
    """
    source_flat, spatial_shapes, anchors_t, vmask_t = dfine_decoder_inputs(
        pan,
        hidden_dim=hidden_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        feat_strides=feat_strides,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
    )
    target, ref_logit = dfine_two_stage_proposals(
        source_flat,
        anchors_t,
        vmask_t,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_queries=num_queries,
    )
    return dfine_fdr_block(
        target,
        ref_logit,
        source_flat,
        spatial_shapes,
        hidden_dim=hidden_dim,
        decoder_num_layers=decoder_num_layers,
        decoder_ffn_dim=decoder_ffn_dim,
        decoder_n_points=decoder_n_points,
        num_feature_levels=num_feature_levels,
    )


def dfine_functional(
    inputs,
    stem_channels,
    stage_in_channels,
    stage_mid_channels,
    stage_out_channels,
    stage_num_blocks,
    stage_numb_of_layers,
    use_lab,
    encoder_in_channels,
    encoder_hidden_dim,
    encoder_ffn_dim,
    encode_proj_layers,
    hidden_expansion,
    ccfm_num_blocks,
    hidden_dim,
    decoder_num_layers,
    decoder_ffn_dim,
    decoder_n_points,
    num_feature_levels,
    feat_strides,
    num_classes,
    num_queries,
    input_shape,
):
    """Build the full D-FINE architecture from an input tensor (no class heads).

    Top-level orchestrator that wires the three architectural stages:

    1. :func:`dfine_backbone`, HGNetV2 backbone produces multi-scale
       features at the levels chosen by ``encoder_in_channels``.
    2. :func:`dfine_hybrid_encoder`, AIFI transformer encoder + CCFM
       (FPN + PAN) for cross-scale fusion.
    3. :func:`dfine_decoder`, token preparation + two-stage query
       initialization + iterative deformable decoder with FDR refinement.

    Per-layer class prediction heads (``class_embed_*``) and LQE
    quality-score heads are intentionally not built here: they are
    added by :class:`DFineDetect`, which composes :class:`DFineModel`
    around this graph.

    Args:
        inputs: Keras input tensor of shape ``(B, H, W, 3)`` (or
            ``(B, 3, H, W)`` for ``channels_first``).
        stem_channels: HGNetV2 stem channel configuration
            ``[in, mid, out]``.
        stage_in_channels: Input channels per backbone stage.
        stage_mid_channels: Middle channels per backbone stage.
        stage_out_channels: Output channels per backbone stage.
        stage_num_blocks: Number of basic blocks per stage.
        stage_numb_of_layers: Conv layers per basic block per stage.
        use_lab: Whether the backbone uses Learnable Affine Blocks.
        encoder_in_channels: Backbone channels fed into the hybrid encoder.
        encoder_hidden_dim: Channel dim inside the hybrid encoder.
        encoder_ffn_dim: FFN dim in the AIFI transformer.
        encode_proj_layers: Feature-level indices where AIFI is applied.
        hidden_expansion: CSP hidden-channel expansion ratio in CCFM.
        ccfm_num_blocks: Number of RepVGG bottleneck blocks per CCFM stage.
        hidden_dim: Decoder model dimension.
        decoder_num_layers: Number of stacked decoder layers.
        decoder_ffn_dim: FFN dim in each decoder layer.
        decoder_n_points: Sampling points per level for deformable attention.
        num_feature_levels: Number of multi-scale levels.
        feat_strides: Feature strides per level.
        num_classes: Number of object classes (used by the two-stage scoring head).
        input_shape: ``(H, W, C)`` (or ``(C, H, W)``) shape of ``inputs``,
            used to compute per-level spatial sizes.

    Returns:
        Tuple ``(hs, last_boxes, last_pred_corners)``: the three
        outputs of :func:`dfine_fdr_block`. ``hs`` is the decoder last
        hidden state, ``last_boxes`` are the final refined boxes, and
        ``last_pred_corners`` is fed into the LQE head of
        :class:`DFineDetect`.
    """
    stage_downsample = (False, True, True, True)
    stage_light_block = (False, False, True, True)
    stage_kernel_size = (3, 3, 5, 5)

    data_format = keras.config.image_data_format()
    channels_axis = -1 if data_format == "channels_last" else 1

    if decoder_n_points is None:
        decoder_n_points = [4] * num_feature_levels

    if data_format == "channels_first":
        spatial_h, spatial_w = input_shape[1], input_shape[2]
    else:
        spatial_h, spatial_w = input_shape[0], input_shape[1]

    out_stage_indices = []
    for enc_ch in encoder_in_channels:
        for si, soc in enumerate(stage_out_channels):
            if soc == enc_ch and si not in out_stage_indices:
                out_stage_indices.append(si)
                break

    bk_feats = dfine_backbone(
        inputs,
        stem_channels=list(stem_channels),
        stage_in_channels=list(stage_in_channels),
        stage_mid_channels=list(stage_mid_channels),
        stage_out_channels=list(stage_out_channels),
        stage_num_blocks=list(stage_num_blocks),
        stage_downsample=list(stage_downsample),
        stage_light_block=list(stage_light_block),
        stage_kernel_size=list(stage_kernel_size),
        stage_numb_of_layers=list(stage_numb_of_layers),
        use_lab=use_lab,
        out_stage_indices=out_stage_indices,
        data_format=data_format,
        channels_axis=channels_axis,
    )

    pan = dfine_hybrid_encoder(
        bk_feats,
        encoder_hidden_dim=encoder_hidden_dim,
        encoder_ffn_dim=encoder_ffn_dim,
        encode_proj_layers=encode_proj_layers,
        hidden_expansion=hidden_expansion,
        ccfm_num_blocks=ccfm_num_blocks,
        num_feature_levels=num_feature_levels,
        feat_strides=feat_strides,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
    )

    return dfine_decoder(
        pan,
        hidden_dim=hidden_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        decoder_num_layers=decoder_num_layers,
        decoder_ffn_dim=decoder_ffn_dim,
        decoder_n_points=decoder_n_points,
        num_feature_levels=num_feature_levels,
        feat_strides=feat_strides,
        num_classes=num_classes,
        num_queries=num_queries,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
    )


@keras.saving.register_keras_serializable(package="kerasformers")
class DFineModel(BaseModel):
    """D-FINE backbone + hybrid encoder + decoder (no class heads).

    Matches the reference ``DFineModel`` pattern: outputs the decoder
    ``last_hidden_state`` with shape ``(B, num_queries, hidden_dim)``. The
    iterative bbox refinement layers stay in the model (they feed back
    into the decoder via reference points); only the per-layer class
    prediction and quality-score heads are pruned from the output graph.
    Use ``DFineDetect`` for full detection outputs.

    Reference:
        - `D-FINE: Redefine Regression Task of DETRs as Fine-grained
          Distribution Refinement <https://arxiv.org/abs/2410.13842>`_
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = ("d_fine", "dfine")

    def __init__(
        self,
        stem_channels=(3, 16, 16),
        stage_in_channels=(16, 64, 256, 512),
        stage_mid_channels=(16, 32, 64, 128),
        stage_out_channels=(64, 256, 512, 1024),
        stage_num_blocks=(1, 1, 2, 1),
        stage_numb_of_layers=(3, 3, 3, 3),
        use_lab=True,
        encoder_in_channels=(256, 512, 1024),
        encoder_hidden_dim=256,
        encoder_ffn_dim=1024,
        encode_proj_layers=(2,),
        hidden_expansion=1.0,
        ccfm_num_blocks=1,
        hidden_dim=256,
        decoder_num_layers=6,
        decoder_ffn_dim=1024,
        decoder_n_points=None,
        num_feature_levels=3,
        feat_strides=(8, 16, 32),
        num_classes=80,
        num_queries=300,
        image_size=640,
        input_tensor=None,
        name="DFineModel",
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

        if decoder_n_points is None:
            decoder_n_points = [4] * num_feature_levels

        hs_last, last_boxes, last_pred_corners = dfine_functional(
            img_input,
            stem_channels=stem_channels,
            stage_in_channels=stage_in_channels,
            stage_mid_channels=stage_mid_channels,
            stage_out_channels=stage_out_channels,
            stage_num_blocks=stage_num_blocks,
            stage_numb_of_layers=stage_numb_of_layers,
            use_lab=use_lab,
            encoder_in_channels=encoder_in_channels,
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_ffn_dim=encoder_ffn_dim,
            encode_proj_layers=encode_proj_layers,
            hidden_expansion=hidden_expansion,
            ccfm_num_blocks=ccfm_num_blocks,
            hidden_dim=hidden_dim,
            decoder_num_layers=decoder_num_layers,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_n_points=decoder_n_points,
            num_feature_levels=num_feature_levels,
            feat_strides=feat_strides,
            num_classes=num_classes,
            num_queries=num_queries,
            input_shape=image_size,
        )

        outputs = {
            "last_hidden_state": hs_last,
            "last_boxes": last_boxes,
            "last_pred_corners": last_pred_corners,
        }
        super().__init__(inputs=img_input, outputs=outputs, name=name, **kwargs)

        self._stem_channels = list(stem_channels)
        self._stage_in_channels = list(stage_in_channels)
        self._stage_mid_channels = list(stage_mid_channels)
        self._stage_out_channels = list(stage_out_channels)
        self._stage_num_blocks = list(stage_num_blocks)
        self._stage_numb_of_layers = list(stage_numb_of_layers)
        self._use_lab = use_lab
        self._encoder_in_channels = list(encoder_in_channels)
        self._encoder_hidden_dim = encoder_hidden_dim
        self._encoder_ffn_dim = encoder_ffn_dim
        self._encode_proj_layers = list(encode_proj_layers)
        self._hidden_expansion = hidden_expansion
        self._ccfm_num_blocks = ccfm_num_blocks
        self._d_model = hidden_dim
        self._decoder_layers = decoder_num_layers
        self._decoder_ffn_dim = decoder_ffn_dim
        self._decoder_n_points = list(decoder_n_points)
        self._num_feature_levels = num_feature_levels
        self._feat_strides = list(feat_strides)
        self._num_classes = num_classes
        self._num_queries = num_queries
        self.image_size = image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stem_channels": self._stem_channels,
                "stage_in_channels": self._stage_in_channels,
                "stage_mid_channels": self._stage_mid_channels,
                "stage_out_channels": self._stage_out_channels,
                "stage_num_blocks": self._stage_num_blocks,
                "stage_numb_of_layers": self._stage_numb_of_layers,
                "use_lab": self._use_lab,
                "encoder_in_channels": self._encoder_in_channels,
                "encoder_hidden_dim": self._encoder_hidden_dim,
                "encoder_ffn_dim": self._encoder_ffn_dim,
                "encode_proj_layers": self._encode_proj_layers,
                "hidden_expansion": self._hidden_expansion,
                "ccfm_num_blocks": self._ccfm_num_blocks,
                "hidden_dim": self._d_model,
                "decoder_num_layers": self._decoder_layers,
                "decoder_ffn_dim": self._decoder_ffn_dim,
                "decoder_n_points": self._decoder_n_points,
                "num_feature_levels": self._num_feature_levels,
                "feat_strides": self._feat_strides,
                "num_classes": self._num_classes,
                "num_queries": self._num_queries,
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
        return DFineDetect.config_from_hf(hf_config)

    @classmethod
    def from_hf(cls, hf_id, load_weights=True, skip_mismatch=False, **kwargs):
        model = super().from_hf(hf_id, load_weights=False, **kwargs)
        if load_weights:
            src = DFineDetect.from_hf(hf_id, skip_mismatch=skip_mismatch)
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
class DFineDetect(BaseModel):
    """D-FINE: Detection with Fine-grained Distribution Refinement.

    A real-time object detection model combining an HGNetV2 backbone with
    a hybrid encoder (AIFI + CCFM) and a decoder with Fine-grained
    Distribution Refinement (FDR) and Localization Quality Estimation
    (LQE).

    Reference:
        - `D-FINE: Redefine Regression Task of DETRs as Fine-grained
          Distribution Refinement <https://arxiv.org/abs/2410.13842>`_

    Args:
        stem_channels: Stem channel configuration ``[in, mid, out]``.
        stage_in_channels: Input channels per backbone stage.
        stage_mid_channels: Middle channels per backbone stage.
        stage_out_channels: Output channels per backbone stage.
        stage_num_blocks: Number of basic blocks per stage.
        stage_numb_of_layers: Conv layers per basic block per stage.
        use_lab: Whether to use Learnable Affine Block.
        encoder_in_channels: Backbone channels fed to encoder.
        encoder_hidden_dim: Hidden dim of hybrid encoder.
        encoder_ffn_dim: FFN dim in AIFI encoder.
        encode_proj_layers: Feature level indices for AIFI.
        hidden_expansion: CSP hidden channel expansion ratio.
        ccfm_num_blocks: Number of RepVGG bottleneck blocks in CCFM.
        hidden_dim: Decoder model dimension.
        decoder_num_layers: Number of decoder layers.
        decoder_ffn_dim: FFN dim in decoder.
        decoder_n_points: List of sampling points per feature level.
        num_feature_levels: Number of multi-scale feature levels.
        feat_strides: Feature strides from backbone.
        num_classes: Number of object classes.
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
    config_class = DFineConfig
    # Weights load by Hub repo id, e.g. from_weights("kerasformers/dfine-nano"),
    # via kf_config.json on the repo (no url table in the package).
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = ("d_fine", "dfine")

    def __init__(
        self,
        stem_channels=(3, 16, 16),
        stage_in_channels=(16, 64, 256, 512),
        stage_mid_channels=(16, 32, 64, 128),
        stage_out_channels=(64, 256, 512, 1024),
        stage_num_blocks=(1, 1, 2, 1),
        stage_numb_of_layers=(3, 3, 3, 3),
        use_lab=True,
        encoder_in_channels=(256, 512, 1024),
        encoder_hidden_dim=256,
        encoder_ffn_dim=1024,
        encode_proj_layers=(2,),
        hidden_expansion=1.0,
        ccfm_num_blocks=1,
        hidden_dim=256,
        decoder_num_layers=6,
        decoder_ffn_dim=1024,
        decoder_n_points=None,
        num_feature_levels=3,
        feat_strides=(8, 16, 32),
        num_classes=80,
        num_queries=300,
        image_size=640,
        input_tensor=None,
        name="DFineDetect",
        **kwargs,
    ):
        if decoder_n_points is None:
            decoder_n_points = [4] * num_feature_levels

        base = DFineModel(
            stem_channels=stem_channels,
            stage_in_channels=stage_in_channels,
            stage_mid_channels=stage_mid_channels,
            stage_out_channels=stage_out_channels,
            stage_num_blocks=stage_num_blocks,
            stage_numb_of_layers=stage_numb_of_layers,
            use_lab=use_lab,
            encoder_in_channels=encoder_in_channels,
            encoder_hidden_dim=encoder_hidden_dim,
            encoder_ffn_dim=encoder_ffn_dim,
            encode_proj_layers=encode_proj_layers,
            hidden_expansion=hidden_expansion,
            ccfm_num_blocks=ccfm_num_blocks,
            hidden_dim=hidden_dim,
            decoder_num_layers=decoder_num_layers,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_n_points=decoder_n_points,
            num_feature_levels=num_feature_levels,
            feat_strides=feat_strides,
            num_classes=num_classes,
            num_queries=num_queries,
            image_size=image_size,
            input_tensor=input_tensor,
            name=f"{name}_model",
        )
        hs_last = base.output["last_hidden_state"]
        last_boxes = base.output["last_boxes"]
        last_pred_corners = base.output["last_pred_corners"]

        max_num_bins = 32

        di_last = decoder_num_layers - 1
        class_logits = layers.Dense(num_classes, name=f"class_embed_{di_last}")(hs_last)

        prob = ops.softmax(
            ops.reshape(last_pred_corners, [-1, num_queries, 4, max_num_bins + 1]),
            axis=-1,
        )
        prob_topk, _ = ops.top_k(prob, k=4)
        prob_mean = ops.mean(prob_topk, axis=-1, keepdims=True)
        stat = ops.concatenate([prob_topk, prob_mean], axis=-1)
        stat = ops.reshape(stat, [-1, num_queries, 4 * 5])
        quality_score = layers.Dense(64, activation="relu", name=f"lqe_{di_last}_0")(
            stat
        )
        quality_score = layers.Dense(1, name=f"lqe_{di_last}_1")(quality_score)
        logits = class_logits + quality_score

        outputs = {"logits": logits, "pred_boxes": last_boxes}
        super().__init__(inputs=base.input, outputs=outputs, name=name, **kwargs)

        self._stem_channels = list(stem_channels)
        self._stage_in_channels = list(stage_in_channels)
        self._stage_mid_channels = list(stage_mid_channels)
        self._stage_out_channels = list(stage_out_channels)
        self._stage_num_blocks = list(stage_num_blocks)
        self._stage_numb_of_layers = list(stage_numb_of_layers)
        self._use_lab = use_lab
        self._encoder_in_channels = list(encoder_in_channels)
        self._encoder_hidden_dim = encoder_hidden_dim
        self._encoder_ffn_dim = encoder_ffn_dim
        self._encode_proj_layers = list(encode_proj_layers)
        self._hidden_expansion = hidden_expansion
        self._ccfm_num_blocks = ccfm_num_blocks
        self._d_model = hidden_dim
        self._decoder_layers = decoder_num_layers
        self._decoder_ffn_dim = decoder_ffn_dim
        self._decoder_n_points = list(decoder_n_points)
        self._num_feature_levels = num_feature_levels
        self._feat_strides = list(feat_strides)
        self._num_classes = num_classes
        self._num_queries = num_queries
        self.image_size = base.image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stem_channels": self._stem_channels,
                "stage_in_channels": self._stage_in_channels,
                "stage_mid_channels": self._stage_mid_channels,
                "stage_out_channels": self._stage_out_channels,
                "stage_num_blocks": self._stage_num_blocks,
                "stage_numb_of_layers": self._stage_numb_of_layers,
                "use_lab": self._use_lab,
                "encoder_in_channels": self._encoder_in_channels,
                "encoder_hidden_dim": self._encoder_hidden_dim,
                "encoder_ffn_dim": self._encoder_ffn_dim,
                "encode_proj_layers": self._encode_proj_layers,
                "hidden_expansion": self._hidden_expansion,
                "ccfm_num_blocks": self._ccfm_num_blocks,
                "hidden_dim": self._d_model,
                "decoder_num_layers": self._decoder_layers,
                "decoder_ffn_dim": self._decoder_ffn_dim,
                "decoder_n_points": self._decoder_n_points,
                "num_feature_levels": self._num_feature_levels,
                "feat_strides": self._feat_strides,
                "num_classes": self._num_classes,
                "num_queries": self._num_queries,
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
        bb = hf_config.get("backbone_config", hf_config)
        return {
            "stem_channels": tuple(bb["stem_channels"]),
            "stage_in_channels": tuple(bb["stage_in_channels"]),
            "stage_mid_channels": tuple(bb["stage_mid_channels"]),
            "stage_out_channels": tuple(bb["stage_out_channels"]),
            "stage_num_blocks": tuple(bb["stage_num_blocks"]),
            "stage_numb_of_layers": tuple(bb["stage_numb_of_layers"]),
            "use_lab": bb.get("use_learnable_affine_block", bb.get("use_lab", True)),
            "encoder_in_channels": tuple(hf_config["encoder_in_channels"]),
            "encoder_hidden_dim": hf_config["encoder_hidden_dim"],
            "encoder_ffn_dim": hf_config["encoder_ffn_dim"],
            "encode_proj_layers": tuple(hf_config["encode_proj_layers"]),
            "hidden_expansion": hf_config["hidden_expansion"],
            "ccfm_num_blocks": hf_config.get("ccfm_num_blocks", 1),
            "hidden_dim": hf_config["d_model"],
            "decoder_num_layers": hf_config["decoder_layers"],
            "decoder_ffn_dim": hf_config["decoder_ffn_dim"],
            "decoder_n_points": list(hf_config["decoder_n_points"]),
            "num_feature_levels": hf_config["num_feature_levels"],
            "feat_strides": tuple(hf_config["feat_strides"]),
            "num_classes": hf_num_classes(hf_config),
            "num_queries": hf_config.get("num_queries", 300),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_dfine_hf_to_keras import transfer_dfine_weights

        transfer_dfine_weights(keras_model, hf_state_dict)
