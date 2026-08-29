import math

import keras
from keras import layers, ops, utils

from zeromodels.base import BaseModel
from zeromodels.utils import standardize_input_shape

from .rf_detr_config import RFDetrConfig, RFDetrSegmentConfig
from .rf_detr_layers import (
    RFDETRChannelLayerNorm,
    RFDETRDecoderLayer,
    RFDETRDinoV2Embeddings,
    RFDETRDinoV2LayerScale,
    RFDETRLearnedEmbedding,
    RFDETRSegmentationBias,
)


def sincos_interleave(x):
    """Interleave sine and cosine components for positional encoding.

    Converts alternating sin/cos components into an interleaved format,
    matching the standard sinusoidal positional encoding format.

    Args:
        x: Input tensor with even indices containing angles for sin,
            odd indices for cos.

    Returns:
        Tensor with interleaved sin/cos values.
    """
    sin_part = ops.sin(x[..., 0::2])
    cos_part = ops.cos(x[..., 1::2])
    stacked = ops.stack([sin_part, cos_part], axis=-1)
    target_shape = [-1 if s is None else s for s in stacked.shape]
    target_shape[-2] = target_shape[-2] * 2
    target_shape.pop()
    return ops.reshape(stacked, target_shape)


def rf_detr_position_embedding_sine(
    x, num_pos_feats=128, temperature=10000, normalize=True
):
    """Generate 2D sinusoidal positional embeddings for feature maps.

    Creates a positional encoding using sine and cosine functions at different
    frequencies, suitable for transformer-based vision models.

    Args:
        x: Input tensor used only for shape reference. Shape is expected to be
            ``(B, H, W, C)`` where H and W determine the spatial dimensions.
        num_pos_feats: Number of positional features per dimension. The output
            will have ``2 * num_pos_feats`` channels. Default 128.
        temperature: Temperature for scaling the frequency bands. Default 10000.
        normalize: Whether to normalize coordinates to [0, 2π]. Default True.

    Returns:
        Positional embedding tensor of shape ``(1, H, W, 2 * num_pos_feats)``.
    """
    scale = 2.0 * math.pi
    shape = ops.shape(x)
    h, w = shape[1], shape[2]

    y_range = ops.cast(ops.arange(h), "float32")
    x_range = ops.cast(ops.arange(w), "float32")

    if normalize:
        eps = 1e-6
        y_range = y_range / (ops.cast(h, "float32") - 1 + eps) * scale
        x_range = x_range / (ops.cast(w, "float32") - 1 + eps) * scale
    else:
        y_range = (y_range + 1) * scale
        x_range = (x_range + 1) * scale

    dim_t = ops.cast(ops.arange(num_pos_feats), "float32")
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_range[:, None] / dim_t[None, :]
    pos_y = y_range[:, None] / dim_t[None, :]

    pos_x_sin = ops.sin(pos_x[:, 0::2])
    pos_x_cos = ops.cos(pos_x[:, 1::2])
    pos_x = ops.reshape(ops.stack([pos_x_sin, pos_x_cos], axis=-1), [w, num_pos_feats])

    pos_y_sin = ops.sin(pos_y[:, 0::2])
    pos_y_cos = ops.cos(pos_y[:, 1::2])
    pos_y = ops.reshape(ops.stack([pos_y_sin, pos_y_cos], axis=-1), [h, num_pos_feats])

    pos_y = ops.expand_dims(pos_y, axis=1)
    pos_y = ops.tile(pos_y, [1, w, 1])
    pos_x = ops.expand_dims(pos_x, axis=0)
    pos_x = ops.tile(pos_x, [h, 1, 1])
    pos = ops.concatenate([pos_y, pos_x], axis=-1)
    pos = ops.expand_dims(pos, axis=0)
    return pos


def rf_detr_gen_sineembed_for_position(pos_tensor, dim=128):
    """Generate sine positional embeddings for bounding box coordinates.

    Creates sinusoidal positional embeddings from normalized coordinate tensors,
    used for query position encoding in the DETR decoder.

    Args:
        pos_tensor: Position tensor of shape ``(..., 2)`` for (x, y) coordinates
            or ``(..., 4)`` for (x, y, w, h) coordinates. Values should be in
            range [0, 1] representing normalized positions.
        dim: Embedding dimension per coordinate axis. Default 128.

    Returns:
        Positional embedding tensor. Shape depends on input:
        - For 2D input: ``(..., 2 * dim)``
        - For 4D input: ``(..., 4 * dim)``

    Raises:
        ValueError: If pos_tensor has last dimension other than 2 or 4.
    """
    scale = 2 * math.pi
    dim_t = ops.cast(ops.arange(dim), "float32")
    dim_t = 10000.0 ** (2 * (dim_t // 2) / dim)

    x_embed = pos_tensor[..., 0:1] * scale
    y_embed = pos_tensor[..., 1:2] * scale

    pos_x = sincos_interleave(x_embed / dim_t)
    pos_y = sincos_interleave(y_embed / dim_t)

    if pos_tensor.shape[-1] == 2:
        return ops.concatenate([pos_y, pos_x], axis=-1)
    elif pos_tensor.shape[-1] == 4:
        w_embed = pos_tensor[..., 2:3] * scale
        h_embed = pos_tensor[..., 3:4] * scale
        pos_w = sincos_interleave(w_embed / dim_t)
        pos_h = sincos_interleave(h_embed / dim_t)
        return ops.concatenate([pos_y, pos_x, pos_w, pos_h], axis=-1)
    else:
        raise ValueError(
            f"pos_tensor last dim must be 2 or 4, got {pos_tensor.shape[-1]}"
        )


def rf_detr_unwindow_features(
    hidden_state,
    num_h,
    num_w,
    num_windows,
    hidden_dim,
    num_register_tokens,
    data_format="channels_last",
):
    """Convert windowed features back to spatial feature map format.

    Transforms features from window-based representation (used in windowed
    attention) back to a standard spatial layout, removing register tokens
    and reorganizing patches.

    Args:
        hidden_state: Windowed feature tensor from DINOv2 encoder.
        num_h: Number of patches in height dimension.
        num_w: Number of patches in width dimension.
        num_windows: Number of windows along each spatial dimension.
        hidden_dim: Feature dimension.
        num_register_tokens: Number of DINOv2 register tokens to remove.
        data_format: String, image data format. Default ``"channels_last"``.

    Returns:
        Unwindowed feature tensor.
    """
    hidden_state = hidden_state[:, num_register_tokens + 1 :, :]

    if num_windows > 1:
        nw2 = num_windows**2
        shape = ops.shape(hidden_state)
        HW_win = shape[1]
        C = shape[2]

        hidden_state = ops.reshape(hidden_state, [-1, nw2 * HW_win, C])
        h_pw = num_h // num_windows
        w_pw = num_w // num_windows
        hidden_state = ops.reshape(hidden_state, [-1, num_windows, h_pw, w_pw, C])
        hidden_state = ops.transpose(hidden_state, [0, 2, 1, 3, 4])

    hidden_state = ops.reshape(hidden_state, [-1, num_h, num_w, hidden_dim])
    if data_format == "channels_first":
        hidden_state = ops.transpose(hidden_state, [0, 3, 1, 2])
    return hidden_state


def rf_detr_encoder_output_proposals(memory, spatial_shapes, bbox_reparam=True):
    """Generate encoder output proposals for two-stage DETR initialization.

    Creates initial reference points (proposals) from spatial locations across
    feature levels, used in the two-stage DETR architecture for query selection.

    Args:
        memory: Encoder output features of shape ``(B, total_tokens, C)``.
        spatial_shapes: List of ``(H, W)`` tuples for each feature level's
            spatial dimensions.
        bbox_reparam: If True, use bbox reparameterization (cx, cy, w, h
            format directly). If False, use sigmoid-sigmoid format for
            numerically stable sigmoid inverse. Default True.

    Returns:
        Tuple of:
        - output_memory: Filtered memory features with invalid proposals zeroed.
        - output_proposals: Generated proposals as (cx, cy, w, h) tensors
            in normalized coordinates [0, 1].
        - valid: Boolean mask ``(B, N, 1)``, True where the proposal is in
            bounds (used to exclude border proposals from the top-k selection,
            matching HF's ``masked_fill(invalid_mask, -inf)``).
    """
    proposals = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        y_range = ops.cast(ops.arange(H_), "float32")
        x_range = ops.cast(ops.arange(W_), "float32")
        grid_y, grid_x = ops.meshgrid(y_range, x_range, indexing="ij")
        grid = ops.stack([grid_x, grid_y], axis=-1)
        grid = ops.reshape(grid, [1, H_ * W_, 2])

        scale = ops.convert_to_tensor([[W_, H_]], dtype="float32")
        scale = ops.reshape(scale, [1, 1, 2])
        grid = (grid + 0.5) / scale

        wh = ops.ones_like(grid) * 0.05 * (2.0**lvl)
        proposal = ops.concatenate([grid, wh], axis=-1)
        proposals.append(proposal)

    output_proposals = ops.concatenate(proposals, axis=1)
    batch_zero = ops.sum(memory * 0, axis=(1, 2), keepdims=True)
    output_proposals = batch_zero + output_proposals

    valid = ops.all(
        (output_proposals > 0.01) & (output_proposals < 0.99),
        axis=-1,
        keepdims=True,
    )

    if bbox_reparam:
        output_proposals = ops.where(
            valid, output_proposals, ops.zeros_like(output_proposals)
        )
    else:
        eps = 1e-7
        clamped = ops.clip(output_proposals, eps, 1.0 - eps)
        unsig = ops.log(clamped / (1.0 - clamped))
        inf_val = ops.full_like(unsig, 1e8)
        output_proposals = ops.where(valid, unsig, inf_val)

    output_memory = ops.where(valid, memory, ops.zeros_like(memory))
    return output_memory, output_proposals, valid


def rf_detr_two_stage_refine_refpoints(
    refpoint_embed, refpoint_embed_ts, bbox_reparam=True, num_queries=300
):
    """Refine reference points for two-stage query initialization.

    Combines learned query reference points with encoder proposals selected
    in the first stage, using either bbox reparameterization or direct addition.

    Args:
        refpoint_embed: Learned query embeddings of shape ``(B, num_queries, 4)``.
        refpoint_embed_ts: Two-stage proposals from encoder of shape
            ``(B, num_proposals, 4)``.
        bbox_reparam: If True, use multiplicative refinement for center and
            exponential for size. If False, use additive refinement. Default True.
        num_queries: Number of queries to select from proposals. Default 300.

    Returns:
        Refined reference points of shape ``(B, num_queries, 4)`` as
        (cx, cy, w, h) in normalized coordinates.
    """
    refpoint_embed_subset = refpoint_embed[:, :num_queries, :]
    if bbox_reparam:
        ref_cxcy = (
            refpoint_embed_subset[..., :2] * refpoint_embed_ts[..., 2:]
            + refpoint_embed_ts[..., :2]
        )
        ref_wh = ops.exp(refpoint_embed_subset[..., 2:]) * refpoint_embed_ts[..., 2:]
        return ops.concatenate([ref_cxcy, ref_wh], axis=-1)
    else:
        return refpoint_embed_subset + refpoint_embed_ts


def rf_detr_conv_bn(
    x,
    filters,
    kernel_size=3,
    strides=1,
    groups=1,
    activation="relu",
    use_layer_norm=False,
    data_format="channels_last",
    channels_axis=-1,
    name="conv_bn",
):
    """Apply convolution followed by batch normalization and activation.

    Args:
        x: Input tensor.
        filters: Number of output filters (channels).
        kernel_size: Convolution kernel size. Default 3.
        strides: Convolution stride. Default 1.
        groups: Number of groups for grouped convolution. Default 1.
        activation: Activation function name. Default "relu".
        use_layer_norm: If True, use RFDETRChannelLayerNorm. Default False.
        data_format: String, image data format. Default ``"channels_last"``.
        channels_axis: Integer, channel axis index. Default ``-1``.
        name: Layer name prefix. Default "conv_bn".

    Returns:
        Output tensor.
    """
    padding = (kernel_size // 2, kernel_size // 2)
    x = layers.ZeroPadding2D(
        padding=padding, data_format=data_format, name=f"{name}_pad"
    )(x)
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="valid",
        groups=groups,
        use_bias=False,
        data_format=data_format,
        name=f"{name}_conv",
    )(x)
    if use_layer_norm:
        x = RFDETRChannelLayerNorm(name=f"{name}_ln")(x)
    else:
        x = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=1e-5,
            momentum=0.1,
            name=f"{name}_bn",
        )(x)
    x = layers.Activation(activation, name=f"{name}_{activation}")(x)
    return x


def rf_detr_bottleneck(
    x,
    out_channels,
    shortcut=True,
    expansion=1.0,
    activation="silu",
    use_layer_norm=False,
    data_format="channels_last",
    channels_axis=-1,
    name="bottleneck",
):
    """Apply a bottleneck block with optional residual connection.

    Args:
        x: Input tensor.
        out_channels: Number of output channels.
        shortcut: Whether to add residual connection. Default True.
        expansion: Hidden channel expansion ratio. Default 1.0.
        activation: Activation function name. Default "silu".
        use_layer_norm: If True, use RFDETRChannelLayerNorm. Default False.
        data_format: String, image data format. Default ``"channels_last"``.
        channels_axis: Integer, channel axis index. Default ``-1``.
        name: Layer name prefix. Default "bottleneck".

    Returns:
        Output tensor.
    """
    hidden = int(out_channels * expansion)
    in_channels = x.shape[channels_axis]
    residual = x
    x = rf_detr_conv_bn(
        x,
        hidden,
        3,
        activation=activation,
        use_layer_norm=use_layer_norm,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_cv1",
    )
    x = rf_detr_conv_bn(
        x,
        out_channels,
        3,
        activation=activation,
        use_layer_norm=use_layer_norm,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_cv2",
    )
    if shortcut and in_channels == out_channels:
        x = x + residual
    return x


def rf_detr_c2f(
    x,
    out_channels,
    depths=1,
    shortcut=False,
    expansion=0.5,
    activation="silu",
    use_layer_norm=False,
    data_format="channels_last",
    channels_axis=-1,
    name="c2f",
):
    """Apply a C2F (CSP Bottleneck with 2 convolutions) module from YOLO.

    Args:
        x: Input tensor.
        out_channels: Number of output channels.
        depths: Number of bottleneck blocks. Default 1.
        shortcut: Whether to use residual connections. Default False.
        expansion: Hidden channel expansion ratio. Default 0.5.
        activation: Activation function name. Default "silu".
        use_layer_norm: If True, use RFDETRChannelLayerNorm. Default False.
        data_format: String, image data format. Default ``"channels_last"``.
        channels_axis: Integer, channel axis index. Default ``-1``.
        name: Layer name prefix. Default "c2f".

    Returns:
        Output tensor.
    """
    c = int(out_channels * expansion)
    x = rf_detr_conv_bn(
        x,
        2 * c,
        1,
        activation=activation,
        use_layer_norm=use_layer_norm,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_cv1",
    )
    chunks = ops.split(x, 2, axis=channels_axis)
    y = [chunks[0], chunks[1]]
    for i in range(depths):
        y.append(
            rf_detr_bottleneck(
                y[-1],
                c,
                shortcut=shortcut,
                expansion=1.0,
                activation=activation,
                use_layer_norm=use_layer_norm,
                data_format=data_format,
                channels_axis=channels_axis,
                name=f"{name}_bottleneck_{i}",
            )
        )
    x = ops.concatenate(y, axis=channels_axis)
    x = rf_detr_conv_bn(
        x,
        out_channels,
        1,
        activation=activation,
        use_layer_norm=use_layer_norm,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_cv2",
    )
    return x


def rf_detr_simple_projector(
    x,
    out_channels,
    data_format="channels_last",
    channels_axis=-1,
    name="projector",
):
    """Apply a simple 2-layer projector with convolution and normalization.

    Args:
        x: Input tensor.
        out_channels: Number of output channels (decoder hidden dimension).
        data_format: String, image data format. Default ``"channels_last"``.
        channels_axis: Integer, channel axis index. Default ``-1``.
        name: Layer name prefix. Default "projector".

    Returns:
        Output tensor.
    """
    in_dim = x.shape[channels_axis]
    x = rf_detr_conv_bn(
        x,
        in_dim * 2,
        3,
        activation="silu",
        use_layer_norm=True,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_convx1",
    )
    x = rf_detr_conv_bn(
        x,
        out_channels,
        3,
        activation="silu",
        use_layer_norm=True,
        data_format=data_format,
        channels_axis=channels_axis,
        name=f"{name}_convx2",
    )
    x = RFDETRChannelLayerNorm(name=f"{name}_ln")(x)
    return x


def rf_detr_dinov2_swiglu_ffn(x, hidden_dim, mlp_ratio=4, name="mlp"):
    """Apply SwiGLU feed-forward network from DINOv2.

    Implements the SwiGLU variant of FFN which uses a gated linear unit
    with Swish activation, providing better performance than standard FFN.

    Args:
        x: Input tensor of shape ``(B, seq_len, hidden_dim)``.
        hidden_dim: Model dimension (input and output size).
        mlp_ratio: Expansion ratio for hidden dimension. Default 4.
        name: Layer name prefix. Default "mlp".

    Returns:
        Output tensor of shape ``(B, seq_len, hidden_dim)``.
    """
    hidden_features = int(hidden_dim * mlp_ratio)
    hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
    x = layers.Dense(2 * hidden_features, name=f"{name}_weights_in")(x)
    x1, x2 = ops.split(x, 2, axis=-1)
    x = ops.silu(x1) * x2
    x = layers.Dense(hidden_dim, name=f"{name}_weights_out")(x)
    return x


def rf_detr_dinov2_mlp(x, hidden_dim, mlp_ratio=4, name="mlp"):
    """Apply standard MLP feed-forward network with GELU activation.

    Implements the classic transformer FFN: two linear layers with GELU
    activation in between.

    Args:
        x: Input tensor of shape ``(B, seq_len, hidden_dim)``.
        hidden_dim: Model dimension (input and output size).
        mlp_ratio: Expansion ratio for hidden dimension. Default 4.
        name: Layer name prefix. Default "mlp".

    Returns:
        Output tensor of shape ``(B, seq_len, hidden_dim)``.
    """
    hidden_features = int(hidden_dim * mlp_ratio)
    x = layers.Dense(hidden_features, name=f"{name}_fc1")(x)
    x = ops.gelu(x, approximate=False)
    x = layers.Dense(hidden_dim, name=f"{name}_fc2")(x)
    return x


def rf_detr_dinov2_block(
    x,
    hidden_dim,
    num_heads,
    mlp_ratio=4,
    use_swiglu=False,
    run_full_attention=False,
    num_windows=1,
    name="layer",
):
    """Apply a single DINOv2 transformer block.

    Implements self-attention followed by feed-forward network with
    pre-norm and residual connections. Supports both windowed and full attention.

    Args:
        x: Input tensor of shape ``(B, seq_len, hidden_dim)`` for full attention
            or ``(B, num_windows^2, seq_per_window, hidden_dim)`` for windowed.
        hidden_dim: Model dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Expansion ratio for FFN hidden dimension. Default 4.
        use_swiglu: If True, use SwiGLU FFN; otherwise use GELU MLP. Default False.
        run_full_attention: If True and num_windows > 1, reshape for full attention.
            Default False.
        num_windows: Number of windows per dimension for windowed attention.
            Default 1 (full attention).
        name: Layer name prefix. Default "layer".

    Returns:
        Output tensor of same shape as input.
    """
    head_dim = hidden_dim // num_heads
    shortcut = x

    if run_full_attention and num_windows > 1:
        nw2 = num_windows**2
        shape = ops.shape(x)
        x = ops.reshape(x, [-1, nw2 * shape[1], shape[2]])

    normed = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm1")(x)

    q = layers.Dense(hidden_dim, name=f"{name}_attention_query")(normed)
    k = layers.Dense(hidden_dim, name=f"{name}_attention_key")(normed)
    v = layers.Dense(hidden_dim, name=f"{name}_attention_value")(normed)

    seq_len = ops.shape(normed)[1]
    q = ops.reshape(q, [-1, seq_len, num_heads, head_dim])
    k = ops.reshape(k, [-1, seq_len, num_heads, head_dim])
    v = ops.reshape(v, [-1, seq_len, num_heads, head_dim])
    q = ops.transpose(q, [0, 2, 1, 3])
    k = ops.transpose(k, [0, 2, 1, 3])
    v = ops.transpose(v, [0, 2, 1, 3])

    scale = ops.cast(head_dim, q.dtype) ** -0.5
    attn_weights = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2])) * scale
    attn_weights = ops.softmax(attn_weights, axis=-1)
    attn_output = ops.matmul(attn_weights, v)
    attn_output = ops.transpose(attn_output, [0, 2, 1, 3])
    attn_output = ops.reshape(attn_output, [-1, seq_len, hidden_dim])
    attn_out = layers.Dense(
        hidden_dim,
        name=f"{name}_attention_out_proj",
    )(attn_output)

    if run_full_attention and num_windows > 1:
        full_shape = ops.shape(attn_out)
        attn_out = ops.reshape(
            attn_out,
            [-1, full_shape[1] // nw2, full_shape[2]],
        )

    attn_out = RFDETRDinoV2LayerScale(hidden_dim, name=f"{name}_layer_scale1")(attn_out)
    x = attn_out + shortcut

    shortcut2 = x
    normed2 = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm2")(x)
    if use_swiglu:
        mlp_out = rf_detr_dinov2_swiglu_ffn(
            normed2,
            hidden_dim,
            mlp_ratio,
            name=f"{name}_mlp",
        )
    else:
        mlp_out = rf_detr_dinov2_mlp(
            normed2,
            hidden_dim,
            mlp_ratio,
            name=f"{name}_mlp",
        )
    mlp_out = RFDETRDinoV2LayerScale(hidden_dim, name=f"{name}_layer_scale2")(mlp_out)
    x = mlp_out + shortcut2
    return x


def rf_detr_windowed_dinov2_encoder(
    x,
    hidden_dim,
    num_heads,
    num_layers,
    mlp_ratio=4,
    use_swiglu=False,
    num_windows=1,
    out_feature_indexes=None,
    window_block_indexes=None,
    name="backbone_encoder",
):
    """Apply a stack of DINOv2 transformer blocks with windowed attention.

    Runs a sequence of DINOv2 blocks, alternating between windowed and full
    attention based on the window_block_indexes, and extracts features at
    specified layer indices.

    Args:
        x: Input tensor from DINOv2 embeddings.
        hidden_dim: Model dimension.
        num_heads: Number of attention heads.
        num_layers: Total number of transformer blocks.
        mlp_ratio: Expansion ratio for FFN hidden dimension. Default 4.
        use_swiglu: If True, use SwiGLU FFN. Default False.
        num_windows: Number of windows per dimension for windowed attention.
            Default 1.
        out_feature_indexes: Layer indices (1-indexed) at which to extract
            intermediate features. Default None.
        window_block_indexes: Layer indices that should use windowed attention.
            Others will use full attention. Default None.
        name: Layer name prefix. Default "backbone_encoder".

    Returns:
        List of feature tensors extracted at out_feature_indexes (after layer
        norm). If out_feature_indexes is empty, returns empty list.
    """
    out_feature_indexes = out_feature_indexes or []
    window_block_indexes = window_block_indexes or []
    max_layer = max(out_feature_indexes) + 1 if out_feature_indexes else num_layers

    shared_layernorm = layers.LayerNormalization(
        epsilon=1e-6,
        name=f"{name}_layernorm",
    )

    features = []
    for i in range(max_layer):
        run_full = i not in window_block_indexes
        x = rf_detr_dinov2_block(
            x,
            hidden_dim,
            num_heads,
            mlp_ratio,
            use_swiglu=use_swiglu,
            run_full_attention=run_full,
            num_windows=num_windows,
            name=f"{name}_layer_{i}",
        )
        if (i + 1) in out_feature_indexes:
            features.append(shared_layernorm(x))
    return features


def rf_detr_backbone(
    inputs,
    hidden_dim,
    num_heads,
    num_layers,
    mlp_ratio,
    use_swiglu,
    num_register_tokens,
    out_feature_indexes,
    patch_size,
    num_windows,
    positional_encoding_size,
    input_shape,
    data_format,
):
    """Build RF-DETR's DINOv2 backbone with windowed attention.

    Patch-embeds the input image with :class:`RFDETRDinoV2Embeddings`, runs
    a stack of ``num_layers`` DINOv2 transformer blocks (windowed
    attention everywhere except at ``out_feature_indexes`` layers,
    which use full attention), extracts intermediate features at each
    output index, and unwindows them back into ``(num_h, num_w)``
    feature maps.

    Args:
        inputs: Keras input tensor of shape ``(B, H, W, 3)`` (or
            ``(B, 3, H, W)`` for channels_first).
        hidden_dim: DINOv2 backbone model dimension.
        num_heads: Number of attention heads in each backbone block.
        num_layers: Total number of DINOv2 transformer blocks.
        mlp_ratio: FFN expansion ratio in each backbone block.
        use_swiglu: Whether backbone FFNs use SwiGLU (True) or GELU MLP.
        num_register_tokens: Number of DINOv2 register tokens.
        out_feature_indexes: 1-indexed block IDs at which to extract
            features (these blocks also run full attention).
        patch_size: Patch size for DINOv2 patch embeddings.
        num_windows: Number of windows per spatial dimension in
            windowed-attention blocks.
        positional_encoding_size: Side length of the learned positional
            encoding grid (interpolated to match the patch grid).
        input_shape: Input shape excluding the batch dim, used to derive
            the patch grid size ``(num_h, num_w)``.
        data_format: ``"channels_last"`` or ``"channels_first"``.

    Returns:
        features: List of unwindowed feature maps, one per
            ``out_feature_indexes`` entry, each with shape
            ``(B, num_h, num_w, hidden_dim)`` for channels_last (or
            ``(B, hidden_dim, num_h, num_w)`` for channels_first).
        spatial_shape: ``(num_h, num_w)`` patch grid size.
    """
    embeddings = RFDETRDinoV2Embeddings(
        hidden_dim=hidden_dim,
        patch_size=patch_size,
        num_channels=3,
        num_register_tokens=num_register_tokens,
        num_windows=num_windows,
        positional_encoding_size=positional_encoding_size,
        name="backbone_embeddings",
    )(inputs)

    window_block_indexes = sorted(
        set(range(max(out_feature_indexes) + 1)) - set(out_feature_indexes)
    )

    encoder_features = rf_detr_windowed_dinov2_encoder(
        embeddings,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_ratio=mlp_ratio,
        use_swiglu=use_swiglu,
        num_windows=num_windows,
        out_feature_indexes=out_feature_indexes,
        window_block_indexes=window_block_indexes,
        name="backbone_encoder",
    )

    if data_format == "channels_first":
        num_h = input_shape[1] // patch_size
        num_w = input_shape[2] // patch_size
    else:
        num_h = input_shape[0] // patch_size
        num_w = input_shape[1] // patch_size

    unwindowed_features = []
    for feat in encoder_features:
        uw = rf_detr_unwindow_features(
            feat,
            num_h,
            num_w,
            num_windows,
            hidden_dim,
            num_register_tokens,
            data_format=data_format,
        )
        unwindowed_features.append(uw)

    return unwindowed_features, (num_h, num_w)


def rf_detr_projector(
    features,
    hidden_dim,
    spatial_shape,
    data_format,
    channels_axis,
):
    """Project multi-scale backbone features into a single decoder memory.

    Concatenates the multi-scale unwindowed features along the channel
    axis, fuses them with a C2F (CSP Bottleneck with 2 convolutions)
    block down to ``hidden_dim`` channels, applies a channel-wise
    layernorm, and flattens the spatial dimensions into the token
    sequence expected by the deformable decoder.

    Args:
        features: List of feature maps from :func:`rf_detr_backbone`,
            all with the same spatial shape.
        hidden_dim: Decoder hidden dimension (projector output channels).
        spatial_shape: ``(num_h, num_w)`` spatial size of the projector
            output (same as backbone feature spatial size).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Channel axis (-1 or 1).

    Returns:
        ``(B, num_h * num_w, hidden_dim)`` flattened token sequence
        used as encoder memory by the deformable decoder.
    """
    num_h, num_w = spatial_shape

    concat_feat = layers.Concatenate(axis=channels_axis, name="concat_features")(
        features
    )
    projected = rf_detr_c2f(
        concat_feat,
        hidden_dim,
        depths=3,
        shortcut=False,
        expansion=0.5,
        activation="silu",
        use_layer_norm=True,
        data_format=data_format,
        channels_axis=channels_axis,
        name="projector_c2f",
    )
    projected = RFDETRChannelLayerNorm(name="projector_ln")(projected)

    if data_format == "channels_first":
        projected = ops.transpose(projected, [0, 2, 3, 1])
    memory = ops.reshape(projected, [-1, num_h * num_w, hidden_dim])
    return memory


def rf_detr_decoder_inputs(
    memory,
    num_queries,
    hidden_dim,
):
    """Initialize learned object queries and reference points.

    Both the query features (``tgt``) and the reference points
    (``refpoint_embed``) are learned per-query embeddings broadcast
    across the batch from ``memory``. Reference points are initialized
    to zero and later refined either by the two-stage proposal
    selection (:func:`rf_detr_two_stage_refpoints`) or directly by the
    decoder's iterative refinement loop.

    Args:
        memory: Encoder memory ``(B, H*W, hidden_dim)``, used only to
            carry the batch dimension into the learned-embedding layers.
        num_queries: Number of object queries.
        hidden_dim: Query feature dimension.

    Returns:
        tgt: ``(B, num_queries, hidden_dim)`` initial query features.
        refpoint_embed: ``(B, num_queries, 4)`` initial reference points
            (cx, cy, w, h), zero-initialized.
    """
    tgt = RFDETRLearnedEmbedding(
        num_queries,
        hidden_dim,
        initializer="glorot_uniform",
        name="query_feat_embed",
    )(memory)
    refpoint_embed = RFDETRLearnedEmbedding(
        num_queries,
        4,
        initializer="zeros",
        name="refpoint_embed_layer",
    )(memory)
    return tgt, refpoint_embed


def rf_detr_two_stage_refpoints(
    memory,
    refpoint_embed,
    hidden_dim,
    num_classes,
    num_queries,
    bbox_reparam,
    spatial_shapes,
):
    """Refine reference points from top-k encoder proposals (two-stage init).

    Generates anchor proposals from spatial grid positions across all
    feature levels (:func:`rf_detr_encoder_output_proposals`), projects
    and normalizes the encoder memory, predicts class scores + bbox
    deltas on top of it, selects the top-``num_queries`` highest-scoring
    anchors, and combines them with the learned ``refpoint_embed`` via
    :func:`rf_detr_two_stage_refine_refpoints`.

    Args:
        memory: Encoder memory ``(B, H*W, hidden_dim)``.
        refpoint_embed: Learned reference points
            ``(B, num_queries, 4)``.
        hidden_dim: Decoder hidden dimension.
        num_classes: Number of classes for the first-stage class scoring
            head (proposal scoring; not the final detection head).
        num_queries: Number of object queries / top-k anchors to keep.
        bbox_reparam: Whether to use bbox reparameterization
            (multiplicative center + exponential size) instead of
            additive refinement.
        spatial_shapes: List of ``(H, W)`` per feature level (single
            level for RF-DETR).

    Returns:
        ``(B, num_queries, 4)`` refined reference points.
    """
    output_memory_filtered, output_proposals, valid = rf_detr_encoder_output_proposals(
        memory,
        spatial_shapes=spatial_shapes,
        bbox_reparam=bbox_reparam,
    )

    enc_output_proj = layers.Dense(hidden_dim, name="enc_output_0")(
        output_memory_filtered
    )
    enc_output_norm = layers.LayerNormalization(epsilon=1e-5, name="enc_output_norm_0")
    output_memory_proj = enc_output_norm(enc_output_proj)

    enc_cls = layers.Dense(num_classes, name="enc_out_class_embed_0")(
        output_memory_proj
    )

    bbox_embed_enc_0 = layers.Dense(hidden_dim, activation="relu", name="enc_bbox_0")
    bbox_embed_enc_1 = layers.Dense(hidden_dim, activation="relu", name="enc_bbox_1")
    bbox_embed_enc_2 = layers.Dense(4, name="enc_bbox_2")

    enc_bbox_delta = bbox_embed_enc_2(
        bbox_embed_enc_1(bbox_embed_enc_0(output_memory_proj))
    )

    if bbox_reparam:
        enc_coord_cxcy = (
            enc_bbox_delta[..., :2] * output_proposals[..., 2:]
            + output_proposals[..., :2]
        )
        enc_coord_wh = ops.exp(enc_bbox_delta[..., 2:]) * output_proposals[..., 2:]
        enc_coords = ops.concatenate([enc_coord_cxcy, enc_coord_wh], axis=-1)
    else:
        enc_coords = enc_bbox_delta + output_proposals

    proj_shape = spatial_shapes[0]
    # Exclude out-of-bounds (border) proposals from selection, matching HF's
    # `enc_outputs_class_proposals.masked_fill(invalid_mask, -inf)` before top-k.
    enc_cls = ops.where(valid, enc_cls, ops.cast(-1e30, enc_cls.dtype))
    enc_cls_max = ops.max(enc_cls, axis=-1)
    topk_indices = ops.top_k(
        enc_cls_max, k=min(num_queries, proj_shape[0] * proj_shape[1])
    )[1]
    topk_idx_4 = ops.repeat(ops.expand_dims(topk_indices, axis=-1), 4, axis=-1)
    refpoint_embed_ts = ops.take_along_axis(enc_coords, topk_idx_4, axis=1)
    refpoint_embed_ts = ops.stop_gradient(refpoint_embed_ts)

    return rf_detr_two_stage_refine_refpoints(
        refpoint_embed,
        refpoint_embed_ts,
        bbox_reparam=bbox_reparam,
        num_queries=num_queries,
    )


def rf_detr_decoder(
    memory,
    tgt,
    refpoints_unsigmoid,
    hidden_dim,
    dec_layers,
    sa_nheads,
    ca_nheads,
    dec_n_points,
    dim_feedforward,
    lite_refpoint_refine,
    bbox_reparam,
    spatial_shapes,
    return_seg_features=False,
):
    """Build the RF-DETR deformable decoder with iterative bbox refinement.

    Runs ``dec_layers`` deformable cross-attention decoder layers
    (:class:`RFDETRDecoderLayer`) on top of the encoder memory. Each
    layer takes the current query features, a query positional
    encoding derived from the current reference points, and a
    reference-point sampling location for deformable attention.

    Two refinement modes are supported:

    - ``lite_refpoint_refine=True`` (default): reference points are
      not refined between layers: the query positional encoding is
      computed once from the initial refpoints, and the bbox head is
      applied only after the final layer.
    - ``lite_refpoint_refine=False``: each decoder layer re-encodes
      the current refpoints into a query positional encoding and
      applies the bbox head to refine refpoints between layers,
      with gradient stopped between layers (standard DETR-style
      iterative refinement).

    Args:
        memory: Encoder memory ``(B, H*W, hidden_dim)``.
        tgt: Initial query features ``(B, num_queries, hidden_dim)``.
        refpoints_unsigmoid: Initial reference points
            ``(B, num_queries, 4)``: either learned (no two-stage) or
            from :func:`rf_detr_two_stage_refpoints`.
        hidden_dim: Decoder hidden dimension.
        dec_layers: Number of decoder layers.
        sa_nheads: Number of self-attention heads.
        ca_nheads: Number of deformable cross-attention heads.
        dec_n_points: Number of sampling points per head in deformable
            cross-attention.
        dim_feedforward: FFN dimension inside each decoder layer.
        lite_refpoint_refine: See above.
        bbox_reparam: Whether to use bbox reparameterization
            (multiplicative + exponential) instead of additive refinement.
        spatial_shapes: List of ``(H, W)`` per feature level (single
            level for RF-DETR).

    Returns:
        last_hidden_state: ``(B, num_queries, hidden_dim)`` post-norm
            query features after the final decoder layer.
        pred_boxes: ``(B, num_queries, 4)`` final bounding box
            predictions in (cx, cy, w, h) normalized coordinates.
    """
    level_start_index = [0]
    num_feature_levels = len(spatial_shapes)

    ref_point_head_0 = layers.Dense(
        hidden_dim, activation="relu", name="ref_point_head_0"
    )
    ref_point_head_1 = layers.Dense(hidden_dim, name="ref_point_head_1")

    decoder_layers_list = []
    for i in range(dec_layers):
        decoder_layers_list.append(
            RFDETRDecoderLayer(
                hidden_dim=hidden_dim,
                sa_nhead=sa_nheads,
                ca_nhead=ca_nheads,
                dim_feedforward=dim_feedforward,
                dropout=0.0,
                num_feature_levels=num_feature_levels,
                dec_n_points=dec_n_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                name=f"decoder_layer_{i}",
            )
        )

    decoder_norm = layers.LayerNormalization(epsilon=1e-5, name="decoder_norm")

    bbox_embed_0 = layers.Dense(hidden_dim, activation="relu", name="bbox_embed_0")
    bbox_embed_1 = layers.Dense(hidden_dim, activation="relu", name="bbox_embed_1")
    bbox_embed_2 = layers.Dense(4, name="bbox_embed_2")

    if bbox_reparam:
        ref_for_query = refpoints_unsigmoid
    else:
        ref_for_query = ops.sigmoid(refpoints_unsigmoid)

    if lite_refpoint_refine:
        query_sine = rf_detr_gen_sineembed_for_position(
            ref_for_query[..., :4], dim=hidden_dim // 2
        )
        query_pos = ref_point_head_1(ref_point_head_0(query_sine))

    output = tgt
    intermediate_hidden_states = []
    for dec_layer in decoder_layers_list:
        if not lite_refpoint_refine:
            if bbox_reparam:
                ref_for_query = refpoints_unsigmoid
            else:
                ref_for_query = ops.sigmoid(refpoints_unsigmoid)
            query_sine = rf_detr_gen_sineembed_for_position(
                ref_for_query[..., :4], dim=hidden_dim // 2
            )
            query_pos = ref_point_head_1(ref_point_head_0(query_sine))

        if bbox_reparam:
            ref_input = refpoints_unsigmoid[:, :, None, :]
        else:
            sig_ref = ops.sigmoid(refpoints_unsigmoid)
            ref_input = sig_ref[:, :, None, :]

        output = dec_layer(
            output,
            memory,
            query_pos,
            ref_input,
            training=False,
        )
        if return_seg_features:
            intermediate_hidden_states.append(decoder_norm(output))

        if not lite_refpoint_refine:
            new_delta = bbox_embed_2(bbox_embed_1(bbox_embed_0(output)))
            if bbox_reparam:
                new_cxcy = (
                    new_delta[..., :2] * refpoints_unsigmoid[..., 2:]
                    + refpoints_unsigmoid[..., :2]
                )
                new_wh = ops.exp(new_delta[..., 2:]) * refpoints_unsigmoid[..., 2:]
                refpoints_unsigmoid = ops.concatenate([new_cxcy, new_wh], axis=-1)
            else:
                refpoints_unsigmoid = refpoints_unsigmoid + new_delta
            refpoints_unsigmoid = ops.stop_gradient(refpoints_unsigmoid)

    last_hidden_state = decoder_norm(output)

    pred_bbox_delta = bbox_embed_2(bbox_embed_1(bbox_embed_0(last_hidden_state)))
    if bbox_reparam:
        pred_cxcy = (
            pred_bbox_delta[..., :2] * refpoints_unsigmoid[..., 2:]
            + refpoints_unsigmoid[..., :2]
        )
        pred_wh = ops.exp(pred_bbox_delta[..., 2:]) * refpoints_unsigmoid[..., 2:]
        pred_boxes = ops.concatenate([pred_cxcy, pred_wh], axis=-1)
    else:
        pred_boxes = ops.sigmoid(pred_bbox_delta + refpoints_unsigmoid)

    if return_seg_features:
        return last_hidden_state, pred_boxes, intermediate_hidden_states
    return last_hidden_state, pred_boxes


def rf_detr_functional(
    inputs,
    hidden_dim,
    backbone_hidden_size,
    backbone_num_heads,
    backbone_num_layers,
    backbone_mlp_ratio,
    backbone_use_swiglu,
    num_register_tokens,
    out_feature_indexes,
    patch_size,
    num_windows,
    positional_encoding_size,
    dec_layers,
    sa_nheads,
    ca_nheads,
    dec_n_points,
    num_queries,
    num_classes,
    two_stage,
    bbox_reparam,
    lite_refpoint_refine,
    dim_feedforward,
    input_shape,
    return_seg_features=False,
):
    """Build the full RF-DETR architecture from an input tensor (no class head).

    Top-level orchestrator that wires the four architectural stages:

    1. :func:`rf_detr_backbone`, DINOv2 windowed-attention backbone
       producing multi-scale features.
    2. :func:`rf_detr_projector`, concat + C2F fusion of the backbone
       features into a single decoder memory.
    3. :func:`rf_detr_decoder_inputs` (+ optional
       :func:`rf_detr_two_stage_refpoints`): initial query features
       and reference points.
    4. :func:`rf_detr_decoder`, deformable decoder with iterative
       bbox refinement and the final bbox prediction head.

    The final class prediction head is intentionally not built here:
    it is added by :class:`RFDETRDetect`, which composes
    :class:`RFDetrModel` around this graph.

    Args:
        inputs: Keras input tensor of shape ``(B, H, W, 3)`` (or
            ``(B, 3, H, W)`` for ``channels_first``).
        hidden_dim: Decoder hidden dimension.
        backbone_hidden_size: DINOv2 backbone hidden size.
        backbone_num_heads: Number of attention heads in backbone.
        backbone_num_layers: Number of transformer layers in backbone.
        backbone_mlp_ratio: MLP expansion ratio in backbone.
        backbone_use_swiglu: Whether backbone uses SwiGLU FFN.
        num_register_tokens: Number of DINOv2 register tokens.
        out_feature_indexes: Backbone layers to extract features from.
        patch_size: Patch size for DINOv2 patch embeddings.
        num_windows: Number of windows for windowed attention.
        positional_encoding_size: Size of positional encoding grid.
        dec_layers: Number of decoder layers.
        sa_nheads: Number of self-attention heads in decoder.
        ca_nheads: Number of cross-attention heads in decoder.
        dec_n_points: Number of sampling points in deformable attention.
        num_queries: Number of object queries.
        num_classes: Number of classes for the first-stage proposal
            scoring head (not the final detection head).
        two_stage: Whether to use two-stage query initialization.
        bbox_reparam: Whether to use bbox reparameterization.
        lite_refpoint_refine: Whether to use lite reference point
            refinement.
        dim_feedforward: FFN dimension in decoder.
        input_shape: Input image shape excluding batch dim.

    Returns:
        last_hidden_state: ``(B, num_queries, hidden_dim)`` decoder
            post-norm hidden states.
        pred_boxes: ``(B, num_queries, 4)`` final bbox predictions in
            (cx, cy, w, h) normalized coordinates.
    """
    data_format = keras.config.image_data_format()
    channels_axis = -1 if data_format == "channels_last" else 1

    features, spatial_shape = rf_detr_backbone(
        inputs,
        hidden_dim=backbone_hidden_size,
        num_heads=backbone_num_heads,
        num_layers=backbone_num_layers,
        mlp_ratio=backbone_mlp_ratio,
        use_swiglu=backbone_use_swiglu,
        num_register_tokens=num_register_tokens,
        out_feature_indexes=out_feature_indexes,
        patch_size=patch_size,
        num_windows=num_windows,
        positional_encoding_size=positional_encoding_size,
        input_shape=input_shape,
        data_format=data_format,
    )

    memory = rf_detr_projector(
        features,
        hidden_dim=hidden_dim,
        spatial_shape=spatial_shape,
        data_format=data_format,
        channels_axis=channels_axis,
    )

    spatial_shapes = [spatial_shape]

    tgt, refpoint_embed = rf_detr_decoder_inputs(
        memory,
        num_queries=num_queries,
        hidden_dim=hidden_dim,
    )

    if two_stage:
        refpoints_unsigmoid = rf_detr_two_stage_refpoints(
            memory,
            refpoint_embed,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_queries=num_queries,
            bbox_reparam=bbox_reparam,
            spatial_shapes=spatial_shapes,
        )
    else:
        refpoints_unsigmoid = refpoint_embed

    decoder_outputs = rf_detr_decoder(
        memory,
        tgt,
        refpoints_unsigmoid,
        hidden_dim=hidden_dim,
        dec_layers=dec_layers,
        sa_nheads=sa_nheads,
        ca_nheads=ca_nheads,
        dec_n_points=dec_n_points,
        dim_feedforward=dim_feedforward,
        lite_refpoint_refine=lite_refpoint_refine,
        bbox_reparam=bbox_reparam,
        spatial_shapes=spatial_shapes,
        return_seg_features=return_seg_features,
    )
    if not return_seg_features:
        return decoder_outputs

    last_hidden_state, pred_boxes, intermediate_hidden_states = decoder_outputs
    num_h, num_w = spatial_shape
    spatial_features = ops.reshape(memory, (-1, num_h, num_w, hidden_dim))
    return last_hidden_state, pred_boxes, spatial_features, intermediate_hidden_states


@keras.saving.register_keras_serializable(package="zeromodels")
class RFDetrModel(BaseModel):
    """RF-DETR backbone + projector + decoder (no class heads).

    bare-backbone ``Model`` variant: outputs the decoder ``last_hidden_state``
    (post layernorm) with shape ``(B, num_queries, hidden_dim)``. The
    final class prediction head is pruned from the output graph; for the
    default ``lite_refpoint_refine=True`` setting, bbox-embed layers are
    also downstream of the output and naturally excluded. Use
    ``RFDETRDetect`` for full detection outputs.

    This bare-backbone variant has no class head, so it does not implement
    ``from_weights("hf:...")``; use :class:`RFDETRDetect` to load the
    ``Roboflow/rf-detr-*`` checkpoints from the model Hub.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = None

    def __init__(
        self,
        hidden_dim=256,
        backbone_hidden_size=384,
        backbone_num_heads=6,
        backbone_num_layers=12,
        backbone_mlp_ratio=4,
        backbone_use_swiglu=False,
        num_register_tokens=0,
        out_feature_indexes=None,
        patch_size=14,
        num_windows=4,
        positional_encoding_size=37,
        resolution=560,
        dec_layers=3,
        sa_nheads=8,
        ca_nheads=16,
        dec_n_points=2,
        num_queries=300,
        num_classes=91,
        two_stage=True,
        bbox_reparam=True,
        lite_refpoint_refine=True,
        group_detr=13,
        dim_feedforward=2048,
        image_size=560,
        input_tensor=None,
        name="RFDetrModel",
        **kwargs,
    ):
        if out_feature_indexes is None:
            out_feature_indexes = [2, 5, 8, 11]

        data_format = keras.config.image_data_format()
        image_size = standardize_input_shape(image_size, data_format)

        if input_tensor is None:
            img_input = layers.Input(shape=image_size)
        else:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=image_size)
            else:
                img_input = input_tensor

        last_hidden_state, pred_boxes = rf_detr_functional(
            img_input,
            hidden_dim=hidden_dim,
            backbone_hidden_size=backbone_hidden_size,
            backbone_num_heads=backbone_num_heads,
            backbone_num_layers=backbone_num_layers,
            backbone_mlp_ratio=backbone_mlp_ratio,
            backbone_use_swiglu=backbone_use_swiglu,
            num_register_tokens=num_register_tokens,
            out_feature_indexes=out_feature_indexes,
            patch_size=patch_size,
            num_windows=num_windows,
            positional_encoding_size=positional_encoding_size,
            dec_layers=dec_layers,
            sa_nheads=sa_nheads,
            ca_nheads=ca_nheads,
            dec_n_points=dec_n_points,
            num_queries=num_queries,
            num_classes=num_classes,
            two_stage=two_stage,
            bbox_reparam=bbox_reparam,
            lite_refpoint_refine=lite_refpoint_refine,
            dim_feedforward=dim_feedforward,
            input_shape=image_size,
        )

        outputs = {"last_hidden_state": last_hidden_state, "pred_boxes": pred_boxes}
        super().__init__(inputs=img_input, outputs=outputs, name=name, **kwargs)

        self.hidden_dim = hidden_dim
        self.backbone_hidden_size = backbone_hidden_size
        self.backbone_num_heads = backbone_num_heads
        self.backbone_num_layers = backbone_num_layers
        self.backbone_mlp_ratio = backbone_mlp_ratio
        self.backbone_use_swiglu = backbone_use_swiglu
        self.num_register_tokens = num_register_tokens
        self.out_feature_indexes = out_feature_indexes
        self.patch_size = patch_size
        self.num_windows = num_windows
        self.positional_encoding_size = positional_encoding_size
        self.resolution = resolution
        self.dec_layers = dec_layers
        self.sa_nheads = sa_nheads
        self.ca_nheads = ca_nheads
        self.dec_n_points = dec_n_points
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.two_stage = two_stage
        self.bbox_reparam = bbox_reparam
        self.lite_refpoint_refine = lite_refpoint_refine
        self.group_detr = group_detr
        self.dim_feedforward = dim_feedforward
        self.image_size = image_size
        self._input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "backbone_hidden_size": self.backbone_hidden_size,
                "backbone_num_heads": self.backbone_num_heads,
                "backbone_num_layers": self.backbone_num_layers,
                "backbone_mlp_ratio": self.backbone_mlp_ratio,
                "backbone_use_swiglu": self.backbone_use_swiglu,
                "num_register_tokens": self.num_register_tokens,
                "out_feature_indexes": self.out_feature_indexes,
                "patch_size": self.patch_size,
                "num_windows": self.num_windows,
                "positional_encoding_size": self.positional_encoding_size,
                "resolution": self.resolution,
                "dec_layers": self.dec_layers,
                "sa_nheads": self.sa_nheads,
                "ca_nheads": self.ca_nheads,
                "dec_n_points": self.dec_n_points,
                "num_queries": self.num_queries,
                "num_classes": self.num_classes,
                "two_stage": self.two_stage,
                "bbox_reparam": self.bbox_reparam,
                "lite_refpoint_refine": self.lite_refpoint_refine,
                "group_detr": self.group_detr,
                "dim_feedforward": self.dim_feedforward,
                "image_size": self.image_size,
                "input_tensor": self._input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def from_hf(cls, hf_id, load_weights=True, **kwargs):
        raise NotImplementedError(
            "RFDetrModel is the headless backbone variant with no Hub checkpoint. "
            "Use RFDETRDetect.from_weights('hf:Roboflow/rf-detr-base') for the "
            "detection model, or pass a local .weights.h5 path to model.load_weights()."
        )


@keras.saving.register_keras_serializable(package="zeromodels")
class RFDETRDetect(BaseModel):
    """RF-DETR: Real-Time Detection Transformer.

    A real-time object detection model based on DINOv2 backbone with windowed
    attention, a simple projector, and a deformable DETR decoder. Uses two-stage
    query initialization and iterative bounding box refinement.

    Reference:
        - RF-DETR (Roboflow, ICLR 2026)
        - https://github.com/roboflow/rf-detr

    Args:
        hidden_dim: Transformer decoder hidden dimension.
        backbone_hidden_size: DINOv2 backbone hidden size.
        backbone_num_heads: Number of attention heads in backbone.
        backbone_num_layers: Number of transformer layers in backbone.
        backbone_mlp_ratio: MLP expansion ratio in backbone.
        backbone_use_swiglu: Whether backbone uses SwiGLU FFN.
        num_register_tokens: Number of register tokens in DINOv2.
        out_feature_indexes: Backbone layer indices to extract features from.
        patch_size: Patch size for DINOv2 patch embeddings.
        num_windows: Number of windows for windowed attention.
        positional_encoding_size: Size of positional encoding grid.
        resolution: Input image resolution.
        dec_layers: Number of decoder layers.
        sa_nheads: Number of self-attention heads in decoder.
        ca_nheads: Number of cross-attention heads in decoder.
        dec_n_points: Number of sampling points in deformable attention.
        num_queries: Number of object queries.
        num_classes: Number of object classes (COCO: 91).
        two_stage: Whether to use two-stage query initialization.
        bbox_reparam: Whether to use bbox reparameterization.
        lite_refpoint_refine: Whether to use lite reference point refinement.
        group_detr: Number of DETR groups (training only, inference uses 1).
        dim_feedforward: FFN dimension in decoder.
        weights: Pre-trained weight identifier or file path.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `resolution` (an N x N x 3
            square at the per-variant resolution).
        input_tensor: Optional input Keras tensor.
        name: Model name.
    """

    BASE_MODEL_CONFIG = None
    # Weights load by Hub repo id, e.g. from_weights("zeromodels/rfdetr-nano"),
    # via zm_config.json on the repo (no url table in the package).
    config_class = RFDetrConfig
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "rf_detr"

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_rf_detr_hf_to_keras import transfer_rf_detr_weights

        # RF-DETR's custom layers create their weights on the first call, so
        # build the functional graph on a dummy input before assigning weights.
        shape = [d if d is not None else 1 for d in keras_model.inputs[0].shape]
        keras_model(ops.zeros(shape, dtype="float32"))
        transfer_rf_detr_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        import math

        bb = hf_config["backbone_config"]
        patch_size = bb["patch_size"]
        num_windows = bb.get("num_windows", 1)
        image_size = bb["image_size"]
        # Input resolution isn't stored in config.json; reconstruct the value the
        # image processor uses: the smallest multiple of patch_size * num_windows
        # that is >= the backbone image_size (windowed attention needs the feature
        # grid divisible by num_windows).
        unit = patch_size * max(num_windows, 1)
        resolution = math.ceil(image_size / unit) * unit
        num_classes = (
            len(hf_config["id2label"])
            if "id2label" in hf_config
            else hf_config.get("num_labels", 91)
        )
        return {
            "hidden_dim": hf_config["d_model"],
            "backbone_hidden_size": bb["hidden_size"],
            "backbone_num_heads": bb["num_attention_heads"],
            "backbone_num_layers": bb["num_hidden_layers"],
            "backbone_mlp_ratio": bb.get("mlp_ratio", 4),
            "backbone_use_swiglu": bb.get("use_swiglu_ffn", False),
            "num_register_tokens": bb.get("num_register_tokens", 0),
            "out_feature_indexes": [
                int(s.removeprefix("stage")) for s in bb["out_features"]
            ],
            "patch_size": patch_size,
            "num_windows": num_windows,
            "positional_encoding_size": image_size // patch_size,
            "resolution": resolution,
            "dec_layers": hf_config["decoder_layers"],
            "sa_nheads": hf_config["decoder_self_attention_heads"],
            "ca_nheads": hf_config["decoder_cross_attention_heads"],
            "dec_n_points": hf_config["decoder_n_points"],
            "num_queries": hf_config["num_queries"],
            "num_classes": num_classes,
            "group_detr": hf_config["group_detr"],
            "dim_feedforward": hf_config["decoder_ffn_dim"],
        }

    def __init__(
        self,
        hidden_dim=256,
        backbone_hidden_size=384,
        backbone_num_heads=6,
        backbone_num_layers=12,
        backbone_mlp_ratio=4,
        backbone_use_swiglu=False,
        num_register_tokens=0,
        out_feature_indexes=None,
        patch_size=14,
        num_windows=4,
        positional_encoding_size=37,
        resolution=560,
        dec_layers=3,
        sa_nheads=8,
        ca_nheads=16,
        dec_n_points=2,
        num_queries=300,
        num_classes=91,
        two_stage=True,
        bbox_reparam=True,
        lite_refpoint_refine=True,
        group_detr=13,
        dim_feedforward=2048,
        image_size=None,
        input_tensor=None,
        name="RFDETRDetect",
        **kwargs,
    ):
        if out_feature_indexes is None:
            out_feature_indexes = [2, 5, 8, 11]

        if image_size is None:
            image_size = resolution

        base = RFDetrModel(
            hidden_dim=hidden_dim,
            backbone_hidden_size=backbone_hidden_size,
            backbone_num_heads=backbone_num_heads,
            backbone_num_layers=backbone_num_layers,
            backbone_mlp_ratio=backbone_mlp_ratio,
            backbone_use_swiglu=backbone_use_swiglu,
            num_register_tokens=num_register_tokens,
            out_feature_indexes=out_feature_indexes,
            patch_size=patch_size,
            num_windows=num_windows,
            positional_encoding_size=positional_encoding_size,
            resolution=resolution,
            dec_layers=dec_layers,
            sa_nheads=sa_nheads,
            ca_nheads=ca_nheads,
            dec_n_points=dec_n_points,
            num_queries=num_queries,
            num_classes=num_classes,
            two_stage=two_stage,
            bbox_reparam=bbox_reparam,
            lite_refpoint_refine=lite_refpoint_refine,
            group_detr=group_detr,
            dim_feedforward=dim_feedforward,
            image_size=image_size,
            input_tensor=input_tensor,
            name=f"{name}_model",
        )
        last_hidden_state = base.output["last_hidden_state"]
        pred_boxes = base.output["pred_boxes"]

        logits = layers.Dense(num_classes, name="class_embed")(last_hidden_state)

        outputs = {"logits": logits, "pred_boxes": pred_boxes}
        super().__init__(inputs=base.input, outputs=outputs, name=name, **kwargs)

        self.hidden_dim = hidden_dim
        self.backbone_hidden_size = backbone_hidden_size
        self.backbone_num_heads = backbone_num_heads
        self.backbone_num_layers = backbone_num_layers
        self.backbone_mlp_ratio = backbone_mlp_ratio
        self.backbone_use_swiglu = backbone_use_swiglu
        self.num_register_tokens = num_register_tokens
        self.out_feature_indexes = out_feature_indexes
        self.patch_size = patch_size
        self.num_windows = num_windows
        self.positional_encoding_size = positional_encoding_size
        self.resolution = resolution
        self.dec_layers = dec_layers
        self.sa_nheads = sa_nheads
        self.ca_nheads = ca_nheads
        self.dec_n_points = dec_n_points
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.two_stage = two_stage
        self.bbox_reparam = bbox_reparam
        self.lite_refpoint_refine = lite_refpoint_refine
        self.group_detr = group_detr
        self.dim_feedforward = dim_feedforward
        self.image_size = base.image_size
        self._input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "backbone_hidden_size": self.backbone_hidden_size,
                "backbone_num_heads": self.backbone_num_heads,
                "backbone_num_layers": self.backbone_num_layers,
                "backbone_mlp_ratio": self.backbone_mlp_ratio,
                "backbone_use_swiglu": self.backbone_use_swiglu,
                "num_register_tokens": self.num_register_tokens,
                "out_feature_indexes": self.out_feature_indexes,
                "patch_size": self.patch_size,
                "num_windows": self.num_windows,
                "positional_encoding_size": self.positional_encoding_size,
                "resolution": self.resolution,
                "dec_layers": self.dec_layers,
                "sa_nheads": self.sa_nheads,
                "ca_nheads": self.ca_nheads,
                "dec_n_points": self.dec_n_points,
                "num_queries": self.num_queries,
                "num_classes": self.num_classes,
                "two_stage": self.two_stage,
                "bbox_reparam": self.bbox_reparam,
                "lite_refpoint_refine": self.lite_refpoint_refine,
                "group_detr": self.group_detr,
                "dim_feedforward": self.dim_feedforward,
                "image_size": self.image_size,
                "input_tensor": self._input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def rf_detr_segmentation_head(
    spatial_features,
    list_query_features,
    *,
    num_queries,
    mask_height,
    mask_width,
    hidden_dim,
    intermediate_size,
    seg_activation,
):
    """RF-DETR mask head: one ConvNeXt-style block per decoder layer, dotted with
    the (projected) per-layer queries.

    Mirrors HF ``RfDetrForInstanceSegmentation.segmentation_head``: the projector
    spatial map is bilinearly resized to ``image_size / mask_downsample_ratio``,
    then for each decoder layer a depthwise-conv block refines it and a
    ``query x spatial`` product (plus a learned scalar bias) yields per-query mask
    logits; the last layer's logits are returned.

    Args:
        spatial_features: ``(B, Hf, Wf, hidden_dim)`` projector feature map.
        list_query_features: One ``(B, num_queries, hidden_dim)`` post-norm decoder
            hidden state per decoder layer.
        num_queries: Number of object queries.
        mask_height: Target mask height (image height // mask_downsample_ratio).
        mask_width: Target mask width (image width // mask_downsample_ratio).
        hidden_dim: Model dimension.
        intermediate_size: Hidden dimension of the query-features MLP.
        seg_activation: Activation used in the blocks and the query MLP.

    Returns:
        ``(B, num_queries, mask_height, mask_width)`` mask logits.
    """
    # Layers shared across decoder layers (applied once per layer in the loop).
    qf_norm = layers.LayerNormalization(
        epsilon=1e-5, name="seg_query_features_block_norm"
    )
    qf_fc1 = layers.Dense(intermediate_size, name="seg_query_features_block_fc1")
    qf_fc2 = layers.Dense(hidden_dim, name="seg_query_features_block_fc2")
    qf_proj = layers.Dense(hidden_dim, name="seg_query_features_proj")
    spatial_proj = layers.Conv2D(
        hidden_dim, 1, data_format="channels_last", name="seg_spatial_features_proj"
    )
    seg_bias = RFDETRSegmentationBias(name="seg_bias")

    sf = ops.image.resize(
        spatial_features,
        (mask_height, mask_width),
        interpolation="bilinear",
        data_format="channels_last",
    )

    mask_logits = None
    for i, query_features in enumerate(list_query_features):
        residual = sf
        x = layers.DepthwiseConv2D(
            3,
            padding="same",
            data_format="channels_last",
            name=f"seg_block_{i}_dwconv",
        )(sf)
        x = layers.LayerNormalization(epsilon=1e-6, name=f"seg_block_{i}_norm")(x)
        x = layers.Dense(hidden_dim, name=f"seg_block_{i}_pwconv")(x)
        x = layers.Activation(seg_activation)(x)
        sf = layers.Add()([x, residual])

        spatial = spatial_proj(sf)  # (B, mask_h, mask_w, hidden_dim)

        q = qf_fc2(layers.Activation(seg_activation)(qf_fc1(qf_norm(query_features))))
        q = layers.Add()([q, query_features])
        q = qf_proj(q)  # (B, num_queries, hidden_dim)

        spatial_flat = ops.reshape(spatial, (-1, mask_height * mask_width, hidden_dim))
        logits = ops.matmul(q, ops.transpose(spatial_flat, (0, 2, 1)))
        logits = ops.reshape(logits, (-1, num_queries, mask_height, mask_width))
        mask_logits = seg_bias(logits)
    return mask_logits


@keras.saving.register_keras_serializable(package="zeromodels")
class RFDETRInstanceSegment(BaseModel):
    """RF-DETR instance segmentation.

    Adds a mask head on top of the :class:`RFDETRDetect` detection architecture
    (DINOv2 backbone + deformable decoder). The projector's spatial feature map and
    every decoder layer's query features are routed through the segmentation head
    to produce per-query masks.

    Takes a ``(B, H, W, 3)`` image (``RFDETRImageProcessor`` output) and returns
    ``{"logits": (B, num_queries, num_classes), "pred_boxes": (B, num_queries,
    4), "pred_masks": (B, num_queries, H // mask_downsample_ratio, W //
    mask_downsample_ratio)}``.

    Reference:
        - RF-DETR (Roboflow): https://github.com/roboflow/rf-detr

    Args:
        See :class:`RFDETRDetect` for the backbone / decoder arguments.
        mask_downsample_ratio: Mask resolution = image size // this. Defaults to `4`.
        intermediate_size: Hidden dim of the segmentation query MLP. Defaults to `1024`.
        seg_activation: Activation in the segmentation head. Defaults to `"gelu"`.
        image_size: Input image specification (see :class:`RFDETRDetect`).
        input_tensor: Optional input Keras tensor.
        name: Model name.
    """

    BASE_MODEL_CONFIG = None
    config_class = RFDetrSegmentConfig
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "rf_detr"

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_rf_detr_hf_to_keras import transfer_rf_detr_seg_weights

        shape = [d if d is not None else 1 for d in keras_model.inputs[0].shape]
        keras_model(ops.zeros(shape, dtype="float32"))
        transfer_rf_detr_seg_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        config = RFDETRDetect.config_from_hf(hf_config)
        config["mask_downsample_ratio"] = hf_config.get("mask_downsample_ratio", 4)
        config["intermediate_size"] = hf_config.get("intermediate_size", 1024)
        config["seg_activation"] = hf_config.get(
            "segmentation_head_activation_function", "gelu"
        )
        return config

    def __init__(
        self,
        hidden_dim=256,
        backbone_hidden_size=384,
        backbone_num_heads=6,
        backbone_num_layers=12,
        backbone_mlp_ratio=4,
        backbone_use_swiglu=False,
        num_register_tokens=0,
        out_feature_indexes=None,
        patch_size=12,
        num_windows=2,
        positional_encoding_size=32,
        resolution=384,
        dec_layers=4,
        sa_nheads=8,
        ca_nheads=16,
        dec_n_points=2,
        num_queries=100,
        num_classes=91,
        two_stage=True,
        bbox_reparam=True,
        lite_refpoint_refine=True,
        group_detr=13,
        dim_feedforward=2048,
        mask_downsample_ratio=4,
        intermediate_size=1024,
        seg_activation="gelu",
        image_size=None,
        input_tensor=None,
        name="RFDETRInstanceSegment",
        **kwargs,
    ):
        if out_feature_indexes is None:
            out_feature_indexes = [3, 6, 9, 12]
        if image_size is None:
            image_size = resolution

        data_format = keras.config.image_data_format()
        image_size = standardize_input_shape(image_size, data_format)

        if input_tensor is None:
            img_input = layers.Input(shape=image_size)
        elif not utils.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=image_size)
        else:
            img_input = input_tensor

        (
            last_hidden_state,
            pred_boxes,
            spatial_features,
            intermediate_hidden_states,
        ) = rf_detr_functional(
            img_input,
            hidden_dim=hidden_dim,
            backbone_hidden_size=backbone_hidden_size,
            backbone_num_heads=backbone_num_heads,
            backbone_num_layers=backbone_num_layers,
            backbone_mlp_ratio=backbone_mlp_ratio,
            backbone_use_swiglu=backbone_use_swiglu,
            num_register_tokens=num_register_tokens,
            out_feature_indexes=out_feature_indexes,
            patch_size=patch_size,
            num_windows=num_windows,
            positional_encoding_size=positional_encoding_size,
            dec_layers=dec_layers,
            sa_nheads=sa_nheads,
            ca_nheads=ca_nheads,
            dec_n_points=dec_n_points,
            num_queries=num_queries,
            num_classes=num_classes,
            two_stage=two_stage,
            bbox_reparam=bbox_reparam,
            lite_refpoint_refine=lite_refpoint_refine,
            dim_feedforward=dim_feedforward,
            input_shape=image_size,
            return_seg_features=True,
        )

        logits = layers.Dense(num_classes, name="class_embed")(last_hidden_state)

        if data_format == "channels_last":
            img_h, img_w = image_size[0], image_size[1]
        else:
            img_h, img_w = image_size[1], image_size[2]

        pred_masks = rf_detr_segmentation_head(
            spatial_features,
            intermediate_hidden_states,
            num_queries=num_queries,
            mask_height=img_h // mask_downsample_ratio,
            mask_width=img_w // mask_downsample_ratio,
            hidden_dim=hidden_dim,
            intermediate_size=intermediate_size,
            seg_activation=seg_activation,
        )

        outputs = {
            "logits": logits,
            "pred_boxes": pred_boxes,
            "pred_masks": pred_masks,
        }
        super().__init__(inputs=img_input, outputs=outputs, name=name, **kwargs)

        self.hidden_dim = hidden_dim
        self.backbone_hidden_size = backbone_hidden_size
        self.backbone_num_heads = backbone_num_heads
        self.backbone_num_layers = backbone_num_layers
        self.backbone_mlp_ratio = backbone_mlp_ratio
        self.backbone_use_swiglu = backbone_use_swiglu
        self.num_register_tokens = num_register_tokens
        self.out_feature_indexes = out_feature_indexes
        self.patch_size = patch_size
        self.num_windows = num_windows
        self.positional_encoding_size = positional_encoding_size
        self.resolution = resolution
        self.dec_layers = dec_layers
        self.sa_nheads = sa_nheads
        self.ca_nheads = ca_nheads
        self.dec_n_points = dec_n_points
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.two_stage = two_stage
        self.bbox_reparam = bbox_reparam
        self.lite_refpoint_refine = lite_refpoint_refine
        self.group_detr = group_detr
        self.dim_feedforward = dim_feedforward
        self.mask_downsample_ratio = mask_downsample_ratio
        self.intermediate_size = intermediate_size
        self.seg_activation = seg_activation
        self.image_size = image_size
        self._input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "backbone_hidden_size": self.backbone_hidden_size,
                "backbone_num_heads": self.backbone_num_heads,
                "backbone_num_layers": self.backbone_num_layers,
                "backbone_mlp_ratio": self.backbone_mlp_ratio,
                "backbone_use_swiglu": self.backbone_use_swiglu,
                "num_register_tokens": self.num_register_tokens,
                "out_feature_indexes": self.out_feature_indexes,
                "patch_size": self.patch_size,
                "num_windows": self.num_windows,
                "positional_encoding_size": self.positional_encoding_size,
                "resolution": self.resolution,
                "dec_layers": self.dec_layers,
                "sa_nheads": self.sa_nheads,
                "ca_nheads": self.ca_nheads,
                "dec_n_points": self.dec_n_points,
                "num_queries": self.num_queries,
                "num_classes": self.num_classes,
                "two_stage": self.two_stage,
                "bbox_reparam": self.bbox_reparam,
                "lite_refpoint_refine": self.lite_refpoint_refine,
                "group_detr": self.group_detr,
                "dim_feedforward": self.dim_feedforward,
                "mask_downsample_ratio": self.mask_downsample_ratio,
                "intermediate_size": self.intermediate_size,
                "seg_activation": self.seg_activation,
                "image_size": self.image_size,
                "input_tensor": self._input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
