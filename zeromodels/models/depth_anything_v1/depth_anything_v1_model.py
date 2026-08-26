import keras
from keras import layers, ops

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.models.vit.vit_layers import (
    ViTAddPositionEmbs,
    ViTClassDistToken,
    ViTMultiHeadSelfAttention,
)
from zeromodels.utils import standardize_input_shape

from .depth_anything_v1_config import DepthAnythingV1Config


def depth_anything_v1_aligned_bilinear_resize(x, target_h, target_w, data_format):
    """bilinear resize matching ``align_corners=True``.

    ``keras.ops.image.resize`` only supports half-pixel alignment, but
    the reference Depth Anything upsamples (inside each fusion block
    and inside the head) use ``torch.nn.functional.interpolate`` with
    ``align_corners=True``. This helper implements the align-corners
    coordinate mapping manually via explicit gather + lerp so the
    Keras graph produces the same outputs on every backend.

    For ``target_h > 1`` the output rows are sampled at source
    coordinates ``i * (H - 1) / (target_h - 1)`` for ``i = 0..target_h - 1``
    (and symmetrically for width). The four surrounding source pixels
    are gathered with ``ops.take`` and blended with the fractional
    offsets ``dy`` / ``dx``.

    Args:
        x: Input tensor. ``(batch, H, W, C)`` when ``data_format`` is
            ``"channels_last"``, ``(batch, C, H, W)`` when
            ``"channels_first"``.
        target_h: Integer, target output height.
        target_w: Integer, target output width.
        data_format: Either ``"channels_last"`` or ``"channels_first"``.

    Returns:
        Tensor with the same layout as ``x`` and spatial dims
        ``(target_h, target_w)``.
    """
    shape = ops.shape(x)
    if data_format == "channels_first":
        h_axis, w_axis = 2, 3
        h = shape[2]
        w = shape[3]
    else:
        h_axis, w_axis = 1, 2
        h = shape[1]
        w = shape[2]

    h_f = ops.cast(h, "float32")
    w_f = ops.cast(w, "float32")

    if target_h > 1:
        y_coords = (
            ops.arange(target_h, dtype="float32") * (h_f - 1.0) / float(target_h - 1)
        )
    else:
        y_coords = ops.zeros((1,), dtype="float32")

    if target_w > 1:
        x_coords = (
            ops.arange(target_w, dtype="float32") * (w_f - 1.0) / float(target_w - 1)
        )
    else:
        x_coords = ops.zeros((1,), dtype="float32")

    y0 = ops.cast(ops.floor(y_coords), "int32")
    x0 = ops.cast(ops.floor(x_coords), "int32")
    y1 = ops.minimum(y0 + 1, h - 1)
    x1 = ops.minimum(x0 + 1, w - 1)
    y0 = ops.minimum(y0, h - 1)
    x0 = ops.minimum(x0, w - 1)

    dy = y_coords - ops.cast(y0, "float32")
    dx = x_coords - ops.cast(x0, "float32")

    top = ops.take(x, y0, axis=h_axis)
    bot = ops.take(x, y1, axis=h_axis)
    tl = ops.take(top, x0, axis=w_axis)
    tr = ops.take(top, x1, axis=w_axis)
    bl = ops.take(bot, x0, axis=w_axis)
    br = ops.take(bot, x1, axis=w_axis)

    if data_format == "channels_first":
        dx_r = ops.reshape(dx, (1, 1, 1, target_w))
        dy_r = ops.reshape(dy, (1, 1, target_h, 1))
    else:
        dx_r = ops.reshape(dx, (1, 1, target_w, 1))
        dy_r = ops.reshape(dy, (1, target_h, 1, 1))

    top_lerp = tl * (1.0 - dx_r) + tr * dx_r
    bot_lerp = bl * (1.0 - dx_r) + br * dx_r
    return top_lerp * (1.0 - dy_r) + bot_lerp * dy_r


def depth_anything_v1_pre_act_residual(x, channels, name, data_format):
    """Pre-activation residual unit used inside DPT fusion blocks.

    Follows the ResNet pre-activation layout: ``ReLU -> 3x3 Conv ->
    ReLU -> 3x3 Conv`` and adds the original input back at the end.
    Both convolutions preserve the spatial dimensions and produce
    ``channels`` output channels.

    Args:
        x: Input feature tensor, shaped according to ``data_format``.
        channels: Integer, number of output channels for both
            convolutions. Must match the input channel count so the
            residual add is well-defined.
        name: String prefix for the layer names. The internal layers
            are named ``{name}_act1``, ``{name}_conv1``, ``{name}_act2``,
            ``{name}_conv2``, and ``{name}_add``.
        data_format: Either ``"channels_last"`` or ``"channels_first"``.

    Returns:
        Output tensor with the same shape as ``x``.
    """
    residual = x
    x = layers.Activation("relu", name=f"{name}_act1")(x)
    x = layers.Conv2D(
        channels, 3, padding="same", data_format=data_format, name=f"{name}_conv1"
    )(x)
    x = layers.Activation("relu", name=f"{name}_act2")(x)
    x = layers.Conv2D(
        channels, 3, padding="same", data_format=data_format, name=f"{name}_conv2"
    )(x)
    return layers.Add(name=f"{name}_add")([x, residual])


@keras.saving.register_keras_serializable(package="zeromodels")
class DepthAnythingV1Resize(layers.Layer):
    """Serializable align-corners bilinear resize to a fixed ``(target_h, target_w)``.

    Wraps :func:`depth_anything_v1_aligned_bilinear_resize` (used by the fusion
    blocks and the head) as a real layer with a ``get_config`` so the model
    round-trips through ``model.save()`` / ``load_model()``: unlike a closure
    ``Lambda``, which Keras cannot serialize.
    """

    def __init__(self, target_h, target_w, data_format, **kwargs):
        super().__init__(**kwargs)
        self.target_h = target_h
        self.target_w = target_w
        self.data_format = data_format

    def call(self, x):
        return depth_anything_v1_aligned_bilinear_resize(
            x, self.target_h, self.target_w, self.data_format
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "target_h": self.target_h,
                "target_w": self.target_w,
                "data_format": self.data_format,
            }
        )
        return config


def depth_anything_v1_fusion_block(
    hidden_state,
    residual,
    target_h,
    target_w,
    fusion_hidden_size,
    data_format,
    name,
):
    """One stage of the DPT bottom-up feature-fusion pyramid.

    Each fusion stage optionally merges a higher-resolution skip
    connection (``residual``) into the running ``hidden_state`` via a
    pre-activation residual unit, passes the merged tensor through a
    second pre-activation residual unit, upsamples it to the next
    pyramid level with aligned-corners bilinear interpolation, and
    finally projects it through a 1x1 convolution. The first stage of
    the pyramid receives ``residual=None`` and just refines and
    upsamples its input.

    Args:
        hidden_state: Running fused tensor from the previous fusion
            stage (or the coarsest projected feature map on the first
            stage).
        residual: Optional higher-resolution skip connection from the
            neck projection layer. Pass ``None`` on the first stage.
        target_h: Integer, output height after the upsample.
        target_w: Integer, output width after the upsample.
        fusion_hidden_size: Integer, channel count used by both
            pre-activation residual units and the final 1x1 projection.
        data_format: Either ``"channels_last"`` or ``"channels_first"``.
        name: String prefix for the layer names of this fusion stage.

    Returns:
        Output tensor with spatial size ``(target_h, target_w)`` and
        ``fusion_hidden_size`` channels, passed into the next fusion
        stage or into the depth-estimation head.
    """
    if residual is not None:
        hidden_state = layers.Add(name=f"{name}_add_residual")(
            [
                hidden_state,
                depth_anything_v1_pre_act_residual(
                    residual, fusion_hidden_size, f"{name}_res1", data_format
                ),
            ]
        )

    hidden_state = depth_anything_v1_pre_act_residual(
        hidden_state, fusion_hidden_size, f"{name}_res2", data_format
    )

    hidden_state = DepthAnythingV1Resize(
        target_h, target_w, data_format, name=f"{name}_upsample"
    )(hidden_state)

    hidden_state = layers.Conv2D(
        fusion_hidden_size,
        1,
        padding="valid",
        data_format=data_format,
        name=f"{name}_projection",
    )(hidden_state)

    return hidden_state


def depth_anything_v1_dino_backbone(
    pixel_values,
    backbone_dim,
    backbone_depth,
    backbone_num_heads,
    out_indices,
    patch_size,
    patch_h,
    patch_w,
    mlp_ratio,
    data_format,
    name="backbone",
):
    """Functional DINOv2 ViT backbone used by Depth Anything V1.

    Runs the standard DINOv2 stack: patch embedding, class token,
    learnable position embeddings, ``backbone_depth`` pre-norm
    transformer blocks with ``DepthAnythingV1LayerScale`` on both branches, and a
    final shared ``LayerNorm``, and returns the intermediate token
    sequences listed in ``out_indices`` as spatial feature maps.

    The final ``LayerNorm`` is applied once per selected block, the
    class token is stripped from the head, and the remaining patch
    tokens are reshaped to ``(patch_h, patch_w, backbone_dim)``. When
    ``data_format`` is ``"channels_first"`` the spatial tensor is
    transposed back to ``(backbone_dim, patch_h, patch_w)`` via a
    ``Permute`` so it can feed the channels-first neck directly.

    Args:
        pixel_values: Input image tensor from the model's ``Input``
            layer, shaped according to ``data_format``.
        backbone_dim: Integer, embedding dimension of the DINOv2 ViT.
        backbone_depth: Integer, number of transformer blocks.
        backbone_num_heads: Integer, number of attention heads per
            block.
        out_indices: Iterable of 1-indexed block numbers whose outputs
            are returned. Depth Anything V1 uses the last four blocks
            of each backbone variant.
        patch_size: Integer, spatial size of the patch-embedding
            convolution (``14`` for DINOv2).
        patch_h: Integer, patch-grid height, equal to
            ``image_height // patch_size``.
        patch_w: Integer, patch-grid width, equal to
            ``image_width // patch_size``.
        mlp_ratio: Float, expansion ratio of the MLP hidden layer
            inside each block. Depth Anything uses ``4.0``.
        data_format: Either ``"channels_last"`` or ``"channels_first"``.
        name: String prefix used to name every layer created by this
            backbone. Defaults to ``"backbone"``.

    Returns:
        List of spatial feature tensors, one per entry in
        ``out_indices``, in the order those entries appear.
    """
    x = layers.Conv2D(
        backbone_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        data_format=data_format,
        name=f"{name}_patch_embed",
    )(pixel_values)
    if data_format == "channels_first":
        x = ops.transpose(x, (0, 2, 3, 1))
    x = layers.Reshape((-1, backbone_dim))(x)
    x = ViTClassDistToken(use_distillation=False, name=f"{name}_cls_token")(x)
    x = ViTAddPositionEmbs(
        name=f"{name}_pos_embed",
        no_embed_class=False,
        use_distillation=False,
        grid_h=patch_h,
        grid_w=patch_w,
    )(x)

    all_features = []
    for i in range(backbone_depth):
        x_norm = layers.LayerNormalization(
            epsilon=1e-6, axis=-1, name=f"{name}_block_{i}_ln1"
        )(x)
        x_attn = ViTMultiHeadSelfAttention(
            embed_dim=backbone_dim,
            num_heads=backbone_num_heads,
            qkv_bias=True,
            qk_norm=False,
            attn_drop=0.0,
            proj_drop=0.0,
            block_prefix=f"{name}_block_{i}",
            name=f"{name}_block_{i}_attn",
        )(x_norm)
        x_attn = DepthAnythingV1LayerScale(
            layer_scale_init=1.0, name=f"{name}_block_{i}_ls1"
        )(x_attn)
        x = layers.Add(name=f"{name}_block_{i}_add1")([x, x_attn])

        y_norm = layers.LayerNormalization(
            epsilon=1e-6, axis=-1, name=f"{name}_block_{i}_ln2"
        )(x)
        y_mlp = layers.Dense(
            int(backbone_dim * mlp_ratio), name=f"{name}_block_{i}_mlp_fc1"
        )(y_norm)
        y_mlp = layers.Activation("gelu", name=f"{name}_block_{i}_gelu")(y_mlp)
        y_mlp = layers.Dense(backbone_dim, name=f"{name}_block_{i}_mlp_fc2")(y_mlp)
        y_mlp = DepthAnythingV1LayerScale(
            layer_scale_init=1.0, name=f"{name}_block_{i}_ls2"
        )(y_mlp)
        x = layers.Add(name=f"{name}_block_{i}_add2")([x, y_mlp])

        if (i + 1) in out_indices:
            all_features.append(x)

    backbone_ln = layers.LayerNormalization(
        epsilon=1e-6, axis=-1, name=f"{name}_layernorm"
    )
    backbone_features = []
    for i, feat in enumerate(all_features):
        feat = backbone_ln(feat)
        feat = layers.Lambda(lambda v: v[:, 1:], name=f"{name}_strip_cls_{i}")(feat)
        feat = layers.Reshape(
            (patch_h, patch_w, backbone_dim), name=f"{name}_reshape_{i}"
        )(feat)
        if data_format == "channels_first":
            feat = layers.Permute((3, 1, 2), name=f"{name}_permute_{i}")(feat)
        backbone_features.append(feat)
    return backbone_features


def depth_anything_v1_neck(
    backbone_features,
    reassemble_factors,
    neck_hidden_sizes,
    fusion_hidden_size,
    patch_h,
    patch_w,
    data_format,
    name="neck",
):
    """Functional DPT-style neck: reassemble + project + bottom-up fusion.

    Applies the DPT neck to the 4 intermediate DINOv2 feature maps and
    returns a single fused tensor. The neck is split into three stages:

    1. **Reassemble.** Each feature map is projected to its
       stage-specific channel count with a 1x1 conv and then up- or
       down-sampled by ``reassemble_factors[i]``. Factors greater than
       one use a ``Conv2DTranspose`` with ``kernel_size == factor``;
       factors less than one use a strided 3x3 conv; factor ``1`` is a
       no-op.
    2. **Project.** A 3x3 conv (no bias) brings every reassembled
       feature to the common ``fusion_hidden_size`` channel count.
    3. **Fusion.** The projected features are walked bottom-up through
       four ``depth_anything_v1_fusion_block`` stages. Each stage
       upsamples to the next coarser pyramid level except for the
       final stage, which upsamples to ``2x`` the coarsest level so
       the head can recover the original image resolution with a
       single additional upsample.

    Args:
        backbone_features: List of 4 spatial feature maps returned by
            ``depth_anything_v1_dino_backbone`` in the same order as
            ``out_indices``.
        reassemble_factors: List of 4 up/down-sampling factors used by
            the reassemble stage. Depth Anything V1 uses
            ``[4, 2, 1, 0.5]``.
        neck_hidden_sizes: List of 4 per-stage channel counts used by
            the reassemble 1x1 projections.
        fusion_hidden_size: Integer, shared channel count used after
            the project stage and throughout fusion.
        patch_h: Integer, patch-grid height from the backbone.
        patch_w: Integer, patch-grid width from the backbone.
        data_format: Either ``"channels_last"`` or ``"channels_first"``.
        name: String prefix used to name every layer created by this
            neck. Defaults to ``"neck"``.

    Returns:
        The fused tensor returned by the final fusion stage, ready to
        feed ``depth_anything_v1_head``.
    """
    reassembled = []
    for i, (feat, factor, out_ch) in enumerate(
        zip(backbone_features, reassemble_factors, neck_hidden_sizes)
    ):
        feat = layers.Conv2D(
            out_ch,
            1,
            padding="valid",
            data_format=data_format,
            name=f"{name}_reassemble_{i}_projection",
        )(feat)
        if factor > 1:
            feat = layers.Conv2DTranspose(
                out_ch,
                kernel_size=int(factor),
                strides=int(factor),
                padding="valid",
                data_format=data_format,
                name=f"{name}_reassemble_{i}_resize",
            )(feat)
        elif factor < 1:
            stride = int(1 / factor)
            feat = layers.Conv2D(
                out_ch,
                kernel_size=3,
                strides=stride,
                padding="same",
                data_format=data_format,
                name=f"{name}_reassemble_{i}_resize",
            )(feat)
        reassembled.append(feat)

    projected = []
    for i, feat in enumerate(reassembled):
        feat = layers.Conv2D(
            fusion_hidden_size,
            3,
            padding="same",
            use_bias=False,
            data_format=data_format,
            name=f"{name}_conv_{i}",
        )(feat)
        projected.append(feat)

    reassembled_sizes = []
    for factor in reassemble_factors:
        if factor > 1:
            reassembled_sizes.append((patch_h * int(factor), patch_w * int(factor)))
        elif factor < 1:
            stride = int(1 / factor)
            reassembled_sizes.append((-(-patch_h // stride), -(-patch_w // stride)))
        else:
            reassembled_sizes.append((patch_h, patch_w))
    reversed_sizes = reassembled_sizes[::-1]
    features_reversed = projected[::-1]

    fused = None
    for idx in range(4):
        if idx < 3:
            target_h, target_w = reversed_sizes[idx + 1]
        else:
            target_h = reversed_sizes[idx][0] * 2
            target_w = reversed_sizes[idx][1] * 2

        fused = depth_anything_v1_fusion_block(
            features_reversed[idx] if fused is None else fused,
            None if fused is None else features_reversed[idx],
            target_h,
            target_w,
            fusion_hidden_size,
            data_format,
            name=f"{name}_fusion_{idx}",
        )
    return fused


def depth_anything_v1_head(
    fused,
    height,
    width,
    fusion_hidden_size,
    head_hidden_size,
    depth_estimation_type,
    max_depth,
    data_format,
    name="head",
):
    """Functional depth-estimation head used by Depth Anything V1.

    Consumes the fused tensor from ``depth_anything_v1_neck`` and
    produces the final single-channel depth map at the original image
    resolution. The head is three convolutions with an aligned-corners
    bilinear upsample between the first and second conv:

    ``3x3 conv (fusion_hidden_size // 2) -> upsample to (height, width)
    -> 3x3 conv (head_hidden_size) -> ReLU -> 1x1 conv (1) -> output``

    The final activation depends on the estimation type:

    - ``"relative"``: a final ``ReLU``, so outputs are non-negative
      disparity-style depth values.
    - ``"metric"``: a final ``sigmoid`` scaled by ``max_depth``, so
      outputs are bounded metric depth values in
      ``[0, max_depth]`` meters.

    Args:
        fused: Fused tensor returned by ``depth_anything_v1_neck``.
        height: Integer, target output height (input image height).
        width: Integer, target output width (input image width).
        fusion_hidden_size: Integer, channel count of the fused tensor.
            The first conv projects to ``fusion_hidden_size // 2``.
        head_hidden_size: Integer, channel count used by the second
            conv. Depth Anything V1 uses ``32``.
        depth_estimation_type: Either ``"relative"`` or ``"metric"``.
        max_depth: Float, metric-depth scale factor applied only when
            ``depth_estimation_type == "metric"``.
        data_format: Either ``"channels_last"`` or ``"channels_first"``.
        name: String prefix used to name every layer created by this
            head. Defaults to ``"head"``.

    Returns:
        Predicted depth tensor shaped ``(batch, height, width, 1)`` for
        ``channels_last`` or ``(batch, 1, height, width)`` for
        ``channels_first``.
    """
    x = layers.Conv2D(
        fusion_hidden_size // 2,
        3,
        padding="same",
        data_format=data_format,
        name=f"{name}_conv1",
    )(fused)
    x = DepthAnythingV1Resize(height, width, data_format, name=f"{name}_upsample")(x)
    x = layers.Conv2D(
        head_hidden_size,
        3,
        padding="same",
        data_format=data_format,
        name=f"{name}_conv2",
    )(x)
    x = layers.Activation("relu", name=f"{name}_act1")(x)
    x = layers.Conv2D(
        1, 1, padding="valid", data_format=data_format, name=f"{name}_conv3"
    )(x)

    if depth_estimation_type == "metric":
        x = layers.Activation("sigmoid", name=f"{name}_act2")(x)
        x = layers.Rescaling(max_depth, name=f"{name}_scale_depth")(x)
    else:
        x = layers.Activation("relu", name=f"{name}_act2")(x)
    return x


PATCH_SIZE = 14
IMAGE_SIZE = 518
MLP_RATIO = 4.0
HEAD_HIDDEN_SIZE = 32


def depth_anything_v1_functional(
    inputs,
    backbone_dim,
    backbone_depth,
    backbone_num_heads,
    out_indices,
    neck_hidden_sizes,
    fusion_hidden_size,
    reassemble_factors,
    height,
    width,
):
    """Build the Depth Anything V1 backbone + neck graph (no depth head).

    Top-level orchestrator wiring the two stages that are shared
    between the relative and metric variants:

    1. :func:`depth_anything_v1_dino_backbone`, DINOv2 ViT backbone
       producing multi-scale patch token sequences at the layers
       listed in ``out_indices``.
    2. :func:`depth_anything_v1_neck`, DPT reassemble + project +
       bottom-up fusion producing a single fused feature map at the
       finest pyramid level.

    The depth-prediction head is intentionally not built here: it is
    added by :class:`DepthAnythingV1DepthEstimation`, which composes
    :class:`DepthAnythingV1Model` around this graph. This separation
    means fine-tuning hyperparameters (``depth_estimation_type``,
    ``max_depth``) live on the head class, not the backbone class.

    Args:
        inputs: Keras input tensor.
        backbone_dim: ViT backbone hidden dimension.
        backbone_depth: Number of ViT transformer blocks.
        backbone_num_heads: Attention heads per block.
        out_indices: 1-indexed block IDs whose outputs feed the neck.
        neck_hidden_sizes: Per-stage channel counts in the reassemble
            projections.
        fusion_hidden_size: Channel count used across the project
            stage and fusion blocks.
        reassemble_factors: Up/down-sampling factors per stage in the
            reassemble step.
        height: Input image height in pixels.
        width: Input image width in pixels.

    Returns:
        Fused feature map at the finest pyramid level.
    """
    data_format = keras.config.image_data_format()
    patch_h = height // PATCH_SIZE
    patch_w = width // PATCH_SIZE

    backbone_features = depth_anything_v1_dino_backbone(
        inputs,
        backbone_dim=backbone_dim,
        backbone_depth=backbone_depth,
        backbone_num_heads=backbone_num_heads,
        out_indices=out_indices,
        patch_size=PATCH_SIZE,
        patch_h=patch_h,
        patch_w=patch_w,
        mlp_ratio=MLP_RATIO,
        data_format=data_format,
        name="backbone",
    )

    fused = depth_anything_v1_neck(
        backbone_features,
        reassemble_factors=reassemble_factors,
        neck_hidden_sizes=neck_hidden_sizes,
        fusion_hidden_size=fusion_hidden_size,
        patch_h=patch_h,
        patch_w=patch_w,
        data_format=data_format,
        name="neck",
    )

    return fused


@keras.saving.register_keras_serializable(package="zeromodels")
class DepthAnythingV1Model(BaseModel):
    """Depth Anything V1 backbone + DPT neck (no depth-prediction head).

    Wraps the functional graph built by
    :func:`depth_anything_v1_functional`: a DINOv2 ViT backbone and a
    DPT reassemble + fusion neck. Outputs the fused feature map at
    the finest pyramid level: the same tensor that
    :class:`DepthAnythingV1DepthEstimation` then feeds into the depth head.

    Use this class when you want the backbone+neck features (e.g. for
    a custom head). Use :class:`DepthAnythingV1DepthEstimation` when you want
    the full monocular depth estimator.

    Reference:
        - `Depth Anything: Unleashing the Power of Large-Scale
          Unlabeled Data <https://arxiv.org/abs/2401.10891>`_
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "depth_anything"

    @classmethod
    def config_from_hf(cls, hf_config):
        bb = hf_config["backbone_config"]
        out_indices = list(bb["out_indices"])
        return {
            "backbone_dim": bb["hidden_size"],
            "backbone_depth": bb.get("num_hidden_layers", max(out_indices)),
            "backbone_num_heads": bb["num_attention_heads"],
            "out_indices": out_indices,
            "neck_hidden_sizes": list(hf_config["neck_hidden_sizes"]),
            "fusion_hidden_size": hf_config["fusion_hidden_size"],
            "reassemble_factors": list(hf_config["reassemble_factors"]),
        }

    @classmethod
    def from_hf(cls, hf_id, load_weights=True, skip_mismatch=False, **kwargs):
        model = super().from_hf(hf_id, load_weights=False, **kwargs)
        if load_weights:
            src = DepthAnythingV1DepthEstimation.from_hf(
                hf_id, skip_mismatch=skip_mismatch
            )
            unmatched = copy_weights_by_path_suffix(src, model)
            if unmatched and not skip_mismatch:
                raise ValueError(
                    f"{cls.__name__}.from_hf: {len(unmatched)} weight(s) not "
                    f"matched from the {type(src).__name__} checkpoint: "
                    f"{unmatched[:5]}"
                )
            del src
        return model

    def __init__(
        self,
        backbone_dim=384,
        backbone_depth=12,
        backbone_num_heads=6,
        out_indices=None,
        neck_hidden_sizes=None,
        fusion_hidden_size=64,
        reassemble_factors=None,
        image_size=IMAGE_SIZE,
        input_tensor=None,
        name="DepthAnythingV1Model",
        **kwargs,
    ):
        if out_indices is None:
            out_indices = [9, 10, 11, 12]
        if neck_hidden_sizes is None:
            neck_hidden_sizes = [48, 96, 192, 384]
        if reassemble_factors is None:
            reassemble_factors = [4, 2, 1, 0.5]

        data_format = keras.config.image_data_format()
        image_size = standardize_input_shape(image_size, data_format)

        if input_tensor is not None:
            if not keras.utils.is_keras_tensor(input_tensor):
                pixel_values = layers.Input(
                    tensor=input_tensor, shape=image_size, name="pixel_values"
                )
            else:
                pixel_values = input_tensor
        else:
            pixel_values = layers.Input(shape=image_size, name="pixel_values")

        if data_format == "channels_first":
            height, width = image_size[1], image_size[2]
        else:
            height, width = image_size[0], image_size[1]

        fused = depth_anything_v1_functional(
            pixel_values,
            backbone_dim=backbone_dim,
            backbone_depth=backbone_depth,
            backbone_num_heads=backbone_num_heads,
            out_indices=out_indices,
            neck_hidden_sizes=neck_hidden_sizes,
            fusion_hidden_size=fusion_hidden_size,
            reassemble_factors=reassemble_factors,
            height=height,
            width=width,
        )

        super().__init__(inputs=pixel_values, outputs=fused, name=name, **kwargs)

        self.backbone_dim = backbone_dim
        self.backbone_depth = backbone_depth
        self.backbone_num_heads = backbone_num_heads
        self.out_indices = list(out_indices)
        self.neck_hidden_sizes = list(neck_hidden_sizes)
        self.fusion_hidden_size = fusion_hidden_size
        self.reassemble_factors = list(reassemble_factors)
        self.image_size = image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone_dim": self.backbone_dim,
                "backbone_depth": self.backbone_depth,
                "backbone_num_heads": self.backbone_num_heads,
                "out_indices": self.out_indices,
                "neck_hidden_sizes": self.neck_hidden_sizes,
                "fusion_hidden_size": self.fusion_hidden_size,
                "reassemble_factors": self.reassemble_factors,
                "image_size": self.image_size,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class DepthAnythingV1DepthEstimation(BaseModel):
    """Depth Anything V1 full monocular depth estimator (backbone + neck + head).

    Composes :class:`DepthAnythingV1Model` and adds the DPT depth
    head. Maps a preprocessed image to a single-channel depth map at
    the original input resolution.

    Two head modes are supported via ``depth_estimation_type``:

    - ``"relative"``: final ReLU, non-negative disparity-style depth.
      Used by the official Depth Anything V1 release.
    - ``"metric"``: final ``sigmoid * max_depth``, bounded metric
      depth in ``[0, max_depth]`` meters. Used by fine-tuned metric
      variants (e.g. Depth Anything V2 indoor/outdoor checkpoints).

    Reference:
        - `Depth Anything: Unleashing the Power of Large-Scale
          Unlabeled Data <https://arxiv.org/abs/2401.10891>`_

    Args:
        backbone_dim: ViT backbone hidden dimension.
        backbone_depth: Number of ViT transformer blocks.
        backbone_num_heads: Attention heads per block.
        out_indices: 1-indexed block IDs whose outputs feed the neck.
        neck_hidden_sizes: Per-stage channel counts in the reassemble
            projections.
        fusion_hidden_size: Channel count shared across project,
            fusion, and head.
        reassemble_factors: Up/down-sampling factors per stage in the
            reassemble step.
        depth_estimation_type: ``"relative"`` or ``"metric"`` (fine-tune
            switch: the metric variants use ``"metric"``).
        max_depth: Metric-depth scale factor; only used when
            ``depth_estimation_type == "metric"``.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `518`.
        input_tensor: Optional pre-existing Keras input tensor.
        name: Model name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = DepthAnythingV1Config
    HF_MODEL_TYPE = "depth_anything"

    @classmethod
    def config_from_hf(cls, hf_config):
        bb = hf_config["backbone_config"]
        out_indices = list(bb["out_indices"])
        return {
            "backbone_dim": bb["hidden_size"],
            "backbone_depth": bb.get("num_hidden_layers", max(out_indices)),
            "backbone_num_heads": bb["num_attention_heads"],
            "out_indices": out_indices,
            "neck_hidden_sizes": list(hf_config["neck_hidden_sizes"]),
            "fusion_hidden_size": hf_config["fusion_hidden_size"],
            "reassemble_factors": list(hf_config["reassemble_factors"]),
            "depth_estimation_type": hf_config.get("depth_estimation_type", "relative"),
            "max_depth": float(hf_config.get("max_depth", 1.0)),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from zeromodels.models.depth_anything_v1.convert_depth_anything_v1_hf_to_keras import (
            transfer_depth_anything_weights,
        )

        transfer_depth_anything_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        backbone_dim=384,
        backbone_depth=12,
        backbone_num_heads=6,
        out_indices=None,
        neck_hidden_sizes=None,
        fusion_hidden_size=64,
        reassemble_factors=None,
        depth_estimation_type="relative",
        max_depth=1.0,
        image_size=IMAGE_SIZE,
        input_tensor=None,
        name="DepthAnythingV1DepthEstimation",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        image_size = standardize_input_shape(image_size, data_format)

        base = DepthAnythingV1Model(
            backbone_dim=backbone_dim,
            backbone_depth=backbone_depth,
            backbone_num_heads=backbone_num_heads,
            out_indices=out_indices,
            neck_hidden_sizes=neck_hidden_sizes,
            fusion_hidden_size=fusion_hidden_size,
            reassemble_factors=reassemble_factors,
            image_size=image_size,
            input_tensor=input_tensor,
            name=f"{name}_model",
        )
        fused = base.output

        if data_format == "channels_first":
            height, width = image_size[1], image_size[2]
        else:
            height, width = image_size[0], image_size[1]

        predicted_depth = depth_anything_v1_head(
            fused,
            height=height,
            width=width,
            fusion_hidden_size=fusion_hidden_size,
            head_hidden_size=HEAD_HIDDEN_SIZE,
            depth_estimation_type=depth_estimation_type,
            max_depth=max_depth,
            data_format=data_format,
            name="head",
        )

        super().__init__(
            inputs=base.input, outputs=predicted_depth, name=name, **kwargs
        )

        self.backbone_dim = backbone_dim
        self.backbone_depth = backbone_depth
        self.backbone_num_heads = backbone_num_heads
        self.out_indices = list(base.out_indices)
        self.neck_hidden_sizes = list(base.neck_hidden_sizes)
        self.fusion_hidden_size = fusion_hidden_size
        self.reassemble_factors = list(base.reassemble_factors)
        self.depth_estimation_type = depth_estimation_type
        self.max_depth = max_depth
        self.image_size = base.image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone_dim": self.backbone_dim,
                "backbone_depth": self.backbone_depth,
                "backbone_num_heads": self.backbone_num_heads,
                "out_indices": self.out_indices,
                "neck_hidden_sizes": self.neck_hidden_sizes,
                "fusion_hidden_size": self.fusion_hidden_size,
                "reassemble_factors": self.reassemble_factors,
                "depth_estimation_type": self.depth_estimation_type,
                "max_depth": self.max_depth,
                "image_size": self.image_size,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class DepthAnythingV1LayerScale(layers.Layer):
    """Learnable per-channel scale (x * gamma), gamma initialized to layer_scale_init."""

    def __init__(self, layer_scale_init, **kwargs):
        super().__init__(**kwargs)
        self.layer_scale_init = layer_scale_init

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer=keras.initializers.Constant(self.layer_scale_init),
            trainable=True,
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update({"layer_scale_init": self.layer_scale_init})
        return config
