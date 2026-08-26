import keras
from keras import layers, ops, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.utils import standardize_input_shape
from zeromodels.utils.image_util import normalize_image_for_classify_models

from .nextvit_config import NextViTConfig
from .nextvit_layers import NextViTEfficientAttention

# The backbone (NextViTModel) and classifier (NextViTImageClassify) share the
# variant's weights repo, whose kf_config.json declares NextViTImageClassify.
NEXTVIT_HUB_SIBLINGS = frozenset({"NextViTModel", "NextViTImageClassify"})


def nextvit_conv_attention(x, out_chs, head_dim, channels_axis, data_format, prefix=""):
    """Multi-Head Convolutional Attention (MHCA) branch for NextViT.

    Args:
        x: Input feature map.
        out_chs: Output channel count.
        head_dim: Per-head channel dimension; ``out_chs // head_dim`` becomes the
            number of grouped-conv groups.
        channels_axis: Channel axis index.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        prefix: String prefix for layer names.

    Returns:
        Output tensor with ``out_chs`` channels and the same spatial size as ``x``.
    """
    num_groups = out_chs // head_dim
    out = layers.Conv2D(
        out_chs,
        3,
        strides=1,
        padding="same",
        groups=num_groups,
        use_bias=False,
        data_format=data_format,
        name=prefix + "mhca_group_conv3x3",
    )(x)
    out = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.9,
        name=prefix + "mhca_norm",
    )(out)
    out = layers.ReLU()(out)
    out = layers.Conv2D(
        out_chs,
        1,
        use_bias=False,
        data_format=data_format,
        name=prefix + "mhca_projection",
    )(out)
    return out


def make_divisible(v, divisor, min_value=None):
    """Snap a (possibly scaled) channel count to a multiple of ``divisor``.

    Args:
        v: Channel count to round.
        divisor: Multiple to snap to.
        min_value: Floor for the rounded value; defaults to ``divisor`` when ``None``.

    Returns:
        Integer channel count that is a multiple of ``divisor`` and at least
        ``min_value``.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def calculate_drop_path_rates(drop_path_rate, depths):
    """Build a per-block linear drop-path schedule.

    Args:
        drop_path_rate: Maximum drop-path probability applied to the last block.
        depths: Iterable of block counts per stage.

    Returns:
        List of per-stage lists, where entry ``[i][j]`` is the drop-path rate for
        the ``j``-th block of stage ``i`` (linearly ramped from 0 to
        ``drop_path_rate``).
    """
    total_depth = sum(depths)
    rates = []
    idx = 0
    for d in depths:
        stage_rates = []
        for i in range(d):
            stage_rates.append(
                drop_path_rate * idx / (total_depth - 1) if total_depth > 1 else 0.0
            )
            idx += 1
        rates.append(stage_rates)
    return rates


def get_stage_out_chs(depths):
    """Compute the per-block output channel count for each NextViT stage.

    Args:
        depths: Iterable of block counts per stage (length 4).

    Returns:
        List of four per-stage channel lists matching the NextViT architecture
        spec (last block in each stage may use an enlarged channel count).
    """
    return [
        [96] * depths[0],
        [192] * (depths[1] - 1) + [256],
        [384, 384, 384, 384, 512] * (depths[2] // 5),
        [768] * (depths[3] - 1) + [1024],
    ]


def get_stage_block_types(depths):
    """Compute the per-block type ("conv" or "transformer") for each NextViT stage.

    Args:
        depths: Iterable of block counts per stage (length 4).

    Returns:
        List of four per-stage lists, where each entry is ``"conv"`` for a
        NextConvBlock or ``"transformer"`` for a NextTransformerBlock.
    """
    return [
        ["conv"] * depths[0],
        ["conv"] * (depths[1] - 1) + ["transformer"],
        ["conv", "conv", "conv", "conv", "transformer"] * (depths[2] // 5),
        ["conv"] * (depths[3] - 1) + ["transformer"],
    ]


def conv_mlp(
    x, in_features, hidden_features, out_features, channels_axis, data_format, prefix=""
):
    """ConvMlp block: two 1x1 convolutions with ReLU activation in between.

    Args:
        x: Input feature map.
        in_features: Unused; kept for parity with the timm signature.
        hidden_features: Channel count of the hidden 1x1 conv.
        out_features: Output channel count.
        channels_axis: Channel axis index (unused; reserved for future use).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        prefix: String prefix for layer names.

    Returns:
        Output tensor with ``out_features`` channels and the same spatial size
        as ``x``.
    """
    x = layers.Conv2D(
        hidden_features,
        1,
        use_bias=True,
        data_format=data_format,
        name=prefix + "mlp_fc1",
    )(x)
    x = layers.Activation("relu", name=prefix + "mlp_act")(x)
    x = layers.Conv2D(
        out_features,
        1,
        use_bias=True,
        data_format=data_format,
        name=prefix + "mlp_fc2",
    )(x)
    return x


def patch_embed_block(
    x, in_chs, out_chs, use_pool, channels_axis, data_format, prefix=""
):
    """Patch embedding: optional 2x average pooling + 1x1 projection + BN.

    The 1x1 projection (and its BN) only runs when the spatial size is being
    reduced or when ``in_chs != out_chs``.

    Args:
        x: Input feature map.
        in_chs: Input channel count.
        out_chs: Output channel count.
        use_pool: If ``True``, apply a 2x2 average pool with stride 2 before the
            projection.
        channels_axis: Channel axis index.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        prefix: String prefix for layer names.

    Returns:
        Output tensor with ``out_chs`` channels.
    """
    if use_pool:
        x = layers.AveragePooling2D(
            pool_size=2,
            strides=2,
            padding="valid",
            data_format=data_format,
            name=prefix + "patch_embed_pool",
        )(x)
    if use_pool or in_chs != out_chs:
        x = layers.Conv2D(
            out_chs,
            1,
            use_bias=False,
            data_format=data_format,
            name=prefix + "patch_embed_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=1e-5,
            momentum=0.9,
            name=prefix + "patch_embed_norm",
        )(x)
    return x


def next_conv_block(
    x,
    in_chs,
    out_chs,
    stride,
    drop_path_rate,
    head_dim,
    mlp_ratio,
    channels_axis,
    data_format,
    prefix="",
):
    """NextConvBlock: patch embedding + MHCA branch + ConvMLP branch.

    Args:
        x: Input feature map.
        in_chs: Input channel count.
        out_chs: Output channel count.
        stride: Spatial stride (``1`` or ``2``); ``2`` triggers average pooling
            in patch embedding.
        drop_path_rate: Drop-path rate for this block (currently unused).
        head_dim: Per-head dimension for the MHCA grouped conv.
        mlp_ratio: Hidden-dim expansion ratio for the ConvMLP.
        channels_axis: Channel axis index.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        prefix: String prefix for layer names.

    Returns:
        Output tensor with ``out_chs`` channels and spatial size reduced by
        ``stride``.
    """
    use_pool = stride == 2
    x = patch_embed_block(
        x, in_chs, out_chs, use_pool, channels_axis, data_format, prefix=prefix
    )
    mhca_out = nextvit_conv_attention(
        x, out_chs, head_dim, channels_axis, data_format, prefix=prefix
    )
    x = layers.Add()([x, mhca_out])

    residual = x
    out = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.9,
        name=prefix + "norm",
    )(x)
    out = conv_mlp(
        out,
        out_chs,
        int(out_chs * mlp_ratio),
        out_chs,
        channels_axis,
        data_format,
        prefix=prefix,
    )
    x = layers.Add()([residual, out])
    return x


def next_transformer_block(
    x,
    in_chs,
    out_chs,
    stride,
    drop_path_rate,
    head_dim,
    sr_ratio,
    mix_block_ratio,
    mlp_ratio,
    channels_axis,
    data_format,
    prefix="",
):
    """NextTransformerBlock: E-MHSA branch + MHCA branch, concatenated then MLP.

    Args:
        x: Input feature map.
        in_chs: Input channel count.
        out_chs: Output channel count (split between E-MHSA and MHCA branches).
        stride: Spatial stride (``1`` or ``2``); ``2`` triggers average pooling
            in patch embedding.
        drop_path_rate: Drop-path rate for this block (currently unused).
        head_dim: Per-head channel dimension for both attention branches.
        sr_ratio: Spatial-reduction ratio for E-MHSA.
        mix_block_ratio: Fraction of ``out_chs`` allocated to the E-MHSA branch
            (rest goes to MHCA).
        mlp_ratio: Hidden-dim expansion ratio for the trailing ConvMLP.
        channels_axis: Channel axis index.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        prefix: String prefix for layer names.

    Returns:
        Output tensor with ``out_chs`` channels and spatial size reduced by
        ``stride``.
    """
    mhsa_out_chs = make_divisible(int(out_chs * mix_block_ratio), 32)
    mhca_out_chs = out_chs - mhsa_out_chs

    use_pool = stride == 2
    x = patch_embed_block(
        x, in_chs, mhsa_out_chs, use_pool, channels_axis, data_format, prefix=prefix
    )

    out = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.9,
        name=prefix + "norm1",
    )(x)
    if data_format == "channels_first":
        out = layers.Permute((2, 3, 1), name=prefix + "to_seq_perm")(out)
    out = layers.Reshape((-1, mhsa_out_chs), name=prefix + "reshape_to_seq")(out)

    out = NextViTEfficientAttention(
        mhsa_out_chs,
        head_dim=head_dim,
        sr_ratio=sr_ratio,
        prefix=prefix,
        name=prefix + "e_mhsa",
    )(out)

    x_shape = ops.shape(x)
    if data_format == "channels_first":
        h_idx, w_idx = 2, 3
    else:
        h_idx, w_idx = 1, 2
    out = layers.Reshape(
        (x_shape[h_idx], x_shape[w_idx], mhsa_out_chs),
        name=prefix + "reshape_to_spatial",
    )(out)
    if data_format == "channels_first":
        out = layers.Permute((3, 1, 2), name=prefix + "from_seq_perm")(out)

    x = layers.Add()([x, out])

    proj_out = layers.Conv2D(
        mhca_out_chs,
        1,
        use_bias=False,
        data_format=data_format,
        name=prefix + "projection_conv",
    )(x)
    proj_out = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.9,
        name=prefix + "projection_norm",
    )(proj_out)

    mhca_out = nextvit_conv_attention(
        proj_out,
        mhca_out_chs,
        head_dim,
        channels_axis,
        data_format,
        prefix=prefix,
    )
    proj_out = layers.Add()([proj_out, mhca_out])

    x = layers.Concatenate(axis=channels_axis)([x, proj_out])

    residual = x
    out = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.9,
        name=prefix + "norm2",
    )(x)
    out = conv_mlp(
        out,
        out_chs,
        int(out_chs * mlp_ratio),
        out_chs,
        channels_axis,
        data_format,
        prefix=prefix,
    )
    x = layers.Add()([residual, out])
    return x


def nextvit_backbone_feature(
    inputs,
    *,
    depths,
    stem_chs,
    head_dim,
    mix_block_ratio,
    sr_ratios,
    drop_path_rate,
    data_format,
    channels_axis,
    return_stages=False,
):
    """NextViT stem + 4 stages + final BN.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` for channels-last
            or ``(B, C, H, W)`` for channels-first.
        depths: Number of blocks per stage (length 4).
        stem_chs: Stem channel widths (length 3); a 4-conv stem reaches
            ``stem_chs[-1]``.
        head_dim: Per-head channel dimension shared across all attention modules.
        mix_block_ratio: Fraction of channels allocated to E-MHSA inside
            transformer blocks.
        sr_ratios: Per-stage spatial-reduction ratios for E-MHSA (length 4).
        drop_path_rate: Maximum stochastic-depth rate (linearly ramped per block).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Channel axis index.
        return_stages: If ``True``, return a list of the 4 per-stage feature
            maps (collected at the end of each stage, BEFORE the trailing
            BatchNorm). Defaults to ``False``.

    Returns:
        Final stage feature map (after BatchNorm) with the last stage's
        channel count at spatial resolution ``H/32`` when
        ``return_stages=False``. When ``return_stages=True``, a list of 4
        per-stage feature maps.
    """
    x = inputs
    stem_configs = [
        (3, stem_chs[0], 2),
        (stem_chs[0], stem_chs[1], 1),
        (stem_chs[1], stem_chs[2], 1),
        (stem_chs[2], stem_chs[2], 2),
    ]
    for i, (in_c, out_c, stride) in enumerate(stem_configs):
        if stride == 2:
            x = layers.ZeroPadding2D(
                padding=1,
                data_format=data_format,
                name=f"stem_{i}_pad",
            )(x)
        x = layers.Conv2D(
            out_c,
            3,
            strides=stride,
            padding="valid" if stride == 2 else "same",
            use_bias=False,
            data_format=data_format,
            name=f"stem_{i}_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=1e-5,
            momentum=0.9,
            name=f"stem_{i}_norm",
        )(x)
        x = layers.Activation("relu", name=f"stem_{i}_act")(x)

    stage_out_chs = get_stage_out_chs(depths)
    stage_block_types = get_stage_block_types(depths)
    dpr = calculate_drop_path_rates(drop_path_rate, depths)
    strides = [1, 2, 2, 2]

    in_chs = stem_chs[-1]

    stages = []
    for stage_idx in range(4):
        block_chs = stage_out_chs[stage_idx]
        block_types = stage_block_types[stage_idx]

        for block_idx in range(depths[stage_idx]):
            stride = strides[stage_idx] if block_idx == 0 else 1
            out_chs = block_chs[block_idx]
            block_type = block_types[block_idx]
            dp_rate = dpr[stage_idx][block_idx]
            prefix = f"stages_{stage_idx}_blocks_{block_idx}_"

            if block_type == "conv":
                x = next_conv_block(
                    x,
                    in_chs,
                    out_chs,
                    stride,
                    dp_rate,
                    head_dim,
                    3.0,
                    channels_axis,
                    data_format,
                    prefix=prefix,
                )
            else:
                x = next_transformer_block(
                    x,
                    in_chs,
                    out_chs,
                    stride,
                    dp_rate,
                    head_dim,
                    sr_ratios[stage_idx],
                    mix_block_ratio,
                    2.0,
                    channels_axis,
                    data_format,
                    prefix=prefix,
                )
            in_chs = out_chs

        stages.append(x)

    if return_stages:
        return stages

    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.9,
        name="norm",
    )(x)

    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class NextViTModel(BaseModel):
    """Instantiates the NextViT backbone.

    NextViT is a next-generation hybrid CNN-Transformer backbone that
    alternates two purpose-built block types across 4 hierarchical
    stages: Next Convolution Blocks (NCB), efficient multi-head
    convolutional attention paired with a Conv-MLP, and Next
    Transformer Blocks (NTB): efficient multi-head self-attention with
    spatial reduction (E-MHSA, linear in the number of tokens) fused
    with a parallel MHCA branch. The four stages follow the standard
    pyramid downsampling schedule and end with a trailing BatchNorm.
    NCB blocks dominate the early/cheap stages while NTB blocks appear
    at the end of each stage to inject global context.

    Output is the last layer output before the classifier head:
    the final stage feature map ``(B, H, W, C)`` (channels-last) /
    ``(B, C, H, W)`` (channels-first) after the trailing BatchNorm,
    unpooled and head-free. :class:`NextViTImageClassify` composes this model
    and appends GAP + Dense.

    References:
    - [Next-ViT: Next Generation Vision Transformer for Efficient Deployment in Realistic Industrial Scenarios](https://arxiv.org/abs/2207.05501)

    Args:
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of the
            4 per-stage feature maps (collected at the end of each stage,
            BEFORE the trailing BatchNorm). Defaults to `False`.
        depths: Tuple of integers, number of blocks per stage (length 4).
            Defaults to `(3, 4, 10, 3)`.
        stem_chs: Tuple of integers, stem channel widths (length 3); a
            4-conv stem reaches ``stem_chs[-1]``. Defaults to `(64, 32, 64)`.
        head_dim: Integer, per-head channel dimension shared across all
            attention modules. Defaults to `32`.
        mix_block_ratio: Float, fraction of channels allocated to E-MHSA
            inside transformer blocks (rest goes to MHCA).
            Defaults to `0.75`.
        sr_ratios: Tuple of integers, per-stage spatial-reduction ratios
            for E-MHSA (length 4). Defaults to `(8, 4, 2, 1)`.
        drop_path_rate: Float, maximum stochastic-depth drop rate.
            Linearly ramped per block. Defaults to `0.1`.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `224`.
        include_normalization: Boolean, whether to prepend an
            image normalization at the start
            of the network. When True, input images should be in uint8
            format with values in `[0, 255]`. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to
            use. Must be one of: `'imagenet'` (default), `'inception'`,
            `'dpn'`, `'clip'`, `'zero_to_one'`, or `'minus_one_to_one'`.
            Only used when ``include_normalization=True``.
        input_tensor: Optional Keras tensor as input. Useful for
            connecting the model to other Keras components.
            Defaults to `None`.
        name: String, the name of the model. Defaults to `"NextViTModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = NextViTConfig
    HUB_REPO_SIBLINGS = NEXTVIT_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with NextViTImageClassify (which the
        # kf_config declares); build from kf_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = NextViTImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_nextvit_timm_to_keras import transfer_nextvit_weights

        transfer_nextvit_weights(keras_model, state_dict)

    def __init__(
        self,
        depths=(3, 4, 10, 3),
        stem_chs=(64, 32, 64),
        head_dim=32,
        mix_block_ratio=0.75,
        sr_ratios=(8, 4, 2, 1),
        drop_path_rate=0.1,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        as_backbone=False,
        name="NextViTModel",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "timm_id"):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else 1

        image_size = standardize_input_shape(image_size, data_format)

        if input_tensor is None:
            img_input = layers.Input(shape=image_size)
        elif not utils.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=image_size)
        else:
            img_input = input_tensor

        x = (
            normalize_image_for_classify_models(img_input, normalization_mode)
            if include_normalization
            else img_input
        )
        x = nextvit_backbone_feature(
            x,
            depths=depths,
            stem_chs=stem_chs,
            head_dim=head_dim,
            mix_block_ratio=mix_block_ratio,
            sr_ratios=sr_ratios,
            drop_path_rate=drop_path_rate,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.depths = list(depths)
        self.stem_chs = list(stem_chs)
        self.head_dim = head_dim
        self.mix_block_ratio = mix_block_ratio
        self.sr_ratios = list(sr_ratios)
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depths": self.depths,
                "stem_chs": self.stem_chs,
                "head_dim": self.head_dim,
                "mix_block_ratio": self.mix_block_ratio,
                "sr_ratios": self.sr_ratios,
                "drop_path_rate": self.drop_path_rate,
                "image_size": self.image_size,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_tensor": self.input_tensor,
                "as_backbone": self.as_backbone,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class NextViTImageClassify(BaseModel):
    """Instantiates the NextViT classifier.

    This classifier wraps a :class:`NextViTModel` backbone and attaches
    a GlobalAveragePooling2D + Dense head to produce ``num_classes``
    class logits. All architectural parameters are forwarded to the
    underlying :class:`NextViTModel`; only ``num_classes`` and
    ``classifier_activation`` are head-specific.

    References:
    - [Next-ViT: Next Generation Vision Transformer for Efficient Deployment in Realistic Industrial Scenarios](https://arxiv.org/abs/2207.05501)

    Args:
        depths: Tuple of integers, number of blocks per stage (length 4).
            Defaults to `(3, 4, 10, 3)`.
        stem_chs: Tuple of integers, stem channel widths (length 3); a
            4-conv stem reaches ``stem_chs[-1]``. Defaults to `(64, 32, 64)`.
        head_dim: Integer, per-head channel dimension shared across all
            attention modules. Defaults to `32`.
        mix_block_ratio: Float, fraction of channels allocated to E-MHSA
            inside transformer blocks (rest goes to MHCA).
            Defaults to `0.75`.
        sr_ratios: Tuple of integers, per-stage spatial-reduction ratios
            for E-MHSA (length 4). Defaults to `(8, 4, 2, 1)`.
        drop_path_rate: Float, maximum stochastic-depth drop rate.
            Linearly ramped per block. Defaults to `0.1`.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `224`.
        include_normalization: Boolean, whether to prepend an
            image normalization at the start
            of the network. When True, input images should be in uint8
            format with values in `[0, 255]`. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to
            use. Must be one of: `'imagenet'` (default), `'inception'`,
            `'dpn'`, `'clip'`, `'zero_to_one'`, or `'minus_one_to_one'`.
            Only used when ``include_normalization=True``.
        input_tensor: Optional Keras tensor as input. Useful for
            connecting the model to other Keras components.
            Defaults to `None`.
        num_classes: Integer, the number of output classes for
            classification. Defaults to `1000`.
        classifier_activation: String or callable, activation function
            for the final Dense layer. Use `"linear"` to return raw
            logits or `"softmax"` to return class probabilities.
            Defaults to `"linear"`.
        name: String, the name of the model. The internal backbone is
            named `f"{name}_backbone"`. Defaults to `"NextViTImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = NextViTConfig
    HUB_REPO_SIBLINGS = NEXTVIT_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_nextvit_timm_to_keras import transfer_nextvit_weights

        transfer_nextvit_weights(keras_model, state_dict)

    def __init__(
        self,
        depths=(3, 4, 10, 3),
        stem_chs=(64, 32, 64),
        head_dim=32,
        mix_block_ratio=0.75,
        sr_ratios=(8, 4, 2, 1),
        drop_path_rate=0.1,
        image_size=224,
        include_normalization=True,
        normalization_mode="imagenet",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="NextViTImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = NextViTModel(
            depths=depths,
            stem_chs=stem_chs,
            head_dim=head_dim,
            mix_block_ratio=mix_block_ratio,
            sr_ratios=sr_ratios,
            drop_path_rate=drop_path_rate,
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.GlobalAveragePooling2D(
            data_format=data_format,
            name="head_global_pool",
        )(backbone.output)
        out = layers.Dense(
            num_classes,
            activation=classifier_activation,
            name="head_fc",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.depths = list(depths)
        self.stem_chs = list(stem_chs)
        self.head_dim = head_dim
        self.mix_block_ratio = mix_block_ratio
        self.sr_ratios = list(sr_ratios)
        self.drop_path_rate = drop_path_rate
        self.image_size = backbone.image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depths": self.depths,
                "stem_chs": self.stem_chs,
                "head_dim": self.head_dim,
                "mix_block_ratio": self.mix_block_ratio,
                "sr_ratios": self.sr_ratios,
                "drop_path_rate": self.drop_path_rate,
                "image_size": self.image_size,
                "include_normalization": self.include_normalization,
                "normalization_mode": self.normalization_mode,
                "input_tensor": self.input_tensor,
                "num_classes": self.num_classes,
                "classifier_activation": self.classifier_activation,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
