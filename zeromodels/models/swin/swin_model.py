import keras
from keras import layers, ops, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.models.swin.swin_layers import (
    SwinRollLayer,
    SwinStochasticDepth,
    SwinWindowAttention,
    SwinWindowPartition,
)
from zeromodels.utils import standardize_input_shape

from .swin_config import SwinConfig

# The backbone (SwinModel) and classifier (SwinImageClassify) share the variant's
# weights repo, whose zm_config.json declares SwinImageClassify.
SWIN_HUB_SIBLINGS = frozenset({"SwinModel", "SwinImageClassify"})


def spatial_layer_norm(x, data_format, epsilon=1.001e-5, name=None):
    """LayerNorm over the channel axis for spatial feature maps.

    For channels_first inputs, permutes to NHWC, normalizes on axis=-1,
    then permutes back, so the normalization always runs on the channel
    dimension regardless of data format.

    Args:
        x: Input feature-map tensor in either channels-last or channels-first
            layout.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        epsilon: Small constant added to the LayerNorm denominator for
            numerical stability.
        name: Optional layer-name prefix.

    Returns:
        Normalized tensor with the same shape and layout as ``x``.
    """
    if data_format == "channels_first":
        x = layers.Permute((2, 3, 1), name=f"{name}_to_cl" if name else None)(x)
    x = layers.LayerNormalization(axis=-1, epsilon=epsilon, name=name)(x)
    if data_format == "channels_first":
        x = layers.Permute((3, 1, 2), name=f"{name}_to_cf" if name else None)(x)
    return x


def mlp_block(inputs, dropout=0.0, name="mlp"):
    """Two-layer GELU MLP with 4x expansion ratio (channels-last input).

    Args:
        inputs: Input tensor of shape ``(..., C)`` in channels-last layout.
        dropout: Dropout rate applied after each Dense layer.
        name: Layer-name prefix for the contained Dense/Dropout layers.

    Returns:
        Output tensor with the same shape as ``inputs``.
    """
    channels = inputs.shape[-1]
    x = layers.Dense(int(channels * 4.0), name=f"{name}_dense_1")(inputs)
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(dropout, name=f"{name}_dropout_1")(x)
    x = layers.Dense(channels, name=f"{name}_dense_2")(x)
    x = layers.Dropout(dropout, name=f"{name}_dropout_2")(x)
    return x


def swin_block(
    inputs,
    shift_size,
    window_size,
    relative_index,
    attention_mask,
    num_heads,
    bias_table_window_size,
    channels_axis,
    data_format,
    dropout_rate=0.0,
    drop_path_rate=0.0,
    name="swin_block",
):
    """Swin Transformer block with (shifted) window self-attention and MLP.

    Runs pre-norm windowed attention with optional cyclic shift, then a
    pre-norm MLP, both wrapped in residual connections with stochastic
    depth.

    Args:
        inputs: Input feature map of shape ``(B, H, W, C)`` or
            ``(B, C, H, W)`` depending on ``data_format``.
        shift_size: Cyclic-shift offset applied before attention (0 = no shift).
        window_size: Local-attention window edge length.
        relative_index: Flattened indices into the relative-position bias table.
        attention_mask: Per-window attention mask precomputed for this stage.
        num_heads: Number of attention heads.
        bias_table_window_size: Window size for the relative-position bias table
            (may differ from ``window_size`` when fine-tuning at a new resolution).
        channels_axis: Axis of the channel dimension.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        dropout_rate: Dropout rate inside attention and MLP.
        drop_path_rate: Stochastic-depth drop rate for the residual branches.
        name: Layer-name prefix.

    Returns:
        Output tensor with the same shape and layout as ``inputs``.
    """
    cf = data_format == "channels_first"
    h_ax, w_ax = (2, 3) if cf else (1, 2)
    feature_dim = ops.shape(inputs)[1] if cf else ops.shape(inputs)[-1]
    img_height = ops.shape(inputs)[h_ax]
    img_width = ops.shape(inputs)[w_ax]

    x = spatial_layer_norm(
        inputs, data_format, epsilon=1.001e-5, name=f"{name}_layernorm_1"
    )

    height_padding = int((window_size - img_height % window_size) % window_size)
    width_padding = int((window_size - img_width % window_size) % window_size)
    if height_padding > 0 or width_padding > 0:
        x = layers.ZeroPadding2D(
            padding=((0, height_padding), (0, width_padding)),
            data_format=data_format,
        )(x)

    padded_x = x
    shifted_x = SwinRollLayer(shift=[-shift_size, -shift_size], axis=[h_ax, w_ax])(
        padded_x
    )

    attention_layer = SwinWindowAttention(
        dim=feature_dim,
        num_heads=num_heads,
        window_size=window_size,
        bias_table_window_size=bias_table_window_size,
        proj_drop=dropout_rate,
        data_format=data_format,
        block_prefix=name,
    )
    attended_x = attention_layer(
        [shifted_x, window_size, relative_index, attention_mask]
    )
    unshifted_x = SwinRollLayer(shift=[shift_size, shift_size], axis=[h_ax, w_ax])(
        attended_x
    )

    if cf:
        trimmed_x = unshifted_x[:, :, :img_height, :img_width]
    else:
        trimmed_x = unshifted_x[:, :img_height, :img_width]

    dropout_layer = SwinStochasticDepth(drop_path_rate=drop_path_rate)
    skip_x1 = inputs + dropout_layer(trimmed_x)

    normalized_x = spatial_layer_norm(
        skip_x1, data_format, epsilon=1.001e-5, name=f"{name}_layernorm_2"
    )

    if cf:
        mlp_in = ops.transpose(normalized_x, [0, 2, 3, 1])
    else:
        mlp_in = normalized_x
    mlp_x = mlp_block(inputs=mlp_in, dropout=dropout_rate, name=f"{name}_mlp")
    if cf:
        mlp_x = ops.transpose(mlp_x, [0, 3, 1, 2])

    return skip_x1 + dropout_layer(mlp_x)


def patch_merging(inputs, channels_axis, data_format, name="patch_merging"):
    """Patch merging downsample: halve H/W, double channels (V1: norm then dense).

    Concatenates each 2x2 group of patches along the channel axis, applies
    LayerNorm, then a linear projection from ``4C`` to ``2C``.

    Args:
        inputs: Input feature map in either channels-last or channels-first layout.
        channels_axis: Axis of the channel dimension.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        name: Layer-name prefix.

    Returns:
        Output tensor with halved spatial dimensions and channel count ``2C``.
    """
    cf = data_format == "channels_first"
    channels = inputs.shape[1] if cf else inputs.shape[-1]
    h_ax, w_ax = (2, 3) if cf else (1, 2)

    height = ops.shape(inputs)[h_ax]
    width = ops.shape(inputs)[w_ax]
    hpad, wpad = height % 2, width % 2

    if cf:
        paddings = [[0, 0], [0, 0], [0, hpad], [0, wpad]]
    else:
        paddings = [[0, 0], [0, hpad], [0, wpad], [0, 0]]
    x = ops.pad(inputs, paddings)

    # Do the 2x2 merge in channels_last, converting channels_first at the
    # boundaries. The merged 4*C dim must be grouped (2, 2, C) for the timm
    # channel permutation below to line up; the channels_last reshape produces
    # exactly that, whereas a direct NCHW reshape would scramble the grouping.
    if cf:
        x = ops.transpose(x, (0, 2, 3, 1))

    h = ops.shape(x)[1] // 2
    w = ops.shape(x)[2] // 2

    x = ops.reshape(x, (-1, h, 2, w, 2, channels))
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    x = ops.reshape(x, (-1, h, w, 4 * channels))

    perm = ops.reshape(ops.arange(channels * 4), (4, -1))
    perm = ops.convert_to_numpy(perm)
    perm[[1, 2]] = perm[[2, 1]]
    perm = perm.ravel()

    x_reshaped = ops.reshape(x, (-1, 4 * channels))
    perm_matrix = ops.zeros((4 * channels, 4 * channels), dtype="float32")
    perm_matrix = ops.convert_to_numpy(perm_matrix)
    for i, j in enumerate(perm):
        perm_matrix[i, j] = 1
    x = ops.matmul(x_reshaped, ops.convert_to_tensor(perm_matrix))
    x = ops.reshape(x, (-1, h, w, 4 * channels))

    x = layers.LayerNormalization(
        epsilon=1.001e-5, name=f"{name}_pm_layernorm", dtype=inputs.dtype, axis=-1
    )(x)
    x = layers.Dense(
        channels * 2, use_bias=False, name=f"{name}_pm_dense", dtype=inputs.dtype
    )(x)

    if cf:
        x = ops.transpose(x, (0, 3, 1, 2))

    return x


def swin_stage(
    inputs,
    depth,
    num_heads,
    window_size,
    bias_table_window_size,
    channels_axis,
    data_format,
    dropout_rate=0.0,
    drop_path_rate=0.0,
    name="swin_stage",
):
    """One Swin stage: ``depth`` blocks with alternating regular / shifted windows.

    Precomputes the relative-position index and the attention masks for the
    shifted-window pattern, then applies that many windowed-attention blocks.

    Args:
        inputs: Stage input feature map.
        depth: Number of Swin blocks in this stage.
        num_heads: Number of attention heads.
        window_size: Local-attention window edge length.
        bias_table_window_size: Window size to index into the relative-position
            bias table (may differ from ``window_size``).
        channels_axis: Axis of the channel dimension.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        dropout_rate: Dropout rate inside each block.
        drop_path_rate: Stochastic-depth rate, either scalar or per-block list.
        name: Layer-name prefix.

    Returns:
        Output feature map with the same spatial / channel dimensions as the input.
    """
    cf = data_format == "channels_first"
    h_ax, w_ax = (2, 3) if cf else (1, 2)

    h = ops.shape(inputs)[h_ax]
    w = ops.shape(inputs)[w_ax]
    min_dim = ops.minimum(h, w)
    win_size = ops.minimum(window_size, min_dim)

    shift_size = window_size // 2
    shift_sz = 0
    if min_dim > window_size:
        shift_sz = shift_size

    pad_h = ((h - 1) // win_size + 1) * win_size
    pad_w = ((w - 1) // win_size + 1) * win_size

    coords = ops.arange(win_size)
    gx, gy = ops.meshgrid(coords, coords, indexing="ij")
    flat_gx = ops.reshape(gx, [-1])
    flat_gy = ops.reshape(gy, [-1])

    rel_pos_x = flat_gx[:, None] - flat_gx[None, :]
    rel_pos_y = flat_gy[:, None] - flat_gy[None, :]

    relative_index = (ops.reshape(rel_pos_x, [-1]) + win_size - 1) * (
        2 * win_size - 1
    ) + (ops.reshape(rel_pos_y, [-1]) + win_size - 1)

    dtype = keras.backend.floatx()
    partitioner = SwinWindowPartition(
        window_size=win_size, fused=False, data_format="channels_last"
    )

    ones = ops.ones((1, h, w, 1), dtype="int32")
    pad_mask = ops.pad(ones, [[0, 0], [0, pad_h - h], [0, pad_w - w], [0, 0]])
    mask_wins = ops.squeeze(partitioner(pad_mask, height=pad_h, width=pad_w), axis=-1)
    win_diffs = mask_wins[:, None] - mask_wins[:, :, None]

    id_mask = ops.where(
        win_diffs == 0,
        ops.zeros_like(win_diffs, dtype=dtype),
        ops.full_like(win_diffs, -100.0, dtype=dtype),
    )[None, :, None]

    if shift_sz > 0:
        pattern = ops.convert_to_tensor(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype="int32"
        )
        expanded_h = ops.concatenate(
            [
                ops.tile(pattern[0:1, :], [pad_h - win_size, 1]),
                ops.tile(pattern[1:2, :], [win_size - shift_sz, 1]),
                ops.tile(pattern[2:3, :], [shift_sz, 1]),
            ],
            axis=0,
        )
        shift_base = ops.concatenate(
            [
                ops.tile(expanded_h[:, 0:1], [1, pad_w - win_size]),
                ops.tile(expanded_h[:, 1:2], [1, win_size - shift_sz]),
                ops.tile(expanded_h[:, 2:3], [1, shift_sz]),
            ],
            axis=1,
        )
        shift_wins = ops.squeeze(
            partitioner(shift_base[None, ..., None], height=pad_h, width=pad_w), axis=-1
        )
        shift_diffs = shift_wins[:, None] - shift_wins[:, :, None]
        shift_mask = ops.where(
            (shift_diffs == 0) & (win_diffs == 0),
            ops.zeros_like(win_diffs, dtype=dtype),
            ops.full_like(win_diffs, -100.0, dtype=dtype),
        )[None, :, None]
    else:
        shift_mask = id_mask

    masks = [id_mask, shift_mask]

    if not isinstance(drop_path_rate, (list, tuple)):
        drop_rates = [drop_path_rate] * depth
    else:
        drop_rates = list(drop_path_rate)

    x = inputs
    for i in range(depth):
        is_odd = i % 2
        current_shift = shift_sz if is_odd else 0
        x = swin_block(
            x,
            current_shift,
            win_size,
            relative_index,
            masks[is_odd],
            num_heads=num_heads,
            bias_table_window_size=bias_table_window_size,
            channels_axis=channels_axis,
            data_format=data_format,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_rates[i],
            name=f"{name}_blocks_{i}",
        )

    return x


def swin_backbone_feature(
    inputs,
    *,
    pretrain_size,
    window_size,
    embed_dim,
    depths,
    num_heads,
    dropout_rate,
    drop_path_rate,
    data_format,
    channels_axis,
    return_stages=False,
):
    """Swin stem (4x4 patch conv) + 4 hierarchical stages with patch merging.

    Args:
        inputs: Input image tensor of shape ``(B, H, W, C)`` or ``(B, C, H, W)``.
        pretrain_size: Image side used during pretraining (drives relative-position
            bias-table sizing for fine-tuning).
        window_size: Local-attention window edge length.
        embed_dim: Stage-0 token embedding dimension.
        depths: Number of blocks per stage (length-4 list).
        num_heads: Number of attention heads per stage (length-4 list).
        dropout_rate: Dropout rate inside attention / MLP.
        drop_path_rate: Maximum stochastic-depth rate (linearly scaled across blocks).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        channels_axis: Axis of the channel dimension.
        return_stages: If True, return a list of the 4 per-stage feature maps
            (each captured post-stage, pre-downsample). If False (default),
            return the final stage feature map only.

    Returns:
        Final stage feature map ``(B, H, W, C)`` (pre-final-norm), or a list of
        4 per-stage feature maps when ``return_stages=True``.
    """
    x = layers.Conv2D(
        embed_dim,
        kernel_size=4,
        strides=4,
        padding="same",
        data_format=data_format,
        name="stem_conv",
    )(inputs)
    x = spatial_layer_norm(x, data_format, epsilon=1.001e-5, name="stem_norm")
    x = layers.Dropout(dropout_rate, name="stem_dropout")(x)

    path_drops = ops.convert_to_numpy(ops.linspace(0.0, drop_path_rate, sum(depths)))
    scale_factors = 2 ** ops.arange(2, 6)
    pretrain_windows = pretrain_size // scale_factors
    bias_table_window_size = ops.minimum(window_size, pretrain_windows)

    stages = []
    for i in range(len(depths)):
        start_idx = sum(depths[:i])
        end_idx = sum(depths[: i + 1])
        path_drop_values = path_drops[start_idx:end_idx].tolist()
        x = swin_stage(
            x,
            depth=depths[i],
            num_heads=num_heads[i],
            window_size=window_size,
            bias_table_window_size=bias_table_window_size[i],
            channels_axis=channels_axis,
            data_format=data_format,
            dropout_rate=dropout_rate,
            drop_path_rate=path_drop_values,
            name=f"layers_{i}",
        )
        stages.append(x)
        if i != len(depths) - 1:
            x = patch_merging(
                x,
                channels_axis=channels_axis,
                data_format=data_format,
                name=f"layers_{i + 1}_downsample",
            )

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class SwinModel(BaseModel):
    """Instantiates the Swin Transformer backbone.

    Swin is a hierarchical vision transformer that computes self-attention
    within shifted local windows, alternating regular and shifted
    windowing yields cross-window connections while keeping compute
    linear in image size. The network has 4 stages with progressive
    patch merging, halving spatial resolution and doubling channels
    between stages.

    Output is the last layer output before the classifier head: the
    final stage feature map ``(B, H, W, C)`` (or ``(B, C, H, W)`` for
    channels_first), pre-final-norm. :class:`SwinImageClassify` composes this
    model and applies a spatial-LayerNorm + GlobalAveragePooling2D +
    Dense head.

    References:
    - [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

    Args:
        pretrain_size: Integer, image side used during pretraining.
            Drives relative-position bias-table sizing for fine-tuning.
            Defaults to `224`.
        window_size: Integer, local-attention window edge length.
            Defaults to `7`.
        embed_dim: Integer, stage-0 token embedding dimension.
            Subsequent stages double this. Defaults to `96`.
        depths: Tuple of integers, number of Swin blocks per stage
            (length-4). Defaults to `(2, 2, 6, 2)`.
        num_heads: Tuple of integers, number of attention heads per
            stage (length-4). Defaults to `(3, 6, 12, 24)`.
        dropout_rate: Float, dropout rate inside attention and MLP.
            Defaults to `0.0`.
        drop_path_rate: Float, maximum stochastic-depth drop rate. The
            rate is linearly scaled from 0 to this value across all
            blocks. Defaults to `0.1`.
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
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of the
            4 per-stage feature maps. Defaults to `False`.
        name: String, the name of the model. Defaults to `"SwinModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = SwinConfig
    HUB_REPO_SIBLINGS = SWIN_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with SwinImageClassify (which the
        # zm_config declares); build from zm_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = SwinImageClassify.from_weights(repo_id, skip_mismatch=skip_mismatch)
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_swin_timm_to_keras import transfer_swin_weights

        transfer_swin_weights(keras_model, state_dict)

    def __init__(
        self,
        pretrain_size=224,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        dropout_rate=0.0,
        drop_path_rate=0.1,
        image_size=224,
        input_tensor=None,
        as_backbone=False,
        name="SwinModel",
        **kwargs,
    ):
        kwargs.pop("include_normalization", None)
        kwargs.pop("normalization_mode", None)
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

        x = img_input
        x = swin_backbone_feature(
            x,
            pretrain_size=pretrain_size,
            window_size=window_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
            data_format=data_format,
            channels_axis=channels_axis,
            return_stages=as_backbone,
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.pretrain_size = pretrain_size
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pretrain_size": self.pretrain_size,
                "window_size": self.window_size,
                "embed_dim": self.embed_dim,
                "depths": self.depths,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "drop_path_rate": self.drop_path_rate,
                "image_size": self.image_size,
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
class SwinImageClassify(BaseModel):
    """Instantiates the Swin Transformer classifier.

    This classifier wraps a :class:`SwinModel` backbone and attaches a
    spatial-LayerNorm + GlobalAveragePooling2D + Dense head on the final
    feature map to produce ``num_classes`` class logits. All
    architectural parameters are forwarded to the underlying
    :class:`SwinModel`; only ``num_classes`` and
    ``classifier_activation`` are head-specific.

    References:
    - [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

    Args:
        pretrain_size: Integer, image side used during pretraining.
            Drives relative-position bias-table sizing for fine-tuning.
            Defaults to `224`.
        window_size: Integer, local-attention window edge length.
            Defaults to `7`.
        embed_dim: Integer, stage-0 token embedding dimension.
            Subsequent stages double this. Defaults to `96`.
        depths: Tuple of integers, number of Swin blocks per stage
            (length-4). Defaults to `(2, 2, 6, 2)`.
        num_heads: Tuple of integers, number of attention heads per
            stage (length-4). Defaults to `(3, 6, 12, 24)`.
        dropout_rate: Float, dropout rate inside attention and MLP.
            Defaults to `0.0`.
        drop_path_rate: Float, maximum stochastic-depth drop rate. The
            rate is linearly scaled from 0 to this value across all
            blocks. Defaults to `0.1`.
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
            named `f"{name}_backbone"`. Defaults to `"SwinImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = SwinConfig
    HUB_REPO_SIBLINGS = SWIN_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_swin_timm_to_keras import transfer_swin_weights

        transfer_swin_weights(keras_model, state_dict)

    def __init__(
        self,
        pretrain_size=224,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        dropout_rate=0.0,
        drop_path_rate=0.1,
        image_size=224,
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="SwinImageClassify",
        **kwargs,
    ):
        kwargs.pop("include_normalization", None)
        kwargs.pop("normalization_mode", None)
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = SwinModel(
            pretrain_size=pretrain_size,
            window_size=window_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
            image_size=image_size,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = spatial_layer_norm(
            backbone.output, data_format, epsilon=1.001e-5, name="final_norm"
        )
        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(x)
        out = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

        self.pretrain_size = pretrain_size
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate
        self.image_size = backbone.image_size
        self.input_tensor = input_tensor
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pretrain_size": self.pretrain_size,
                "window_size": self.window_size,
                "embed_dim": self.embed_dim,
                "depths": self.depths,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "drop_path_rate": self.drop_path_rate,
                "image_size": self.image_size,
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
