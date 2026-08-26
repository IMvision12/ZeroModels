import keras
from keras import layers, utils

from zeromodels.base import BaseModel
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.utils import standardize_input_shape
from zeromodels.utils.image_util import normalize_image_for_classify_models

from .inceptionv3_config import InceptionV3Config

# The backbone (InceptionV3Model) and classifier (InceptionV3ImageClassify) share the
# variant's repo, whose zm_config.json declares InceptionV3ImageClassify.
INCEPTIONV3_HUB_SIBLINGS = frozenset({"InceptionV3Model", "InceptionV3ImageClassify"})


def conv_block(
    inputs,
    filters=None,
    kernel_size=1,
    strides=1,
    bn_momentum=0.9,
    bn_epsilon=1e-3,
    padding="valid",
    name="conv2d_block",
):
    """Conv -> BatchNorm -> ReLU with optional asymmetric ZeroPadding.

    Args:
        inputs: Input Keras tensor.
        filters: Number of output filters for the convolution.
        kernel_size: Int or 2-tuple kernel size. Scalars are expanded to
            ``(k, k)``.
        strides: Stride of the convolution.
        bn_momentum: Momentum for the BatchNormalization layer.
        bn_epsilon: Epsilon for the BatchNormalization layer.
        padding: Padding mode. ``None`` triggers explicit asymmetric
            zero-padding to match the timm reference layout.
        name: Name prefix used for the conv / bn / activation layers.

    Returns:
        Output tensor after Conv -> BN -> ReLU.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    else:
        kernel_size = tuple(kernel_size)
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    x = inputs
    if padding is None:

        def calculate_padding(kernel_dim):
            pad_total = kernel_dim - 1
            pad_size = pad_total // 2
            pad_extra = (kernel_dim - 1) % 2
            return pad_size, pad_extra

        pad_h, extra_h = calculate_padding(kernel_size[0])
        pad_w, extra_w = calculate_padding(kernel_size[1])

        if strides > 1:
            padding_config = ((pad_h + extra_h, pad_h), (pad_w + extra_w, pad_w))
        else:
            padding_config = ((pad_h, pad_h), (pad_w, pad_w))

        x = layers.ZeroPadding2D(padding=padding_config, name=f"{name}_padding")(x)
        padding = "valid"

    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        data_format=keras.config.image_data_format(),
        name=f"{name}_conv2d",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        momentum=bn_momentum,
        epsilon=bn_epsilon,
        name=f"{name}_batchnorm",
    )(x)
    x = layers.Activation("relu", name=name)(x)
    return x


def inception_blocka(inputs, pool_channels, name="inception_block_a"):
    """Inception-A block (35×35 stage of InceptionV3, used at Mixed_5b/5c/5d).

    Four parallel branches concatenated along the channel axis:

    1. **1×1**: 64 channels.
    2. **5×5 via 1×1 → 5×5**: 48 → 64 channels (factorized
       dimensionality reduction).
    3. **Double-3×3 via 1×1 → 3×3 → 3×3**: 64 → 96 → 96 channels
       (replaces the original 5×5 with a stack of 3×3s).
    4. **Avg-pool 1×1**: same spatial → ``pool_channels`` channels.
       Pool-branch width varies across Mixed_5b (32), Mixed_5c (64),
       Mixed_5d (64).

    Spatial size is preserved (no stride). Output channels =
    ``64 + 64 + 96 + pool_channels``.

    Args:
        inputs: Input Keras tensor.
        pool_channels: Number of filters for the average-pool 1×1 projection
            branch (differs across Mixed_5b/5c/5d).
        name: Name prefix for layers in the block.

    Returns:
        Output tensor formed by concatenating the four branch outputs along
        the channel axis.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch1x1 = conv_block(inputs, 64, 1, name=f"{name}_branch1x1")

    branch5x5 = conv_block(inputs, 48, 1, name=f"{name}_branch5x5_1")
    branch5x5 = conv_block(branch5x5, 64, 5, padding=None, name=f"{name}_branch5x5_2")

    branch3x3dbl = conv_block(inputs, 64, 1, name=f"{name}_branch3x3dbl_1")
    branch3x3dbl = conv_block(
        branch3x3dbl, 96, 3, padding=None, name=f"{name}_branch3x3dbl_2"
    )
    branch3x3dbl = conv_block(
        branch3x3dbl, 96, 3, padding=None, name=f"{name}_branch3x3dbl_3"
    )

    branch_pool = layers.ZeroPadding2D(
        data_format=keras.config.image_data_format(), padding=1
    )(inputs)
    branch_pool = layers.AveragePooling2D(
        pool_size=3,
        strides=1,
        data_format=keras.config.image_data_format(),
    )(branch_pool)
    branch_pool = conv_block(
        branch_pool,
        pool_channels,
        name=f"{name}_branch_pool",
    )

    return layers.Concatenate(axis=channels_axis)(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool]
    )


def inception_blockb(inputs, name="inception_block_b"):
    """Reduction-A block: halves spatial size between Inception-A and -C stages.

    Three parallel branches concatenated along the channel axis:

    1. **Strided 3×3**: 384 channels, stride-2.
    2. **Double-3×3 with stride**: 64 → 96 → 96 channels, last conv
       stride-2.
    3. **Strided max-pool**: 3×3 max-pool, stride-2 (zero-channel
       contribution; relays input channels).

    Used once in InceptionV3 at the ``Mixed_6a`` position to transition
    35×35 → 17×17 feature maps before the Inception-C stack.

    Args:
        inputs: Input Keras tensor.
        name: Name prefix for layers in the block.

    Returns:
        Spatially down-sampled output tensor (3 concatenated branches).
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch3x3 = conv_block(inputs, 384, 3, 2, name=f"{name}_branch3x3")

    branch3x3dbl = conv_block(inputs, 64, 1, name=f"{name}_branch3x3dbl_1")
    branch3x3dbl = conv_block(
        branch3x3dbl, 96, 3, padding=None, name=f"{name}_branch3x3dbl_2"
    )
    branch3x3dbl = conv_block(
        branch3x3dbl, 96, 3, strides=2, name=f"{name}_branch3x3dbl_3"
    )

    branch_pool = layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        data_format=keras.config.image_data_format(),
        name=f"{name}_branch_pool",
    )(inputs)

    return layers.Concatenate(axis=channels_axis)(
        [branch3x3, branch3x3dbl, branch_pool]
    )


def inception_blockc(inputs, branch7x7_channels, name="inception_block_c"):
    """Inception-C block (17×17 stage of InceptionV3, used at Mixed_6b/6c/6d/6e).

    Four parallel branches with factorized 7×7 convolutions
    (``7×7 = 1×7 → 7×1``), concatenated along the channel axis:

    1. **1×1**: 192 channels.
    2. **Single 7×7** via ``1×1 → 1×7 → 7×1``: ``c7 → c7 → 192``
       channels.
    3. **Double 7×7** via ``1×1 → 7×1 → 1×7 → 7×1 → 1×7``:
       ``c7 → c7 → c7 → c7 → 192`` channels.
    4. **Avg-pool 1×1**: 192 channels.

    The inner width ``c7`` widens with depth: 128 at Mixed_6b, 160 at
    Mixed_6c/6d, 192 at Mixed_6e. Spatial size is preserved. Output
    channels = ``192 × 4 = 768``.

    Args:
        inputs: Input Keras tensor.
        branch7x7_channels: Inner channel count for the 7×7 / 7×7-double
            branches (differs across Mixed_6b/6c/6d/6e).
        name: Name prefix for layers in the block.

    Returns:
        Output tensor concatenating the four branch outputs along the
        channel axis.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    c7 = branch7x7_channels

    branch1x1 = conv_block(inputs, 192, 1, name=f"{name}_branch1x1")

    branch7x7 = conv_block(inputs, c7, 1, name=f"{name}_branch7x7_1")
    branch7x7 = conv_block(
        branch7x7, c7, (1, 7), padding=None, name=f"{name}_branch7x7_2"
    )
    branch7x7 = conv_block(
        branch7x7, 192, (7, 1), padding=None, name=f"{name}_branch7x7_3"
    )

    branch7x7dbl = conv_block(inputs, c7, 1, name=f"{name}_branch7x7dbl_1")
    branch7x7dbl = conv_block(
        branch7x7dbl, c7, (7, 1), padding=None, name=f"{name}_branch7x7dbl_2"
    )
    branch7x7dbl = conv_block(
        branch7x7dbl, c7, (1, 7), padding=None, name=f"{name}_branch7x7dbl_3"
    )
    branch7x7dbl = conv_block(
        branch7x7dbl, c7, (7, 1), padding=None, name=f"{name}_branch7x7dbl_4"
    )
    branch7x7dbl = conv_block(
        branch7x7dbl, 192, (1, 7), padding=None, name=f"{name}_branch7x7dbl_5"
    )

    branch_pool = layers.ZeroPadding2D(
        data_format=keras.config.image_data_format(), padding=1
    )(inputs)
    branch_pool = layers.AveragePooling2D(
        pool_size=3, strides=1, data_format=keras.config.image_data_format()
    )(branch_pool)
    branch_pool = conv_block(branch_pool, 192, 1, name=f"{name}_branch_pool")

    return layers.Concatenate(axis=channels_axis)(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool]
    )


def inception_blockd(inputs, name="inception_block_d"):
    """Reduction-B block: halves spatial size between Inception-C and -E stages.

    Three parallel branches concatenated along the channel axis:

    1. **1×1 → strided 3×3**: 192 → 320 channels, stride-2.
    2. **1×1 → 1×7 → 7×1 → strided 3×3**: 192 → 192 → 192 → 192
       channels (factorized 7×7 followed by a stride-2 3×3).
    3. **Strided max-pool**: 3×3, stride-2.

    Used once in InceptionV3 at the ``Mixed_7a`` position to transition
    17×17 → 8×8 feature maps before the Inception-E stack.

    Args:
        inputs: Input Keras tensor.
        name: Name prefix for layers in the block.

    Returns:
        Spatially down-sampled output tensor (3 concatenated branches).
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch3x3 = conv_block(inputs, 192, 1, name=f"{name}_branch3x3_1")
    branch3x3 = conv_block(branch3x3, 320, 3, strides=2, name=f"{name}_branch3x3_2")

    branch7x7x3 = conv_block(inputs, 192, 1, name=f"{name}_branch7x7x3_1")
    branch7x7x3 = conv_block(
        branch7x7x3, 192, (1, 7), padding=None, name=f"{name}_branch7x7x3_2"
    )
    branch7x7x3 = conv_block(
        branch7x7x3, 192, (7, 1), padding=None, name=f"{name}_branch7x7x3_3"
    )
    branch7x7x3 = conv_block(
        branch7x7x3, 192, 3, strides=2, name=f"{name}_branch7x7x3_4"
    )

    branch_pool = layers.MaxPooling2D(
        data_format=keras.config.image_data_format(), pool_size=3, strides=2
    )(inputs)

    return layers.Concatenate(axis=channels_axis)([branch3x3, branch7x7x3, branch_pool])


def inception_blocke(inputs, name="inception_block_e"):
    """Inception-E block (8×8 stage of InceptionV3, used at Mixed_7b/7c).

    Four parallel branches with the **parallel** (not stacked) 3×3
    factorization: each ``3×3`` is replaced by ``1×3 || 3×1``
    branches whose outputs are concatenated, producing a wider feature
    map per branch instead of a deeper stack. Branches:

    1. **1×1**: 320 channels.
    2. **1×1 → (1×3 ∥ 3×1)**: 384 → 384 || 384 channels (concat = 768).
    3. **1×1 → 3×3 → (1×3 ∥ 3×1)**: 448 → 384 → 384 || 384 channels
       (concat = 768).
    4. **Avg-pool 1×1**: 192 channels.

    Output channels = ``320 + 768 + 768 + 192 = 2048``: the final
    InceptionV3 feature width.

    Args:
        inputs: Input Keras tensor.
        name: Name prefix for layers in the block.

    Returns:
        Output tensor concatenating the four expanded branch outputs along
        the channel axis.
    """
    channels_axis = -1 if keras.config.image_data_format() == "channels_last" else 1

    branch1x1 = conv_block(inputs, 320, 1, name=f"{name}_branch1x1")

    branch3x3 = conv_block(inputs, 384, 1, name=f"{name}_branch3x3_1")
    branch3x3_a = conv_block(
        branch3x3,
        filters=384,
        kernel_size=(1, 3),
        padding=None,
        name=f"{name}_branch3x3_2a",
    )
    branch3x3_b = conv_block(
        branch3x3,
        filters=384,
        kernel_size=(3, 1),
        padding=None,
        name=f"{name}_branch3x3_2b",
    )
    branch3x3 = layers.Concatenate(axis=channels_axis)([branch3x3_a, branch3x3_b])

    branch3x3dbl = conv_block(inputs, 448, 1, name=f"{name}_branch3x3dbl_1")
    branch3x3dbl = conv_block(
        branch3x3dbl, 384, 3, padding=None, name=f"{name}_branch3x3dbl_2"
    )
    branch3x3dbl_a = conv_block(
        branch3x3dbl,
        filters=384,
        kernel_size=(1, 3),
        padding=None,
        name=f"{name}_branch3x3dbl_3a",
    )
    branch3x3dbl_b = conv_block(
        branch3x3dbl,
        filters=384,
        kernel_size=(3, 1),
        padding=None,
        name=f"{name}_branch3x3dbl_3b",
    )
    branch3x3dbl = layers.Concatenate(axis=channels_axis)(
        [branch3x3dbl_a, branch3x3dbl_b]
    )

    branch_pool = layers.ZeroPadding2D(
        data_format=keras.config.image_data_format(), padding=1
    )(inputs)
    branch_pool = layers.AveragePooling2D(
        pool_size=3,
        strides=1,
        data_format=keras.config.image_data_format(),
    )(branch_pool)
    branch_pool = conv_block(branch_pool, 192, 1, name=f"{name}_branch_pool")

    return layers.Concatenate(axis=channels_axis)(
        [branch1x1, branch3x3, branch3x3dbl, branch_pool]
    )


def inceptionv3_backbone_feature(inputs, *, data_format, return_stages=False):
    """InceptionV3 stem + 5-block backbone, returns final stage feature map.

    Args:
        inputs: Input image tensor.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        return_stages: If True, return a list of per-stage feature maps
            taken at natural downsample boundaries (after Pool1, after
            Conv2d_4a, after Mixed_5d, after Mixed_6e, after Mixed_7c).
            If False (default), return only the final stage map.

    Returns:
        Final stage feature tensor (after Mixed_7c), or a list of per-stage
        feature maps when ``return_stages=True``.
    """
    stages = []
    # Stem
    x = conv_block(inputs, 32, 3, strides=2, name="Conv2d_1a_3x3")
    x = conv_block(x, 32, 3, name="Conv2d_2a_3x3")
    x = conv_block(x, 64, 3, padding=None, name="Conv2d_2b_3x3")

    x = layers.MaxPooling2D(3, 2, name="Pool1")(x)
    stages.append(x)  # stage 1: after Pool1
    x = conv_block(x, 80, 1, name="Conv2d_3b_1x1")
    x = conv_block(x, 192, 3, name="Conv2d_4a_3x3")

    x = layers.MaxPooling2D(3, 2, name="Pool2")(x)
    x = inception_blocka(x, 32, "Mixed_5b")
    x = inception_blocka(x, 64, "Mixed_5c")
    x = inception_blocka(x, 64, "Mixed_5d")
    stages.append(x)  # stage 2: after Inception-A group

    x = inception_blockb(x, "Mixed_6a")
    x = inception_blockc(x, 128, "Mixed_6b")
    x = inception_blockc(x, 160, "Mixed_6c")
    x = inception_blockc(x, 160, "Mixed_6d")
    x = inception_blockc(x, 192, "Mixed_6e")
    stages.append(x)  # stage 3: after Inception-C group

    x = inception_blockd(x, "Mixed_7a")
    x = inception_blocke(x, "Mixed_7b")
    x = inception_blocke(x, "Mixed_7c")
    stages.append(x)  # stage 4: after Inception-E group

    if return_stages:
        return stages
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class InceptionV3Model(BaseModel):
    """Instantiates the Inception V3 backbone.

    Inception V3 refines the original GoogLeNet recipe by factorizing
    large nxn convolutions into asymmetric 1xn + nx1 stacks, replacing
    5x5 convs with double 3x3 stacks, and using dedicated grid-reduction
    blocks; the original training procedure also relied on auxiliary
    classifiers, label smoothing, and the RMSProp optimizer. The output
    tensor is the last layer output before the classifier head: the
    final-stage feature map ``(B, H, W, C)`` (after the last Mixed_7c
    Inception-E block), unpooled and head-free.
    :class:`InceptionV3ImageClassify` composes this model and applies a
    GlobalAveragePooling2D + Dense head to produce logits.

    References:
    - [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

    Args:
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `299`.
        include_normalization: Boolean, whether to prepend an
            image normalization at the start
            of the network. When True, input images should be in uint8
            format with values in `[0, 255]`. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to
            use. Must be one of: `'imagenet'`, `'inception'` (default),
            `'dpn'`, `'clip'`, `'zero_to_one'`, or `'minus_one_to_one'`.
            Only used when ``include_normalization=True``.
        input_tensor: Optional Keras tensor as input. Useful for
            connecting the model to other Keras components.
            Defaults to `None`.
        as_backbone: Boolean, whether to output intermediate features for
            use as a backbone network. When True, returns a list of
            per-stage feature maps at natural downsample boundaries
            (after Pool1, after the Inception-A group, after the
            Inception-C group, and after the Inception-E group).
            Defaults to `False`.
        name: String, the name of the model. Defaults to `"InceptionV3Model"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = InceptionV3Config
    HUB_REPO_SIBLINGS = INCEPTIONV3_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        # Backbone shares the variant's repo with InceptionV3ImageClassify (which the
        # zm_config declares); build from zm_config, then copy the backbone weights.
        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if load_weights:
            src = InceptionV3ImageClassify.from_weights(
                repo_id, skip_mismatch=skip_mismatch
            )
            copy_weights_by_path_suffix(src, model)
            del src
        return model

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_inceptionv3_timm_to_keras import transfer_inceptionv3_weights

        transfer_inceptionv3_weights(keras_model, state_dict)

    def __init__(
        self,
        image_size=299,
        include_normalization=True,
        normalization_mode="inception",
        input_tensor=None,
        as_backbone=False,
        name="InceptionV3Model",
        **kwargs,
    ):
        for k in ("num_classes", "classifier_activation", "timm_id"):
            kwargs.pop(k, None)

        data_format = keras.config.image_data_format()

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
        x = inceptionv3_backbone_feature(
            x, data_format=data_format, return_stages=as_backbone
        )

        super().__init__(inputs=img_input, outputs=x, name=name, **kwargs)

        self.image_size = image_size
        self.include_normalization = include_normalization
        self.normalization_mode = normalization_mode
        self.input_tensor = input_tensor
        self.as_backbone = as_backbone

    def get_config(self):
        config = super().get_config()
        config.update(
            {
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
class InceptionV3ImageClassify(BaseModel):
    """Instantiates the Inception V3 classifier.

    This classifier wraps an :class:`InceptionV3Model` backbone and
    attaches a GlobalAveragePooling2D + Dense head to produce
    ``num_classes`` class logits. All architectural parameters are
    forwarded to the underlying :class:`InceptionV3Model`; only
    ``num_classes`` and ``classifier_activation`` are head-specific.

    References:
    - [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

    Args:
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `299`.
        include_normalization: Boolean, whether to prepend an
            image normalization at the start
            of the network. When True, input images should be in uint8
            format with values in `[0, 255]`. Defaults to `True`.
        normalization_mode: String, specifying the normalization mode to
            use. Must be one of: `'imagenet'`, `'inception'` (default),
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
            named `f"{name}_backbone"`. Defaults to `"InceptionV3ImageClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    config_class = InceptionV3Config
    HUB_REPO_SIBLINGS = INCEPTIONV3_HUB_SIBLINGS
    HF_MODEL_TYPE = None

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        from .convert_inceptionv3_timm_to_keras import transfer_inceptionv3_weights

        transfer_inceptionv3_weights(keras_model, state_dict)

    def __init__(
        self,
        image_size=299,
        include_normalization=True,
        normalization_mode="inception",
        input_tensor=None,
        num_classes=1000,
        classifier_activation="linear",
        name="InceptionV3ImageClassify",
        **kwargs,
    ):
        kwargs.pop("timm_id", None)

        data_format = keras.config.image_data_format()

        backbone = InceptionV3Model(
            image_size=image_size,
            include_normalization=include_normalization,
            normalization_mode=normalization_mode,
            input_tensor=input_tensor,
            name=f"{name}_backbone",
        )

        x = layers.GlobalAveragePooling2D(data_format=data_format, name="avg_pool")(
            backbone.output
        )
        out = layers.Dense(
            num_classes,
            activation=classifier_activation,
            name="classifier",
        )(x)

        super().__init__(inputs=backbone.input, outputs=out, name=name, **kwargs)

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
