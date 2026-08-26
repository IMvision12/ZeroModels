import keras
from keras import layers, ops

# Batch-norm momentum is inference-irrelevant (moving stats are loaded, never updated
# here); 0.9 is the Keras equivalent of timm's torch default (0.1).
BN_MOMENTUM = 0.9


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    """Snap a (possibly scaled) channel count to a multiple of ``divisor``.

    Mirrors timm's ``make_divisible`` used for MobileNetV4 expansion channels.
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def conv2d_pad(x, kernel_size, stride, data_format):
    """Apply PyTorch-symmetric padding, returning ``(x, padding_mode)``.

    timm MobileNetV4 uses ``pad_type=''`` (symmetric static padding,
    ``pad = (kernel_size - 1) // 2`` each side). For stride 1 that matches Keras
    ``"same"``; for stride > 1 Keras ``"same"`` pads asymmetrically, so an explicit
    symmetric :class:`ZeroPadding2D` + ``"valid"`` is used instead.
    """
    if stride > 1:
        pad = kernel_size // 2
        x = layers.ZeroPadding2D(
            padding=((pad, pad), (pad, pad)), data_format=data_format
        )(x)
        return x, "valid"
    return x, "same"


@keras.saving.register_keras_serializable(package="zeromodels.mobilenetv4")
class LayerScale2D(layers.Layer):
    """Per-channel learnable scale (timm ``LayerScale2d``): ``x * gamma``.

    ``gamma`` is a length-``C`` vector initialized to ``init_value`` (1e-5 for the
    MobileNetV4 hybrid variants), broadcast over the spatial dimensions.
    """

    def __init__(self, init_value=1e-5, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value
        self.data_format = data_format or keras.config.image_data_format()

    def build(self, input_shape):
        axis = -1 if self.data_format == "channels_last" else 1
        dim = input_shape[axis]
        self.gamma = self.add_weight(
            name="gamma",
            shape=(dim,),
            initializer=keras.initializers.Constant(self.init_value),
            trainable=True,
        )
        self.built = True

    def call(self, x):
        if self.data_format == "channels_last":
            return x * self.gamma
        return x * ops.reshape(self.gamma, (1, -1, 1, 1))

    def get_config(self):
        config = super().get_config()
        config.update({"init_value": self.init_value, "data_format": self.data_format})
        return config


def conv_bn_act(
    x,
    filters,
    kernel_size,
    stride,
    activation,
    prefix,
    data_format,
    channels_axis,
    bn_epsilon=1e-5,
):
    """ConvBnAct block (timm ``cn_``): conv -> BN -> activation.

    MobileNetV4 ``cn_`` blocks never carry a residual (skip token absent).
    """
    x, padding = conv2d_pad(x, kernel_size, stride, data_format)
    x = layers.Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        use_bias=False,
        data_format=data_format,
        name=f"{prefix}_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=bn_epsilon,
        momentum=BN_MOMENTUM,
        name=f"{prefix}_bn1",
    )(x)
    x = layers.Activation(activation, name=f"{prefix}_act")(x)
    return x


def edge_residual(
    x,
    filters,
    exp_kernel_size,
    stride,
    exp_ratio,
    activation,
    prefix,
    data_format,
    channels_axis,
    bn_epsilon=1e-5,
    noskip=False,
):
    """EdgeResidual / FusedIB block (timm ``er_``).

    conv_exp (KxK, in->mid) -> BN+act -> conv_pwl (1x1, mid->out) -> BN, with a
    residual when ``in == out`` and ``stride == 1``.
    """
    shortcut = x
    in_filters = x.shape[channels_axis]
    mid_filters = make_divisible(in_filters * exp_ratio)

    xp, padding = conv2d_pad(x, exp_kernel_size, stride, data_format)
    xp = layers.Conv2D(
        mid_filters,
        kernel_size=exp_kernel_size,
        strides=stride,
        padding=padding,
        use_bias=False,
        data_format=data_format,
        name=f"{prefix}_conv_exp",
    )(xp)
    xp = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=bn_epsilon,
        momentum=BN_MOMENTUM,
        name=f"{prefix}_bn1",
    )(xp)
    xp = layers.Activation(activation, name=f"{prefix}_act")(xp)

    xp = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{prefix}_conv_pwl",
    )(xp)
    xp = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=bn_epsilon,
        momentum=BN_MOMENTUM,
        name=f"{prefix}_bn2",
    )(xp)

    if not noskip and stride == 1 and in_filters == filters:
        xp = layers.Add(name=f"{prefix}_add")([shortcut, xp])
    return xp


def universal_inverted_residual(
    x,
    filters,
    dw_start_kernel,
    dw_mid_kernel,
    stride,
    exp_ratio,
    activation,
    prefix,
    data_format,
    channels_axis,
    layer_scale_init=None,
    bn_epsilon=1e-5,
    noskip=False,
):
    """Universal Inverted Bottleneck (timm ``uir_``).

    Optional dw_start (KxK, BN, no act) -> pw_exp (1x1, BN+act) -> optional dw_mid
    (KxK, BN+act) -> pw_proj (1x1, BN, no act) -> optional LayerScale -> residual.
    The stride lands on dw_mid when present, otherwise on dw_start.
    """
    shortcut = x
    in_filters = x.shape[channels_axis]
    mid_filters = make_divisible(in_filters * exp_ratio)

    xp = x
    if dw_start_kernel:
        # dw_start carries the stride only when there is no dw_mid.
        dw_start_stride = stride if not dw_mid_kernel else 1
        xp, padding = conv2d_pad(xp, dw_start_kernel, dw_start_stride, data_format)
        xp = layers.DepthwiseConv2D(
            dw_start_kernel,
            strides=dw_start_stride,
            padding=padding,
            use_bias=False,
            data_format=data_format,
            name=f"{prefix}_dw_start_conv",
        )(xp)
        xp = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=bn_epsilon,
            momentum=BN_MOMENTUM,
            name=f"{prefix}_dw_start_bn",
        )(xp)

    xp = layers.Conv2D(
        mid_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{prefix}_pw_exp_conv",
    )(xp)
    xp = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=bn_epsilon,
        momentum=BN_MOMENTUM,
        name=f"{prefix}_pw_exp_bn",
    )(xp)
    xp = layers.Activation(activation, name=f"{prefix}_pw_exp_act")(xp)

    if dw_mid_kernel:
        xp, padding = conv2d_pad(xp, dw_mid_kernel, stride, data_format)
        xp = layers.DepthwiseConv2D(
            dw_mid_kernel,
            strides=stride,
            padding=padding,
            use_bias=False,
            data_format=data_format,
            name=f"{prefix}_dw_mid_conv",
        )(xp)
        xp = layers.BatchNormalization(
            axis=channels_axis,
            epsilon=bn_epsilon,
            momentum=BN_MOMENTUM,
            name=f"{prefix}_dw_mid_bn",
        )(xp)
        xp = layers.Activation(activation, name=f"{prefix}_dw_mid_act")(xp)

    xp = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{prefix}_pw_proj_conv",
    )(xp)
    xp = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=bn_epsilon,
        momentum=BN_MOMENTUM,
        name=f"{prefix}_pw_proj_bn",
    )(xp)

    if layer_scale_init is not None:
        xp = LayerScale2D(
            layer_scale_init, data_format=data_format, name=f"{prefix}_layer_scale"
        )(xp)

    if not noskip and stride == 1 and in_filters == filters:
        xp = layers.Add(name=f"{prefix}_add")([shortcut, xp])
    return xp


def multi_query_attention(
    x,
    dim_out,
    num_heads,
    key_dim,
    value_dim,
    kv_stride,
    dw_kernel_size,
    prefix,
    data_format,
    bn_epsilon=1e-5,
):
    """Mobile Multi-Query Attention 2D (timm ``MultiQueryAttention2d``).

    A single shared key/value head is broadcast across ``num_heads`` query heads.
    Key/value are spatially downsampled by a depthwise conv (stride ``kv_stride``)
    before their 1x1 projections. Computed in channels-last layout internally.
    """
    channels_first = data_format == "channels_first"
    if channels_first:
        x = ops.transpose(x, (0, 2, 3, 1))

    height = x.shape[1]
    width = x.shape[2]
    num_tokens = height * width
    scale = key_dim**-0.5

    query = layers.Conv2D(
        num_heads * key_dim,
        kernel_size=1,
        use_bias=False,
        data_format="channels_last",
        name=f"{prefix}_attn_query_proj",
    )(x)
    query = ops.reshape(query, (-1, num_tokens, num_heads, key_dim))
    query = ops.transpose(query, (0, 2, 1, 3))

    key_in = x
    value_in = x
    if kv_stride > 1:
        pad = dw_kernel_size // 2
        key_in = layers.ZeroPadding2D(
            padding=((pad, pad), (pad, pad)), data_format="channels_last"
        )(key_in)
        key_in = layers.DepthwiseConv2D(
            dw_kernel_size,
            strides=kv_stride,
            padding="valid",
            use_bias=False,
            data_format="channels_last",
            name=f"{prefix}_attn_key_down_conv",
        )(key_in)
        key_in = layers.BatchNormalization(
            axis=-1,
            epsilon=bn_epsilon,
            momentum=BN_MOMENTUM,
            name=f"{prefix}_attn_key_norm",
        )(key_in)
        value_in = layers.ZeroPadding2D(
            padding=((pad, pad), (pad, pad)), data_format="channels_last"
        )(value_in)
        value_in = layers.DepthwiseConv2D(
            dw_kernel_size,
            strides=kv_stride,
            padding="valid",
            use_bias=False,
            data_format="channels_last",
            name=f"{prefix}_attn_value_down_conv",
        )(value_in)
        value_in = layers.BatchNormalization(
            axis=-1,
            epsilon=bn_epsilon,
            momentum=BN_MOMENTUM,
            name=f"{prefix}_attn_value_norm",
        )(value_in)

    key = layers.Conv2D(
        key_dim,
        kernel_size=1,
        use_bias=False,
        data_format="channels_last",
        name=f"{prefix}_attn_key_proj",
    )(key_in)
    num_kv = key.shape[1] * key.shape[2]
    key = ops.reshape(key, (-1, 1, num_kv, key_dim))

    value = layers.Conv2D(
        value_dim,
        kernel_size=1,
        use_bias=False,
        data_format="channels_last",
        name=f"{prefix}_attn_value_proj",
    )(value_in)
    value = ops.reshape(value, (-1, 1, num_kv, value_dim))

    attn = ops.matmul(query * scale, ops.transpose(key, (0, 1, 3, 2)))
    attn = ops.softmax(attn, axis=-1)
    out = ops.matmul(attn, value)
    out = ops.transpose(out, (0, 2, 1, 3))
    out = ops.reshape(out, (-1, height, width, num_heads * value_dim))
    out = layers.Conv2D(
        dim_out,
        kernel_size=1,
        use_bias=False,
        data_format="channels_last",
        name=f"{prefix}_attn_output_proj",
    )(out)

    if channels_first:
        out = ops.transpose(out, (0, 3, 1, 2))
    return out


def mobile_attention(
    x,
    dim_out,
    num_heads,
    key_dim,
    value_dim,
    kv_stride,
    dw_kernel_size,
    prefix,
    data_format,
    channels_axis,
    layer_scale_init=None,
    bn_epsilon=1e-5,
    noskip=False,
):
    """MobileAttention block (timm ``mqa_``): BN norm -> MQA -> LayerScale -> residual.

    The block stride is always 1 in the MobileNetV4 arch defs, so the residual is
    added whenever ``in == out`` (spatial resolution is preserved by the attention).
    """
    shortcut = x
    in_filters = x.shape[channels_axis]

    xn = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=bn_epsilon,
        momentum=BN_MOMENTUM,
        name=f"{prefix}_norm",
    )(x)
    attn = multi_query_attention(
        xn,
        dim_out=dim_out,
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        kv_stride=kv_stride,
        dw_kernel_size=dw_kernel_size,
        prefix=prefix,
        data_format=data_format,
        bn_epsilon=bn_epsilon,
    )

    if layer_scale_init is not None:
        attn = LayerScale2D(
            layer_scale_init, data_format=data_format, name=f"{prefix}_layer_scale"
        )(attn)

    if not noskip and in_filters == dim_out:
        attn = layers.Add(name=f"{prefix}_add")([shortcut, attn])
    return attn


def decode_block_str(block_str):
    """Parse a timm arch-def block string into ``(block_type, options, noskip)``.

    Handles the option tokens used by MobileNetV4: ``r`` (repeat), ``k`` (kernel /
    dw_mid / mqa dw kernel), ``a`` (dw_start kernel), ``s`` (stride), ``e`` (expand
    ratio), ``c`` (out channels), ``h`` (num heads), ``v`` (kv stride), ``d`` (key /
    value dim), ``p`` (dw_end kernel), plus ``noskip`` / ``skip``.
    """
    parts = block_str.split("_")
    block_type = parts[0]
    options = {}
    noskip = False
    for token in parts[1:]:
        if token == "noskip":
            noskip = True
        elif token == "skip":
            noskip = False
        else:
            options[token[0]] = token[1:]
    return block_type, options, noskip
