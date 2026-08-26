import keras
import numpy as np
from keras import layers, ops


@keras.saving.register_keras_serializable(package="zeromodels.tipsv2_dpt")
def gelu_tanh(x):
    """tanh-approximate GELU (``gelu_pytorch_tanh``), the DPT readout activation."""
    return keras.activations.gelu(x, approximate=True)


def aligned_bilinear_resize(x, target_h, target_w, data_format):
    """Bilinear resize matching torch ``interpolate(..., align_corners=True)``.

    ``keras.ops.image.resize`` only offers half-pixel alignment; DPT fusion upsamples
    with ``align_corners=True``. This implements the align-corners coordinate mapping
    with explicit gather + lerp so every backend agrees. (Same math as the Depth
    Anything port.)
    """
    shape = ops.shape(x)
    if data_format == "channels_first":
        h_axis, w_axis = 2, 3
        h, w = shape[2], shape[3]
    else:
        h_axis, w_axis = 1, 2
        h, w = shape[1], shape[2]
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


@keras.saving.register_keras_serializable(package="zeromodels")
class Tipsv2DptResize(layers.Layer):
    """Serializable align-corners bilinear resize to a fixed ``(target_h, target_w)``."""

    def __init__(self, target_h, target_w, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.target_h = target_h
        self.target_w = target_w
        self.data_format = data_format or keras.config.image_data_format()

    def call(self, x):
        return aligned_bilinear_resize(
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


@keras.saving.register_keras_serializable(package="zeromodels")
class Tipsv2DptReadout(layers.Layer):
    """Concatenate patch tokens with the broadcast CLS readout, along the channel axis."""

    def call(self, patch_tokens, cls_token):
        readout = ops.broadcast_to(cls_token[:, None, :], ops.shape(patch_tokens))
        return ops.concatenate([patch_tokens, readout], axis=-1)


@keras.saving.register_keras_serializable(package="zeromodels")
class Tipsv2DptFeaturesToDepth(layers.Layer):
    """Soft-argmax depth: ``relu(logits)+min -> normalize over bins -> @ bin_centers``.

    Input logits ``(B, H, W, num_bins)`` (channels-last), output depth ``(B, H, W)``.
    """

    def __init__(self, num_depth_bins=256, min_depth=0.001, max_depth=10.0, **kwargs):
        super().__init__(**kwargs)
        self.num_depth_bins = num_depth_bins
        self.min_depth = min_depth
        self.max_depth = max_depth

    def build(self, input_shape):
        centers = np.linspace(
            self.min_depth, self.max_depth, self.num_depth_bins, dtype="float32"
        )
        self.bin_centers = self.add_weight(
            name="bin_centers",
            shape=(self.num_depth_bins,),
            initializer=lambda shape, dtype=None: ops.convert_to_tensor(centers),
            trainable=False,
        )
        self.built = True

    def call(self, depth_logits):
        probs = ops.relu(depth_logits) + self.min_depth
        probs = probs / ops.sum(probs, axis=-1, keepdims=True)
        return ops.tensordot(
            probs, ops.cast(self.bin_centers, probs.dtype), axes=[[-1], [0]]
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_depth_bins": self.num_depth_bins,
                "min_depth": self.min_depth,
                "max_depth": self.max_depth,
            }
        )
        return config
