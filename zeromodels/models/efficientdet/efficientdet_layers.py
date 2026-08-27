import math

import keras
from keras import layers, ops

from .efficientdet_config import bifpn_nodes


def act(x, act_type="swish"):
    if act_type == "swish":
        return ops.silu(x)
    if act_type == "relu":
        return ops.relu(x)
    return layers.Activation(act_type)(x)


def sep_conv(filters, kernel_size=3, use_bias=True, name=None):
    return layers.SeparableConv2D(
        filters,
        kernel_size=kernel_size,
        padding="same",
        depth_multiplier=1,
        use_bias=use_bias,
        name=name,
    )


@keras.saving.register_keras_serializable(package="zeromodels")
class EfficientDetResample(layers.Layer):
    """Resample a feature map to a target level: 1x1 conv (only when the channel
    count changes) + BatchNorm, then max-pool (downsample) or nearest-neighbour
    upsample to the target spatial size. ``conv_after_downsample`` moves the 1x1
    to after the pool."""

    def __init__(
        self,
        target_channels,
        apply_bn=True,
        conv_after_downsample=False,
        name=None,
    ):
        super().__init__(name=name)
        self.target_channels = target_channels
        self.apply_bn = apply_bn
        self.conv_after_downsample = conv_after_downsample
        self.conv2d = None
        self.bn = None

    def build(self, input_shape):
        in_channels = input_shape[-1]
        if in_channels != self.target_channels:
            self.conv2d = layers.Conv2D(
                self.target_channels, 1, padding="same", name="conv2d"
            )
            if self.apply_bn:
                self.bn = layers.BatchNormalization(name="bn")
        self.built = True

    def apply_1x1(self, feat, training):
        if self.conv2d is not None:
            feat = self.conv2d(feat)
            if self.bn is not None:
                feat = self.bn(feat, training=training)
        return feat

    def call(self, feat, target_height, target_width, training=False):
        height, width = int(feat.shape[1]), int(feat.shape[2])
        if height > target_height and width > target_width:
            if not self.conv_after_downsample:
                feat = self.apply_1x1(feat, training)
            h_stride = (height - 1) // target_height + 1
            w_stride = (width - 1) // target_width + 1
            feat = layers.MaxPooling2D(
                pool_size=(h_stride + 1, w_stride + 1),
                strides=(h_stride, w_stride),
                padding="same",
            )(feat)
            if self.conv_after_downsample:
                feat = self.apply_1x1(feat, training)
        else:
            feat = self.apply_1x1(feat, training)
            if height < target_height or width < target_width:
                feat = ops.image.resize(
                    feat, (target_height, target_width), interpolation="nearest"
                )
        return feat

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "target_channels": self.target_channels,
                "apply_bn": self.apply_bn,
                "conv_after_downsample": self.conv_after_downsample,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class FNode(layers.Layer):
    """One BiFPN node: resample each input feature to this node's level, fuse them
    with a weighted sum (``fastattn`` = normalized ReLU weights, ``attn`` = softmax,
    ``sum`` = plain), then act -> separable 3x3 conv -> BN. Appends the new feature
    to the running list."""

    def __init__(
        self,
        feat_level,
        inputs_offsets,
        fpn_num_filters,
        weight_method,
        act_type,
        apply_bn_for_resampling,
        conv_after_downsample,
        conv_bn_act_pattern,
        name=None,
    ):
        super().__init__(name=name)
        self.feat_level = feat_level
        self.inputs_offsets = list(inputs_offsets)
        self.fpn_num_filters = fpn_num_filters
        self.weight_method = weight_method
        self.act_type = act_type
        self.apply_bn_for_resampling = apply_bn_for_resampling
        self.conv_after_downsample = conv_after_downsample
        self.conv_bn_act_pattern = conv_bn_act_pattern
        self.resamples = [
            EfficientDetResample(
                fpn_num_filters,
                apply_bn=apply_bn_for_resampling,
                conv_after_downsample=conv_after_downsample,
                name=f"resample_{i}_{off}",
            )
            for i, off in enumerate(self.inputs_offsets)
        ]
        self.conv = sep_conv(
            fpn_num_filters, 3, use_bias=not conv_bn_act_pattern, name="conv"
        )
        self.bn = layers.BatchNormalization(name="bn")

    def build(self, input_shape):
        self.edge_weights = None
        if self.weight_method in ("fastattn", "attn"):
            self.edge_weights = [
                self.add_weight(
                    name="WSM" + ("" if i == 0 else f"_{i}"),
                    shape=(),
                    initializer="ones",
                    trainable=True,
                )
                for i in range(len(self.inputs_offsets))
            ]
        self.built = True

    def fuse(self, nodes):
        if self.weight_method == "fastattn":
            weights = [ops.relu(w) for w in self.edge_weights]
            total = ops.sum(ops.stack(weights)) + 1e-4
            return ops.sum(
                ops.stack([n * (w / total) for n, w in zip(nodes, weights)]), axis=0
            )
        if self.weight_method == "attn":
            weights = ops.softmax(ops.stack(self.edge_weights))
            return ops.sum(
                ops.stack([n * weights[i] for i, n in enumerate(nodes)]), axis=0
            )
        return ops.sum(ops.stack(nodes), axis=0)

    def call(self, feats, target_height, target_width, training=False):
        nodes = [
            self.resamples[i](
                feats[off],
                target_height=target_height,
                target_width=target_width,
                training=training,
            )
            for i, off in enumerate(self.inputs_offsets)
        ]
        new_node = self.fuse(nodes)
        if not self.conv_bn_act_pattern:
            new_node = act(new_node, self.act_type)
        new_node = self.conv(new_node)
        new_node = self.bn(new_node, training=training)
        if self.conv_bn_act_pattern:
            new_node = act(new_node, self.act_type)
        return feats + [new_node]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "feat_level": self.feat_level,
                "inputs_offsets": self.inputs_offsets,
                "fpn_num_filters": self.fpn_num_filters,
                "weight_method": self.weight_method,
                "act_type": self.act_type,
                "apply_bn_for_resampling": self.apply_bn_for_resampling,
                "conv_after_downsample": self.conv_after_downsample,
                "conv_bn_act_pattern": self.conv_bn_act_pattern,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class FPNCells(layers.Layer):
    """The full BiFPN: ``fpn_cell_repeats`` stacked cells of ``FNode``s over levels
    ``min_level``..``max_level``. Takes and returns a list of ``max_level-min_level+1``
    feature maps (all at ``fpn_num_filters`` channels), each level at its own
    resolution."""

    def __init__(
        self,
        min_level,
        max_level,
        fpn_num_filters,
        fpn_cell_repeats,
        weight_method,
        act_type,
        apply_bn_for_resampling,
        conv_after_downsample,
        conv_bn_act_pattern,
        name="fpn_cells",
    ):
        super().__init__(name=name)
        self.min_level = min_level
        self.max_level = max_level
        self.fpn_num_filters = fpn_num_filters
        self.fpn_cell_repeats = fpn_cell_repeats
        self.weight_method = weight_method
        self.act_type = act_type
        self.nodes_cfg = bifpn_nodes(min_level, max_level)
        self.cells = []
        for rep in range(fpn_cell_repeats):
            fnodes = [
                FNode(
                    cfg["feat_level"] - min_level,
                    cfg["inputs_offsets"],
                    fpn_num_filters,
                    weight_method,
                    act_type,
                    apply_bn_for_resampling,
                    conv_after_downsample,
                    conv_bn_act_pattern,
                    name=f"cell_{rep}_fnode{i}",
                )
                for i, cfg in enumerate(self.nodes_cfg)
            ]
            self.cells.append(fnodes)

    def call(self, feats, level_sizes, training=False):
        """``level_sizes``: list of (height, width) per level (min_level..max_level)."""
        for fnodes in self.cells:
            cur = list(feats)
            for fnode, cfg in zip(fnodes, self.nodes_cfg):
                lvl = cfg["feat_level"] - self.min_level
                th, tw = level_sizes[lvl]
                cur = fnode(cur, target_height=th, target_width=tw, training=training)
            # keep the last node produced at each level (reverse-scan, like Google's).
            feats = []
            for level in range(self.min_level, self.max_level + 1):
                for i, cfg in enumerate(reversed(self.nodes_cfg)):
                    if cfg["feat_level"] == level:
                        feats.append(cur[-1 - i])
                        break
        return feats


@keras.saving.register_keras_serializable(package="zeromodels")
class PredictionHead(layers.Layer):
    """Shared class or box head: ``repeats`` separable 3x3 convs, each with a
    per-level BatchNorm and activation (conv weights shared across levels, BN not),
    then a final separable 3x3 predictor with ``out_channels`` outputs per anchor."""

    def __init__(
        self,
        out_channels,
        num_filters,
        repeats,
        num_levels,
        act_type,
        head_name,
        min_level=3,
        predict_bias_init=0.0,
        name=None,
    ):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.repeats = repeats
        self.num_levels = num_levels
        self.act_type = act_type
        self.head_name = head_name
        self.min_level = min_level
        self.predict_bias_init = predict_bias_init
        self.convs = [
            sep_conv(num_filters, 3, use_bias=True, name=f"{head_name}-{i}")
            for i in range(repeats)
        ]
        # one BatchNorm per (repeat, level); conv weights are shared across levels.
        # BN names carry the absolute pyramid level (min_level + lvl), like Google's.
        self.bns = [
            [
                layers.BatchNormalization(name=f"{head_name}-{i}-bn-{min_level + lvl}")
                for lvl in range(num_levels)
            ]
            for i in range(repeats)
        ]
        self.predict = layers.SeparableConv2D(
            out_channels,
            3,
            padding="same",
            depth_multiplier=1,
            bias_initializer=keras.initializers.Constant(predict_bias_init),
            name=f"{head_name}-predict",
        )

    def call(self, feats, training=False):
        outputs = []
        for level_id in range(self.num_levels):
            x = feats[level_id]
            for i in range(self.repeats):
                x = self.convs[i](x)
                x = self.bns[i][level_id](x, training=training)
                x = act(x, self.act_type)
            outputs.append(self.predict(x))
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_channels": self.out_channels,
                "num_filters": self.num_filters,
                "repeats": self.repeats,
                "num_levels": self.num_levels,
                "act_type": self.act_type,
                "head_name": self.head_name,
                "min_level": self.min_level,
                "predict_bias_init": self.predict_bias_init,
            }
        )
        return config


def class_predict_bias():
    """Focal-loss prior bias for the class predictor: ``-log((1 - 0.01) / 0.01)``."""
    return -math.log((1 - 0.01) / 0.01)
