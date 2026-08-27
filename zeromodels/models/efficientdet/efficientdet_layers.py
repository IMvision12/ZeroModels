import math

import keras
import numpy as np
from keras import layers, ops

from zeromodels.utils.image_util import get_data_format

from .efficientdet_config import bifpn_nodes


def channel_axis(data_format):
    return 1 if data_format == "channels_first" else -1


def spatial_dims(shape, data_format):
    """(height, width) of a 4D feature-map shape under ``data_format``."""
    if data_format == "channels_first":
        return int(shape[2]), int(shape[3])
    return int(shape[1]), int(shape[2])


def act(x, act_type="swish"):
    if act_type == "swish":
        return ops.silu(x)
    if act_type == "relu":
        return ops.relu(x)
    return layers.Activation(act_type)(x)


def sep_conv(filters, kernel_size=3, use_bias=True, data_format=None, name=None):
    return layers.SeparableConv2D(
        filters,
        kernel_size=kernel_size,
        padding="same",
        depth_multiplier=1,
        use_bias=use_bias,
        data_format=data_format,
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
        data_format=None,
        name=None,
    ):
        super().__init__(name=name)
        self.target_channels = target_channels
        self.apply_bn = apply_bn
        self.conv_after_downsample = conv_after_downsample
        self.data_format = get_data_format(data_format)
        self.conv2d = None
        self.bn = None

    def build(self, input_shape):
        in_channels = input_shape[channel_axis(self.data_format)]
        if in_channels != self.target_channels:
            self.conv2d = layers.Conv2D(
                self.target_channels,
                1,
                padding="same",
                data_format=self.data_format,
                name="conv2d",
            )
            if self.apply_bn:
                self.bn = layers.BatchNormalization(
                    axis=channel_axis(self.data_format), name="bn"
                )
        self.built = True

    def apply_1x1(self, feat, training):
        if self.conv2d is not None:
            feat = self.conv2d(feat)
            if self.bn is not None:
                feat = self.bn(feat, training=training)
        return feat

    def call(self, feat, target_height, target_width, training=False):
        height, width = spatial_dims(feat.shape, self.data_format)
        if height > target_height and width > target_width:
            if not self.conv_after_downsample:
                feat = self.apply_1x1(feat, training)
            h_stride = (height - 1) // target_height + 1
            w_stride = (width - 1) // target_width + 1
            feat = layers.MaxPooling2D(
                pool_size=(h_stride + 1, w_stride + 1),
                strides=(h_stride, w_stride),
                padding="same",
                data_format=self.data_format,
            )(feat)
            if self.conv_after_downsample:
                feat = self.apply_1x1(feat, training)
        else:
            feat = self.apply_1x1(feat, training)
            if height < target_height or width < target_width:
                feat = ops.image.resize(
                    feat,
                    (target_height, target_width),
                    interpolation="nearest",
                    data_format=self.data_format,
                )
        return feat

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "target_channels": self.target_channels,
                "apply_bn": self.apply_bn,
                "conv_after_downsample": self.conv_after_downsample,
                "data_format": self.data_format,
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
        data_format=None,
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
        self.data_format = get_data_format(data_format)
        self.resamples = [
            EfficientDetResample(
                fpn_num_filters,
                apply_bn=apply_bn_for_resampling,
                conv_after_downsample=conv_after_downsample,
                data_format=self.data_format,
                name=f"resample_{i}_{off}",
            )
            for i, off in enumerate(self.inputs_offsets)
        ]
        self.conv = sep_conv(
            fpn_num_filters,
            3,
            use_bias=not conv_bn_act_pattern,
            data_format=self.data_format,
            name="conv",
        )
        self.bn = layers.BatchNormalization(
            axis=channel_axis(self.data_format), name="bn"
        )

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
                "data_format": self.data_format,
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
        data_format=None,
        name="fpn_cells",
    ):
        super().__init__(name=name)
        self.min_level = min_level
        self.max_level = max_level
        self.fpn_num_filters = fpn_num_filters
        self.fpn_cell_repeats = fpn_cell_repeats
        self.weight_method = weight_method
        self.act_type = act_type
        self.data_format = get_data_format(data_format)
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
                    data_format=self.data_format,
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
        data_format=None,
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
        self.data_format = get_data_format(data_format)
        self.convs = [
            sep_conv(
                num_filters,
                3,
                use_bias=True,
                data_format=self.data_format,
                name=f"{head_name}-{i}",
            )
            for i in range(repeats)
        ]
        # one BatchNorm per (repeat, level); conv weights are shared across levels.
        # BN names carry the absolute pyramid level (min_level + lvl), like Google's.
        self.bns = [
            [
                layers.BatchNormalization(
                    axis=channel_axis(self.data_format),
                    name=f"{head_name}-{i}-bn-{min_level + lvl}",
                )
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
            data_format=self.data_format,
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
                "data_format": self.data_format,
            }
        )
        return config


def class_predict_bias():
    """Focal-loss prior bias for the class predictor: ``-log((1 - 0.01) / 0.01)``."""
    return -math.log((1 - 0.01) / 0.01)


def feat_sizes(image_size, max_level):
    """Per-level (height, width) of the feature pyramid, ceil-halving from the
    input size (index 0 = input). Matches Google's ``utils.get_feat_sizes``."""
    if isinstance(image_size, int):
        h = w = image_size
    else:
        h, w = image_size
    sizes = [(h, w)]
    for _ in range(1, max_level + 1):
        h, w = (h - 1) // 2 + 1, (w - 1) // 2 + 1
        sizes.append((h, w))
    return sizes


def generate_anchor_boxes(
    min_level, max_level, num_scales, aspect_ratios, anchor_scale, image_size
):
    """Multiscale anchor boxes as ``(N, 4)`` in ``[ymin, xmin, ymax, xmax]`` pixel
    coordinates, ordered position-major then (octave, aspect)-minor per level, then
    concatenated over levels. A direct port of Google AutoML ``anchors.Anchors``."""
    if isinstance(image_size, int):
        image_h = image_w = image_size
    else:
        image_h, image_w = image_size
    sizes = feat_sizes(image_size, max_level)
    if isinstance(anchor_scale, (list, tuple)):
        anchor_scales = list(anchor_scale)
    else:
        anchor_scales = [anchor_scale] * (max_level - min_level + 1)

    boxes_all = []
    for level in range(min_level, max_level + 1):
        stride_h = sizes[0][0] / float(sizes[level][0])
        stride_w = sizes[0][1] / float(sizes[level][1])
        boxes_level = []
        for scale_octave in range(num_scales):
            for aspect in aspect_ratios:
                base_x = anchor_scales[level - min_level] * stride_w * (
                    2 ** (scale_octave / float(num_scales))
                )
                base_y = anchor_scales[level - min_level] * stride_h * (
                    2 ** (scale_octave / float(num_scales))
                )
                aspect_x = math.sqrt(aspect)
                aspect_y = 1.0 / aspect_x
                half_x = base_x * aspect_x / 2.0
                half_y = base_y * aspect_y / 2.0
                x = np.arange(stride_w / 2, image_w, stride_w)
                y = np.arange(stride_h / 2, image_h, stride_h)
                xv, yv = np.meshgrid(x, y)
                xv, yv = xv.reshape(-1), yv.reshape(-1)
                boxes = np.vstack(
                    (yv - half_y, xv - half_x, yv + half_y, xv + half_x)
                )
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))
        boxes_level = np.concatenate(boxes_level, axis=1)
        boxes_all.append(boxes_level.reshape([-1, 4]))
    return np.vstack(boxes_all).astype("float32")


def decode_box_outputs(pred_boxes, anchor_boxes):
    """Invert the anchor box regression: ``(ty, tx, th, tw)`` relative to anchors ->
    absolute ``[ymin, xmin, ymax, xmax]``. ``anchor_boxes`` are ``[ymin, xmin, ymax,
    xmax]``. Matches Google AutoML ``anchors.decode_box_outputs``."""
    ycenter_a = (anchor_boxes[..., 0] + anchor_boxes[..., 2]) / 2
    xcenter_a = (anchor_boxes[..., 1] + anchor_boxes[..., 3]) / 2
    ha = anchor_boxes[..., 2] - anchor_boxes[..., 0]
    wa = anchor_boxes[..., 3] - anchor_boxes[..., 1]
    ty, tx, th, tw = (
        pred_boxes[..., 0],
        pred_boxes[..., 1],
        pred_boxes[..., 2],
        pred_boxes[..., 3],
    )
    w = ops.exp(tw) * wa
    h = ops.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    return ops.stack(
        [
            ycenter - h / 2.0,
            xcenter - w / 2.0,
            ycenter + h / 2.0,
            xcenter + w / 2.0,
        ],
        axis=-1,
    )


@keras.saving.register_keras_serializable(package="zeromodels")
class DecodeBoxes(layers.Layer):
    """Turn raw per-anchor box regressions ``(B, N, 4)`` into absolute
    ``[ymin, xmin, ymax, xmax]`` boxes using pre-generated anchors held as a
    non-trainable constant."""

    def __init__(
        self,
        min_level,
        max_level,
        num_scales,
        aspect_ratios,
        anchor_scale,
        image_size,
        name="decode_boxes",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.min_level = min_level
        self.max_level = max_level
        self.num_scales = num_scales
        self.aspect_ratios = tuple(aspect_ratios)
        self.anchor_scale = anchor_scale
        self.image_size = image_size
        self.anchors_np = generate_anchor_boxes(
            min_level, max_level, num_scales, aspect_ratios, anchor_scale, image_size
        )

    def build(self, input_shape):
        self.built = True

    def call(self, box_outputs):
        # Anchors are a fixed function of the config, so they are baked into the graph
        # as a constant rather than stored as a weight. This keeps EfficientDetDetect's
        # weight set identical to EfficientDetModel's, so both load one hosted file.
        anchors = ops.convert_to_tensor(self.anchors_np, dtype=box_outputs.dtype)
        return decode_box_outputs(box_outputs, anchors)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "min_level": self.min_level,
                "max_level": self.max_level,
                "num_scales": self.num_scales,
                "aspect_ratios": self.aspect_ratios,
                "anchor_scale": self.anchor_scale,
                "image_size": self.image_size,
            }
        )
        return config


def iou_against(box, boxes):
    """IoU of one ``[ymin, xmin, ymax, xmax]`` box against an array of boxes."""
    ymin = np.maximum(box[0], boxes[:, 0])
    xmin = np.maximum(box[1], boxes[:, 1])
    ymax = np.minimum(box[2], boxes[:, 2])
    xmax = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, ymax - ymin) * np.maximum(0.0, xmax - xmin)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-8)


def greedy_nms(boxes, scores, iou_threshold, max_output):
    """Greedy hard NMS. Returns kept indices (highest score first)."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_output:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        overlap = iou_against(boxes[i], boxes[rest])
        order = rest[overlap <= iou_threshold]
    return keep


@keras.saving.register_keras_serializable(package="zeromodels")
class EfficientDetNMS(layers.Layer):
    """Hard non-max suppression over decoded detections.

    Given decoded boxes ``(B, N, 4)`` in ``[ymin, xmin, ymax, xmax]`` and per-class
    sigmoid scores ``(B, N, num_classes)``, returns a padded ``(B, max_detections, 6)``
    tensor of ``[ymin, xmin, ymax, xmax, score, class_id]`` (rows past the detection
    count are zero).

    With ``class_agnostic=True`` (the default, matching Google AutoML's
    ``postprocess_global``) each anchor keeps only its single highest-scoring class
    and one NMS runs across all classes together, so a single object cannot yield two
    boxes under different labels (e.g. a dog also reported as a cat). With
    ``class_agnostic=False`` NMS runs independently per class
    (``postprocess_per_class``). Runs eagerly (detection post-processing), applied
    outside the symbolic graph."""

    def __init__(
        self,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        pre_nms_top_k=5000,
        class_agnostic=True,
        name="nms",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.pre_nms_top_k = pre_nms_top_k
        self.class_agnostic = class_agnostic

    def top_k_prefilter(self, boxes, scores):
        """Keep the ``pre_nms_top_k`` anchors with the highest best-class score."""
        best = scores.max(axis=-1)
        if best.shape[0] > self.pre_nms_top_k:
            top = np.argpartition(-best, self.pre_nms_top_k)[: self.pre_nms_top_k]
            return boxes[top], scores[top]
        return boxes, scores

    def detections_global(self, boxes, scores):
        """Class-agnostic: one best class per anchor, then a single NMS."""
        class_ids = scores.argmax(axis=-1)
        class_scores = scores.max(axis=-1)
        mask = class_scores > self.score_threshold
        if not mask.any():
            return []
        bx, sc, cl = boxes[mask], class_scores[mask], class_ids[mask]
        keep = greedy_nms(bx, sc, self.iou_threshold, self.max_detections)
        return [
            (bx[k][0], bx[k][1], bx[k][2], bx[k][3], sc[k], float(cl[k])) for k in keep
        ]

    def detections_per_class(self, boxes, scores):
        """Independent NMS per class."""
        dets = []
        for c in range(scores.shape[-1]):
            cls_scores = scores[:, c]
            mask = cls_scores > self.score_threshold
            if not mask.any():
                continue
            bx, sc = boxes[mask], cls_scores[mask]
            keep = greedy_nms(bx, sc, self.iou_threshold, self.max_detections)
            dets.extend(
                (bx[k][0], bx[k][1], bx[k][2], bx[k][3], sc[k], float(c)) for k in keep
            )
        return dets

    def call(self, boxes, scores):
        boxes = ops.convert_to_numpy(ops.convert_to_tensor(boxes))
        scores = ops.convert_to_numpy(ops.convert_to_tensor(scores))
        batch = boxes.shape[0]
        out = np.zeros((batch, self.max_detections, 6), dtype="float32")
        for b in range(batch):
            bx, sc = self.top_k_prefilter(boxes[b], scores[b])
            if self.class_agnostic:
                dets = self.detections_global(bx, sc)
            else:
                dets = self.detections_per_class(bx, sc)
            if dets:
                dets.sort(key=lambda d: d[4], reverse=True)
                dets = dets[: self.max_detections]
                out[b, : len(dets)] = np.array(dets, dtype="float32")
        return ops.convert_to_tensor(out)

    def compute_output_shape(self, boxes_shape, scores_shape):
        return (boxes_shape[0], self.max_detections, 6)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "iou_threshold": self.iou_threshold,
                "score_threshold": self.score_threshold,
                "max_detections": self.max_detections,
                "pre_nms_top_k": self.pre_nms_top_k,
                "class_agnostic": self.class_agnostic,
            }
        )
        return config
