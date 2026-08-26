import math

import numpy as np
from keras import ops

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None


def inverse_sigmoid(x, eps=1e-3):
    """Numerically stable inverse sigmoid (logit) function.

    Clips input to ``[eps, 1-eps]`` before computing
    ``log(x / (1 - x))``.

    Args:
        x: Input tensor with values in ``(0, 1)``.
        eps (float): Clamping epsilon. Defaults to ``1e-3``.

    Returns:
        Logit tensor with the same shape as ``x``.
    """
    x = ops.clip(x, eps, 1.0 - eps)
    return ops.log(x / (1.0 - x))


def box_cxcywh_to_xyxy(boxes):
    """Convert boxes from center-size ``(cx, cy, w, h)`` to corner
    ``(x0, y0, x1, y1)`` format.

    Args:
        boxes: Tensor or array with last dimension 4.

    Returns:
        Boxes in ``(x0, y0, x1, y1)`` format.
    """
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return ops.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=-1)


def box_xyxy_to_cxcywh(boxes):
    """Convert boxes from corner ``(x0, y0, x1, y1)`` to center-size
    ``(cx, cy, w, h)`` format.

    Args:
        boxes: Tensor or array with last dimension 4.

    Returns:
        Boxes in ``(cx, cy, w, h)`` format.
    """
    x0, y0, x1, y1 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return ops.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], axis=-1)


def sigmoid(x):
    """Numerically stable sigmoid using numpy.

    Clips input to ``[-88, 88]`` to prevent overflow.

    Args:
        x: Numpy array.

    Returns:
        Sigmoid-activated array in ``(0, 1)``.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def scale_boxes(boxes, target_size):
    """Scale normalized ``[0, 1]`` boxes to absolute pixel coordinates.

    Assumes boxes are in ``(x0, y0, x1, y1)`` format normalized to
    the model input size.

    Args:
        boxes: ``(N, 4)`` numpy array of normalized boxes.
        target_size: ``(height, width)`` tuple of the original image.

    Returns:
        ``(N, 4)`` numpy array in absolute pixel coordinates.
    """
    h, w = target_size
    return boxes * np.array([w, h, w, h], dtype=np.float32)


def compute_scores(pred_logits, presence_logits):
    """Compute detection scores from logits and optional presence logits.

    Applies sigmoid to ``pred_logits`` and multiplies by
    ``sigmoid(presence_logits)`` if provided (from the last decoder
    layer).

    Args:
        pred_logits: ``(B, Q)`` or ``(Q,)`` classification logits.
        presence_logits: ``(B, 1)`` or ``(num_layers,)`` presence
            logits, or ``None``.

    Returns:
        ``(B, Q)`` or ``(Q,)`` numpy array of final scores.
    """
    scores = sigmoid(pred_logits)
    if presence_logits is not None:
        presence = np.asarray(presence_logits)
        if presence.ndim == 1:
            p = sigmoid(presence[-1:]).reshape(1, 1)
        else:
            p = sigmoid(presence)
        scores = scores * p
    return scores


def rotate_pairwise(x):
    """Rotate adjacent pairs for rotary position embedding.

    Transforms ``[..., x0, x1, x2, x3, ...]`` into
    ``[..., -x1, x0, -x3, x2, ...]``.

    Args:
        x: Tensor with even-sized last dimension.

    Returns:
        Pairwise-rotated tensor with the same shape.
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = ops.stack([-x2, x1], axis=-1)
    shape = ops.shape(x)
    return ops.reshape(rotated, shape)


def apply_rotary_pos_emb_2d(q, k, cos, sin):
    """Apply 2-D rotary position embeddings to queries and keys.

    Args:
        q: Query tensor ``(B, heads, seq, head_dim)``.
        k: Key tensor ``(B, heads, seq, head_dim)``.
        cos: Cosine frequencies ``(1, 1, seq, head_dim)``.
        sin: Sine frequencies ``(1, 1, seq, head_dim)``.

    Returns:
        Tuple of ``(q_rotated, k_rotated)`` with same shapes.
    """
    q_embed = q * cos + rotate_pairwise(q) * sin
    k_embed = k * cos + rotate_pairwise(k) * sin
    return q_embed, k_embed


def sine_encode_boxes(boxes, num_pos_feats=128, temperature=10000):
    """Sine positional encoding for bounding boxes.

    Encodes each of the 4 box coordinates ``(cx, cy, w, h)`` using
    interleaved sin/cos features in ``(y, x, w, h)`` order to match
    the reference implementation.

    Args:
        boxes: ``(B, Q, 4)`` tensor of boxes in ``(cx, cy, w, h)``.
        num_pos_feats (int): Features per coordinate. Defaults to ``128``.
        temperature (int): Frequency base. Defaults to ``10000``.

    Returns:
        ``(B, Q, 4 * num_pos_feats)`` positional encoding tensor.
    """
    scale = 2 * 3.141592653589793
    dim_t = ops.cast(ops.arange(num_pos_feats), dtype="float32")
    dim_t = temperature ** (2 * ops.floor(dim_t / 2) / num_pos_feats)

    def _encode_coord(coord):
        c = coord * scale
        c = ops.expand_dims(c, axis=-1) / dim_t
        c_sin = ops.sin(c[..., 0::2])
        c_cos = ops.cos(c[..., 1::2])
        half = num_pos_feats // 2
        parts = []
        for j in range(half):
            parts.append(c_sin[:, :, j : j + 1])
            parts.append(c_cos[:, :, j : j + 1])
        return ops.concatenate(parts, axis=-1)

    pos_y = _encode_coord(boxes[:, :, 1])
    pos_x = _encode_coord(boxes[:, :, 0])
    pos_w = _encode_coord(boxes[:, :, 2])
    pos_h = _encode_coord(boxes[:, :, 3])
    return ops.concatenate([pos_y, pos_x, pos_w, pos_h], axis=-1)


def compute_sine_pos_encoding(
    height,
    width,
    num_pos_feats,
    temperature=10000,
    normalize=True,
    data_format="channels_last",
):
    """2-D sine-cosine positional encoding grid.

    Generates a spatial positional encoding by computing interleaved
    sin/cos features along the height and width axes.

    Args:
        height (int): Grid height.
        width (int): Grid width.
        num_pos_feats (int): Number of features per spatial axis.
        temperature (int): Frequency base. Defaults to ``10000``.
        normalize (bool): Whether to normalize coordinates to
            ``[0, 2*pi]``. Defaults to ``True``.
        data_format (str): ``"channels_last"`` returns
            ``(1, H, W, 2*num_pos_feats)``; ``"channels_first"``
            returns ``(1, 2*num_pos_feats, H, W)``.

    Returns:
        Positional encoding tensor.
    """
    scale = 2 * math.pi
    ones = ops.ones((1, height, width), dtype="float32")
    y_embed = ops.cumsum(ones, axis=1)
    x_embed = ops.cumsum(ones, axis=2)

    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = ops.cast(ops.arange(num_pos_feats), dtype="float32")
    dim_t = temperature ** (2 * ops.floor(dim_t / 2) / num_pos_feats)

    pos_x = ops.expand_dims(x_embed, axis=-1) / dim_t
    pos_y = ops.expand_dims(y_embed, axis=-1) / dim_t

    pos_x_sin = ops.sin(pos_x[:, :, :, 0::2])
    pos_x_cos = ops.cos(pos_x[:, :, :, 1::2])
    pos_y_sin = ops.sin(pos_y[:, :, :, 0::2])
    pos_y_cos = ops.cos(pos_y[:, :, :, 1::2])

    pos_x = ops.reshape(
        ops.stack([pos_x_sin, pos_x_cos], axis=4),
        (1, height, width, num_pos_feats),
    )
    pos_y = ops.reshape(
        ops.stack([pos_y_sin, pos_y_cos], axis=4),
        (1, height, width, num_pos_feats),
    )

    pos = ops.concatenate([pos_y, pos_x], axis=-1)
    if data_format == "channels_first":
        pos = ops.transpose(pos, (0, 3, 1, 2))
    return pos


def resize_mask(mask, target_h, target_w):
    """Resize a single 2-D mask to target size using PIL bilinear.

    Args:
        mask: ``(H, W)`` float array with values in ``[0, 1]``.
        target_h (int): Target height.
        target_w (int): Target width.

    Returns:
        ``(target_h, target_w)`` float32 numpy array.
    """
    if Image is None:
        raise ImportError("PIL required for mask resizing")
    pil_m = Image.fromarray((mask * 255).astype(np.uint8))
    pil_m = pil_m.resize((target_w, target_h), Image.BILINEAR)
    return np.array(pil_m, dtype=np.float32) / 255.0


def resize_masks_batch(masks, target_h, target_w):
    """Resize a batch of masks to target size.

    Args:
        masks: ``(N, H, W)`` float array.
        target_h (int): Target height.
        target_w (int): Target width.

    Returns:
        ``(N, target_h, target_w)`` float32 numpy array.
    """
    if len(masks) == 0:
        return masks
    return np.stack([resize_mask(m, target_h, target_w) for m in masks])


COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
]


def draw_detections(image, results, title=""):
    """Draw detection boxes and scores on an image.

    Args:
        image: PIL Image.
        results: Dict with ``"scores"`` and ``"boxes"`` arrays.
        title (str): Optional title text drawn at top-left.

    Returns:
        PIL Image with drawn bounding boxes and score labels.
    """
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    for i, (score, box) in enumerate(zip(results["scores"], results["boxes"])):
        color = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - 12), f"{score:.2f}", fill=color)
    if title:
        draw.text((10, 10), title, fill=(255, 255, 255))
    return vis


def draw_instance_masks(image, results, title=""):
    """Draw instance segmentation masks, boxes, and scores on an image.

    Overlays each instance mask with a semi-transparent color and
    draws bounding boxes with score labels.

    Args:
        image: PIL Image.
        results: Dict with ``"scores"``, ``"boxes"``, and ``"masks"``
            arrays.
        title (str): Optional title text drawn at top-left.

    Returns:
        PIL Image with colored mask overlays and bounding boxes.
    """
    vis = np.array(image, dtype=np.float32)
    for i in range(len(results["scores"])):
        mask = np.asarray(results["masks"][i])
        color = np.array(COLORS[i % len(COLORS)], dtype=np.float32)
        vis[mask > 0] = vis[mask > 0] * 0.5 + color * 0.5

    vis = Image.fromarray(vis.astype(np.uint8))
    draw = ImageDraw.Draw(vis)
    for i, (score, box) in enumerate(zip(results["scores"], results["boxes"])):
        color = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, y1 - 12), f"{score:.2f}", fill=color)
    if title:
        draw.text((10, 10), title, fill=(255, 255, 255))
    return vis


def draw_semantic_mask(image, mask, title="", color=(0, 200, 255)):
    """Draw a semantic segmentation mask overlay on an image.

    Overlays the binary mask region with a semi-transparent color and
    displays the coverage percentage.

    Args:
        image: PIL Image.
        mask: ``(H, W)`` binary mask array.
        title (str): Category label for the mask.
        color (tuple): RGB color for the overlay.
            Defaults to ``(0, 200, 255)``.

    Returns:
        PIL Image with the mask overlay and coverage label.
    """
    vis = np.array(image, dtype=np.float32)
    mask = np.asarray(mask)
    vis[mask > 0] = vis[mask > 0] * 0.4 + np.array(color, dtype=np.float32) * 0.6

    vis = Image.fromarray(vis.astype(np.uint8))
    draw = ImageDraw.Draw(vis)
    pct = 100 * mask.sum() / mask.size
    label = f"{title} ({pct:.1f}%)" if title else f"{pct:.1f}%"
    draw.text((10, 10), label, fill=(255, 255, 255))
    return vis
