from __future__ import annotations

from typing import Optional, Sequence, Tuple

import keras
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "plot_detections",
    "plot_segmentation",
    "plot_depth",
    "plot_sam_masks",
]


def get_axes(ax, figsize=(8, 8)):
    if ax is not None:
        return ax
    _, ax = plt.subplots(figsize=figsize)
    return ax


def to_numpy(x):
    """Convert any backend tensor or array-like input to a numpy array.

    Lets callers pass torch / tf / jax tensors from model outputs directly
    into the viz helpers without manually moving them to CPU first.
    """
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    return keras.ops.convert_to_numpy(x)


def plot_image(image, ax=None, title: Optional[str] = None, figsize=(8, 8)):
    """Display an image on an axis with the axes hidden.

    Args:
        image: Any array-like that ``matplotlib.pyplot.imshow`` accepts.
        ax: Optional existing ``matplotlib.axes.Axes``. A new figure is
            created when ``None``.
        title: Optional title shown above the axis.
        figsize: Used only when a new figure is created.
    """
    ax = get_axes(ax, figsize=figsize)
    ax.imshow(to_numpy(image))
    if title:
        ax.set_title(title)
    ax.axis("off")
    return ax


def plot_boxes(
    boxes,
    labels: Optional[Sequence] = None,
    scores: Optional[Sequence[float]] = None,
    ax=None,
    color: str = "red",
    linewidth: float = 2.0,
    fontsize: int = 9,
):
    """Draw ``xyxy`` bounding boxes on an axis.

    Args:
        boxes: ``(N, 4)`` array-like of ``[x0, y0, x1, y1]`` in pixel space.
        labels: Optional per-box label strings.
        scores: Optional per-box confidence scores (rendered to 2 decimals).
        ax: Target axis.
        color: Edge color for the rectangles.
        linewidth: Rectangle edge width.
        fontsize: Font size for label text.
    """
    ax = get_axes(ax)
    boxes = to_numpy(boxes).reshape(-1, 4)
    if scores is not None:
        scores = to_numpy(scores).reshape(-1)

    for i, (x0, y0, x1, y1) in enumerate(boxes):
        rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        text_parts = []
        if labels is not None:
            text_parts.append(str(labels[i]))
        if scores is not None:
            text_parts.append(f"{float(scores[i]):.2f}")
        if text_parts:
            ax.text(
                x0,
                max(y0 - 4, 0),
                " ".join(text_parts),
                color="white",
                fontsize=fontsize,
                bbox={"facecolor": color, "edgecolor": "none", "pad": 1},
            )

    return ax


def plot_masks(
    masks,
    ax=None,
    colors=None,
    alpha: float = 0.5,
):
    """Overlay one or more binary masks on an axis.

    Args:
        masks: ``(H, W)`` single mask or ``(N, H, W)`` stack of binary masks
            (boolean or 0/1 int/float).
        ax: Target axis.
        colors: Optional list of per-mask RGB(A) colors. Defaults to the
            ``tab20`` colormap cycled across masks.
        alpha: Fill transparency for non-background pixels.
    """
    ax = get_axes(ax)
    masks = to_numpy(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise ValueError(
            f"Expected masks of shape (H,W) or (N,H,W), got {masks.shape}."
        )

    n, h, w = masks.shape
    if colors is None:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(n)]

    for mask, color in zip(masks, colors):
        rgb = np.asarray(color)[:3]
        overlay = np.zeros((h, w, 4), dtype=np.float32)
        active = mask > 0
        overlay[active, :3] = rgb
        overlay[active, 3] = alpha
        ax.imshow(overlay)

    return ax


def plot_points(
    points,
    labels=None,
    ax=None,
    pos_color: str = "lime",
    neg_color: str = "red",
    marker_size: int = 120,
):
    """Overlay SAM-style point prompts.

    Args:
        points: ``(N, 2)`` array of ``(x, y)`` pixel coordinates.
        labels: Optional ``(N,)`` array where ``1`` is a positive (foreground)
            prompt and ``0`` is a negative (background) prompt. Defaults to
            all-positive.
        ax: Target axis.
        pos_color: Marker color for positive points.
        neg_color: Marker color for negative points.
        marker_size: Scatter marker size.
    """
    ax = get_axes(ax)
    points = to_numpy(points).reshape(-1, 2)
    if labels is None:
        labels = np.ones(len(points), dtype=int)
    else:
        labels = to_numpy(labels).reshape(-1)

    pos = points[labels == 1]
    neg = points[labels == 0]

    if len(pos):
        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            c=pos_color,
            s=marker_size,
            marker="*",
            edgecolors="black",
            linewidths=1.0,
        )
    if len(neg):
        ax.scatter(
            neg[:, 0],
            neg[:, 1],
            c=neg_color,
            s=marker_size,
            marker="*",
            edgecolors="black",
            linewidths=1.0,
        )
    return ax


def overlay_depth(
    depth,
    cmap: str = "inferno",
    ax=None,
    title: Optional[str] = None,
    figsize=(8, 8),
):
    """Display a depth map with min-max normalization and a colormap.

    Args:
        depth: ``(H, W)`` array of depth values.
        cmap: Matplotlib colormap name.
        ax: Target axis.
        title: Optional title.
        figsize: Used only when a new figure is created.
    """
    ax = get_axes(ax, figsize=figsize)
    depth = to_numpy(depth).astype(np.float32)
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax > dmin:
        depth = (depth - dmin) / (dmax - dmin)
    ax.imshow(depth, cmap=cmap)
    if title:
        ax.set_title(title)
    ax.axis("off")
    return ax


def plot_detections(
    image,
    boxes,
    labels: Optional[Sequence] = None,
    scores: Optional[Sequence[float]] = None,
    ax=None,
    color: str = "red",
    linewidth: float = 2.0,
    fontsize: int = 9,
    title: Optional[str] = None,
    figsize=(10, 7),
):
    """Render an image with object-detection bounding boxes.

    Convenience wrapper around :func:`plot_image` + :func:`plot_boxes`.
    Accepts the unpacked outputs from any zeromodels detection
    post-processor (DETR, D-FINE, RT-DETR, RT-DETRv2, RF-DETR, SAM3
    detection branch).

    Args:
        image: ``(H, W, 3)`` array displayed as the background.
        boxes: ``(N, 4)`` ``[x0, y0, x1, y1]`` pixel-coordinate boxes
            from ``result["boxes"]``.
        labels: Optional per-box class names (``result["label_names"]``)
            or integer class IDs (``result["labels"]``).
        scores: Optional per-box confidence scores
            (``result["scores"]``).
        ax: Target axis. A new figure is created when ``None``.
        color: Edge color for the rectangles.
        linewidth: Rectangle edge width.
        fontsize: Label-text font size.
        title: Optional title.
        figsize: Used only when a new figure is created.

    Example:
        ```python
        result = processor.post_process_object_detection(
            outputs, threshold=0.5, target_sizes=[(h, w)],
        )[0]
        plot_detections(
            image, result["boxes"], result["label_names"], result["scores"],
        )
        ```
    """
    ax = get_axes(ax, figsize=figsize)
    plot_image(image, ax=ax, title=title)
    plot_boxes(
        boxes,
        labels=labels,
        scores=scores,
        ax=ax,
        color=color,
        linewidth=linewidth,
        fontsize=fontsize,
    )
    return ax


def plot_segmentation(
    image,
    segmentation,
    class_names: Optional[Sequence[str]] = None,
    ax=None,
    alpha: float = 0.55,
    cmap: str = "tab20",
    show_legend: bool = True,
    legend_top_k: int = 8,
    title: Optional[str] = None,
    figsize=(10, 7),
    seed: int = 42,
):
    """Render a class-id segmentation map on top of an image.

    Works for semantic, instance, and panoptic segmentation: any
    output where ``segmentation`` is an integer label map. Each unique
    label gets a deterministic color. Matches the visualization used in
    SegFormer / DeepLabV3 / EoMT docs.

    Args:
        image: ``(H, W, 3)`` array displayed as the background.
        segmentation: ``(H, W)`` integer array of class / instance IDs.
        class_names: Optional list mapping label index → name. When
            provided, the legend shows readable names; otherwise the
            label IDs are shown.
        ax: Target axis.
        alpha: Overlay opacity over the source image.
        cmap: Matplotlib colormap used to color labels. ``"tab20"`` is
            good for ≤20 classes; for larger label sets a random palette
            is used (deterministic via ``seed``).
        show_legend: Whether to draw a legend of the largest segments.
        legend_top_k: Cap on legend entries (sorted by area).
        title: Optional title.
        figsize: Used only when a new figure is created.
        seed: RNG seed for the color palette when the label space
            exceeds the colormap.

    Example:
        ```python
        result = processor.post_process_semantic_segmentation(
            outputs, target_size=(h, w),
        )
        plot_segmentation(image, result["segmentation"], result["class_names"])
        ```
    """
    ax = get_axes(ax, figsize=figsize)
    image = to_numpy(image).astype(np.uint8)
    segmentation = to_numpy(segmentation).astype(np.int64)

    unique = np.unique(segmentation)
    n_unique = int(unique.max()) + 1 if unique.size else 0

    palette_cmap = plt.get_cmap(cmap)
    if n_unique <= palette_cmap.N:
        colors = (
            np.array([palette_cmap(i % palette_cmap.N) for i in range(n_unique)])[:, :3]
            * 255
        ).astype(np.uint8)
    else:
        rng = np.random.default_rng(seed)
        colors = rng.integers(50, 220, size=(n_unique, 3), dtype=np.uint8)

    # Color the segmentation map; treat negative IDs (e.g. EoMT "no class") as
    # background: leave the source image visible there.
    valid = segmentation >= 0
    colored = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
    if valid.any():
        idx = np.clip(segmentation[valid], 0, n_unique - 1)
        colored[valid] = colors[idx]

    overlay = image.copy()
    if valid.any():
        overlay[valid] = (overlay[valid] * (1 - alpha) + colored[valid] * alpha).astype(
            np.uint8
        )

    ax.imshow(overlay)
    if title:
        ax.set_title(title)
    ax.axis("off")

    if show_legend and unique.size:
        # Sort by per-label pixel area descending, keep top-k.
        areas = [(int(c), int((segmentation == c).sum())) for c in unique if c >= 0]
        areas.sort(key=lambda t: -t[1])
        top = areas[:legend_top_k]

        def _name(label_id: int) -> str:
            if class_names is None:
                return f"class_{label_id}"
            if 0 <= label_id < len(class_names):
                return str(class_names[label_id])
            return f"class_{label_id}"

        patches_ = [
            plt.Rectangle((0, 0), 1, 1, fc=tuple(colors[c] / 255.0)) for c, _ in top
        ]
        names = [_name(c) for c, _ in top]
        ax.legend(patches_, names, loc="upper right", fontsize=10)

    return ax


def plot_depth(
    image,
    depth,
    side_by_side: bool = True,
    cmap: str = "inferno",
    ax=None,
    alpha: float = 0.55,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6),
):
    """Render a depth map alongside or overlaid on the source image.

    Args:
        image: ``(H, W, 3)`` source image.
        depth: ``(H, W)`` depth array (any range, min-max normalized
            internally).
        side_by_side: If True, returns a 2-axis figure (image | depth).
            If False, alpha-blends the colormapped depth onto the image
            on the supplied / created single axis.
        cmap: Matplotlib colormap for the depth map.
        ax: Target axis when ``side_by_side=False``. Ignored otherwise.
        alpha: Overlay opacity (only when ``side_by_side=False``).
        title: Title placed above the figure (or single axis).
        figsize: Figure size when a new figure is created.

    Returns:
        - ``side_by_side=True``: tuple ``(fig, (ax_image, ax_depth))``
        - ``side_by_side=False``: the single ``ax`` used.

    Example:
        ```python
        depth = depth_processor.post_process_depth_estimation(
            output, target_sizes=[(h, w)],
        )[0]["predicted_depth"]
        plot_depth(image, depth)
        ```
    """
    image_np = to_numpy(image).astype(np.uint8)
    depth_np = to_numpy(depth).astype(np.float32)
    if depth_np.ndim == 3 and depth_np.shape[0] == 1:
        depth_np = depth_np[0]
    if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
        depth_np = depth_np[..., 0]
    if depth_np.ndim != 2:
        raise ValueError(f"Expected depth of shape (H, W); got {depth_np.shape}.")

    if side_by_side:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        plot_image(image_np, ax=axes[0], title="Input")
        overlay_depth(depth_np, cmap=cmap, ax=axes[1], title="Depth")
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        return fig, axes

    ax = get_axes(ax, figsize=figsize)
    dmin, dmax = float(depth_np.min()), float(depth_np.max())
    if dmax > dmin:
        depth_norm = (depth_np - dmin) / (dmax - dmin)
    else:
        depth_norm = np.zeros_like(depth_np)
    cmap_obj = plt.get_cmap(cmap)
    depth_rgb = (cmap_obj(depth_norm)[..., :3] * 255).astype(np.uint8)
    overlay = (image_np * (1 - alpha) + depth_rgb * alpha).astype(np.uint8)
    ax.imshow(overlay)
    if title:
        ax.set_title(title)
    ax.axis("off")
    return ax


def plot_sam_masks(
    image,
    masks,
    scores: Optional[Sequence[float]] = None,
    points: Optional[Sequence] = None,
    point_labels: Optional[Sequence] = None,
    boxes: Optional[Sequence] = None,
    ax=None,
    alpha: float = 0.55,
    colors=None,
    title: Optional[str] = None,
    figsize=(10, 7),
):
    """Render SAM-family mask predictions with optional prompt overlays.

    Works for SAM, SAM 2, SAM 2 Video, and SAM 3 instance masks. Pass
    binary masks ``(N, H, W)`` already thresholded; optionally render
    the prompt point/box that produced them.

    Args:
        image: ``(H, W, 3)`` source image.
        masks: ``(H, W)`` single mask or ``(N, H, W)`` stack of binary
            masks (any truthy value is foreground).
        scores: Optional per-mask IoU / confidence scores. Annotated in
            the title when supplied as a single value, otherwise
            ignored at this level (use a multi-axis layout if you want
            per-mask scores).
        points: Optional ``(K, 2)`` array of prompt point coordinates
            in pixel space.
        point_labels: Optional ``(K,)`` array; ``1`` for foreground,
            ``0`` for background prompts. Defaults to all-foreground.
        boxes: Optional ``(M, 4)`` array of prompt boxes drawn in
            yellow.
        ax: Target axis.
        alpha: Mask overlay opacity.
        colors: Optional list of mask RGB(A) colors. Defaults to the
            ``tab20`` colormap.
        title: Optional title.
        figsize: Used only when a new figure is created.

    Example:
        ```python
        # SAM v1 single point prompt
        masks_full = sam_post_process_masks(
            outputs["pred_masks"],
            inputs["original_size"],
            inputs["reshaped_size"],
        )
        # pick best mask from the 3 multi-mask outputs
        best = keras.ops.convert_to_numpy(
            masks_full[0, 0, keras.ops.argmax(outputs["iou_scores"][0, 0])]
        )
        binary = (best > 0).astype("uint8")
        plot_sam_masks(image, binary, points=[[x, y]], point_labels=[1])
        ```
    """
    ax = get_axes(ax, figsize=figsize)
    plot_image(image, ax=ax, title=title)
    plot_masks(masks, ax=ax, colors=colors, alpha=alpha)
    if boxes is not None:
        plot_boxes(boxes, ax=ax, color="yellow", linewidth=2.0)
    if points is not None:
        plot_points(points, labels=point_labels, ax=ax)

    if title is None and scores is not None:
        scores_np = to_numpy(scores).reshape(-1)
        if scores_np.size == 1:
            ax.set_title(f"score: {float(scores_np[0]):.2f}")

    return ax
