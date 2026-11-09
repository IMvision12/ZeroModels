from typing import Dict, List, Optional, Tuple

import keras
import numpy as np

from zeromodels.models.detr.detr_image_processor import DETRImageProcessor
from zeromodels.utils.labels_util import COCO_91_CLASSES


@keras.saving.register_keras_serializable(package="zeromodels")
class ConditionalDETRImageProcessor(DETRImageProcessor):
    """Preprocess images for Conditional DETR and decode its detections.

    Preprocessing is DETR's (see :class:`DETRImageProcessor`): an
    aspect-preserving resize (shortest edge 800, longest 1333), rescale to
    ``[0, 1]`` and ImageNet normalization. Post-processing differs: Conditional
    DETR scores every class with a sigmoid (no "no-object" class), so
    :meth:`post_process_object_detection` keeps the ``top_k`` highest
    (query, class) scores across all queries, as the reference does.
    """

    def post_process_object_detection(
        self, outputs, threshold=0.5, target_sizes=None, label_names=None, top_k=100
    ):
        return sigmoid_topk_post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
            label_names=label_names,
            top_k=top_k,
        )


def sigmoid_topk_post_process_object_detection(
    outputs: Dict,
    threshold: float = 0.5,
    target_sizes: Optional[List[Tuple[int, int]]] = None,
    label_names: Optional[List[str]] = None,
    top_k: int = 100,
) -> List[Dict[str, np.ndarray]]:
    """Decode sigmoid-scored DETR outputs (Conditional / Deformable DETR).

    Takes the ``top_k`` highest sigmoid scores over the flattened
    ``(num_queries, num_classes)`` grid of each image, converts the matching
    boxes from normalized ``(cx, cy, w, h)`` to ``(x_min, y_min, x_max, y_max)``
    (scaled to ``target_sizes`` ``(height, width)`` when given), and keeps the
    detections scoring above ``threshold``.

    Returns:
        One dict per image with ``scores``, ``labels``, ``label_names`` and
        ``boxes``, in descending score order.
    """
    logits = keras.ops.convert_to_numpy(outputs["logits"]).astype("float32")
    boxes = keras.ops.convert_to_numpy(outputs["pred_boxes"]).astype("float32")
    batch_size, _, num_classes = logits.shape
    names = label_names if label_names is not None else COCO_91_CLASSES

    probs = 1.0 / (1.0 + np.exp(-logits))
    flat = probs.reshape(batch_size, -1)
    k = min(top_k, flat.shape[1])

    cx, cy, w, h = np.moveaxis(boxes, -1, 0)
    corners = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)

    results = []
    for i in range(batch_size):
        order = np.argsort(-flat[i], kind="stable")[:k]
        scores = flat[i][order]
        labels = order % num_classes
        image_boxes = corners[i][order // num_classes]
        if target_sizes is not None:
            img_h, img_w = target_sizes[i]
            image_boxes = image_boxes * np.array(
                [img_w, img_h, img_w, img_h], dtype=np.float32
            )
        keep = scores > threshold
        labels = labels[keep]
        results.append(
            {
                "scores": scores[keep],
                "labels": labels,
                "label_names": [
                    names[label] if label < len(names) else f"class_{label}"
                    for label in labels
                ],
                "boxes": image_boxes[keep],
            }
        )
    return results
