import keras

from zeromodels.models.conditional_detr.conditional_detr_image_processor import (
    sigmoid_topk_post_process_object_detection,
)
from zeromodels.models.detr.detr_image_processor import DETRImageProcessor


@keras.saving.register_keras_serializable(package="zeromodels")
class DeformableDETRImageProcessor(DETRImageProcessor):
    """Preprocess images for Deformable DETR and decode its detections.

    Preprocessing is DETR's (see :class:`DETRImageProcessor`): an
    aspect-preserving resize (shortest edge 800, longest 1333), rescale to
    ``[0, 1]`` and ImageNet normalization. Deformable DETR scores every class
    with a sigmoid, so :meth:`post_process_object_detection` keeps the ``top_k``
    highest (query, class) scores across all queries, as the reference does.
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
