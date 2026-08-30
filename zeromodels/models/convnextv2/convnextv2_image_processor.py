import keras

from zeromodels.base import BaseImageProcessor


@keras.saving.register_keras_serializable(package="zeromodels")
class ConvNeXtV2ImageProcessor(BaseImageProcessor):
    """Image processor for ConvNeXtV2: resize + rescale + normalize (ImageNet 0.485/0.229).

    A thin subclass of :class:`BaseImageProcessor`'s generalized classification
    pipeline (ImageNet defaults). See :class:`BaseImageProcessor` for the full
    constructor and pipeline.
    """
