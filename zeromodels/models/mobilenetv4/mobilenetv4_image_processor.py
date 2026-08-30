import keras

from zeromodels.base import BaseImageProcessor


@keras.saving.register_keras_serializable(package="zeromodels")
class MobileNetV4ImageProcessor(BaseImageProcessor):
    """Image processor for MobileNetV4: resize + rescale + normalize (ImageNet 0.485/0.229).

    A thin subclass of :class:`BaseImageProcessor`'s generalized classification
    pipeline; only the resize size and/or normalization statistics differ from the
    defaults. See :class:`BaseImageProcessor` for the full constructor and pipeline.
    """
