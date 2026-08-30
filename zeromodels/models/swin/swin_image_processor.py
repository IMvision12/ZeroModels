import keras

from zeromodels.base import BaseImageProcessor


@keras.saving.register_keras_serializable(package="zeromodels")
class SwinImageProcessor(BaseImageProcessor):
    """Image processor for Swin: resize + rescale + normalize (ImageNet 0.485/0.229).

    A thin subclass of :class:`BaseImageProcessor`'s generalized classification
    pipeline; only the resize size and/or normalization statistics differ from the
    defaults. See :class:`BaseImageProcessor` for the full constructor and pipeline.
    """
