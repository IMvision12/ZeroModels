import keras

from zeromodels.base import BaseImageProcessor


@keras.saving.register_keras_serializable(package="zeromodels")
class InceptionV3ImageProcessor(BaseImageProcessor):
    """Image processor for InceptionV3: resize + rescale + normalize (inception 0.5/0.5).

    A thin subclass of :class:`BaseImageProcessor`'s generalized classification
    pipeline; only the resize size and/or normalization statistics differ from the
    defaults. See :class:`BaseImageProcessor` for the full constructor and pipeline.
    """

    DEFAULT_SIZE = 299
    DEFAULT_MEAN = BaseImageProcessor.IMAGENET_INCEPTION_MEAN
    DEFAULT_STD = BaseImageProcessor.IMAGENET_INCEPTION_STD
