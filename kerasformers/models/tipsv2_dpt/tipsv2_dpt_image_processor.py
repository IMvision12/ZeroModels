import keras

from kerasformers.models.tipsv2.tipsv2_image_processor import Tipsv2ImageProcessor


@keras.saving.register_keras_serializable(package="kerasformers")
class Tipsv2DptImageProcessor(Tipsv2ImageProcessor):
    """Image processor for TIPSv2-DPT.

    Identical preprocessing to :class:`Tipsv2ImageProcessor`: resize to
    ``image_resolution`` (bilinear) and rescale to ``[0, 1]``, with no mean/std
    normalization (the backbone consumes ``[0, 1]`` pixels). Defaults to 448.
    """
