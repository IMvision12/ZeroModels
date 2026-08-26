import keras

from zeromodels.models.mobilevit.mobilevit_image_processor import (
    MobileViTImageProcessor,
)


@keras.saving.register_keras_serializable(package="zeromodels")
class MobileViTV2ImageProcessor(MobileViTImageProcessor):
    """Preprocess images for MobileViTV2 inference.

    Functionally identical to :class:`MobileViTImageProcessor`,
    The reference ships a single ``MobileViTImageProcessor`` class that
    serves both V1 and V2 checkpoints. The two only differ in default
    sizes (carried over from the parent), which are loaded from the source
    config if you instantiate via ``from_pretrained`` and otherwise match
    the V1 classification defaults: ``size["shortest_edge"]=288`` /
    ``crop_size=256x256``. For V2 segmentation, override with
    ``size={"shortest_edge": 544}`` and ``crop_size={"height": 512,
    "width": 512}``.
    """

    pass
