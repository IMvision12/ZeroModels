from typing import Optional

import keras

from zeromodels.models.siglip.siglip_image_processor import SigLIPImageProcessor


@keras.saving.register_keras_serializable(package="zeromodels")
class SigLIP2ImageProcessor(SigLIPImageProcessor):
    """Image processor for SigLIP 2 models.

    Identical preprocessing to :class:`SigLIPImageProcessor` (resize +
    center crop + normalize with mean=std=0.5). Exposed as a separate
    class so SigLIP 2 code paths can import a SigLIP2-named processor.
    """

    def __init__(
        self,
        image_resolution: int = 224,
        mean=SigLIPImageProcessor.IMAGENET_INCEPTION_MEAN,
        std=SigLIPImageProcessor.IMAGENET_INCEPTION_STD,
        do_center_crop: bool = True,
        do_normalize: bool = True,
        do_resize: bool = True,
        data_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            image_resolution=image_resolution,
            mean=mean,
            std=std,
            do_center_crop=do_center_crop,
            do_normalize=do_normalize,
            do_resize=do_resize,
            data_format=data_format,
            **kwargs,
        )
