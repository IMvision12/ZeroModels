from typing import List, Optional, Union

import keras

from zeromodels.base import BaseProcessor
from zeromodels.models.owlvit.owlvit_image_processor import OwlViTImageProcessor
from zeromodels.models.owlvit.owlvit_tokenizer import OwlViTTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class OwlViTProcessor(BaseProcessor):
    """Composite processor that bundles an image processor and CLIP tokenizer.

    Load it by Hub repo id, which pulls the model's own ``tokenizer.json`` and
    ``zm_preprocessor.json``
    (``OwlViTProcessor.from_weights("zeromodels/owlvit-base-patch32")``), or pass
    pre-built ``tokenizer`` / ``image_processor`` components. Text queries are
    flattened across the batch and the model uses the per-row argmax of
    ``input_ids`` to pool, so padded queries (first token = pad id ``0``) are
    detected by the class predictor.

    Args:
        hf_id: Hub repo carrying the model's ``tokenizer.json`` (OwlViT shares
            CLIP's BPE vocab, hosted per-model on the zeromodels org).
        tokenizer / image_processor: Optional pre-built components.
    """

    TOKENIZER_CLS = OwlViTTokenizer
    IMAGE_PROCESSOR_CLS = OwlViTImageProcessor

    def __init__(
        self,
        hf_id: Optional[str] = None,
        tokenizer=None,
        image_processor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hf_id = hf_id
        self.tokenizer = tokenizer or OwlViTTokenizer(hf_id=hf_id)
        self.image_processor = image_processor or OwlViTImageProcessor()

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(hf_id=repo, **kwargs)

    # from_hub_repo("zeromodels/owlvit-...") is inherited from BaseProcessor: it
    # loads the tokenizer from the repo's tokenizer.json and the image processor
    # from its zm_preprocessor.json (two separate files, one each, like CLIP /
    # SigLIP), so no override is needed here.

    def get_config(self):
        config = super().get_config()
        config.update({"hf_id": self.hf_id})
        return config

    def call(
        self,
        text: Optional[Union[str, List[str], List[List[str]]]] = None,
        images=None,
    ):
        if text is None and images is None:
            raise ValueError("At least one of `text` or `images` must be provided.")

        out = {}

        if text is not None:
            if isinstance(text, str):
                flat = [text]
            elif (
                isinstance(text, (list, tuple))
                and len(text)
                and isinstance(text[0], (list, tuple))
            ):
                flat = [q for inner in text for q in inner]
            else:
                flat = list(text)

            text_enc = self.tokenizer(inputs=flat)
            out["input_ids"] = text_enc["input_ids"]
            out["attention_mask"] = text_enc["attention_mask"]

        if images is not None:
            out["pixel_values"] = self.image_processor(images)["pixel_values"]

        return out
