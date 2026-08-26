from typing import List, Optional, Sequence, Union

import keras

from zeromodels.base import BaseProcessor
from zeromodels.models.owlv2.owlv2_image_processor import Owlv2ImageProcessor
from zeromodels.models.owlv2.owlv2_tokenizer import Owlv2Tokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class Owlv2Processor(BaseProcessor):
    """Composite processor that bundles an image processor and CLIP tokenizer.

    Text queries are flattened across
    the batch and the model uses the per-row argmax of ``input_ids``
    to pool, so padded queries (whose first token is the pad id ``0``)
    are detected by the class predictor. Image preprocessing follows
    the OWLv2 spec: rescale → pad-to-square → resize → normalize.

    Args:
        size: Image size dict; forwarded to :class:`Owlv2ImageProcessor`.
        resample: Image interpolation mode. Defaults to ``"bicubic"``.
        do_rescale: Whether to divide pixel values by 255.
        rescale_factor: Rescale factor.
        do_pad: Whether to pad images to a square. Defaults to ``True``.
        do_normalize: Whether to apply CLIP normalization.
        image_mean: Per-channel normalization mean.
        image_std: Per-channel normalization std.
        return_tensor: If True, return Keras tensors.
        data_format: Image data format string.
        tokenizer_file: Optional explicit CLIP ``tokenizer.json`` path; ``None``
            uses the CLIP release default (OWLv2 shares CLIP's BPE vocab).
        max_seq_len: Maximum tokenized text length. Defaults to ``16``
            to match the reference.
        unk_token: Unknown token. Defaults to ``"<|endoftext|>"``.
        bos_token: Beginning-of-sequence token. Defaults to
            ``"<|startoftext|>"``.
        eos_token: End-of-sequence token. Defaults to ``"<|endoftext|>"``.
        pad_token: Padding token. Defaults to ``"!"`` to match the reference.
    """

    TOKENIZER_CLS = Owlv2Tokenizer
    IMAGE_PROCESSOR_CLS = Owlv2ImageProcessor

    def __init__(
        self,
        size: Optional[dict] = None,
        resample: str = "bicubic",
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_pad: bool = True,
        do_normalize: bool = True,
        image_mean: Sequence[float] = (0.48145466, 0.4578275, 0.40821073),
        image_std: Sequence[float] = (0.26862954, 0.26130258, 0.27577711),
        return_tensor: bool = True,
        data_format: Optional[str] = None,
        tokenizer_file: Optional[str] = None,
        max_seq_len: int = 16,
        unk_token: str = "<|endoftext|>",
        bos_token: str = "<|startoftext|>",
        eos_token: str = "<|endoftext|>",
        pad_token: str = "!",
        tokenizer=None,
        image_processor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_processor = image_processor or Owlv2ImageProcessor(
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_pad=do_pad,
            do_normalize=do_normalize,
            image_mean=tuple(image_mean),
            image_std=tuple(image_std),
            return_tensor=return_tensor,
            data_format=data_format,
        )

        self.tokenizer = tokenizer or Owlv2Tokenizer(
            tokenizer_file=tokenizer_file,
            max_seq_len=max_seq_len,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
        )

    @classmethod
    def from_hf(cls, repo, **kwargs):
        """Load the image processor (``preprocessor_config.json``) and, when the
        ``repo`` publishes one, its ``tokenizer.json``. The canonical OWL repos
        ship slow tokenizer files only, and their BPE is byte-identical to
        CLIP's, so absent a ``tokenizer.json`` the CLIP release default is the
        faithful fallback."""
        from huggingface_hub import hf_hub_download

        kwargs.setdefault("image_processor", cls.IMAGE_PROCESSOR_CLS.from_hf(repo))
        if "tokenizer_file" not in kwargs:
            try:
                kwargs["tokenizer_file"] = hf_hub_download(repo, "tokenizer.json")
            except Exception:
                pass
        return cls(**kwargs)

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
