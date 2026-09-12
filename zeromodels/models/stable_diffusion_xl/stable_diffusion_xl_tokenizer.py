import keras

from zeromodels.models.stable_diffusion.stable_diffusion_tokenizer import (
    StableDiffusionTokenizer,
)


@keras.saving.register_keras_serializable(package="zeromodels")
class StableDiffusionXLTokenizer(StableDiffusionTokenizer):
    """Stable Diffusion XL text tokenizer: the CLIP ViT-L/14 BPE tokenizer.

    SDXL tokenizes the prompt twice, once per text encoder, with the same
    byte-pair encoding and ``<|startoftext|>`` / ``<|endoftext|>`` framing: the
    two tokenizers differ only in the pad token (``<|endoftext|>`` for the CLIP
    ViT-L/14 tower, ``!`` for the OpenCLIP ViT-bigG/14 one). This is the first;
    :class:`StableDiffusionXLTextToImage` derives the second tower's ids from the
    returned ``attention_mask``, so one call feeds ``generate``. Loads by repo id
    like the model: ``from_weights("zeromodels/stable-diffusion-xl-base-1.0")``.

    Args:
        hf_id: Hosted repo to read ``tokenizer.json`` from. Required unless
            ``tokenizer_file`` is given (no default repo).
        tokenizer_file: Explicit ``tokenizer.json`` path (overrides ``hf_id``).
        max_seq_len: Padded / truncated length (default 77).
        pad_token: Pad token string (``<|endoftext|>``).
    """

    def __init__(
        self,
        hf_id=None,
        tokenizer_file=None,
        max_seq_len=77,
        pad_token="<|endoftext|>",
        **kwargs,
    ):
        super().__init__(
            hf_id=hf_id,
            tokenizer_file=tokenizer_file,
            max_seq_len=max_seq_len,
            pad_token=pad_token,
            **kwargs,
        )
