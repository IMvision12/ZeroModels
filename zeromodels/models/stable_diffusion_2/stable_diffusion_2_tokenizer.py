import keras

from zeromodels.models.stable_diffusion.stable_diffusion_tokenizer import (
    StableDiffusionTokenizer,
)


@keras.saving.register_keras_serializable(package="zeromodels")
class StableDiffusion2Tokenizer(StableDiffusionTokenizer):
    """Stable Diffusion 2.x text tokenizer: the OpenCLIP ViT-H/14 BPE tokenizer.

    The same byte-pair encoding and ``<|startoftext|>`` / ``<|endoftext|>`` framing
    as CLIP ViT-L/14's, truncated to 77 tokens, but padded with ``!`` (id 0) the
    way OpenCLIP does, which is what the SD 2 text encoder was trained on. Loads by
    repo id like the model: ``from_weights("zeromodels/stable-diffusion-2-1")``.

    Args:
        hf_id: Hosted repo to read ``tokenizer.json`` from. Required unless
            ``tokenizer_file`` is given (no default repo).
        tokenizer_file: Explicit ``tokenizer.json`` path (overrides ``hf_id``).
        max_seq_len: Padded / truncated length (default 77).
        pad_token: Pad token string (``!``).
    """

    def __init__(
        self, hf_id=None, tokenizer_file=None, max_seq_len=77, pad_token="!", **kwargs
    ):
        super().__init__(
            hf_id=hf_id,
            tokenizer_file=tokenizer_file,
            max_seq_len=max_seq_len,
            pad_token=pad_token,
            **kwargs,
        )
