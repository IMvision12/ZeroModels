import keras
from keras import ops

from zeromodels.models.stable_diffusion.stable_diffusion_tokenizer import (
    StableDiffusionTokenizer,
)
from zeromodels.models.t5.t5_tokenizer import T5Tokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class StableDiffusion3Tokenizer(StableDiffusionTokenizer):
    """Stable Diffusion 3 text tokenizer: the CLIP ViT-L/14 BPE tokenizer plus the
    T5 SentencePiece tokenizer, in one call.

    SD 3 tokenizes the prompt three times: with the two CLIP tokenizers (the same
    BPE, padded with ``<|endoftext|>`` and ``!`` respectively; this class is the
    first, and :class:`StableDiffusion3TextToImage` derives the second tower's ids
    from the returned ``attention_mask``, as SDXL does) and with the T5 tokenizer
    (``</s>``-terminated, ``<pad>``-padded to ``max_sequence_length``, 256). The
    returned dict is ``{"input_ids", "attention_mask", "input_ids_3"}``, what
    ``generate`` takes. Loads by repo id like the model, reading the hosted repo's
    ``tokenizer.json`` (CLIP) and ``tokenizer_3.json`` (T5):
    ``from_weights("zeromodels/stable-diffusion-3-medium")``.

    Args:
        hf_id: Hosted repo to read the tokenizer files from. Required unless the
            files are given (no default repo).
        tokenizer_file: Explicit CLIP ``tokenizer.json`` path (overrides ``hf_id``).
        tokenizer_file_3: Explicit T5 ``tokenizer.json`` path (overrides ``hf_id``).
        max_seq_len: CLIP padded / truncated length (default 77).
        max_sequence_length: T5 padded / truncated length (default 256).
        pad_token: CLIP pad token string (``<|endoftext|>``).
    """

    def __init__(
        self,
        hf_id=None,
        tokenizer_file=None,
        tokenizer_file_3=None,
        max_seq_len=77,
        max_sequence_length=256,
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
        if tokenizer_file_3 is None:
            tokenizer_file_3 = self.download_tokenizer_json(hf_id, "tokenizer_3.json")
        self.tokenizer_file_3 = tokenizer_file_3
        self.max_sequence_length = max_sequence_length
        self.tokenizer_3 = T5Tokenizer(
            tokenizer_file=tokenizer_file_3, max_seq_len=max_sequence_length
        )

    @staticmethod
    def download_tokenizer_json(hf_id, filename="tokenizer.json"):
        import os

        from huggingface_hub import hf_hub_download

        return hf_hub_download(hf_id, filename, token=os.environ.get("HF_TOKEN"))

    def call(self, inputs):
        out = super().call(inputs)
        # the T5 tokenizer truncates to max_sequence_length and pads to the longest
        # prompt; the transformer takes the fixed 256-token length, so pad the rest
        ids = self.tokenizer_3(inputs)["input_ids"]
        short = self.max_sequence_length - int(ids.shape[1])
        if short > 0:
            ids = ops.pad(
                ids, ((0, 0), (0, short)), constant_values=self.tokenizer_3.pad_token_id
            )
        out["input_ids_3"] = ids
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "tokenizer_file_3": self.tokenizer_file_3,
                "max_sequence_length": self.max_sequence_length,
            }
        )
        return config
