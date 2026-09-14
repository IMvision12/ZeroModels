import keras

from zeromodels.models.stable_diffusion_3.stable_diffusion_3_tokenizer import (
    StableDiffusion3Tokenizer,
)


@keras.saving.register_keras_serializable(package="zeromodels")
class StableDiffusion3_5Tokenizer(StableDiffusion3Tokenizer):
    """Stable Diffusion 3.5 text tokenizer: the SD 3 tokenizer (CLIP BPE plus the T5
    SentencePiece tokenizer, ``{"input_ids", "attention_mask", "input_ids_3"}``);
    the checkpoints share their tokenizers with SD 3. Loads by repo id like the
    model: ``from_weights("zeromodels/stable-diffusion-3.5-large")``.

    Args:
        hf_id: Hosted repo to read the tokenizer files from. Required unless the
            files are given (no default repo).
        tokenizer_file / tokenizer_file_3: Explicit CLIP / T5 ``tokenizer.json`` paths.
        max_seq_len: CLIP padded / truncated length (default 77).
        max_sequence_length: T5 padded / truncated length (default 256).
    """

    def __init__(
        self,
        hf_id=None,
        tokenizer_file=None,
        tokenizer_file_3=None,
        max_seq_len=77,
        max_sequence_length=256,
        **kwargs,
    ):
        super().__init__(
            hf_id=hf_id,
            tokenizer_file=tokenizer_file,
            tokenizer_file_3=tokenizer_file_3,
            max_seq_len=max_seq_len,
            max_sequence_length=max_sequence_length,
            **kwargs,
        )
