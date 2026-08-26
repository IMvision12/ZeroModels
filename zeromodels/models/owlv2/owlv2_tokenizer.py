import keras

from zeromodels.models.clip.clip_tokenizer import CLIPTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class Owlv2Tokenizer(CLIPTokenizer):
    """OWLv2 text tokenizer.

    OWLv2 reuses CLIP's byte-level BPE vocabulary, so this is the
    :class:`~zeromodels.models.clip.clip_tokenizer.CLIPTokenizer` with OWLv2's
    reference defaults: a shorter ``max_seq_len`` of 16 and ``"!"`` as the pad
    token. Load it by Hub repo id, which downloads that repo's ``tokenizer.json``
    (``Owlv2Tokenizer.from_weights("zeromodels/owlv2-base-patch16")``), or pass
    an explicit ``tokenizer_file``.

    Args:
        variant: Optional variant label, kept for reference.
        tokenizer_file: Optional explicit ``tokenizer.json`` path.
        max_seq_len: Padded / truncated length (default 16).
        unk_token / bos_token / eos_token / pad_token: Special token strings.
    """

    def __init__(
        self,
        variant: str = None,
        tokenizer_file: str = None,
        max_seq_len: int = 16,
        unk_token: str = "<|endoftext|>",
        bos_token: str = "<|startoftext|>",
        eos_token: str = "<|endoftext|>",
        pad_token: str = "!",
        **kwargs,
    ):
        super().__init__(
            variant=variant,
            tokenizer_file=tokenizer_file,
            max_seq_len=max_seq_len,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )
