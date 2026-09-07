import keras

from zeromodels.models.clip.clip_tokenizer import CLIPTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class OwlViTTokenizer(CLIPTokenizer):
    """OWL-ViT text tokenizer.

    OWL-ViT reuses CLIP's byte-level BPE vocabulary, so this is the
    :class:`~zeromodels.models.clip.clip_tokenizer.CLIPTokenizer` with OWL's
    reference defaults: a shorter ``max_seq_len`` of 16 and ``"!"`` as the pad
    token. Load it by Hub repo id (``hf_id``), which downloads that repo's
    ``tokenizer.json`` (``OwlViTTokenizer.from_weights("zeromodels/owlvit-base-patch32")``),
    or pass an explicit ``tokenizer_file``.

    Args:
        hf_id: Hub repo carrying this OWL-ViT model's ``tokenizer.json``; takes
            precedence over ``variant``.
        variant: Optional variant label (resolves to ``zeromodels/<variant>``).
        tokenizer_file: Optional explicit ``tokenizer.json`` path.
        max_seq_len: Padded / truncated length (default 16).
        unk_token / bos_token / eos_token / pad_token: Special token strings.
    """

    def __init__(
        self,
        hf_id: str = None,
        variant: str = None,
        tokenizer_file: str = None,
        max_seq_len: int = 16,
        unk_token: str = "<|endoftext|>",
        bos_token: str = "<|startoftext|>",
        eos_token: str = "<|endoftext|>",
        pad_token: str = "!",
        **kwargs,
    ):
        # hf_id (a full Hub repo) is the primary knob, like SAM3/Gemma4; resolve
        # it to this repo's tokenizer.json here so CLIPTokenizer stays untouched.
        if hf_id is not None and tokenizer_file is None:
            tokenizer_file = self.resolve_tokenizer_json_from_hf(hf_id, None)
        self.hf_id = hf_id
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

    def get_config(self):
        config = super().get_config()
        config["hf_id"] = self.hf_id
        return config
