import keras

from zeromodels.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class GraniteSpeechTokenizer(BaseTokenizer):
    """Granite byte-level BPE tokenizer (``tokenizers`` backend) with ``<|audio|>``.

    Downloads the fast-tokenizer ``tokenizer.json`` for ``variant`` from that
    variant's zeromodels Hub repo (``zeromodels/<variant>``), or takes an
    explicit ``tokenizer_file``; the audio / eos ids are read from the loaded vocab.
    Load by repo id like weights:
    ``GraniteSpeechTokenizer.from_weights("zeromodels/granite_speech_3_3_2b")``.

    Args:
        variant: Granite Speech variant key (no default; pass this or ``tokenizer_file``);
            resolves to the ``zeromodels/<variant>`` repo's ``tokenizer.json``.
        tokenizer_file: Optional explicit ``tokenizer.json`` path (overrides variant).
        audio_token: The audio placeholder token string.
    """

    def __init__(
        self, variant=None, tokenizer_file=None, audio_token="<|audio|>", **kwargs
    ):
        super().__init__(**kwargs)
        from tokenizers import Tokenizer

        self.variant = variant
        tokenizer_file = self.resolve_tokenizer_json_from_hf(
            (f"zeromodels/{self.variant}" if self.variant else None), tokenizer_file
        )
        self.tokenizer_file = tokenizer_file
        self._tok = Tokenizer.from_file(tokenizer_file)
        self.register_special_ids(audio_token)

    def register_special_ids(self, audio_token):
        from tokenizers import AddedToken

        self.audio_token = audio_token
        if self._tok.token_to_id(audio_token) is None:
            self._tok.add_special_tokens(
                [AddedToken(audio_token, special=True, normalized=False)]
            )
        self.audio_token_id = self._tok.token_to_id(audio_token)
        self.eos_token = "<|end_of_text|>"
        eos_id = self._tok.token_to_id(self.eos_token)
        self.eos_token_id = eos_id if eos_id is not None else 0

    @classmethod
    def from_hf(cls, repo, **kwargs):
        from huggingface_hub import hf_hub_download

        return cls(tokenizer_file=hf_hub_download(repo, "tokenizer.json"), **kwargs)

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text, add_special_tokens=False).ids

    def call(self, inputs):
        texts = self.normalize_texts(inputs)
        return {"input_ids": [self.encode(t) for t in texts]}

    def decode(self, ids, skip_special_tokens=True):
        return self._tok.decode(
            self.to_id_list(ids), skip_special_tokens=skip_special_tokens
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "variant": self.variant,
                "tokenizer_file": self.tokenizer_file,
                "audio_token": self.audio_token,
            }
        )
        return config
