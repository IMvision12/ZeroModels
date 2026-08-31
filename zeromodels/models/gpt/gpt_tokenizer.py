import keras
from tokenizers import Tokenizer

from zeromodels.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class GptTokenizer(BaseTokenizer):
    """Original GPT BPE tokenizer (``tokenizers`` backend).

    Loads the model's ``tokenizer.json`` from a Hub repo (``zeromodels/<variant>``
    by default, or an explicit ``hf_id`` / ``tokenizer_file``). ``call`` pads batches
    to the longest sequence. GPT is a base LM with no end-of-text token. Load by repo
    id like weights: ``GptTokenizer.from_weights("zeromodels/gpt")``.

    Args:
        variant: GPT variant key (no default; pass this or ``tokenizer_file``); resolves to the
            ``zeromodels/<variant>`` repo's tokenizer.json.
        hf_id: Explicit Hub repo to pull ``tokenizer.json`` from (overrides the
            variant).
        tokenizer_file: Optional explicit ``tokenizer.json`` path (overrides the
            download).
        unk_token: Unknown-token string (default ``"<unk>"``).
    """

    def __init__(
        self,
        variant=None,
        hf_id=None,
        tokenizer_file=None,
        unk_token="<unk>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.variant = variant
        self.hf_id = hf_id
        repo = (
            hf_id
            if hf_id is not None
            else (f"zeromodels/{self.variant}" if self.variant else None)
        )
        tokenizer_file = self.resolve_tokenizer_json_from_hf(repo, tokenizer_file)
        self.tokenizer_file = tokenizer_file
        self.unk_token = unk_token
        self._tok = Tokenizer.from_file(tokenizer_file)
        self.eos_token_id = None

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
        input_ids, attention_mask = self.pad_batch([self.encode(t) for t in texts])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens=True):
        return self._tok.decode(
            self.to_id_list(ids), skip_special_tokens=skip_special_tokens
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "variant": self.variant,
                "hf_id": self.hf_id,
                "tokenizer_file": self.tokenizer_file,
                "unk_token": self.unk_token,
            }
        )
        return config
