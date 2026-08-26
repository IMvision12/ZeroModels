import keras
from tokenizers import Tokenizer

from zeromodels.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class GPT2Tokenizer(BaseTokenizer):
    """GPT-2 byte-level BPE tokenizer (``tokenizers`` backend).

    Loads the model's ``tokenizer.json`` from a Hub repo (``zeromodels/<variant>``
    by default, or an explicit ``hf_id`` / ``tokenizer_file``). ``<|endoftext|>`` is
    the single special token; ``call`` pads batches to the longest sequence. GPT-2 is
    a base LM with no chat template. Load by repo id like weights:
    ``GPT2Tokenizer.from_weights("zeromodels/gpt2")``.

    Args:
        variant: GPT-2 variant key (default ``"gpt2"``); resolves to the
            ``zeromodels/<variant>`` repo's tokenizer.json.
        hf_id: Explicit Hub repo to pull ``tokenizer.json`` from (overrides the
            variant default).
        tokenizer_file: Optional explicit ``tokenizer.json`` path (overrides the
            download).
        eos_token: End-of-text token string (default ``"<|endoftext|>"``).
    """

    DEFAULT_VARIANT = "gpt2"

    def __init__(
        self,
        variant=None,
        hf_id=None,
        tokenizer_file=None,
        eos_token="<|endoftext|>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.variant = variant or self.DEFAULT_VARIANT
        self.hf_id = hf_id
        repo = hf_id if hf_id is not None else f"zeromodels/{self.variant}"
        tokenizer_file = self.resolve_tokenizer_json_from_hf(repo, tokenizer_file)
        self.tokenizer_file = tokenizer_file
        self.eos_token = eos_token
        self._tok = Tokenizer.from_file(tokenizer_file)
        self.eos_token_id = self._tok.token_to_id(eos_token)

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
                "eos_token": self.eos_token,
            }
        )
        return config
