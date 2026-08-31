from typing import List, Union

import keras
from tokenizers import Tokenizer

from zeromodels.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class T5Tokenizer(BaseTokenizer):
    """T5 SentencePiece (Unigram) tokenizer via the ``tokenizers`` Rust backend.

    Loads the HuggingFace fast-tokenizer ``tokenizer.json`` for ``variant`` (or an
    explicit ``tokenizer_file``): the Unigram model, the ``▁`` metaspace, the
    ``$A </s>`` post-processing (T5 appends the EOS), and the 100 ``<extra_id_N>``
    sentinel tokens are baked into the file. ``call`` returns the ``input_ids`` /
    ``attention_mask`` dict expected by the T5 models (T5 has no token-type ids).

    Args:
        variant: T5 variant key (no default; pass this or ``tokenizer_file``).
        tokenizer_file: Optional explicit ``tokenizer.json`` path (overrides variant).
        max_seq_len: Truncation length (default 512); batches pad to the longest.
        eos_token / pad_token / unk_token: Special tokens.
    """

    def __init__(
        self,
        variant: str = None,
        tokenizer_file: str = None,
        max_seq_len: int = 512,
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.variant = variant
        tokenizer_file = self.resolve_tokenizer_json_from_hf(
            (f"zeromodels/{self.variant}" if self.variant else None), tokenizer_file
        )
        self.tokenizer_file = tokenizer_file
        self.max_seq_len = max_seq_len
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token

        tok = Tokenizer.from_file(tokenizer_file)
        self.eos_token_id = tok.token_to_id(eos_token)
        self.pad_token_id = tok.token_to_id(pad_token)
        self.unk_token_id = tok.token_to_id(unk_token)
        tok.enable_truncation(max_length=max_seq_len)
        tok.enable_padding(pad_id=self.pad_token_id, pad_token=pad_token)
        self._tok = tok

    @classmethod
    def from_hf(cls, repo, **kwargs):
        from huggingface_hub import hf_hub_download

        return cls(tokenizer_file=hf_hub_download(repo, "tokenizer.json"), **kwargs)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def tokenize(
        self, text: Union[str, List[str]]
    ) -> Union[List[int], List[List[int]]]:
        if isinstance(text, str):
            return self._tok.encode(text, add_special_tokens=False).ids
        encs = self._tok.encode_batch(text, add_special_tokens=False)
        return [e.ids for e in encs]

    def detokenize(self, token_ids, skip_special_tokens: bool = True) -> str:
        return self.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return self._tok.decode(
            self.to_id_list(ids), skip_special_tokens=skip_special_tokens
        )

    def call(self, inputs: Union[str, List[str]], text_pair=None):
        return self.encode_batch_to_inputs(inputs, text_pair, token_type_ids=False)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "variant": self.variant,
                "tokenizer_file": self.tokenizer_file,
                "max_seq_len": self.max_seq_len,
                "eos_token": self.eos_token,
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
            }
        )
        return config
