from typing import List, Union

import keras
from tokenizers import Tokenizer

from zeromodels.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class DebertaTokenizer(BaseTokenizer):
    """DeBERTa (v1) GPT-2 byte-level BPE tokenizer (``tokenizers`` Rust backend).

    Loads the HuggingFace fast-tokenizer ``tokenizer.json`` for ``variant`` from the
    ``deberta`` release tag (or an explicit ``tokenizer_file``), with DeBERTa's
    ``[CLS] A [SEP]`` / ``[CLS] A [SEP] B [SEP]`` post-processing baked in, plus
    truncation + padding. ``call`` returns the ``input_ids`` / ``attention_mask`` /
    ``token_type_ids`` dict expected by :class:`DebertaModel` (``token_type_ids`` is
    produced for parity but the model has no token-type embeddings).

    Args:
        variant: DeBERTa variant key (default ``"deberta_base"``).
        tokenizer_file: Optional explicit ``tokenizer.json`` path (overrides variant).
        max_seq_len: Truncation length (default 512); batches pad to the longest.
        cls_token / sep_token / pad_token / unk_token / mask_token: Special tokens.
    """

    DEFAULT_VARIANT = "deberta_base"

    def __init__(
        self,
        variant: str = None,
        tokenizer_file: str = None,
        max_seq_len: int = 512,
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        mask_token: str = "[MASK]",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.variant = variant or self.DEFAULT_VARIANT
        tokenizer_file = self.resolve_tokenizer_json_from_hf(
            f"zeromodels/{self.variant}", tokenizer_file
        )
        self.tokenizer_file = tokenizer_file
        self.max_seq_len = max_seq_len
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token

        tok = Tokenizer.from_file(tokenizer_file)
        self.cls_token_id = tok.token_to_id(cls_token)
        self.sep_token_id = tok.token_to_id(sep_token)
        self.pad_token_id = tok.token_to_id(pad_token)
        self.unk_token_id = tok.token_to_id(unk_token)
        self.mask_token_id = tok.token_to_id(mask_token)
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
        return self.encode_batch_to_inputs(inputs, text_pair)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "variant": self.variant,
                "tokenizer_file": self.tokenizer_file,
                "max_seq_len": self.max_seq_len,
                "cls_token": self.cls_token,
                "sep_token": self.sep_token,
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "mask_token": self.mask_token,
            }
        )
        return config
