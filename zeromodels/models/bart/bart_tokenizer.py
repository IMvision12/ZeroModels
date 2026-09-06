from typing import List, Union

import keras
from tokenizers import Tokenizer

from zeromodels.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class BartTokenizer(BaseTokenizer):
    """BART byte-level BPE tokenizer (``tokenizers`` Rust backend).

    Same tokenizer family as RoBERTa / GPT-2: byte-level BPE with the
    ``<s> A </s>`` (single) and ``<s> A </s></s> B </s>`` (pair, e.g. NLI) special
    tokens baked into the fast ``tokenizer.json`` post-processor, plus truncation +
    padding. Loads ``tokenizer.json`` from a Hub repo (``zeromodels/<variant>`` by
    default, or an explicit ``hf_id`` / ``tokenizer_file``); load by repo id like
    weights: ``BartTokenizer.from_weights("zeromodels/bart-base")``. ``call`` returns
    the ``input_ids`` / ``attention_mask`` dict expected by :class:`BartModel` and the
    task heads (BART has no ``token_type_ids``); the decoder ids are produced by the
    model, not here.

    Args:
        variant: BART variant key (no default; pass this, ``hf_id`` or
            ``tokenizer_file``); resolves to the ``zeromodels/<variant>`` repo's
            ``tokenizer.json``.
        hf_id: Explicit Hub repo to pull ``tokenizer.json`` from (overrides variant).
        tokenizer_file: Optional explicit ``tokenizer.json`` path (overrides the
            download).
        max_seq_len: Truncation length (default 1024, BART's ``max_position_embeddings``);
            batches pad to the longest sequence.
        bos_token / eos_token / unk_token / pad_token / mask_token: Special tokens.
    """

    def __init__(
        self,
        variant: str = None,
        hf_id: str = None,
        tokenizer_file: str = None,
        max_seq_len: int = 1024,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
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
        self.max_seq_len = max_seq_len
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.mask_token = mask_token

        tok = Tokenizer.from_file(tokenizer_file)
        self.bos_token_id = tok.token_to_id(bos_token)
        self.eos_token_id = tok.token_to_id(eos_token)
        self.unk_token_id = tok.token_to_id(unk_token)
        self.pad_token_id = tok.token_to_id(pad_token)
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

    def call(self, inputs: Union[str, List[str]], text_pair=None):
        return self.encode_batch_to_inputs(inputs, text_pair, token_type_ids=False)

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
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
                "max_seq_len": self.max_seq_len,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "unk_token": self.unk_token,
                "pad_token": self.pad_token,
                "mask_token": self.mask_token,
            }
        )
        return config
