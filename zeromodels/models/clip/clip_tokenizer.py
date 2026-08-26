from typing import Dict, List, Union

import keras
from tokenizers import Tokenizer

from zeromodels.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class CLIPTokenizer(BaseTokenizer):
    """CLIP byte-level BPE tokenizer (``tokenizers`` Rust backend).

    Downloads the fast-tokenizer ``tokenizer.json`` for ``variant`` from that
    variant's zeromodels Hub repo (``zeromodels/<variant>``), or takes an
    explicit ``tokenizer_file``, and re-enables CLIP's truncation +
    ``<|endoftext|>`` padding to ``max_seq_len`` (77). Load by repo id the same way
    weights load: ``CLIPTokenizer.from_weights("zeromodels/clip_vit_base_16")``.
    The openai / open_clip CLIP variants tokenize identically.

    Args:
        variant: CLIP variant key (default ``"clip_vit_base_16"``); resolves to the
            ``zeromodels/<variant>`` repo's ``tokenizer.json``.
        tokenizer_file: Optional explicit ``tokenizer.json`` path (overrides variant).
        max_seq_len: Padded / truncated length (default 77).
        unk_token / bos_token / eos_token / pad_token: Special token strings.
    """

    DEFAULT_VARIANT = "clip_vit_base_16"

    def __init__(
        self,
        variant: str = None,
        tokenizer_file: str = None,
        max_seq_len: int = 77,
        unk_token: str = "<|endoftext|>",
        bos_token: str = "<|startoftext|>",
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|endoftext|>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.variant = variant or self.DEFAULT_VARIANT
        tokenizer_file = self.resolve_tokenizer_json_from_hf(
            f"zeromodels/{self.variant}", tokenizer_file
        )
        self.tokenizer_file = tokenizer_file
        self.max_seq_len = max_seq_len
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        tok = Tokenizer.from_file(tokenizer_file)
        self.bos_token_id = tok.token_to_id(bos_token)
        self.eos_token_id = tok.token_to_id(eos_token)
        self.pad_token_id = tok.token_to_id(pad_token)
        self.unk_token_id = tok.token_to_id(unk_token)
        tok.enable_truncation(max_length=max_seq_len)
        tok.enable_padding(
            pad_id=self.pad_token_id, pad_token=pad_token, length=max_seq_len
        )
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

    def detokenize(self, token_ids) -> str:
        ids = self.to_id_list(token_ids)
        skip = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
        keep = [i for i in ids if i not in skip]
        text = self._tok.decode(keep, skip_special_tokens=False)
        return text.replace("</w>", " ").strip()

    def prepare_for_model(self, text: str) -> Dict[str, List[int]]:
        enc = self._tok.encode(text)
        return {"input_ids": enc.ids, "attention_mask": enc.attention_mask}

    def call(self, inputs: Union[str, List[str]]):
        return self.encode_batch_to_inputs(
            inputs, token_type_ids=False, mask_dtype="bool"
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "variant": self.variant,
                "tokenizer_file": self.tokenizer_file,
                "max_seq_len": self.max_seq_len,
                "unk_token": self.unk_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "pad_token": self.pad_token,
            }
        )
        return config
