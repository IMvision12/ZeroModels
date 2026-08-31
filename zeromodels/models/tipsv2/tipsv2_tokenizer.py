from typing import List, Union

import keras
from keras import ops
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Metaspace

from zeromodels.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class Tipsv2Tokenizer(BaseTokenizer):
    """TIPSv2 text tokenizer (``tokenizers`` Rust backend, byte-fallback BPE).

    Loads the fast-tokenizer ``tokenizer.json`` (lowercase + Metaspace normalization
    baked in). ``call`` returns fixed-length (``max_seq_len``) ``input_ids`` padded
    with ``<pad>`` (id 0) plus an ``attention_mask`` (1 for real tokens). Load via
    ``Tipsv2Tokenizer.from_weights("hf:google/tipsv2-b14")`` or by zeromodels repo id.
    """

    def __init__(
        self,
        variant: str = None,
        tokenizer_file: str = None,
        max_seq_len: int = 64,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.variant = variant
        tokenizer_file = self.resolve_tokenizer_json_from_hf(
            (f"zeromodels/{self.variant}" if self.variant else None), tokenizer_file
        )
        self.tokenizer_file = tokenizer_file
        self.max_seq_len = max_seq_len
        self.unk_token = unk_token
        self.pad_token = pad_token

        tok = Tokenizer.from_file(tokenizer_file)
        tok.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="never")
        self.unk_token_id = tok.token_to_id(unk_token)
        self.pad_token_id = tok.token_to_id(pad_token)
        tok.enable_truncation(max_length=max_seq_len)
        tok.enable_padding(
            pad_id=self.pad_token_id, pad_token=pad_token, length=max_seq_len
        )
        self._tok = tok

    @classmethod
    def from_hf(cls, repo, **kwargs):
        import os

        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(repo, "tokenizer.json")
        except Exception:
            from transformers import AutoTokenizer

            src = hf_hub_download(repo, "tokenizer.model")
            fast = AutoTokenizer.from_pretrained(repo)
            path = os.path.join(os.path.dirname(src), "tokenizer.json")
            fast._tokenizer.save(path)
        return cls(tokenizer_file=path, **kwargs)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def tokenize(
        self, text: Union[str, List[str]]
    ) -> Union[List[int], List[List[int]]]:
        if isinstance(text, str):
            return self._tok.encode(text, add_special_tokens=True).ids
        return [e.ids for e in self._tok.encode_batch(text, add_special_tokens=True)]

    def detokenize(
        self, token_ids, skip_special_tokens: bool = True
    ) -> Union[str, List[str]]:
        if hasattr(token_ids, "numpy"):
            token_ids = ops.convert_to_numpy(token_ids)
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        def decode_row(ids):
            ids = [int(i) for i in ids]
            if skip_special_tokens:
                skip = {self.pad_token_id, self.unk_token_id}
                ids = [i for i in ids if i not in skip]
            return self._tok.decode(ids, skip_special_tokens=False).strip()

        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            return [decode_row(r) for r in token_ids]
        return decode_row(token_ids)

    def call(self, inputs):
        """Encode text -> ``{"input_ids", "attention_mask"}`` (fixed ``max_seq_len``)."""
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        encs = self._tok.encode_batch(texts, add_special_tokens=True)
        return {
            "input_ids": ops.convert_to_tensor([e.ids for e in encs], dtype="int32"),
            "attention_mask": ops.convert_to_tensor(
                [e.attention_mask for e in encs], dtype="int32"
            ),
        }

    def batch_decode(
        self, token_ids_batch, skip_special_tokens: bool = True
    ) -> List[str]:
        if hasattr(token_ids_batch, "numpy"):
            token_ids_batch = ops.convert_to_numpy(token_ids_batch)
        out = []
        for row in token_ids_batch:
            row = row.tolist() if hasattr(row, "tolist") else list(row)
            out.append(self.detokenize(row, skip_special_tokens))
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "variant": self.variant,
                "tokenizer_file": self.tokenizer_file,
                "max_seq_len": self.max_seq_len,
                "unk_token": self.unk_token,
                "pad_token": self.pad_token,
            }
        )
        return config
