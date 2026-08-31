import json
from typing import List, Union

import keras
from tokenizers import Tokenizer

from zeromodels.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class WhisperTokenizer(BaseTokenizer):
    """Whisper byte-level BPE tokenizer (``tokenizers`` Rust backend).

    Downloads the fast-tokenizer ``tokenizer.json`` for ``variant`` from that
    variant's zeromodels Hub repo (``zeromodels/<variant>``), or takes an
    explicit ``tokenizer_file``: the byte-level BPE and Whisper's ~1600 special
    tokens (languages, task, timestamps) are baked into the file. ``added_tokens``
    (content -> id) is rebuilt from it for the processor's prompt helpers. ``call``
    tokenizes into padded ``{input_ids, attention_mask}``. Load by repo id like
    weights: ``WhisperTokenizer.from_weights("zeromodels/whisper_tiny")``.

    Args:
        variant: Whisper variant key (no default; pass this or ``tokenizer_file``); resolves to the
            ``zeromodels/<variant>`` repo's tokenizer.json (v3 has a 51866 vocab).
        tokenizer_file: Optional explicit ``tokenizer.json`` path (overrides variant).
        bos_token_id / eos_token_id / pad_token_id: Whisper special ids (all 50257).
    """

    def __init__(
        self,
        variant: str = None,
        tokenizer_file: str = None,
        bos_token_id: int = 50257,
        eos_token_id: int = 50257,
        pad_token_id: int = 50257,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.variant = variant
        tokenizer_file = self.resolve_tokenizer_json_from_hf(
            (f"zeromodels/{self.variant}" if self.variant else None), tokenizer_file
        )
        self.tokenizer_file = tokenizer_file
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self._tok = Tokenizer.from_file(tokenizer_file)
        with open(tokenizer_file, encoding="utf-8") as f:
            payload = json.load(f)
        self.added_tokens = {
            t["content"]: t["id"] for t in payload.get("added_tokens", [])
        }
        self._special_id_set = set(self.added_tokens.values())
        self._special_id_set.add(eos_token_id)
        self._special_id_set.add(pad_token_id)

    @classmethod
    def from_hf(cls, repo, **kwargs):
        from huggingface_hub import hf_hub_download

        return cls(tokenizer_file=hf_hub_download(repo, "tokenizer.json"), **kwargs)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size(with_added_tokens=True)

    def tokenize(
        self, text: Union[str, List[str]]
    ) -> Union[List[int], List[List[int]]]:
        if isinstance(text, str):
            return self._tok.encode(text, add_special_tokens=False).ids
        encs = self._tok.encode_batch(text, add_special_tokens=False)
        return [e.ids for e in encs]

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        ids = self.to_id_list(token_ids)
        if skip_special_tokens:
            ids = [i for i in ids if i not in self._special_id_set]
        return self._tok.decode(ids, skip_special_tokens=False)

    def batch_decode(
        self, token_ids_batch, skip_special_tokens: bool = True
    ) -> List[str]:
        if hasattr(token_ids_batch, "numpy"):
            token_ids_batch = token_ids_batch.numpy()
        out = []
        for row in token_ids_batch:
            row = row.tolist() if hasattr(row, "tolist") else list(row)
            out.append(self.decode(row, skip_special_tokens=skip_special_tokens))
        return out

    def call(self, inputs: Union[str, List[str]]):
        texts = self.normalize_texts(inputs)
        encs = self._tok.encode_batch(texts, add_special_tokens=False)
        ids, mask = self.pad_batch([e.ids for e in encs], pad_value=self.pad_token_id)
        return {
            "input_ids": keras.ops.convert_to_tensor(ids, dtype="int32"),
            "attention_mask": keras.ops.convert_to_tensor(mask, dtype="int32"),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "variant": self.variant,
                "tokenizer_file": self.tokenizer_file,
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
            }
        )
        return config
