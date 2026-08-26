from typing import Dict, List, Union

import keras
import numpy as np
from keras import ops
from tokenizers import Tokenizer

from zeromodels.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class SigLIPTokenizer(BaseTokenizer):
    """SigLIP SentencePiece Unigram tokenizer (``tokenizers`` Rust backend).

    Downloads the fast-tokenizer ``tokenizer.json`` for ``variant`` from that
    variant's zeromodels Hub repo (``zeromodels/<variant>``), or takes an
    explicit ``tokenizer_file``. The reference canonicalization (lowercase +
    ASCII-punctuation strip + whitespace collapse) and the trailing ``</s>`` are
    baked into the file's normalizer / post-processor. ``call`` returns fixed-length
    (``max_seq_len``) ``input_ids`` padded with ``</s>`` (SigLIP's pad == eos), with
    no attention mask. Load by repo id like weights:
    ``SigLIPTokenizer.from_weights("zeromodels/siglip_base_p16_224")``.

    Args:
        variant: SigLIP variant key (default ``"siglip_base_p16_224"``); resolves to
            the ``zeromodels/<variant>`` repo's ``tokenizer.json``.
        tokenizer_file: Optional explicit ``tokenizer.json`` path (overrides variant).
        max_seq_len: Fixed sequence length (default 64).
        unk_token / pad_token / eos_token: Special token strings.
    """

    DEFAULT_VARIANT = "siglip_base_p16_224"

    def __init__(
        self,
        variant: str = None,
        tokenizer_file: str = None,
        max_seq_len: int = 64,
        unk_token: str = "<unk>",
        pad_token: str = "</s>",
        eos_token: str = "</s>",
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
        self.pad_token = pad_token
        self.eos_token = eos_token

        tok = Tokenizer.from_file(tokenizer_file)
        self.unk_token_id = tok.token_to_id(unk_token)
        self.pad_token_id = tok.token_to_id(pad_token)
        self.eos_token_id = tok.token_to_id(eos_token)
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
            return self._tok.encode(text, add_special_tokens=True).ids
        encs = self._tok.encode_batch(text, add_special_tokens=True)
        return [e.ids for e in encs]

    def detokenize(
        self, token_ids, skip_special_tokens: bool = True
    ) -> Union[str, List[str]]:
        if hasattr(token_ids, "numpy"):
            token_ids = token_ids.numpy()
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        def _one(ids):
            ids = [int(i) for i in ids]
            if skip_special_tokens:
                skip = {self.pad_token_id, self.eos_token_id, self.unk_token_id}
                ids = [i for i in ids if i not in skip]
            return self._tok.decode(ids, skip_special_tokens=False).strip()

        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            return [_one(r) for r in token_ids]
        return _one(token_ids)

    def build_inputs_with_special_tokens(self, token_ids: List[int]) -> List[int]:
        return list(token_ids) + [self.eos_token_id]

    def prepare_for_model(self, text: Union[str, List[int]]) -> Dict[str, List[int]]:
        token_ids = self.tokenize(text) if isinstance(text, str) else list(text)
        token_ids = token_ids[: self.max_seq_len]
        pad_len = self.max_seq_len - len(token_ids)
        if pad_len > 0:
            token_ids = token_ids + [self.pad_token_id] * pad_len
        return {"input_ids": token_ids}

    def prepare_for_model_tensor(
        self, token_ids_list: List[List[int]]
    ) -> Dict[str, keras.KerasTensor]:
        padded = []
        for seq in token_ids_list:
            seq = list(seq)[: self.max_seq_len]
            pad_len = self.max_seq_len - len(seq)
            if pad_len > 0:
                seq = seq + [self.pad_token_id] * pad_len
            padded.append(seq)
        ids = np.array(padded, dtype=np.int32)
        return {"input_ids": ops.convert_to_tensor(ids, dtype="int32")}

    def call(self, inputs):
        return self.encode_batch_to_inputs(
            inputs, token_type_ids=False, mask_dtype=None
        )

    def batch_decode(
        self, token_ids_batch, skip_special_tokens: bool = True
    ) -> List[str]:
        if hasattr(token_ids_batch, "numpy"):
            token_ids_batch = token_ids_batch.numpy()
        out = []
        for row in token_ids_batch:
            row = row.tolist() if hasattr(row, "tolist") else list(row)
            out.append(self.detokenize(row, skip_special_tokens))
        return out

    def get_sequence_length(self, input_ids: keras.KerasTensor) -> keras.KerasTensor:
        pad = ops.convert_to_tensor(self.pad_token_id, dtype="int32")
        mask = ops.not_equal(input_ids, pad)
        return ops.sum(ops.cast(mask, dtype="int32"), axis=1)

    def truncate_sequences(
        self, input_ids: keras.KerasTensor, max_length: int
    ) -> keras.KerasTensor:
        if max_length >= input_ids.shape[1]:
            return input_ids
        return input_ids[:, :max_length]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "variant": self.variant,
                "tokenizer_file": self.tokenizer_file,
                "max_seq_len": self.max_seq_len,
                "unk_token": self.unk_token,
                "pad_token": self.pad_token,
                "eos_token": self.eos_token,
            }
        )
        return config
