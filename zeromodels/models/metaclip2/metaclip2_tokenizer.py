from typing import List, Union

import keras
from keras import ops
from tokenizers import Tokenizer

from zeromodels.base import BaseTokenizer

METACLIP2_EOS_TOKEN_ID = 2
METACLIP2_BOS_TOKEN_ID = 0
METACLIP2_PAD_TOKEN_ID = 1
METACLIP2_UNK_TOKEN_ID = 3
METACLIP2_MASK_TOKEN_ID = 901628


@keras.saving.register_keras_serializable(package="zeromodels")
class MetaClip2Tokenizer(BaseTokenizer):
    """XLM-RoBERTa tokenizer for MetaCLIP 2 worldwide variants (``tokenizers`` backend).

    Downloads the fast-tokenizer ``tokenizer.json`` for ``variant`` from that
    variant's zeromodels Hub repo (``zeromodels/<variant>``), or takes an
    explicit ``tokenizer_file``: the Unigram model, the fairseq id offset and the
    ``<s> A </s>`` post-processing are baked into the file (901629-token
    multilingual vocab). ``call`` returns fixed-length (``max_seq_len`` = 77)
    ``token_ids`` + ``padding_mask``. The text backbone pools by an explicit
    ``token == eos_token_id`` (=2) match (MASK id 901628 > EOS). Load by repo id
    like weights:
    ``MetaClip2Tokenizer.from_weights("zeromodels/metaclip2_worldwide_b16_224")``.

    Args:
        variant: MetaCLIP 2 worldwide variant key; resolves to the
            ``zeromodels/<variant>`` repo's ``tokenizer.json``.
        tokenizer_file: Optional explicit ``tokenizer.json`` path (overrides variant).
        max_seq_len: Fixed sequence length (default 77).
        bos_token_id / eos_token_id / pad_token_id / unk_token_id: XLM-R special ids.
    """

    DEFAULT_VARIANT = "metaclip2_worldwide_b16_224"

    def __init__(
        self,
        variant: str = None,
        tokenizer_file: str = None,
        max_seq_len: int = 77,
        bos_token_id: int = METACLIP2_BOS_TOKEN_ID,
        eos_token_id: int = METACLIP2_EOS_TOKEN_ID,
        pad_token_id: int = METACLIP2_PAD_TOKEN_ID,
        unk_token_id: int = METACLIP2_UNK_TOKEN_ID,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.variant = variant or self.DEFAULT_VARIANT
        tokenizer_file = self.resolve_tokenizer_json_from_hf(
            f"zeromodels/{self.variant}", tokenizer_file
        )
        self.tokenizer_file = tokenizer_file
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id

        tok = Tokenizer.from_file(tokenizer_file)
        tok.enable_truncation(max_length=max_seq_len)
        tok.enable_padding(pad_id=pad_token_id, pad_token="<pad>", length=max_seq_len)
        self._tok = tok

    @classmethod
    def from_hf(cls, repo, **kwargs):
        from huggingface_hub import hf_hub_download

        return cls(tokenizer_file=hf_hub_download(repo, "tokenizer.json"), **kwargs)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def call(self, inputs: Union[str, List[str]]):
        texts = self.normalize_texts(inputs)
        encs = self._tok.encode_batch(texts)
        token_ids = [e.ids for e in encs]
        mask = [e.attention_mask for e in encs]
        return {
            "token_ids": ops.convert_to_tensor(token_ids, dtype="int32"),
            "padding_mask": ops.convert_to_tensor(mask, dtype="int32"),
        }

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        # `to_id_list` rather than `.numpy()`: the latter raises on a CUDA
        # tensor, which is what the torch backend hands back on a GPU.
        ids = self.to_id_list(ids)
        if skip_special_tokens:
            skip = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
            ids = [i for i in ids if i not in skip]
        return self._tok.decode(ids, skip_special_tokens=False)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "variant": self.variant,
                "tokenizer_file": self.tokenizer_file,
                "max_seq_len": self.max_seq_len,
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
                "unk_token_id": self.unk_token_id,
            }
        )
        return config
