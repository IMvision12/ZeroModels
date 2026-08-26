import os
import string
from typing import List, Union

import keras
import numpy as np

from zeromodels.base import BaseTokenizer
from zeromodels.conversion import download_file

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

METACLIP2_MT5_EOS_TOKEN_ID = 1
METACLIP2_MT5_PAD_TOKEN_ID = 1
METACLIP2_MT5_UNK_TOKEN_ID = 2

DEFAULT_MT5_SENTENCEPIECE_URL = (
    "https://huggingface.co/zeromodels/metaclip2_mt5_worldwide_s16_224/"
    "resolve/main/spiece.model"
)


@keras.saving.register_keras_serializable(package="zeromodels")
class MetaClip2Mt5Tokenizer(BaseTokenizer):
    """SigLIP-style SentencePiece tokenizer for the MetaCLIP 2 mT5 variants.

    Replicates the reference ``SiglipTokenizer`` bit-close on the mT5-tokenizer
    MetaCLIP 2 checkpoints (250 100-token vocab built on top of mT5's
    SentencePiece model).

    Tokenization pipeline:

    1. **Lowercase** the input string and strip ASCII punctuation
       (matches the reference SigLIP preprocessing: without this the token ids
       drift from the reference).
    2. SentencePiece encode to integer pieces (no fairseq offset, no
       BOS).
    3. Append ``eos_token_id`` (``1``).
    4. Truncate (replacing the last token with EOS) or pad with
       ``pad_token_id`` to ``max_seq_len``: note that PAD and EOS
       share the same id ``1`` here, so the attention mask is the
       only reliable signal for "real vs padding".
    5. Build an attention mask: ``1`` for real tokens, ``0`` for
       padding.

    Args:
        sentencepiece_model_file: Path to ``spiece.model``. When
            ``None``, downloads from the default MetaCLIP 2 release
            URL on first use.
        max_seq_len: Maximum sequence length. Defaults to ``77``.
        eos_token_id: EOS token id. Defaults to ``1``.
        pad_token_id: PAD token id. Defaults to ``1`` (same as EOS,
            the SigLIP convention).
        unk_token_id: UNK token id. Defaults to ``2``.

    Example:
        >>> from zeromodels.models.metaclip2 import MetaClip2Mt5Tokenizer
        >>> tok = MetaClip2Mt5Tokenizer.from_weights("metaclip2_mt5_worldwide_s16_224")
        >>> out = tok(["A photo of a cat."])
        >>> out["token_ids"].shape       # (1, 77)
        >>> out["padding_mask"].shape    # (1, 77)
    """

    def __init__(
        self,
        sentencepiece_model_file: str = None,
        max_seq_len: int = 77,
        eos_token_id: int = METACLIP2_MT5_EOS_TOKEN_ID,
        pad_token_id: int = METACLIP2_MT5_PAD_TOKEN_ID,
        unk_token_id: int = METACLIP2_MT5_UNK_TOKEN_ID,
        **kwargs,
    ):
        super().__init__(**kwargs)
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise ImportError(
                "MetaClip2Mt5Tokenizer requires the `sentencepiece` package. "
                "Install with `pip install sentencepiece`."
            ) from exc

        if sentencepiece_model_file is None:
            sentencepiece_model_file = download_file(DEFAULT_MT5_SENTENCEPIECE_URL)
        if not os.path.exists(sentencepiece_model_file):
            raise FileNotFoundError(sentencepiece_model_file)

        self.sentencepiece_model_file = sentencepiece_model_file
        self.max_seq_len = max_seq_len
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id

        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(sentencepiece_model_file)

    def _encode_one(self, text: str):
        text = text.lower().translate(_PUNCT_TABLE)
        ids = self._sp.encode(text, out_type=int)
        ids = ids + [self.eos_token_id]
        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len - 1] + [self.eos_token_id]
        attention_mask = [1] * len(ids) + [0] * (self.max_seq_len - len(ids))
        ids = ids + [self.pad_token_id] * (self.max_seq_len - len(ids))
        return ids, attention_mask

    def call(self, inputs: Union[str, List[str]]):
        texts = self.normalize_texts(inputs)
        token_ids = np.zeros((len(texts), self.max_seq_len), dtype=np.int32)
        attention_mask = np.zeros((len(texts), self.max_seq_len), dtype=np.int32)
        for i, t in enumerate(texts):
            row_ids, row_mask = self._encode_one(t)
            token_ids[i] = row_ids
            attention_mask[i] = row_mask
        return {
            "token_ids": keras.ops.convert_to_tensor(token_ids, dtype="int32"),
            "padding_mask": keras.ops.convert_to_tensor(attention_mask, dtype="int32"),
        }

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        # `to_id_list` rather than `.numpy()`: the latter raises on a CUDA
        # tensor, which is what the torch backend hands back on a GPU.
        ids = self.to_id_list(ids)
        if skip_special_tokens:
            ids = [i for i in ids if i not in (self.eos_token_id, self.pad_token_id)]
        return self._sp.decode(ids)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sentencepiece_model_file": self.sentencepiece_model_file,
                "max_seq_len": self.max_seq_len,
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
                "unk_token_id": self.unk_token_id,
            }
        )
        return config
