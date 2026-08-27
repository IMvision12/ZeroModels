import itertools
from typing import List, Union

import keras
from tokenizers import Tokenizer

from zeromodels.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class GraniteSpeech5Tokenizer(BaseTokenizer):
    """CTC tokenizer for Granite Speech 5.0 (Parakeet-style, ``tokenizers`` backend).

    Loads the fast ``tokenizer.json``. ``decode`` performs CTC decoding of the
    model's per-frame argmax ids: group consecutive duplicate ids (keep one per
    run), drop the blank id (``<|blank|>`` = ``pad_token_id`` = 0), then map the
    surviving ids to text. ``tokenize`` runs the plain encoder, for building CTC
    training labels. Load via ``GraniteSpeech5Tokenizer.from_weights(...)`` or
    ``from_release("hf:ibm-granite/granite-speech-5.0-470m-turboctc")``.
    """

    DEFAULT_VARIANT = "granite-speech-5.0-470m-turboctc"

    def __init__(
        self,
        variant: str = None,
        tokenizer_file: str = None,
        pad_token: str = "<|blank|>",
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.variant = variant or self.DEFAULT_VARIANT
        tokenizer_file = self.resolve_tokenizer_json_from_hf(
            f"zeromodels/{self.variant}", tokenizer_file
        )
        self.tokenizer_file = tokenizer_file
        self.pad_token = pad_token
        tok = Tokenizer.from_file(tokenizer_file)
        self._tok = tok
        resolved = tok.token_to_id(pad_token)
        self.pad_token_id = resolved if resolved is not None else pad_token_id

    @classmethod
    def from_hf(cls, repo, **kwargs):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo, "tokenizer.json")
        return cls(tokenizer_file=path, **kwargs)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def tokenize(
        self, text: Union[str, List[str]]
    ) -> Union[List[int], List[List[int]]]:
        if isinstance(text, str):
            return self._tok.encode(text, add_special_tokens=False).ids
        return [e.ids for e in self._tok.encode_batch(text, add_special_tokens=False)]

    def ctc_collapse(self, ids) -> List[int]:
        ids = [int(i) for i in ids]
        ids = [group[0] for group in itertools.groupby(ids)]  # group consecutive
        return [i for i in ids if i != self.pad_token_id]  # drop the blank

    def decode(self, token_ids, skip_special_tokens: bool = True):
        if hasattr(token_ids, "cpu"):
            token_ids = token_ids.cpu()
        if hasattr(token_ids, "numpy"):
            token_ids = token_ids.numpy()
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        def decode_row(ids):
            return self._tok.decode(
                self.ctc_collapse(ids), skip_special_tokens=skip_special_tokens
            )

        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            return [decode_row(r) for r in token_ids]
        return decode_row(token_ids)

    def detokenize(self, token_ids, skip_special_tokens: bool = True):
        return self.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self, token_ids_batch, skip_special_tokens: bool = True
    ) -> List[str]:
        if hasattr(token_ids_batch, "cpu"):
            token_ids_batch = token_ids_batch.cpu()
        if hasattr(token_ids_batch, "numpy"):
            token_ids_batch = token_ids_batch.numpy()
        out = []
        for row in token_ids_batch:
            row = row.tolist() if hasattr(row, "tolist") else list(row)
            out.append(self.decode(row, skip_special_tokens))
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "variant": self.variant,
                "tokenizer_file": self.tokenizer_file,
                "pad_token": self.pad_token,
                "pad_token_id": self.pad_token_id,
            }
        )
        return config
