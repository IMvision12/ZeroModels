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

    def ctc_offsets(self, ids, frame_seconds: float) -> List[dict]:
        """Word-level timestamps from per-frame CTC argmax ids.

        Collapses each run of identical ids into one token carrying its ``[start,
        end]`` frame span (the blank is dropped), groups the subword tokens into words
        at the space marker (``Ġ`` for byte-level BPE, ``▁`` for sentencepiece), and
        converts frame spans to seconds via ``frame_seconds`` (the audio time one CTC
        frame covers). CTC alignment is spiky, so a timestamp marks roughly where a word
        is emitted. Returns ``[{"text": word, "timestamp": (start, end)}, ...]``.
        """
        ids = [int(i) for i in ids]
        # Runs of identical ids collapse to one token with an inclusive frame span; a
        # trailing blank sentinel flushes the final run. Blanks are dropped.
        tokens = []  # (id, start_frame, end_frame)
        prev, run_start = self.pad_token_id, 0
        for frame, tid in enumerate(ids + [self.pad_token_id]):
            if tid != prev:
                if prev != self.pad_token_id:
                    tokens.append((prev, run_start, frame - 1))
                prev, run_start = tid, frame

        words = []  # each: {"ids": [...], "start": frame, "end": frame}
        for tid, start, end in tokens:
            piece = self._tok.id_to_token(tid) or ""
            if not words or piece.startswith(("Ġ", "▁")):
                words.append({"ids": [tid], "start": start, "end": end})
            else:
                words[-1]["ids"].append(tid)
                words[-1]["end"] = end

        offsets = []
        for word in words:
            text = self._tok.decode(word["ids"], skip_special_tokens=True).strip()
            if not text:
                continue
            offsets.append(
                {
                    "text": text,
                    "timestamp": (
                        round(word["start"] * frame_seconds, 2),
                        round((word["end"] + 1) * frame_seconds, 2),
                    ),
                }
            )
        return offsets

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
