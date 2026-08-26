import base64

import keras

from zeromodels.base import BaseTokenizer

DEFAULT_HF_ID = "moonshotai/Kimi-K2.6"
VOCAB_FILE = "tiktoken.model"
NUM_RESERVED_SPECIAL_TOKENS = 256

# The tokenizer is tiktoken-based and ships no tokenizer.json, so the repo's
# resolve_tokenizer_json path does not apply -- the BPE ranks come from
# tiktoken.model and the special-token names from tokenizer_config.json.
PAT_STR = "|".join(
    [
        r"[\p{Han}]+",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)

# Names of the 256 reserved slots that Kimi actually uses; the rest stay
# <|reserved_token_{i}|>. Offsets are relative to the first special id.
SPECIAL_TOKEN_NAMES = {
    0: "[BOS]",
    1: "[EOS]",
    2: "<|im_end|>",
    3: "<|im_user|>",
    4: "<|im_assistant|>",
    6: "<|start_header_id|>",
    7: "<|end_header_id|>",
    9: "[EOT]",
    10: "<|im_system|>",
    11: "<|tool_calls_section_begin|>",
    12: "<|tool_calls_section_end|>",
    13: "<|tool_call_begin|>",
    14: "<|tool_call_argument_begin|>",
    15: "<|tool_call_end|>",
    17: "<|im_middle|>",
    18: "<|media_begin|>",
    19: "<|media_content|>",
    20: "<|media_end|>",
    21: "<|media_pad|>",
    22: "<think>",
    23: "</think>",
    254: "[UNK]",
    255: "[PAD]",
}


def load_bpe_ranks(path):
    """Parse a tiktoken ``.model`` file: one ``<base64 token> <rank>`` per line.

    Read directly rather than through ``tiktoken.load.load_tiktoken_bpe``, which
    pulls in ``blobfile`` for its cached reader.
    """
    ranks = {}
    with open(path, "rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            token, rank = line.split()
            ranks[base64.b64decode(token)] = int(rank)
    return ranks


@keras.saving.register_keras_serializable(package="zeromodels")
class KimiK25Tokenizer(BaseTokenizer):
    """Kimi K2.5 / K2.6 / K2.7-Code tiktoken tokenizer (163840 tokens).

    All three checkpoints share this tokenizer byte for byte. The 163584 BPE
    ranks come from ``tiktoken.model``; the last 256 ids are reserved special
    slots, of which Kimi names 23 (``[BOS]`` 163584 ... ``[PAD]`` 163839). Note
    the model's ``video_token_id`` (163840) is *outside* the vocabulary -- the
    processor splices it in rather than encoding it.

    Args:
        vocab_file: Path to a local ``tiktoken.model``.
        hf_id: Hub repo to pull ``tiktoken.model`` from when ``vocab_file`` is
            omitted.
    """

    HF_ID = DEFAULT_HF_ID

    def __init__(self, vocab_file=None, hf_id=None, **kwargs):
        super().__init__(**kwargs)
        self.hf_id = hf_id or self.HF_ID
        self.vocab_file = vocab_file or self.download_vocab(self.hf_id)

        ranks = load_bpe_ranks(self.vocab_file)
        base = len(ranks)
        self.special_tokens = {
            SPECIAL_TOKEN_NAMES.get(i, f"<|reserved_token_{i}|>"): base + i
            for i in range(NUM_RESERVED_SPECIAL_TOKENS)
        }

        import tiktoken

        self.encoding = tiktoken.Encoding(
            name="kimi_k25",
            pat_str=PAT_STR,
            mergeable_ranks=ranks,
            special_tokens=self.special_tokens,
        )
        self.vocab_size = self.encoding.n_vocab
        self.bos_token_id = self.special_tokens["[BOS]"]
        self.eos_token_id = self.special_tokens["<|im_end|>"]
        self.pad_token_id = self.special_tokens["[PAD]"]
        self.unk_token_id = self.special_tokens["[UNK]"]

    @staticmethod
    def download_vocab(hf_id):
        from huggingface_hub import hf_hub_download

        return hf_hub_download(hf_id, VOCAB_FILE)

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(hf_id=repo.removeprefix("hf:"), **kwargs)

    def encode(self, text, allowed_special="all"):
        return self.encoding.encode(text, allowed_special=allowed_special)

    def call(self, inputs, max_length=None):
        texts = self.normalize_texts(inputs)
        sequences = [self.encode(text) for text in texts]
        if max_length is not None:
            sequences = [s[:max_length] for s in sequences]
        input_ids, attention_mask = self.pad_batch(sequences, self.pad_token_id)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        ids = self.to_id_list(ids)
        if skip_special_tokens:
            specials = set(self.special_tokens.values())
            ids = [i for i in ids if i not in specials and i < self.vocab_size]
        else:
            ids = [i for i in ids if i < self.vocab_size]
        return self.encoding.decode(ids)

    def get_config(self):
        config = super().get_config()
        config.update({"vocab_file": self.vocab_file, "hf_id": self.hf_id})
        return config
