import keras
import numpy as np
from tokenizers import Tokenizer

from zeromodels.base import BaseTokenizer

SAM3_CONTEXT_LENGTH = 32
SAM3_VOCAB_SIZE = 49408
SAM3_BOS_TOKEN_ID = 49406
SAM3_EOS_TOKEN_ID = 49407
SAM3_PAD_TOKEN_ID = 49407
SAM3_HF_REPO = "zeromodels/sam3"


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM3CLIPTokenizer(BaseTokenizer):
    """BPE tokenizer for SAM3's CLIP text encoder (max_seq_len=32).

    SAM3's text encoder reuses the OpenAI CLIP tokenizer. The ``tokenizer.json``
    is pulled on the fly from the ungated ``zeromodels/sam3`` repo (identical to
    the gated ``facebook/sam3`` file, no license gate). Pass an explicit ``hf_id``
    (e.g. ``"facebook/sam3"`` or ``"openai/clip-vit-base-patch16"``) or
    ``tokenizer_file`` to override. CLIP truncation + ``<|endoftext|>`` padding to
    ``max_seq_len`` (32) is re-enabled on load.

    Args:
        hf_id: HF repo to pull ``tokenizer.json`` from (default ``"zeromodels/sam3"``).
        tokenizer_file: Optional explicit ``tokenizer.json`` path (overrides hf_id).
        max_seq_len: Max sequence length (default 32 for SAM3).

    Usage:
        tokenizer = SAM3CLIPTokenizer()                 # pulls facebook/sam3
        input_ids, attention_mask = tokenizer.encode("a cat")
        # input_ids: (1, 32) int32, attention_mask: (1, 32) float32
    """

    def __init__(
        self,
        hf_id=SAM3_HF_REPO,
        tokenizer_file=None,
        max_seq_len=SAM3_CONTEXT_LENGTH,
        **kwargs,
    ):
        super().__init__(**kwargs)
        tokenizer_file = self.resolve_tokenizer_json_from_hf(hf_id, tokenizer_file)
        self.hf_id = hf_id
        self.tokenizer_file = tokenizer_file
        self.max_seq_len = max_seq_len
        self.bos_token_id = SAM3_BOS_TOKEN_ID
        self.eos_token_id = SAM3_EOS_TOKEN_ID
        self.pad_token_id = SAM3_PAD_TOKEN_ID

        tok = Tokenizer.from_file(tokenizer_file)
        tok.enable_truncation(max_length=max_seq_len)
        tok.enable_padding(
            pad_id=self.pad_token_id, pad_token="<|endoftext|>", length=max_seq_len
        )
        self._tok = tok

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(hf_id=repo, **kwargs)

    def encode(self, text):
        texts = self.normalize_texts(text)
        encs = self._tok.encode_batch(texts)
        input_ids = np.array([e.ids for e in encs], dtype=np.int32)
        attention_mask = np.array([e.attention_mask for e in encs], dtype=np.float32)
        return input_ids, attention_mask

    def decode(self, token_ids):
        skip = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
        keep = [i for i in self.to_id_list(token_ids) if i not in skip]
        text = self._tok.decode(keep, skip_special_tokens=False)
        return text.replace("</w>", " ").strip()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hf_id": self.hf_id,
                "tokenizer_file": self.tokenizer_file,
                "max_seq_len": self.max_seq_len,
            }
        )
        return config
