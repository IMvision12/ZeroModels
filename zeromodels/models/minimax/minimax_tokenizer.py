import keras

from zeromodels.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class MiniMaxTokenizer(BaseTokenizer):
    """MiniMax-Text-01 GPT-2-style byte-level BPE tokenizer (~200k vocab).

    Built from the checkpoint's ``vocab.json`` + ``merges.txt`` with the
    classic GPT-2 byte-level pre-tokenizer, matching the slow
    ``GPT2Tokenizer`` the HF checkpoint resolves to (the repo's
    ``tokenizer.json`` encodes a different pre-tokenizer and diverges from
    it). No BOS/EOS is added.

    Args:
        hf_id: Hub repo to pull ``vocab.json`` / ``merges.txt`` from.
        vocab_file / merges_file: Explicit paths (override the download).
    """

    def __init__(self, hf_id=None, vocab_file=None, merges_file=None, **kwargs):
        super().__init__(**kwargs)
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers

        if vocab_file is None or merges_file is None:
            from huggingface_hub import hf_hub_download

            vocab_file = vocab_file or hf_hub_download(hf_id, "vocab.json")
            merges_file = merges_file or hf_hub_download(hf_id, "merges.txt")
        self.hf_id = hf_id
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self._tok = Tokenizer(models.BPE.from_file(vocab_file, merges_file))
        self._tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self._tok.decoder = decoders.ByteLevel()
        self.eos_token = "<end_of_sentence>"
        # The specials live in the slow tokenizer's added_tokens_decoder, not
        # vocab.json, so the lookup misses; the checkpoint id is 200020.
        self.eos_token_id = self._tok.token_to_id(self.eos_token)
        if self.eos_token_id is None:
            self.eos_token_id = 200020

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text, add_special_tokens=False).ids

    def call(self, inputs):
        texts = self.normalize_texts(inputs)
        return {"input_ids": [self.encode(t) for t in texts]}

    def decode(self, ids, skip_special_tokens=True):
        return self._tok.decode(
            self.to_id_list(ids), skip_special_tokens=skip_special_tokens
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hf_id": self.hf_id,
                "vocab_file": self.vocab_file,
                "merges_file": self.merges_file,
            }
        )
        return config
