import keras

from zeromodels.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen2VLTokenizer(BaseTokenizer):
    """Qwen2 BPE tokenizer (``tokenizers`` backend).

    Args:
        hf_id: Hub repo to pull ``tokenizer.json`` from.
        tokenizer_file: Explicit path to a ``tokenizer.json`` (overrides the
            download).
    """

    def __init__(self, hf_id=None, tokenizer_file=None, **kwargs):
        super().__init__(**kwargs)
        from tokenizers import AddedToken, Tokenizer

        tokenizer_file = self.resolve_tokenizer_json_from_hf(hf_id, tokenizer_file)
        self.hf_id = hf_id
        self.tokenizer_file = tokenizer_file
        self._tok = Tokenizer.from_file(tokenizer_file)

        for pad_token in ("<|image_pad|>", "<|video_pad|>"):
            if self._tok.token_to_id(pad_token) is None:
                self._tok.add_special_tokens(
                    [AddedToken(pad_token, special=True, normalized=False)]
                )

        self.image_token = "<|image_pad|>"
        self.video_token = "<|video_pad|>"
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"
        self.eos_token = "<|im_end|>"
        self.image_token_id = self._tok.token_to_id(self.image_token)
        self.eos_token_id = self._tok.token_to_id(self.eos_token)

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        """Text -> list[int] (no auto special tokens; the template carries them)."""
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
        config.update({"hf_id": self.hf_id, "tokenizer_file": self.tokenizer_file})
        return config
