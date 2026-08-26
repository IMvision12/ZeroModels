import keras

from zeromodels.base import BaseTokenizer

DEFAULT_SYSTEM = "You are a helpful assistant."


@keras.saving.register_keras_serializable(package="zeromodels")
class Qwen3NextTokenizer(BaseTokenizer):
    """Qwen3.5 BPE tokenizer (``tokenizers`` backend)."""

    def __init__(self, hf_id=None, tokenizer_file=None, **kwargs):
        super().__init__(**kwargs)
        from tokenizers import Tokenizer

        tokenizer_file = self.resolve_tokenizer_json_from_hf(hf_id, tokenizer_file)
        self.hf_id = hf_id
        self.tokenizer_file = tokenizer_file
        self._tok = Tokenizer.from_file(tokenizer_file)
        self.eos_token = "<|im_end|>"
        self.eos_token_id = self._tok.token_to_id(self.eos_token)

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text, add_special_tokens=False).ids

    def apply_chat_template(
        self, messages, add_generation_prompt=True, system=DEFAULT_SYSTEM
    ):
        text = ""
        if system is not None and not any(m.get("role") == "system" for m in messages):
            text += f"<|im_start|>system\n{system}<|im_end|>\n"
        for m in messages:
            text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def call(self, inputs):
        """Tokenize text(s) or a chat ``messages`` list into model inputs.

        Accepts a single string, a list of strings, or a ChatML-style
        conversation (``[{"role": ..., "content": ...}, ...]``), the latter is
        run through :meth:`apply_chat_template` first. Returns padded
        ``{"input_ids", "attention_mask"}`` (numpy) ready for ``model.generate``.
        """
        texts = self.normalize_texts(inputs)
        input_ids, attention_mask = self.pad_batch([self.encode(t) for t in texts])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens=True):
        return self._tok.decode(
            self.to_id_list(ids), skip_special_tokens=skip_special_tokens
        )

    def get_config(self):
        config = super().get_config()
        config.update({"hf_id": self.hf_id, "tokenizer_file": self.tokenizer_file})
        return config
