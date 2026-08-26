import keras

from zeromodels.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class LlamaTokenizer(BaseTokenizer):
    """Llama 3 family BPE tokenizer (``tokenizers`` backend).

    Loads the model's ``tokenizer.json`` (downloaded on the fly from ``hf_id``
    when no explicit file is given; the official meta-llama repos are gated,
    public mirrors like ``unsloth/Llama-3.2-1B-Instruct`` work too) and
    exposes ``encode`` / ``decode`` plus a ``call`` that tokenizes text(s) or
    a chat ``messages`` list (rendered with the ``<|start_header_id|>``
    template) into padded ``{"input_ids", "attention_mask"}`` with the
    ``<|begin_of_text|>`` bos prepended, matching the Hugging Face tokenizer's
    default output.
    """

    def __init__(self, hf_id=None, tokenizer_file=None, **kwargs):
        super().__init__(**kwargs)
        from tokenizers import Tokenizer

        tokenizer_file = self.resolve_tokenizer_json_from_hf(hf_id, tokenizer_file)
        self.hf_id = hf_id
        self.tokenizer_file = tokenizer_file
        self._tok = Tokenizer.from_file(tokenizer_file)
        self.bos_token = "<|begin_of_text|>"
        self.eos_token = "<|eot_id|>"
        self.bos_token_id = self._tok.token_to_id(self.bos_token)
        self.eos_token_id = self._tok.token_to_id(self.eos_token)

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text, add_special_tokens=False).ids

    def apply_chat_template(self, messages, add_generation_prompt=True, system=None):
        text = ""
        if system is not None and not any(m.get("role") == "system" for m in messages):
            text += f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        for m in messages:
            text += (
                f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n"
                f"{m['content']}<|eot_id|>"
            )
        if add_generation_prompt:
            text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return text

    def call(self, inputs):
        texts = self.normalize_texts(inputs)
        input_ids, attention_mask = self.pad_batch(
            [[self.bos_token_id] + self.encode(t) for t in texts]
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens=True):
        return self._tok.decode(
            self.to_id_list(ids), skip_special_tokens=skip_special_tokens
        )

    def get_config(self):
        config = super().get_config()
        config.update({"hf_id": self.hf_id, "tokenizer_file": self.tokenizer_file})
        return config
