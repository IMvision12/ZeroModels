import keras

from zeromodels.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3Tokenizer(BaseTokenizer):
    """Gemma 3 SentencePiece-BPE tokenizer (``tokenizers`` backend).

    Loads the model's ``tokenizer.json`` (downloaded on the fly from ``hf_id``
    when no explicit file is given; the official google repos are gated:
    public mirrors like ``unsloth/gemma-3-4b-it`` work too) and exposes
    ``encode`` / ``decode`` plus the image special tokens
    (``<start_of_image>`` / ``<end_of_image>`` / ``<image_soft_token>``).
    ``call`` tokenizes text(s) or a chat ``messages`` list (rendered with the
    ``<start_of_turn>`` template) into padded
    ``{"input_ids", "attention_mask"}`` with ``<bos>`` prepended.
    """

    def __init__(self, hf_id=None, tokenizer_file=None, **kwargs):
        super().__init__(**kwargs)
        from tokenizers import Tokenizer

        tokenizer_file = self.resolve_tokenizer_json_from_hf(hf_id, tokenizer_file)
        self.hf_id = hf_id
        self.tokenizer_file = tokenizer_file
        self._tok = Tokenizer.from_file(tokenizer_file)
        self.bos_token = "<bos>"
        self.eos_token = "<end_of_turn>"
        self.image_token = "<image_soft_token>"
        self.boi_token = "<start_of_image>"
        self.eoi_token = "<end_of_image>"
        self.bos_token_id = self._tok.token_to_id(self.bos_token)
        self.eos_token_id = self._tok.token_to_id(self.eos_token)
        self.image_token_id = self._tok.token_to_id(self.image_token)

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text, add_special_tokens=False).ids

    def apply_chat_template(self, messages, add_generation_prompt=True):
        # Gemma renames the assistant role to "model"; system content is folded
        # into the first user turn; image content items become a single
        # <start_of_image> marker (expanded by the processor).
        system = None
        if messages and messages[0].get("role") == "system":
            system = messages[0]["content"]
            messages = messages[1:]
        text = ""
        for i, m in enumerate(messages):
            role = "model" if m["role"] == "assistant" else m["role"]
            content = m["content"]
            if not isinstance(content, str):
                body = ""
                for item in content:
                    if item.get("type") == "image":
                        body += self.boi_token
                    elif item.get("type") == "text":
                        body += item["text"]
                content = body
            if i == 0 and system is not None and role == "user":
                content = f"{system}\n\n{content}"
            text += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
        if add_generation_prompt:
            text += "<start_of_turn>model\n"
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
