import keras

from zeromodels.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3nTokenizer(BaseTokenizer):
    """Gemma 3n SentencePiece-BPE tokenizer (``tokenizers`` backend).

    Loads the model's ``tokenizer.json`` (downloaded on the fly from ``hf_id``
    when no explicit file is given) and exposes ``encode`` / ``decode`` plus a
    ``call`` that tokenizes text(s) or a chat ``messages`` list (rendered with the
    ``<start_of_turn>`` template) into padded ``{"input_ids", "attention_mask"}``
    with ``<bos>`` prepended. A ``TemplateProcessing`` post-processor prepends
    ``<bos>`` and ``enable_padding`` right-pads the batch, so ``call`` delegates to
    :meth:`BaseTokenizer.encode_batch_to_inputs`. Image / audio content items in a
    chat message become single soft-token markers that :class:`Gemma3nProcessor`
    later expands to the full ``<start_of_image>`` + ``<image_soft_token>`` * n +
    ``<end_of_image>`` (and audio) sequences.
    """

    def __init__(self, hf_id=None, tokenizer_file=None, **kwargs):
        super().__init__(**kwargs)
        from tokenizers import Tokenizer
        from tokenizers.processors import TemplateProcessing

        tokenizer_file = self.resolve_tokenizer_json_from_hf(hf_id, tokenizer_file)
        self.hf_id = hf_id
        self.tokenizer_file = tokenizer_file
        self._tok = Tokenizer.from_file(tokenizer_file)
        self.bos_token = "<bos>"
        self.eos_token = "<end_of_turn>"
        self.pad_token = "<pad>"
        self.bos_token_id = self._tok.token_to_id(self.bos_token)
        self.eos_token_id = self._tok.token_to_id(self.eos_token)
        pad_id = self._tok.token_to_id(self.pad_token)
        self.pad_token_id = 0 if pad_id is None else pad_id
        # Image / audio soft-token markers.
        self.image_token = "<image_soft_token>"
        self.boi_token = "<start_of_image>"
        self.eoi_token = "<end_of_image>"
        self.audio_token = "<audio_soft_token>"
        self.boa_token = "<start_of_audio>"
        self.eoa_token = "<end_of_audio>"
        self.image_token_id = self._tok.token_to_id(self.image_token)
        self.audio_token_id = self._tok.token_to_id(self.audio_token)
        self._tok.post_processor = TemplateProcessing(
            single=f"{self.bos_token}:0 $A:0",
            pair=f"{self.bos_token}:0 $A:0 $B:1",
            special_tokens=[(self.bos_token, self.bos_token_id)],
        )
        self._tok.enable_padding(pad_id=self.pad_token_id, pad_token=self.pad_token)

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text, add_special_tokens=False).ids

    def apply_chat_template(self, messages, add_generation_prompt=True):
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
                    item_type = item.get("type")
                    if item_type in ("image", "image_url"):
                        body += self.image_token
                    elif item_type in ("audio", "input_audio"):
                        body += self.audio_token
                    elif item_type == "text":
                        body += item["text"]
                content = body
            if i == 0 and system is not None and role == "user":
                content = f"{system}\n\n{content}"
            text += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
        if add_generation_prompt:
            text += "<start_of_turn>model\n"
        return text

    def call(self, inputs):
        return self.encode_batch_to_inputs(inputs, token_type_ids=False)

    def decode(self, ids, skip_special_tokens=True):
        return self._tok.decode(
            self.to_id_list(ids), skip_special_tokens=skip_special_tokens
        )

    def get_config(self):
        config = super().get_config()
        config.update({"hf_id": self.hf_id, "tokenizer_file": self.tokenizer_file})
        return config
