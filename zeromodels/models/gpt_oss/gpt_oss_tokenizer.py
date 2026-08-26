import keras

from zeromodels.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class GptOssTokenizer(BaseTokenizer):
    """GPT-OSS tokenizer (``o200k_harmony``, ``tokenizers`` backend).

    Loads the model's ``tokenizer.json`` and exposes ``encode`` / ``decode`` plus
    a ``call`` that tokenizes text(s) or a chat ``messages`` list (rendered with a
    minimal Harmony template) into padded ``{"input_ids", "attention_mask"}``.
    """

    def __init__(self, hf_id=None, tokenizer_file=None, **kwargs):
        super().__init__(**kwargs)
        from tokenizers import Tokenizer

        tokenizer_file = self.resolve_tokenizer_json_from_hf(hf_id, tokenizer_file)
        self.hf_id = hf_id
        self.tokenizer_file = tokenizer_file
        self._tok = Tokenizer.from_file(tokenizer_file)
        # Harmony assistant turns end with <|return|>; fall back to <|endoftext|>.
        self.eos_token = "<|return|>"
        self.eos_token_id = self._tok.token_to_id(self.eos_token)
        if self.eos_token_id is None:
            self.eos_token = "<|endoftext|>"
            self.eos_token_id = self._tok.token_to_id(self.eos_token)

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text, add_special_tokens=False).ids

    def apply_chat_template(self, messages, add_generation_prompt=True):
        text = ""
        for m in messages:
            text += f"<|start|>{m['role']}<|message|>{m['content']}<|end|>"
        if add_generation_prompt:
            text += "<|start|>assistant"
        return text

    def call(self, inputs):
        texts = self.normalize_texts(inputs)
        input_ids, attention_mask = self.pad_batch([self.encode(t) for t in texts])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens=True):
        return self._tok.decode(
            self.to_id_list(ids), skip_special_tokens=skip_special_tokens
        )

    def channel_text(self, raw, channel):
        """Text of a Harmony ``channel`` in a marker-carrying decode, or ``None``.

        A turn is ``<|channel|>{name}<|message|>{text}{stop}``; return ``{text}``
        for the requested channel, cut at the next control token.
        """
        marker = f"<|channel|>{channel}<|message|>"
        start = raw.find(marker)
        if start == -1:
            return None
        tail = raw[start + len(marker) :]
        stops = (
            "<|end|>",
            "<|return|>",
            "<|call|>",
            "<|start|>",
            "<|channel|>",
            "<|endoftext|>",
        )
        cuts = [i for i in (tail.find(s) for s in stops) if i != -1]
        return (tail[: min(cuts)] if cuts else tail).strip()

    def parse_harmony(self, ids):
        """Split a generated Harmony turn into ``(final_answer, reasoning)``.

        Either is ``None`` when its channel is absent (a response truncated before
        the ``final`` channel, or plain non-Harmony text)."""
        raw = self.decode(ids, skip_special_tokens=False)
        return self.channel_text(raw, "final"), self.channel_text(raw, "analysis")

    def decode_message(self, ids, role="assistant", skip_special_tokens=True):
        """Decode a generated turn into a chat-message ``dict``, Harmony-aware.

        ``content`` is the ``final`` channel (the answer), and ``thinking`` (added
        only when present) is the ``analysis`` channel (the chain-of-thought), so
        the reasoning is separated out instead of munged into the content. Falls
        back to a plain decode for non-Harmony text.
        """
        final, thinking = self.parse_harmony(ids)
        if final is not None:
            content = final
        elif thinking is not None:
            content = ""  # ran out of tokens before the final channel
        else:
            content = self.decode(ids, skip_special_tokens)
        message = {"role": role, "content": content}
        if thinking:
            message["thinking"] = thinking
        return message

    def get_config(self):
        config = super().get_config()
        config.update({"hf_id": self.hf_id, "tokenizer_file": self.tokenizer_file})
        return config
