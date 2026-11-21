import keras

from zeromodels.base import BaseTokenizer

IMAGE_TOKEN = "<|unused_200054|>"
EOS_TOKEN = "<|content_model_end_sampling|>"
PAD_TOKEN = "<|endoftext|>"

EFFORT_MAP = {
    "none": 0.0,
    "minimal": 0.1,
    "low": 0.2,
    "medium": 0.7,
    "high": 0.9,
    "xhigh": 0.99,
    "max": 0.99,
}

ROLE_TOKEN = {
    "user": "<|message_user|>",
    "assistant": "<|message_model|>",
    "system": "<|message_system|>",
    "tool": "<|message_tool|>",
}


@keras.saving.register_keras_serializable(package="zeromodels")
class Inkling2LoopTokenizer(BaseTokenizer):
    """Inkling-2-Mini tokenizer backed by Hugging Face ``tokenizers`` (``tokenizer.json``)."""

    def __init__(self, hf_id=None, tokenizer_file=None, **kwargs):
        super().__init__(**kwargs)
        from tokenizers import Tokenizer

        tokenizer_file = self.resolve_tokenizer_json_from_hf(hf_id, tokenizer_file)
        self.hf_id = hf_id
        self.tokenizer_file = tokenizer_file
        self._tok = Tokenizer.from_file(tokenizer_file)
        self.image_token = IMAGE_TOKEN
        self.eos_token = EOS_TOKEN
        self.pad_token = PAD_TOKEN
        self.image_token_id = self._tok.token_to_id(IMAGE_TOKEN)
        self.eos_token_id = self._tok.token_to_id(EOS_TOKEN)
        self.pad_token_id = self._tok.token_to_id(PAD_TOKEN)

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text, add_special_tokens=False).ids

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt=True,
        reasoning_effort="high",
    ):
        """Render OpenAI-style chat messages using Inkling-2-Mini's message tags."""
        if isinstance(reasoning_effort, str):
            key = reasoning_effort.strip()
            if key not in EFFORT_MAP:
                raise ValueError(f"Unknown reasoning_effort: {reasoning_effort!r}")
            eff = EFFORT_MAP[key]
        else:
            eff = float(reasoning_effort)
        eff_str = "0" if eff == 0.0 else str(eff)
        effort_block = (
            f"<|message_system|><|content_text|>Thinking effort level: {eff_str}<|end_message|>"
        )

        parts = []
        effort_emitted = False
        for msg in messages:
            role = msg["role"]
            if role not in ROLE_TOKEN:
                raise ValueError(f"Unknown message role: {role!r}")
            if not effort_emitted and role != "system":
                parts.append(effort_block)
                effort_emitted = True
            parts.append(ROLE_TOKEN[role])
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(f"<|content_text|>{content}")
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        parts.append(f"<|content_text|>{item}")
                    elif item.get("type") == "text":
                        parts.append(f"<|content_text|>{item.get('text', '')}")
                    elif item.get("type") == "image":
                        parts.append(f"<|content_image|>{self.image_token}")
            parts.append("<|end_message|>")

        if not effort_emitted:
            parts.append(effort_block)
        if add_generation_prompt:
            parts.append("<|message_model|>")
        return "".join(parts)

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
