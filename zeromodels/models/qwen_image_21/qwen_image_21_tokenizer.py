import keras

from zeromodels.models.qwen3.qwen3_tokenizer import Qwen3Tokenizer

SYS_PROMPT = "Comprehend and analyze the provided prompt."
PROMPT_TEMPLATE = (
    f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n"
    f"<|im_start|>user\n{{}}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_START_IDX = 14


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImage21Tokenizer(Qwen3Tokenizer):
    """Tokenizer for Qwen-Image-2.1 text-to-image.

    Wraps each prompt in the Diffusers ChatML template (system prompt
    ``Comprehend and analyze the provided prompt.``), then BPE-encodes.
    Returns ``input_ids`` / ``attention_mask`` for
    :meth:`QwenImage21TextToImage.generate`.

    Load::

        tok = QwenImage21Tokenizer.from_weights("zeromodels/qwen-image-2.1")
        model.generate(**tok("a photo of a cat"))
    """

    prompt_template = PROMPT_TEMPLATE
    prompt_template_start_idx = PROMPT_TEMPLATE_START_IDX

    def __init__(
        self,
        hf_id=None,
        tokenizer_file=None,
        max_seq_len=1024,
        **kwargs,
    ):
        self.max_seq_len = max_seq_len
        self.tokenizer_max_length = max_seq_len
        super().__init__(hf_id=hf_id, tokenizer_file=tokenizer_file, **kwargs)

    def format_prompt(self, text):
        return self.prompt_template.format(text if text else " ")

    def call(self, inputs):
        texts = self.normalize_texts(inputs)
        templated = [self.format_prompt(t) for t in texts]
        max_length = self.tokenizer_max_length + self.prompt_template_start_idx
        encoded = [self.encode(t)[:max_length] for t in templated]
        input_ids, attention_mask = self.pad_batch(encoded)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def get_config(self):
        config = super().get_config()
        config.update({"max_seq_len": self.max_seq_len})
        return config
