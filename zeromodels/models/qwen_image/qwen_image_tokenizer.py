"""Qwen-Image tokenizer: Qwen2 ChatML template for prompt encoding."""

import keras

from zeromodels.models.qwen2.qwen2_tokenizer import Qwen2Tokenizer

# Diffusers ``QwenImagePipeline.prompt_template_encode``
PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_START_IDX = 34


@keras.saving.register_keras_serializable(package="zeromodels")
class QwenImageTokenizer(Qwen2Tokenizer):
    """Tokenizer for Qwen-Image text-to-image.

    Wraps each prompt in the Diffusers ChatML template, then BPE-encodes with
    the Qwen2 tokenizer. Returns ``input_ids`` / ``attention_mask`` for
    :meth:`QwenImageTextToImage.generate` (the task drops the template prefix
    after the text encoder).

    Load::

        tok = QwenImageTokenizer.from_weights("zeromodels/qwen-image")
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
        # Diffusers pads to tokenizer_max_length + drop_idx (= 1024 + 34).
        self.max_seq_len = max_seq_len
        self.tokenizer_max_length = max_seq_len
        super().__init__(hf_id=hf_id, tokenizer_file=tokenizer_file, **kwargs)

    def format_prompt(self, text):
        return self.prompt_template.format(text)

    def call(self, inputs):
        texts = self.normalize_texts(inputs)
        templated = [self.format_prompt(t) for t in texts]
        # Pad/truncate to max_seq_len + template prefix budget (Diffusers).
        max_length = self.tokenizer_max_length + self.prompt_template_start_idx
        encoded = [self.encode(t)[:max_length] for t in templated]
        input_ids, attention_mask = self.pad_batch(encoded)
        # Cap length for the static text-encoder graph when shorter.
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def get_config(self):
        config = super().get_config()
        config.update({"max_seq_len": self.max_seq_len})
        return config
