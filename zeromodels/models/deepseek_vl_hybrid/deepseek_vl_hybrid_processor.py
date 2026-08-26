import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseProcessor
from zeromodels.models.deepseek_vl.deepseek_vl_processor import DeepseekVLProcessor

from .deepseek_vl_hybrid_image_processor import DeepseekVLHybridImageProcessor
from .deepseek_vl_hybrid_tokenizer import DeepseekVLHybridTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class DeepseekVLHybridProcessor(DeepseekVLProcessor):
    """Image + text -> model inputs for DeepSeek-VL Hybrid (7B).

    Same chat formatting and ``<image_placeholder>`` expansion as
    :class:`~zeromodels.models.deepseek_vl.DeepseekVLProcessor`, but drives the
    dual-resolution image processor: ``call`` returns ``high_res_pixel_values``
    (1024, SAM tower) alongside ``pixel_values`` (384, SigLIP tower).
    """

    TOKENIZER_CLS = DeepseekVLHybridTokenizer
    IMAGE_PROCESSOR_CLS = DeepseekVLHybridImageProcessor

    def __init__(
        self,
        variant=None,
        hf_id=None,
        num_image_tokens=576,
        tokenizer=None,
        image_processor=None,
        **kwargs,
    ):
        # Skip DeepseekVLProcessor.__init__ (it builds the 1.3B single-resolution
        # image processor); build the hybrid components instead.
        BaseProcessor.__init__(self, **kwargs)
        self.variant = variant
        self.hf_id = hf_id
        self.num_image_tokens = num_image_tokens
        self.image_processor = image_processor or DeepseekVLHybridImageProcessor()
        self.tokenizer = tokenizer or DeepseekVLHybridTokenizer(
            variant=variant, hf_id=hf_id
        )
        self.image_token = self.tokenizer.image_token

    def call(
        self,
        conversation=None,
        text=None,
        images=None,
        messages=None,
        add_generation_prompt=True,
    ):
        if conversation is not None:
            texts, extracted = self.render_conversations(
                conversation, add_generation_prompt
            )
            if images is None:
                images = extracted
        elif messages is not None:
            texts = [self.apply_chat_template(messages, add_generation_prompt)]
        elif text is not None:
            texts = [text] if isinstance(text, str) else list(text)
        else:
            raise ValueError("Provide a `conversation`, `messages`, or `text`.")
        texts = [self.expand_image_tokens(t) for t in texts]

        out = {}
        if images is not None:
            feats = self.image_processor(images)
            out["pixel_values"] = ops.convert_to_tensor(feats["pixel_values"])
            out["high_res_pixel_values"] = ops.convert_to_tensor(
                feats["high_res_pixel_values"]
            )
        ids = [self.tokenizer.encode(t) for t in texts]
        max_len = max(len(x) for x in ids)
        input_ids = np.zeros((len(ids), max_len), dtype="int32")
        attention_mask = np.zeros((len(ids), max_len), dtype="int32")
        for i, seq in enumerate(ids):
            input_ids[i, : len(seq)] = seq
            attention_mask[i, : len(seq)] = 1
        out["input_ids"] = ops.convert_to_tensor(input_ids)
        out["attention_mask"] = ops.convert_to_tensor(attention_mask)
        return out
