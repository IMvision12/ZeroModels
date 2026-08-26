import keras
import numpy as np

from zeromodels.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class OneFormerTokenizer(BaseTokenizer):
    """CLIP-BPE task tokenizer for OneFormer.

    Renders the chosen ``task`` (``"semantic"`` / ``"instance"`` /
    ``"panoptic"``) as ``"the task is {task}"``, tokenizes it with the CLIP BPE
    vocabulary and pads to ``task_seq_len`` with ``<|endoftext|>``: the float
    task-id vector the model's task MLP consumes.

    Loads ``tokenizer.json`` from a Hub repo id
    (``from_weights("zeromodels/oneformer_ade20k_swin_tiny")``), on the fly
    from ``hf_id``, or from an explicit ``tokenizer_file``.

    Args:
        variant: Optional variant label, kept for reference; all variants share
            the same CLIP tokenizer.
        task_seq_len: Padded task-prompt length in tokens (77).
        hf_id: Hub repo to pull ``tokenizer.json`` from (on-the-fly path).
        tokenizer_file: Explicit path to a ``tokenizer.json`` (overrides both).
    """

    def __init__(
        self, variant=None, task_seq_len=77, hf_id=None, tokenizer_file=None, **kwargs
    ):
        super().__init__(**kwargs)
        from tokenizers import Tokenizer

        self.variant = variant
        self.task_seq_len = task_seq_len
        self.hf_id = hf_id
        tokenizer_file = self.resolve_tokenizer_json_from_hf(hf_id, tokenizer_file)
        self.tokenizer_file = tokenizer_file
        self._tok = Tokenizer.from_file(tokenizer_file)
        self.eot_token_id = self._tok.token_to_id("<|endoftext|>")

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def tokenize_task(self, task):
        ids = self._tok.encode(f"the task is {task}").ids
        ids = ids[: self.task_seq_len]
        ids = ids + [self.eot_token_id] * (self.task_seq_len - len(ids))
        return np.asarray(ids, dtype="float32")

    def call(self, task="panoptic"):
        tasks = task if isinstance(task, (list, tuple)) else [task]
        ids = np.stack([self.tokenize_task(t) for t in tasks])
        return {"task_inputs": keras.ops.convert_to_tensor(ids)}

    def decode(self, ids, skip_special_tokens=True):
        return self._tok.decode(
            self.to_id_list(ids), skip_special_tokens=skip_special_tokens
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "variant": self.variant,
                "task_seq_len": self.task_seq_len,
                "hf_id": self.hf_id,
                "tokenizer_file": self.tokenizer_file,
            }
        )
        return config
