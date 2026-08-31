import keras

from zeromodels.base import BaseTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class JanusTokenizer(BaseTokenizer):
    """Janus-Pro BPE tokenizer (``tokenizers`` backend).

    Loads the model's ``tokenizer.json`` from a Hub repo (``zeromodels/<variant>``
    by default, or an explicit ``hf_id``) and exposes ``encode`` / ``decode`` plus
    the image special tokens (``<image_placeholder>`` / ``<begin_of_image>`` /
    ``<end_of_image>``). ``encode`` prepends the BOS id (the checkpoints use
    ``add_bos_token=True``); ``call`` returns unpadded id lists: the
    :class:`JanusProcessor` expands image placeholders and pads. Load by repo
    id like weights: ``JanusTokenizer.from_weights("zeromodels/janus_pro_1b")``.

    Args:
        variant: Janus variant key (no default; pass this or ``tokenizer_file``); resolves to the
            ``zeromodels/<variant>`` repo's tokenizer.json.
        hf_id: Explicit Hub repo to pull ``tokenizer.json`` from (overrides
            the variant).
        tokenizer_file: Explicit path to a ``tokenizer.json`` (overrides the
            download).
    """

    def __init__(self, variant=None, hf_id=None, tokenizer_file=None, **kwargs):
        super().__init__(**kwargs)
        from tokenizers import Tokenizer

        self.variant = variant
        self.hf_id = hf_id
        repo = (
            hf_id
            if hf_id is not None
            else (f"zeromodels/{self.variant}" if self.variant else None)
        )
        tokenizer_file = self.resolve_tokenizer_json_from_hf(repo, tokenizer_file)
        self.tokenizer_file = tokenizer_file
        self._tok = Tokenizer.from_file(tokenizer_file)
        self.image_token = "<image_placeholder>"
        self.boi_token = "<begin_of_image>"
        self.eoi_token = "<end_of_image>"
        self.bos_token = "<｜begin▁of▁sentence｜>"
        self.eos_token = "<｜end▁of▁sentence｜>"
        self.image_token_id = self._tok.token_to_id(self.image_token)
        self.bos_token_id = self._tok.token_to_id(self.bos_token)
        self.eos_token_id = self._tok.token_to_id(self.eos_token)

    @classmethod
    def from_hf(cls, repo, **kwargs):
        from huggingface_hub import hf_hub_download

        return cls(tokenizer_file=hf_hub_download(repo, "tokenizer.json"), **kwargs)

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text, add_bos=True):
        ids = self._tok.encode(text, add_special_tokens=False).ids
        return [self.bos_token_id] + ids if add_bos else ids

    def call(self, inputs):
        texts = self.normalize_texts(inputs)
        return {"input_ids": [self.encode(t) for t in texts]}

    def decode(self, ids, skip_special_tokens=True):
        return self._tok.decode(
            self.to_id_list(ids), skip_special_tokens=skip_special_tokens
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "variant": self.variant,
                "hf_id": self.hf_id,
                "tokenizer_file": self.tokenizer_file,
            }
        )
        return config
