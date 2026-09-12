import keras

from zeromodels.models.clip.clip_tokenizer import CLIPTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class StableDiffusionTokenizer(CLIPTokenizer):
    """Stable Diffusion 1.x text tokenizer: the CLIP ViT-L/14 BPE tokenizer.

    SD 1.x conditions on OpenAI's CLIP ViT-L/14 text encoder, so this is that
    tokenizer byte for byte: byte-pair encoding with ``<|startoftext|>`` /
    ``<|endoftext|>`` framing, truncated and ``<|endoftext|>``-padded to
    ``max_seq_len`` (77), the fixed text length the UNet's cross-attention is built
    for. Loads by repo id like the model, reading the hosted repo's root
    ``tokenizer.json``: ``from_weights("zeromodels/stable-diffusion-v1-5")``.

    Args:
        hf_id: Hosted repo to read ``tokenizer.json`` from. Required unless
            ``tokenizer_file`` is given (no default repo).
        tokenizer_file: Explicit ``tokenizer.json`` path (overrides ``hf_id``).
        max_seq_len: Padded / truncated length (default 77).
    """

    def __init__(self, hf_id=None, tokenizer_file=None, max_seq_len=77, **kwargs):
        if tokenizer_file is None and hf_id is None:
            raise ValueError(
                f"{type(self).__name__}() needs an hf_id (a hosted repo, read from its "
                "tokenizer.json) or an explicit tokenizer_file: there is no default "
                "repo. Load it by repo id, e.g. "
                "from_weights('zeromodels/stable-diffusion-v1-5')."
            )
        if tokenizer_file is None:
            tokenizer_file = self.download_tokenizer_json(hf_id)
        super().__init__(
            tokenizer_file=tokenizer_file, max_seq_len=max_seq_len, **kwargs
        )
        self.hf_id = hf_id

    @staticmethod
    def download_tokenizer_json(hf_id):
        import os

        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            hf_id, "tokenizer.json", token=os.environ.get("HF_TOKEN")
        )

    @classmethod
    def from_hf(cls, repo, **kwargs):
        raise NotImplementedError(
            "On-the-fly `hf:` loading is not supported for diffusion models: use "
            f"{cls.__name__}.from_weights('zeromodels/<variant>')."
        )

    def get_config(self):
        config = super().get_config()
        config["hf_id"] = self.hf_id
        return config
