import os

import numpy as np
from keras import ops

from zeromodels.base.base_mixin import PreprocessorMixin


class BaseTokenizer(PreprocessorMixin):
    """Abstract base for zeromodels tokenizers.

    Subclasses implement ``call`` (text -> ids) and ``decode`` (ids -> text);
    ``batch_decode`` is a pure-Python loop over ``decode``. The loading API
    (``from_weights`` / ``from_variant`` / ``from_hf``) and the ``__call__`` ->
    ``call`` forwarder are inherited from :class:`PreprocessorMixin`.

    Shared ``call`` / ``decode`` plumbing is provided as helpers so concrete
    tokenizers keep only their backend-specific encode/decode:

    * :meth:`normalize_texts`, coerce ``call`` input to a list of strings (with
      ChatML-messages dispatch when the subclass defines ``apply_chat_template``);
    * :meth:`pad_batch`, right-pad ragged id sequences to a rectangular
      ``input_ids`` + ``attention_mask``;
    * :meth:`to_id_list`, normalize a ``decode`` argument (tensor / numpy / int /
      list) to a flat list of ints;
    * :meth:`encode_batch_to_inputs`, for ``tokenizers``-padded backends, encode
      a batch (optionally as text pairs) straight to an ``input_ids`` (+ optional
      ``attention_mask`` / ``token_type_ids``) tensor dict.
    * :meth:`resolve_tokenizer_json`, resolve a ``variant`` to a local
      ``tokenizer.json`` path (an explicit ``tokenizer_file`` if given, else the
      per-variant release file from the tokenizer's ``TOKENIZER_URLS`` dict) for
      ``Tokenizer.from_file``.
    * :meth:`resolve_tokenizer_json_from_hf`, resolve a local ``tokenizer.json``
      path from a Hub ``hf_id`` (or an explicit ``tokenizer_file``), raising when
      neither is given (no default repo, like ``AutoTokenizer.from_pretrained``).

    Concrete tokenizers add their own state (vocab path, merges, special-token
    ids, BPE / SentencePiece backend) and ``get_config`` payload: the base
    intentionally bakes in no defaults.
    """

    @classmethod
    def from_hub_repo(cls, repo_id, **kwargs):
        """Build the tokenizer from a Hub repo id by downloading its tokenizer.json.

        Mirrors model loading by repo id (``from_weights("org/repo")``): the
        ``tokenizer.json`` is pulled straight from the repo, so no per-variant URL
        table is needed. Used for ``from_weights("zeromodels/grounding_dino_tiny")``.
        """
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id.rstrip("/"), "tokenizer.json", token=os.environ.get("HF_TOKEN")
        )
        return cls(tokenizer_file=path, **kwargs)

    def normalize_texts(self, inputs):
        if inputs is None:
            raise ValueError(f"No text inputs provided to {type(self).__name__}.")
        if (
            isinstance(inputs, (list, tuple))
            and inputs
            and isinstance(inputs[0], dict)
            and "role" in inputs[0]
            and hasattr(self, "apply_chat_template")
        ):
            return [self.apply_chat_template(inputs)]
        return [inputs] if isinstance(inputs, str) else list(inputs)

    def pad_batch(self, sequences, pad_value=0):
        max_len = max((len(s) for s in sequences), default=0)
        input_ids = np.full((len(sequences), max_len), pad_value, dtype=np.int32)
        attention_mask = np.zeros((len(sequences), max_len), dtype=np.int32)
        for i, s in enumerate(sequences):
            input_ids[i, : len(s)] = s
            attention_mask[i, : len(s)] = 1
        return input_ids, attention_mask

    def to_id_list(self, ids):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        return [int(i) for i in ids]

    def encode_batch_to_inputs(
        self, inputs, text_pair=None, token_type_ids=True, mask_dtype="int32"
    ):
        texts = self.normalize_texts(inputs)
        if text_pair is None:
            encs = self._tok.encode_batch(texts)
        else:
            pairs = [text_pair] if isinstance(text_pair, str) else list(text_pair)
            encs = self._tok.encode_batch(list(zip(texts, pairs)))
        out = {"input_ids": ops.convert_to_tensor([e.ids for e in encs], dtype="int32")}
        if mask_dtype is not None:
            out["attention_mask"] = ops.convert_to_tensor(
                [e.attention_mask for e in encs], dtype=mask_dtype
            )
        if token_type_ids:
            out["token_type_ids"] = ops.convert_to_tensor(
                [e.type_ids for e in encs], dtype="int32"
            )
        return out

    def resolve_tokenizer_json(self, variant, tokenizer_file=None):
        if tokenizer_file is not None and os.path.exists(tokenizer_file):
            return tokenizer_file
        from zeromodels.conversion import download_file

        cfg = getattr(self, "TOKENIZER_URLS", None)
        if not cfg:
            raise AttributeError(
                f"{type(self).__name__} must set a TOKENIZER_URLS class attr "
                f"(variant -> {{'tokenizer_json': url}}) to use resolve_tokenizer_json."
            )
        if variant not in cfg:
            raise ValueError(
                f"Unknown variant {variant!r} for {type(self).__name__}; "
                f"available: {list(cfg)}"
            )
        url = cfg[variant].get("tokenizer_json")
        if url is None:
            raise ValueError(
                f"Variant {variant!r} has no 'tokenizer_json' URL in "
                f"{type(self).__name__}.TOKENIZER_URLS."
            )
        return download_file(url)

    def resolve_tokenizer_json_from_hf(
        self, hf_id, tokenizer_file=None, filename="tokenizer.json"
    ):
        """Resolve a local ``tokenizer.json`` path from a Hub repo id.

        Returns ``tokenizer_file`` when given; otherwise downloads ``filename``
        from the ``hf_id`` Hub repo. There is no default repo: exactly one of
        ``hf_id`` / ``tokenizer_file`` must be supplied (mirroring
        ``AutoTokenizer.from_pretrained``), else a ``ValueError`` is raised.
        """
        if tokenizer_file is not None:
            return tokenizer_file
        if not hf_id:
            raise ValueError(
                f"{type(self).__name__} requires `hf_id` (a Hub repo containing a "
                f"{filename}) or an explicit `tokenizer_file`."
            )
        from huggingface_hub import hf_hub_download

        return hf_hub_download(hf_id, filename)

    def call(self, inputs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement `call(inputs)`."
        )

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} must implement `decode(ids, skip_special_tokens)`."
        )

    def batch_decode(self, ids_batch, skip_special_tokens: bool = True):
        return [self.decode(ids, skip_special_tokens) for ids in ids_batch]
