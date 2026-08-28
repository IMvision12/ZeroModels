import keras

from zeromodels.base import BaseTokenizer

DEFAULT_SYSTEM = "You are a helpful assistant."


@keras.saving.register_keras_serializable(package="zeromodels")
class LocateAnythingTokenizer(BaseTokenizer):
    """Qwen2.5 BPE tokenizer extended with LocateAnything's grounding tokens.

    Same ``tokenizers`` backend as :class:`Qwen2Tokenizer`, plus the box / ref /
    coordinate special-token ids and ``parse_grounding``, which turns a generated id
    sequence into structured results (each ``<ref>`` label paired with the boxes /
    points that follow it; a box is four coordinates, a point is two). Each coordinate
    is ``coord_start_token_id + v`` for v in [0, 1000]. The processor's
    ``post_process_generation`` wraps this with pixel rescaling.
    """

    DEFAULT_VARIANT = "locateanything_3b"

    def __init__(self, variant=None, hf_id=None, tokenizer_file=None, **kwargs):
        super().__init__(**kwargs)
        from tokenizers import Tokenizer

        if tokenizer_file is None and hf_id is not None:
            tokenizer_file = self.resolve_tokenizer_json_from_hf(hf_id, tokenizer_file)
        else:
            tokenizer_file = self.resolve_tokenizer_json_from_hf(
                f"zeromodels/{variant or self.DEFAULT_VARIANT}", tokenizer_file
            )
        self.variant = variant
        self.hf_id = hf_id
        self.tokenizer_file = tokenizer_file
        self._tok = Tokenizer.from_file(tokenizer_file)
        self.eos_token = "<|im_end|>"
        self.eos_token_id = self._tok.token_to_id(self.eos_token)
        self.image_token = "<IMG_CONTEXT>"
        self.image_token_id = 151665
        self.image_start_token = "<img>"
        self.image_end_token = "</img>"
        self.box_start_token_id = 151668
        self.box_end_token_id = 151669
        self.ref_start_token_id = 151672
        self.ref_end_token_id = 151673
        self.coord_start_token_id = 151677
        self.coord_end_token_id = 152677
        self.none_token_id = 4064
        self.text_mask_token_id = 151676
        self.null_token_id = 152678
        self.switch_token_id = 152679

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text, add_special_tokens=False).ids

    def apply_chat_template(
        self, messages, add_generation_prompt=True, system=DEFAULT_SYSTEM
    ):
        text = ""
        if system is not None and not any(m.get("role") == "system" for m in messages):
            text += f"<|im_start|>system\n{system}<|im_end|>\n"
        for m in messages:
            text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def call(self, inputs):
        texts = self.normalize_texts(inputs)
        input_ids, attention_mask = self.pad_batch([self.encode(t) for t in texts])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens=True):
        return self._tok.decode(
            self.to_id_list(ids), skip_special_tokens=skip_special_tokens
        )

    def parse_grounding(self, ids):
        """Parse a generated id sequence into grounding results, pairing each
        ``<ref>`` label with the boxes/points that follow it. Returns a list of
        dicts, each ``{"label": str | None, "box": [x1, y1, x2, y2]}`` or
        ``{"label": str | None, "point": [x, y]}``, coordinates in the [0, 1000]
        grid. ``label`` is ``None`` when the model emits no ``<ref>`` (e.g.
        detection/pointing whose target is named in the prompt). Use this for
        multi-object referring, OCR, and layout grounding."""
        ids = self.to_id_list(ids)
        cs, ce = self.coord_start_token_id, self.coord_end_token_id
        bs, be, rs, ref_end = (
            self.box_start_token_id,
            self.box_end_token_id,
            self.ref_start_token_id,
            self.ref_end_token_id,
        )
        stops = {bs, rs, self.null_token_id, self.eos_token_id}
        results, label, i, n = [], None, 0, len(ids)
        while i < n:
            tid = ids[i]
            if tid == rs:
                j, buf = i + 1, []
                while j < n and ids[j] not in stops:
                    if ids[j] != ref_end:
                        buf.append(ids[j])
                    j += 1
                label = self.decode(buf, skip_special_tokens=True).strip() or None
                i = j
            elif tid == bs:
                j, coords = i + 1, []
                while j < n and ids[j] != be:
                    if cs <= ids[j] <= ce:
                        coords.append(ids[j] - cs)
                    j += 1
                if len(coords) == 4:
                    results.append({"label": label, "box": coords})
                elif len(coords) == 2:
                    results.append({"label": label, "point": coords})
                i = j + 1
            else:
                i += 1
        return results

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
