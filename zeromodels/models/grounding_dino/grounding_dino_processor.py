import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseProcessor

from .grounding_dino_image_processor import GroundingDinoImageProcessor
from .grounding_dino_tokenizer import GroundingDinoTokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class GroundingDinoProcessor(BaseProcessor):
    """Image + text-prompt -> model inputs for Grounding DINO (open-set detection).

    Composes the BERT tokenizer and the DETR-style image processor. ``call``
    lowercases candidate labels and joins them into a single ``". "``-separated
    prompt ending in ``"."`` (the convention the model was trained on), then
    tokenizes and preprocesses the image(s).

    ``post_process_object_detection`` turns ``logits`` / ``pred_boxes`` into
    per-image ``{scores, labels, boxes}``, matching the other detection
    processors, plus ``text_labels`` when ``input_ids`` is supplied.
    """

    TOKENIZER_CLS = GroundingDinoTokenizer
    IMAGE_PROCESSOR_CLS = GroundingDinoImageProcessor
    COMPONENTS = ("tokenizer", "image_processor")

    def __init__(
        self,
        hf_id=None,
        shortest_edge=800,
        longest_edge=1333,
        tokenizer=None,
        image_processor=None,
        variant=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hf_id = hf_id
        self.variant = variant
        self.shortest_edge = shortest_edge
        self.longest_edge = longest_edge
        self.image_processor = image_processor or GroundingDinoImageProcessor(
            shortest_edge=shortest_edge, longest_edge=longest_edge
        )
        self.tokenizer = tokenizer or GroundingDinoTokenizer(
            variant=variant, hf_id=hf_id
        )

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(hf_id=repo, **kwargs)

    def format_text(self, text):
        if isinstance(text, (list, tuple)):
            return ". ".join(t.strip().lower() for t in text) + "."
        text = text.strip().lower()
        return text if text.endswith(".") else text + "."

    def call(self, images=None, text=None):
        if text is None:
            raise ValueError("Grounding DINO requires a `text` prompt.")
        if isinstance(text, str) or (
            isinstance(text, (list, tuple)) and text and isinstance(text[0], str)
        ):
            prompts = [self.format_text(text)]
        else:
            prompts = [self.format_text(t) for t in text]

        tok = self.tokenizer(prompts)
        out = {
            "input_ids": ops.convert_to_tensor(
                np.array(tok["input_ids"], dtype="int64")
            ),
            "attention_mask": ops.convert_to_tensor(
                np.array(tok["attention_mask"], dtype="int64")
            ),
            "token_type_ids": ops.convert_to_tensor(
                np.array(tok["token_type_ids"], dtype="int64")
            ),
        }
        if images is not None:
            img = self.image_processor(images)
            out["pixel_values"] = ops.convert_to_tensor(img["pixel_values"])
            out["pixel_mask"] = ops.convert_to_tensor(img["pixel_mask"])
        return out

    def post_process_object_detection(
        self, outputs, threshold=0.3, target_sizes=None, input_ids=None
    ):
        """Decode detector outputs to per-image ``{scores, labels, boxes}``.

        Boxes are converted (cx, cy, w, h) -> (x0, y0, x1, y1), and scaled to
        each ``target_sizes`` ``(height, width)`` when given (otherwise left
        normalized). Queries whose max token score exceeds ``threshold`` are
        kept.

        ``labels`` is the prompt-token *position* each query matched, which is
        what the open-set head actually predicts. Pass ``input_ids`` to also get
        ``text_labels``, those positions decoded back to strings. Note that a
        query matches one token, not a span, so a multi-word phrase such as
        "paddle board" is reported by whichever single token scored highest.
        Prompts read best without articles: "a paddle" lets the "a" outscore
        the noun.
        """
        logits = np.asarray(ops.convert_to_numpy(outputs["logits"]))
        boxes = np.asarray(ops.convert_to_numpy(outputs["pred_boxes"]))
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        scores = probs.max(axis=-1)  # (B, num_queries)
        ids = None
        if input_ids is not None:
            ids = np.asarray(ops.convert_to_numpy(ops.convert_to_tensor(input_ids)))
        results = []
        for i in range(scores.shape[0]):
            keep = scores[i] > threshold
            cx, cy, bw, bh = (
                boxes[i, :, 0],
                boxes[i, :, 1],
                boxes[i, :, 2],
                boxes[i, :, 3],
            )
            xyxy = np.stack(
                [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=-1
            )
            if target_sizes is not None:
                h, w = target_sizes[i]
                xyxy = xyxy * np.array([w, h, w, h], dtype="float32")
            token_argmax = probs[i].argmax(axis=-1)
            labels = [int(token_argmax[j]) for j in np.where(keep)[0]]
            result = {
                "scores": scores[i][keep],
                "labels": labels,
                "boxes": xyxy[keep],
            }
            if ids is not None:
                row = ids[i] if ids.ndim == 2 else ids
                limit = len(row) - 1
                result["text_labels"] = [
                    self.tokenizer.decode([int(row[min(label, limit)])])
                    for label in labels
                ]
            results.append(result)
        return results

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hf_id": self.hf_id,
                "variant": self.variant,
                "shortest_edge": self.shortest_edge,
                "longest_edge": self.longest_edge,
            }
        )
        return config
