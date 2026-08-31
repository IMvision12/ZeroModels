import keras
import numpy as np
from keras import ops
from PIL import Image

from zeromodels.base import BaseProcessor

from .sam3_clip_tokenizer import SAM3CLIPTokenizer
from .sam3_image_processor import SAM3ImageProcessor
from .sam3_utils import box_xyxy_to_cxcywh, compute_scores, scale_boxes, sigmoid

IMAGE_SIZE = 1008
RESCALE_FACTOR = 1.0 / 255.0
IMAGE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
IMAGE_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def preprocess_image(
    image,
    target_size=IMAGE_SIZE,
    image_mean=IMAGE_MEAN,
    image_std=IMAGE_STD,
    rescale_factor=RESCALE_FACTOR,
):
    """Preprocess an image for SAM3 inference.

    Resizes to a square target size using backend-native bilinear
    interpolation, applies rescaling via float64 intermediate to
    match the reference precision, and normalizes with ImageNet-style mean/std.

    Args:
        image: PIL Image, numpy array ``(H, W, 3)``, or file path.
        target_size (int): Target square size. Defaults to ``1008``.
        image_mean: Per-channel normalization mean. Defaults to ``(0.5, 0.5, 0.5)``.
        image_std: Per-channel normalization std. Defaults to ``(0.5, 0.5, 0.5)``.
        rescale_factor (float): Pixel rescale factor. Defaults to ``1/255``.

    Returns:
        Tuple of ``(pixel_values, original_size)`` where
        ``pixel_values`` is ``(1, H, W, 3)`` float32 and
        ``original_size`` is ``(height, width)``.
    """
    image_mean = np.asarray(image_mean, dtype=np.float32)
    image_std = np.asarray(image_std, dtype=np.float32)
    if isinstance(image, str):
        if Image is None:
            raise ImportError("PIL required for loading images from paths")
        image = Image.open(image).convert("RGB")

    if Image is not None and isinstance(image, Image.Image):
        original_size = (image.height, image.width)
        image = np.array(image)
    else:
        image = np.asarray(image)
        original_size = (image.shape[0], image.shape[1])
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

    image_t = ops.convert_to_tensor(image.astype(np.float32) / 256.0)
    image_4d = ops.expand_dims(image_t, 0)
    resized = ops.image.resize(
        image_4d, (target_size, target_size), interpolation="bilinear"
    )
    resized = resized * 256.0
    resized = ops.clip(resized, 0, 255)
    resized = ops.round(resized)
    resized = ops.convert_to_numpy(resized)[0]

    image = (resized.astype(np.float64) * rescale_factor).astype(np.float32)
    image = (image - image_mean) / image_std
    return image[np.newaxis], original_size


def preprocess_text_with_encoder(text, text_encoder_model, tokenizer=None):
    """Tokenize and encode text using the CLIP text encoder.

    Args:
        text: String or list of strings to encode.
        text_encoder_model: Keras CLIP text encoder model (functional).
        tokenizer: ``SAM3CLIPTokenizer`` instance. If ``None``, creates
            one automatically.

    Returns:
        Tuple of ``(text_features, attention_mask)`` where
        ``text_features`` is ``(batch, 32, 1024)`` and
        ``attention_mask`` is ``(batch, 32)`` float32.
    """
    if tokenizer is None:
        tokenizer = SAM3CLIPTokenizer()

    input_ids, attention_mask = tokenizer.encode(text)

    text_features = text_encoder_model.predict(
        {"input_ids": input_ids, "attention_mask": attention_mask.astype(np.int32)},
        verbose=0,
    )
    return text_features, attention_mask


POINT_PAD_VALUE = -10


def preprocess_boxes(input_boxes, input_boxes_labels, original_sizes):
    """Normalize and convert box prompts to model input format.

    Converts absolute pixel coordinates to normalized ``(cx, cy, w, h)``
    format and pads to the maximum number of boxes in the batch.

    Args:
        input_boxes: List of list of ``[x1, y1, x2, y2]`` boxes per
            image in absolute pixel coordinates.
        input_boxes_labels: List of list of int labels (0 or 1) per
            image, or ``None`` for all-positive.
        original_sizes: List of ``(H, W)`` tuples per image.

    Returns:
        Tuple of ``(boxes_cxcywh, box_labels, box_mask)`` where
        shapes are ``(batch, max_boxes, 4)``, ``(batch, max_boxes)``,
        and ``(batch, max_boxes)`` respectively.
    """
    if isinstance(input_boxes, np.ndarray):
        input_boxes = input_boxes.tolist()
    if isinstance(input_boxes_labels, np.ndarray):
        input_boxes_labels = input_boxes_labels.tolist()

    batch_size = len(input_boxes)

    max_boxes = max(len(boxes) for boxes in input_boxes)

    all_boxes = []
    all_labels = []
    all_masks = []

    for img_idx in range(batch_size):
        boxes = input_boxes[img_idx]
        labels = input_boxes_labels[img_idx] if input_boxes_labels else [1] * len(boxes)
        h, w = original_sizes[img_idx]

        normalized = []
        for box in boxes:
            x1, y1, x2, y2 = box
            normalized.append([x1 / w, y1 / h, x2 / w, y2 / h])

        if normalized:
            norm_arr = np.array(normalized, dtype=np.float32)
            cxcywh = np.asarray(ops.convert_to_numpy(box_xyxy_to_cxcywh(norm_arr)))
        else:
            cxcywh = np.zeros((0, 4), dtype=np.float32)

        num_boxes = len(boxes)
        pad_count = max_boxes - num_boxes
        if pad_count > 0:
            pad_boxes = np.full((pad_count, 4), POINT_PAD_VALUE, dtype=np.float32)
            cxcywh = np.concatenate([cxcywh, pad_boxes], axis=0)
            labels = list(labels) + [0] * pad_count

        mask = np.array([1.0] * num_boxes + [0.0] * pad_count, dtype=np.float32)

        all_boxes.append(cxcywh)
        all_labels.append(np.array(labels, dtype=np.int32))
        all_masks.append(mask)

    return (
        np.stack(all_boxes),
        np.stack(all_labels),
        np.stack(all_masks),
    )


def post_process_object_detection(outputs, threshold=0.3, target_sizes=None):
    """Convert raw model outputs to detection results.

    Applies sigmoid scoring with optional presence logits, scales
    boxes to target image sizes, and filters by confidence threshold.

    Args:
        outputs: Dict with ``"pred_logits"`` ``(B, Q)`` and
            ``"pred_boxes"`` ``(B, Q, 4)`` in normalized cxcywh.
        threshold (float): Minimum score to keep. Defaults to ``0.3``.
        target_sizes: List of ``(H, W)`` tuples for box scaling,
            or ``None``.

    Returns:
        List of dicts, each with ``"scores"`` and ``"boxes"`` arrays.
    """
    pred_logits = np.asarray(outputs["pred_logits"])
    pred_boxes = np.asarray(outputs["pred_boxes"])
    presence = outputs.get("presence_logits")
    batch_scores = compute_scores(pred_logits, presence)

    results = []
    for idx in range(pred_logits.shape[0]):
        scores = batch_scores[idx]
        boxes = pred_boxes[idx].copy()
        if target_sizes is not None:
            boxes = scale_boxes(boxes, target_sizes[idx])
        keep = scores > threshold
        results.append({"scores": scores[keep], "boxes": boxes[keep]})
    return results


def post_process_instance_segmentation(
    outputs, threshold=0.3, mask_threshold=0.5, target_sizes=None
):
    """Convert raw model outputs to instance segmentation results.

    Applies sigmoid to masks, resizes to target sizes using PIL
    bilinear interpolation, and binarizes with ``mask_threshold``.

    Args:
        outputs: Dict with ``"pred_logits"``, ``"pred_boxes"``, and
            ``"pred_masks"`` ``(B, Q, H, W)``.
        threshold (float): Minimum score to keep. Defaults to ``0.3``.
        mask_threshold (float): Binarization threshold for masks.
            Defaults to ``0.5``.
        target_sizes: List of ``(H, W)`` tuples, or ``None``.

    Returns:
        List of dicts, each with ``"scores"``, ``"boxes"``, and
        ``"masks"`` arrays.
    """
    pred_logits = np.asarray(outputs["pred_logits"])
    pred_boxes = np.asarray(outputs["pred_boxes"])
    pred_masks = np.asarray(outputs["pred_masks"])
    presence = outputs.get("presence_logits")
    batch_scores = compute_scores(pred_logits, presence)
    batch_masks = sigmoid(pred_masks)

    results = []
    for idx in range(pred_logits.shape[0]):
        scores = batch_scores[idx]
        boxes = pred_boxes[idx].copy()
        masks = batch_masks[idx]

        if target_sizes is not None:
            boxes = scale_boxes(boxes, target_sizes[idx])

        keep = scores > threshold
        scores = scores[keep]
        boxes = boxes[keep]
        masks = masks[keep]

        if target_sizes is not None and len(masks) > 0 and Image is not None:
            th, tw = target_sizes[idx]
            resized = []
            for m in masks:
                pil_m = Image.fromarray((m * 255).astype(np.uint8))
                pil_m = pil_m.resize((tw, th), Image.BILINEAR)
                resized.append(np.array(pil_m, dtype=np.float32) / 255.0)
            masks = np.stack(resized)

        masks = (masks > mask_threshold).astype(np.int32)
        results.append({"scores": scores, "boxes": boxes, "masks": masks})
    return results


def post_process_semantic_segmentation(outputs, target_sizes=None, threshold=0.5):
    """Convert raw model outputs to semantic segmentation maps.

    Applies sigmoid to the single-channel semantic output, resizes
    to target sizes, and binarizes with ``threshold``.

    Args:
        outputs: Dict with ``"semantic_seg"`` ``(B, 1, H, W)`` or
            ``(B, H, W, 1)``.
        target_sizes: List of ``(H, W)`` tuples, or ``None``.
        threshold (float): Binarization threshold. Defaults to ``0.5``.

    Returns:
        List of ``(H, W)`` int32 binary mask arrays.
    """
    semantic = np.asarray(outputs["semantic_seg"])
    probs = sigmoid(semantic)

    results = []
    for idx in range(semantic.shape[0]):
        mask = probs[idx, 0]
        if target_sizes is not None and Image is not None:
            th, tw = target_sizes[idx]
            pil_m = Image.fromarray((mask * 255).astype(np.uint8))
            pil_m = pil_m.resize((tw, th), Image.BILINEAR)
            mask = np.array(pil_m, dtype=np.float32) / 255.0
        mask = (mask > threshold).astype(np.int32)
        results.append(mask)
    return results


@keras.saving.register_keras_serializable(package="zeromodels")
class SAM3Processor(BaseProcessor):
    """Compose the CLIP tokenizer + image processor into SAM3 model inputs.

    Mirrors transformers' ``Sam3Processor``: one call turns raw images, a text
    noun phrase, and optional box prompts into the tensors the model consumes.
    Load it alongside the model with ``SAM3Processor.from_weights("zeromodels/sam3")``
    (the image config comes from ``zm_preprocessor.json``, the tokenizer from
    ``tokenizer.json``); both are hosted ungated.

        proc = SAM3Processor.from_weights("zeromodels/sam3")
        inputs = proc(images=img, text="cat")
        # -> pixel_values, input_ids, attention_mask, original_sizes

    The high-level ``SAM3Detect`` / ``SAM3InstanceSegment`` / ``SAM3SemanticSegment``
    ``.predict(...)`` wrappers reuse the same image processor internally, so this
    class is the transformers-style entry point plus the ``post_process_*`` helpers.

    Args:
        hf_id: Hub repo for the tokenizer's ``tokenizer.json`` (default
            ``"zeromodels/sam3"``, ungated).
        tokenizer / image_processor: Optional pre-built components.
        target_size: Square resolution; defaults to the image processor's.
        point_pad_value: Padding sentinel for batched box prompts (default -10).
    """

    TOKENIZER_CLS = SAM3CLIPTokenizer
    IMAGE_PROCESSOR_CLS = SAM3ImageProcessor
    COMPONENTS = ("tokenizer", "image_processor")

    def __init__(
        self,
        hf_id=None,
        tokenizer=None,
        image_processor=None,
        target_size=None,
        point_pad_value=POINT_PAD_VALUE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hf_id = hf_id
        self.tokenizer = tokenizer or SAM3CLIPTokenizer(hf_id=hf_id)
        self.image_processor = image_processor or SAM3ImageProcessor()
        self.point_pad_value = point_pad_value
        self.target_size = (
            target_size
            if target_size is not None
            else self.image_processor.image_resolution
        )

    @classmethod
    def from_hf(cls, repo, **kwargs):
        return cls(hf_id=repo, **kwargs)

    @classmethod
    def from_hub_repo(cls, repo_id, **kwargs):
        # tokenizer.json for the tokenizer, zm_preprocessor.json for the image
        # processor (the composed processor is not itself described by a file).
        import inspect

        from zeromodels.conversion.zm_config import load_zm_preprocessor

        repo_id = repo_id.rstrip("/")
        tokenizer = cls.TOKENIZER_CLS.from_weights(repo_id)
        spec = load_zm_preprocessor(repo_id) or {}
        img_params = set(inspect.signature(cls.IMAGE_PROCESSOR_CLS.__init__).parameters)
        img_kwargs = {k: v for k, v in spec.items() if k in img_params}
        return cls(
            hf_id=repo_id,
            tokenizer=tokenizer,
            image_processor=cls.IMAGE_PROCESSOR_CLS(**img_kwargs),
            **kwargs,
        )

    def resolve_text(self, text, input_boxes):
        """Fill missing text with ``"visual"`` for any image that has boxes."""
        if text is None:
            return "visual" if input_boxes else None
        if not isinstance(text, (list, tuple)):
            return text
        text = list(text)
        if input_boxes is not None and len(text) != len(input_boxes):
            raise ValueError(
                f"Got {len(text)} text prompts but {len(input_boxes)} box entries."
            )
        for i, value in enumerate(text):
            if (
                value is None
                and input_boxes is not None
                and i < len(input_boxes)
                and input_boxes[i] is not None
            ):
                text[i] = "visual"
        return text

    def call(
        self,
        images=None,
        text=None,
        input_boxes=None,
        input_boxes_labels=None,
        original_sizes=None,
    ):
        out = {}
        if images is not None:
            imgs = images if isinstance(images, (list, tuple)) else [images]
            pixels, sizes = [], []
            for img in imgs:
                res = self.image_processor.call(img)
                pixels.append(np.asarray(res["pixel_values"])[0])
                sizes.append(res["original_size"])
            out["pixel_values"] = np.stack(pixels)
            original_sizes = sizes
        if original_sizes is not None:
            out["original_sizes"] = list(original_sizes)

        text = self.resolve_text(text, input_boxes)
        if text is not None:
            input_ids, attention_mask = self.tokenizer.encode(text)
            out["input_ids"] = input_ids
            out["attention_mask"] = attention_mask

        if input_boxes is not None:
            if original_sizes is None:
                raise ValueError(
                    "`images` or `original_sizes` is required to normalize `input_boxes`."
                )
            boxes, labels, mask = preprocess_boxes(
                input_boxes, input_boxes_labels, original_sizes
            )
            out["input_boxes"] = boxes
            out["input_boxes_labels"] = labels
            out["box_mask"] = mask
        return out

    def post_process_object_detection(self, outputs, threshold=0.3, target_sizes=None):
        return self.image_processor.post_process_object_detection(
            outputs, threshold, target_sizes
        )

    def post_process_instance_segmentation(
        self, outputs, threshold=0.3, mask_threshold=0.5, target_sizes=None
    ):
        return self.image_processor.post_process_instance_segmentation(
            outputs, threshold, mask_threshold, target_sizes
        )

    def post_process_semantic_segmentation(
        self, outputs, target_sizes=None, threshold=0.5
    ):
        return self.image_processor.post_process_semantic_segmentation(
            outputs, target_sizes, threshold
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hf_id": self.hf_id,
                "target_size": self.target_size,
                "point_pad_value": self.point_pad_value,
            }
        )
        return config
