import os

import keras
import numpy as np
import pytest
from keras import ops

from kerasformers.base import BaseProcessor
from tests.base.model_test_registry import (
    MODEL_TEST_CONFIGS,
    create_test_input,
    import_model_class,
)

BACKEND = os.environ.get("KERAS_BACKEND", "torch")
MODEL_IDS = list(MODEL_TEST_CONFIGS.keys())

# Models that don't support runtime channels_first/channels_last switching:
# - Whisper* / Speech2Text* / Moonshine*: audio models, no spatial image
#   dim; the channels_first conversion doesn't apply.
# - MaskFormerUniversalSegment / Mask2FormerUniversalSegment: the HF-aligned Swin backbone
#   port works in channels_last only (the conv → reshape → flatten path
#   assumes (B, H, W, C)). HF MaskFormer / Mask2Former checkpoints are
#   only released for channels_last; supporting channels_first would
#   require an alternate code path in the backbone.
SKIP_DATA_FORMAT = {
    "WhisperModel",
    "WhisperConditionalGenerate",
    "WhisperAudioClassify",
    "Speech2TextModel",
    "Speech2TextConditionalGenerate",
    "MoonshineModel",
    "MoonshineConditionalGenerate",
    "GraniteSpeechModel",
    "GraniteSpeechConditionalGenerate",
    "GraniteSpeechPlusModel",
    "GraniteSpeechPlusConditionalGenerate",
    "MaskFormerUniversalSegment",
    "Mask2FormerUniversalSegment",
    # DPT reassemble reshapes tokens to a channels-last grid -> channels_last only.
    "Tipsv2DptDensePredict",
    "Tipsv2DptDepthEstimation",
    "Tipsv2DptSemanticSegment",
    # Qwen-VL inputs are pre-patchified (no spatial axes) -> layout-agnostic.
    "Qwen2VLModel",
    "Qwen2_5VLModel",
    "Qwen3VLModel",
    "Qwen3_5MoeModel",
    "Qwen3VLMoeModel",
    "Qwen2VLConditionalGenerate",
    "Qwen2_5VLConditionalGenerate",
    "Qwen3VLConditionalGenerate",
    "Qwen3_5MoeConditionalGenerate",
    "Qwen3VLMoeConditionalGenerate",
    # Text LLMs are token-id only -> no image data format.
    "Qwen2Model",
    "Qwen3Model",
    "Qwen3_5Model",
    "Qwen2TextGenerate",
    "Qwen3TextGenerate",
    "Qwen3_5TextGenerate",
}


def _adapt_input_shape_for_format(init_kwargs, data_format):
    kwargs = init_kwargs.copy()
    if data_format == "channels_first" and "input_shape" in kwargs:
        h, w, c = kwargs["input_shape"]
        kwargs["input_shape"] = (c, h, w)
    if data_format == "channels_first" and "image_size" in kwargs:
        spec = kwargs["image_size"]
        if isinstance(spec, (tuple, list)) and len(spec) == 3:
            h, w, c = spec
            kwargs["image_size"] = (c, h, w)
    return kwargs


def _transpose_input(input_data, data_format):
    if data_format != "channels_first":
        return input_data
    if isinstance(input_data, dict):
        result = {}
        for k, v in input_data.items():
            if k in ("pixel_values", "images") and len(v.shape) == 4:
                result[k] = ops.transpose(v, (0, 3, 1, 2))
            else:
                result[k] = v
        return result
    if len(input_data.shape) == 4:
        return ops.transpose(input_data, (0, 3, 1, 2))
    return input_data


@pytest.mark.data_format
@pytest.mark.parametrize("model_name", MODEL_IDS)
def test_channels_last(model_name):
    if model_name in SKIP_DATA_FORMAT:
        pytest.skip(f"{model_name} doesn't support data format switching")

    original = keras.config.image_data_format()
    try:
        keras.config.set_image_data_format("channels_last")
        config = MODEL_TEST_CONFIGS[model_name]
        model_cls = import_model_class(config)
        model = model_cls(**config["init_kwargs"])
        input_data = create_test_input(config)
        output = model(input_data)

        if isinstance(output, dict):
            for key, value in output.items():
                assert not bool(ops.any(ops.isnan(value))), (
                    f"{model_name}[{key}] has NaNs in channels_last"
                )
        elif isinstance(output, (list, tuple)):
            for i, value in enumerate(output):
                assert not bool(ops.any(ops.isnan(value))), (
                    f"{model_name}[{i}] has NaNs in channels_last"
                )
        else:
            assert not bool(ops.any(ops.isnan(output))), (
                f"{model_name} has NaNs in channels_last"
            )
    finally:
        keras.config.set_image_data_format(original)


@pytest.mark.data_format
@pytest.mark.parametrize("model_name", MODEL_IDS)
def test_channels_first(model_name):
    if model_name in SKIP_DATA_FORMAT:
        pytest.skip(f"{model_name} doesn't support data format switching")

    if BACKEND == "tensorflow":
        try:
            import tensorflow as tf

            if not tf.config.list_physical_devices("GPU"):
                pytest.skip("TF channels_first conv2d requires GPU (cuDNN)")
        except ImportError:
            pytest.skip("TensorFlow not installed")

    original = keras.config.image_data_format()
    try:
        keras.config.set_image_data_format("channels_first")
        config = MODEL_TEST_CONFIGS[model_name]
        model_cls = import_model_class(config)
        adapted_kwargs = _adapt_input_shape_for_format(
            config["init_kwargs"], "channels_first"
        )
        model = model_cls(**adapted_kwargs)
        input_data = create_test_input(config)
        input_data = _transpose_input(input_data, "channels_first")
        output = model(input_data)

        if isinstance(output, dict):
            for key, value in output.items():
                assert not bool(ops.any(ops.isnan(value))), (
                    f"{model_name}[{key}] has NaNs in channels_first"
                )
        elif isinstance(output, (list, tuple)):
            for i, value in enumerate(output):
                assert not bool(ops.any(ops.isnan(value))), (
                    f"{model_name}[{i}] has NaNs in channels_first"
                )
        else:
            assert not bool(ops.any(ops.isnan(output))), (
                f"{model_name} has NaNs in channels_first"
            )
    finally:
        keras.config.set_image_data_format(original)


# ---------------------------------------------------------------------------
# Processor input handling.
#
# A processor takes one conversation or a list of them, and vision inputs
# arrive as one flat batch-wide list, so each prompt must expand only the
# inputs its own markers claim. Getting that wrong is silent: the token totals
# can still add up while the patches land against the wrong prompt. These lock
# down the batch rendering, the dealing, and the mismatch guard. No GPU or
# network needed, so they run on every backend.
# ---------------------------------------------------------------------------


class _MarkerProcessor(BaseProcessor):
    """Smallest processor that exercises the shared batching helpers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_token = "<img>"

    def apply_chat_template(self, messages, add_generation_prompt=True):
        text = ""
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                text += content
                continue
            for item in content:
                text += self.image_token if item["type"] == "image" else item["text"]
        return text

    def extract_images(self, conversation):
        images = []
        for msg in conversation:
            content = msg.get("content")
            if isinstance(content, (list, tuple)):
                images.extend(
                    item["image"] for item in content if item.get("type") == "image"
                )
        return images or None


def _conv(*images, text="q"):
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def test_processor_batch_detection():
    proc = _MarkerProcessor()
    one = _conv("a")
    assert not proc.is_conversation_batch(one), "a message list is one conversation"
    assert proc.is_conversation_batch([one, _conv("b")]), "a list of lists is a batch"
    assert proc.is_conversation_batch((one, _conv("b"))), "tuples count too"
    assert not proc.is_conversation_batch([]), "empty is not a batch"
    assert proc.normalize_conversations(one) == [one]
    assert proc.normalize_conversations([one]) == [one]


def test_processor_renders_each_conversation_separately():
    proc = _MarkerProcessor()
    first, second = _conv("a", text="q1"), _conv("b", "c", text="q2")

    texts_one, images_one = proc.render_conversations(first)
    texts_batch, images_batch = proc.render_conversations([first, second])

    assert len(texts_one) == 1, "one conversation renders one prompt"
    assert len(texts_batch) == 2, "a batch renders one prompt per conversation"
    assert texts_batch[0] == texts_one[0], "batching must not change a prompt"
    assert images_one == ["a"]
    assert images_batch == ["a", "b", "c"], "images flatten in marker order"


def test_processor_deals_vision_inputs_per_text():
    proc = _MarkerProcessor()
    # First conversation has one image, second has two: the uneven case is what
    # exposes a processor expanding against the whole batch's list.
    texts, _ = proc.render_conversations([_conv("a"), _conv("b", "c")])
    dealt = proc.deal_per_text(texts, proc.image_token, ["g1", "g2", "g3"])
    assert dealt == [["g1"], ["g2", "g3"]], f"grids dealt wrong: {dealt}"

    single, _ = proc.render_conversations(_conv("a", "b"))
    assert proc.deal_per_text(single, proc.image_token, ["g1", "g2"]) == [["g1", "g2"]]


@pytest.mark.parametrize(
    "markers,items",
    [
        (1, 2),  # extra inputs would ship with nowhere to scatter
        (2, 1),  # a marker would survive unexpanded into the token ids
        (3, 2),
    ],
)
def test_processor_rejects_marker_input_mismatch(markers, items):
    proc = _MarkerProcessor()
    texts, _ = proc.render_conversations(_conv(*[f"i{n}" for n in range(markers)]))
    with pytest.raises(ValueError, match="placeholder"):
        proc.deal_per_text(texts, proc.image_token, [f"g{n}" for n in range(items)])


class _FakeTokenizer:
    """Codepoint tokenizer, so the string paths run without a tokenizer.json.

    Every marker encodes to ``placeholder_id`` and nothing else does, which is
    what lets a test count placeholders per row.
    """

    video_token = "<|video|>"
    pad_token_id = 0
    placeholder_id = 5

    def __init__(self, image_token="<|image|>"):
        self.image_token = image_token

    def encode(self, text, **kwargs):
        ids = []
        for chunk in text.split(self.image_token):
            ids.extend((ord(c) % 100) + 20 for c in chunk)
            ids.append(self.placeholder_id)
        return ids[:-1]


def _glm4v_processor():
    from kerasformers.models.glm4v.glm4v_image_processor import Glm4vImageProcessor
    from kerasformers.models.glm4v.glm4v_processor import Glm4vProcessor

    return Glm4vProcessor(
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokenizer=_FakeTokenizer(),
        image_processor=Glm4vImageProcessor(
            patch_size=16, spatial_merge_size=2, temporal_patch_size=2
        ),
    )


def test_processor_batches_conversations_end_to_end():
    from PIL import Image

    proc = _glm4v_processor()
    # Different sizes on purpose: the grids, and so the placeholder counts, differ.
    first = _conv(Image.new("RGB", (32, 32)), text="q1")
    second = _conv(Image.new("RGB", (64, 32)), text="q2")

    single_first = proc(conversation=first)
    single_second = proc(conversation=second)
    batch = proc(conversation=[first, second])

    grids = np.asarray(ops.convert_to_numpy(batch["image_grid_thw"]))
    assert grids.shape[0] == 2, "one grid per image, in order"
    for grid, single in ((grids[0], single_first), (grids[1], single_second)):
        alone = np.asarray(ops.convert_to_numpy(single["image_grid_thw"]))[0]
        assert np.array_equal(grid, alone), "batching changed an image's grid"

    ids = np.asarray(ops.convert_to_numpy(batch["input_ids"]))
    mask = np.asarray(ops.convert_to_numpy(batch["attention_mask"]))
    rows = [row[m == 1] for row, m in zip(ids, mask)]
    for row, single, label in (
        (rows[0], single_first, "first"),
        (rows[1], single_second, "second"),
    ):
        alone = np.asarray(ops.convert_to_numpy(single["input_ids"]))[0]
        assert np.array_equal(row, alone), (
            f"{label} batch row differs from the same conversation alone"
        )

    counts = [int((row == _FakeTokenizer.placeholder_id).sum()) for row in rows]
    merged = [int(np.prod(grid)) // 4 for grid in grids]
    assert counts == merged, f"placeholders {counts} do not match grids {merged}"
    assert counts[0] != counts[1], "test is only meaningful if the rows differ"


def test_processor_batch_rejects_mismatched_images():
    from PIL import Image

    proc = _glm4v_processor()
    img = Image.new("RGB", (32, 32))
    with pytest.raises(ValueError, match="placeholder"):
        proc(conversation=_conv(img), images=[img, img])


class _FakePatcher:
    """Stands in for an image processor: only the merge factor is read here."""

    merge_size = 2
    temporal_patch_size = 2


def test_kimi_expands_each_image_against_its_own_grid():
    # Expanding markers with repeated replace() puts the second image's span
    # inside the first one's, which keeps the token total right while moving the
    # patches: 4 + 1 stays 5 either way, so only the layout catches it.
    from kerasformers.models.kimi_k25.kimi_k25_processor import (
        IMAGE_TOKEN,
        KimiK25Processor,
    )

    proc = KimiK25Processor(
        tokenizer=_FakeTokenizer(),
        image_processor=_FakePatcher(),
    )
    # (1, 4, 4) -> 4 merged tokens, (1, 2, 2) -> 1
    out = proc.expand_images(f"a{IMAGE_TOKEN}b{IMAGE_TOKEN}c", [(1, 4, 4), (1, 2, 2)])
    assert out == f"a{IMAGE_TOKEN * 4}b{IMAGE_TOKEN}c", "image spans overlap"

    texts = [f"a{IMAGE_TOKEN}b", f"c{IMAGE_TOKEN}d"]
    dealt = proc.deal_per_text(texts, IMAGE_TOKEN, [(1, 4, 4), (1, 2, 2)])
    counts = [proc.expand_images(t, g).count(IMAGE_TOKEN) for t, g in zip(texts, dealt)]
    assert counts == [4, 1], f"batch rows expanded wrong: {counts}"

    with pytest.raises(ValueError, match="placeholders"):
        proc.expand_images(f"a{IMAGE_TOKEN}b", [(1, 4, 4), (1, 2, 2)])
