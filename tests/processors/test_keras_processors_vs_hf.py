from __future__ import annotations

import keras
import numpy as np
import pytest
from PIL import Image

from tests.fixtures.model_repos import build_from_repo

transformers = pytest.importorskip("transformers")

from transformers import AutoProcessor

MM_TEXTS = ["a photo of a cat", "two dogs running on the beach"]


def _as_numpy(x) -> np.ndarray:
    # .cpu() first: a CUDA tensor has both, and its .numpy() raises, so the
    # other order breaks this whole suite on any GPU box (CI is CPU-only).
    if hasattr(x, "cpu"):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else x.cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return keras.ops.convert_to_numpy(x)


def _to_channels_last(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[1] == 3 and arr.shape[-1] != 3:
        return np.transpose(arr, (0, 2, 3, 1))
    return arr


def _max_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = _to_channels_last(a)
    b = _to_channels_last(b)
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _rgb(side, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((side, side, 3)) * 255).astype("uint8")


def _rgb_shape(shape, seed=0):
    """Non-square image: a square one at the target size skips the resize."""
    rng = np.random.default_rng(seed)
    return (rng.random((*shape, 3)) * 255).astype("uint8")


def _strip_pad(ids, mask):
    ids = np.asarray(_as_numpy(ids))
    mask = np.asarray(_as_numpy(mask)).astype(bool)
    return [[int(t) for t, m in zip(row, mrow) if m] for row, mrow in zip(ids, mask)]


def _auto_processor(repo):
    try:
        return AutoProcessor.from_pretrained(repo)
    except Exception as e:
        pytest.skip(f"HF AutoProcessor for {repo!r} unavailable: {e}")


def _legs(cls, repo):
    return [
        ("native", build_from_repo(cls, cls.__name__)),
        ("from_hf", cls.from_weights(f"hf:{repo}")),
    ]


def test_clip_processor_three_way():
    from zeromodels.models.clip.clip_processor import CLIPProcessor

    repo = "openai/clip-vit-base-patch16"
    hf = _auto_processor(repo)
    img = _rgb(224)
    h = hf(
        text=MM_TEXTS, images=Image.fromarray(img), padding=True, return_tensors="np"
    )
    hf_rows = _strip_pad(h["input_ids"], h["attention_mask"])
    for leg, ours in _legs(CLIPProcessor, repo):
        o = ours(text=MM_TEXTS, images=img)
        assert _strip_pad(o["input_ids"], o["attention_mask"]) == hf_rows, (
            f"clip[{leg}]: input_ids differ from HF"
        )
        diff = _max_diff(_as_numpy(o["images"]), h["pixel_values"])
        assert diff < 1e-4, f"clip[{leg}]: pixel max|diff|={diff:.3e}"
        print(f"[{leg:>7} clip processor      ] ids ok, pixel max|diff|={diff:.3e}")


def test_siglip_processor_three_way():
    from zeromodels.models.siglip.siglip_processor import SigLIPProcessor

    repo = "google/siglip-base-patch16-224"
    hf = _auto_processor(repo)
    img = _rgb(224)
    h = hf(
        text=MM_TEXTS,
        images=Image.fromarray(img),
        padding="max_length",
        max_length=64,
        return_tensors="np",
    )
    for leg, ours in _legs(SigLIPProcessor, repo):
        o = ours(text=MM_TEXTS, images=img)
        # SigLIP pads with the eos id and returns no attention mask, so compare
        # the full fixed-length id arrays.
        assert np.array_equal(np.asarray(_as_numpy(o["input_ids"])), h["input_ids"]), (
            f"siglip[{leg}]: input_ids differ from HF"
        )
        diff = _max_diff(_as_numpy(o["images"]), h["pixel_values"])
        assert diff < 1e-4, f"siglip[{leg}]: pixel max|diff|={diff:.3e}"
        print(f"[{leg:>7} siglip processor    ] ids ok, pixel max|diff|={diff:.3e}")


def test_owlvit_processor_three_way():
    from zeromodels.models.owlvit.owlvit_processor import OwlViTProcessor

    repo = "google/owlvit-base-patch32"
    hf = _auto_processor(repo)
    img = _rgb(768)
    queries = [["a photo of a cat", "a photo of a dog"]]
    h = hf(text=queries, images=Image.fromarray(img), return_tensors="np")
    for leg, ours in _legs(OwlViTProcessor, repo):
        o = ours(text=queries, images=img)
        # Both pad to the fixed query length (16) with the "!" pad id.
        assert np.array_equal(np.asarray(_as_numpy(o["input_ids"])), h["input_ids"]), (
            f"owlvit[{leg}]: input_ids differ from HF"
        )
        diff = _max_diff(_as_numpy(o["pixel_values"]), h["pixel_values"])
        assert diff < 1e-4, f"owlvit[{leg}]: pixel max|diff|={diff:.3e}"
        print(f"[{leg:>7} owlvit processor    ] ids ok, pixel max|diff|={diff:.3e}")


def test_whisper_processor_three_way():
    from zeromodels.models.whisper.whisper_processor import WhisperProcessor

    repo = "openai/whisper-tiny"
    hf = _auto_processor(repo)
    t = np.arange(16000 * 2, dtype="float32") / 16000.0
    wave = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype("float32")
    h_feat = hf.feature_extractor(wave, sampling_rate=16000, return_tensors="np")[
        "input_features"
    ]
    h_ids = [hf.tokenizer(x, add_special_tokens=False)["input_ids"] for x in MM_TEXTS]
    for leg, ours in _legs(WhisperProcessor, repo):
        o = ours(audio=wave, text=MM_TEXTS)
        o_feat = np.asarray(_as_numpy(o["input_features"]))
        assert o_feat.shape == h_feat.shape, (
            f"whisper[{leg}]: features shape {o_feat.shape} vs HF {h_feat.shape}"
        )
        diff = float(np.max(np.abs(o_feat - h_feat)))
        assert diff < 5e-3, f"whisper[{leg}]: features max|diff|={diff:.3e}"
        assert _strip_pad(o["input_ids"], o["attention_mask"]) == h_ids, (
            f"whisper[{leg}]: input_ids differ from HF"
        )
        print(f"[{leg:>7} whisper processor   ] ids ok, mel max|diff|={diff:.3e}")


# ---------------------------------------------------------------------------
# Snapshots + a non-square parity case.
#
# Every parity test above feeds a square image at exactly the target
# resolution, which makes the resize a no-op (scale == 1). A broken resize
# therefore matched the reference anyway: CLIP shipped a 0.735 max|diff|
# against HF on any real photo while this suite stayed green. The cases below
# use a non-square image so the resize actually runs.
# ---------------------------------------------------------------------------

from tests.fixtures import snapshot_util  # noqa: E402


def _pil_processor(repo):
    """The PIL-backed HF reference.

    transformers 5.x makes the torchvision backend the default under the plain
    name, and the two disagree by ~1.5e-2 on a resize. zeromodels implements
    PIL semantics (what the original models shipped), so compare against PIL.
    """
    try:
        return AutoProcessor.from_pretrained(repo, use_fast=False)
    except Exception as e:
        pytest.skip(f"HF PIL processor for {repo!r} unavailable: {e}")


def test_clip_processor_non_square_parity():
    from zeromodels.models.clip.clip_processor import CLIPProcessor

    repo = "openai/clip-vit-base-patch16"
    hf = _pil_processor(repo)
    for shape in [(64, 48), (48, 96), (300, 500), (223, 225)]:
        img = _rgb_shape(shape)
        ref = _as_numpy(
            hf(text=["x"], images=Image.fromarray(img), return_tensors="np")[
                "pixel_values"
            ]
        )
        clip = build_from_repo(CLIPProcessor, "CLIPProcessor")
        ours = _as_numpy(clip(text=["x"], images=img)["images"])
        diff = _max_diff(ours, ref)
        assert diff < 1e-4, f"clip{shape}: pixel max|diff|={diff:.3e}"


import inspect  # noqa: E402


def _all_processors():
    """Every exported composed processor (``*Processor``, not ``*ImageProcessor``)."""
    import zeromodels.models as models

    found = {}
    for family in sorted(n for n in dir(models) if not n.startswith("_")):
        package = getattr(models, family)
        for name in getattr(package, "__all__", []):
            obj = getattr(package, name, None)
            if (
                inspect.isclass(obj)
                and name.endswith("Processor")
                and not name.endswith("ImageProcessor")
            ):
                found[name] = obj
    return found


ALL_PROCESSORS = _all_processors()
_SNAPSHOT_IMAGE = _rgb_shape((64, 48))


def _snapshot_audio():
    t = np.arange(16000, dtype="float32") / 16000.0
    return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype("float32")


def _run_processor(cls, name):
    """Build ``cls`` from model_repos.json and drive it with a kind-appropriate input.

    Dispatches on the ``call`` signature: ``audio=`` + ``text=`` for the ASR
    processors, ``text=`` + ``images=`` for the image-text ones. Processors that
    need a richer input (VLM conversations, SAM point/box prompts) or have no repo
    in ``model_repos.json`` are skipped rather than fabricating a call.
    """
    try:
        proc = build_from_repo(cls, name)
    except Exception as e:
        pytest.skip(f"{name}: cannot build from model_repos.json ({type(e).__name__})")
    if proc is None:
        pytest.skip(f"{name}: no repo in model_repos.json")
    try:
        params = set(inspect.signature(proc.call).parameters)
    except (TypeError, ValueError):
        pytest.skip(f"{name}: call signature not introspectable")
    try:
        if "audio" in params:
            return proc(audio=_snapshot_audio(), text=MM_TEXTS)
        if "images" in params and "text" in params:
            return proc(text=MM_TEXTS, images=_SNAPSHOT_IMAGE)
    except Exception as e:
        pytest.skip(f"{name}: text/image/audio call failed ({type(e).__name__}: {e})")
    pytest.skip(f"{name}: not a plain text+image / audio+text processor")


@pytest.mark.parametrize("name", sorted(ALL_PROCESSORS))
def test_processor_snapshot(name):
    """Pin ids + pixel/feature stats for every buildable composed processor.

    Auto-enumerated (like the tokenizer / image-processor snapshots), so a new
    processor is guarded the day it lands. HF is not needed at compare time, so the
    shared golden also cross-checks torch / jax / tf preprocessing.
    """
    out = _run_processor(ALL_PROCESSORS[name], name)
    record = {}
    ids_key = next((k for k in ("input_ids", "token_ids") if k in out), None)
    if ids_key is not None:
        ids = np.asarray(_as_numpy(out[ids_key]))
        rows = ids if ids.ndim == 2 else ids[None]
        record["ids"] = [[int(i) for i in row] for row in rows]
    pixel_key = next(
        (k for k in ("images", "pixel_values", "input_features") if k in out), None
    )
    if pixel_key is not None:
        record[pixel_key] = snapshot_util.stats(_as_numpy(out[pixel_key]))
    snapshot_util.check("processor", name, record)


def _proc_rows(out, ids_key):
    ids = np.asarray(_as_numpy(out[ids_key]))
    if ids.ndim == 1:
        ids = ids[None]
    mask = out.get("attention_mask", out.get("padding_mask"))
    if mask is not None:
        return _strip_pad(out[ids_key], mask)
    return [[int(t) for t in row] for row in ids]


@pytest.mark.parametrize("name", sorted(ALL_PROCESSORS))
def test_processor_text_batch_matches_individual(name):
    """Each text in a batch encodes the same as encoding it alone (padding aside).

    A processor that expands image/audio placeholders per row can silently shift a
    row's real tokens; batching must not move them. Needs no HF reference.
    """
    cls = ALL_PROCESSORS[name]
    batch = _run_processor(cls, name)  # builds + runs over MM_TEXTS (skips if unfit)
    ids_key = next((k for k in ("input_ids", "token_ids") if k in batch), None)
    if ids_key is None:
        pytest.skip(f"{name}: no token ids to compare")
    batch_rows = _proc_rows(batch, ids_key)
    proc = build_from_repo(cls, name)
    audio_kind = "audio" in inspect.signature(proc.call).parameters
    for text, batched in zip(MM_TEXTS, batch_rows):
        one = (
            proc(audio=_snapshot_audio(), text=[text])
            if audio_kind
            else proc(text=[text], images=_SNAPSHOT_IMAGE)
        )
        alone = _proc_rows(one, ids_key)[0]
        assert batched == alone, (
            f"{name}: {text!r} encodes differently in a batch than alone"
        )
