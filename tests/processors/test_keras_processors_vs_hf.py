from __future__ import annotations

import keras
import numpy as np
import pytest
from PIL import Image

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
    return [("native", cls()), ("from_hf", cls.from_weights(f"hf:{repo}"))]


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
        ours = _as_numpy(CLIPProcessor()(text=["x"], images=img)["images"])
        diff = _max_diff(ours, ref)
        assert diff < 1e-4, f"clip{shape}: pixel max|diff|={diff:.3e}"


@pytest.mark.parametrize(
    "name,repo,cls_path",
    [
        ("clip", "openai/clip-vit-base-patch16", "clip.clip_processor.CLIPProcessor"),
        (
            "siglip",
            "google/siglip-base-patch16-224",
            "siglip.siglip_processor.SigLIPProcessor",
        ),
    ],
)
def test_processor_snapshot(name, repo, cls_path):
    """Pin ids + pixel stats for a fixed text/image pair, HF not required."""
    import importlib

    module, cls_name = cls_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(f"zeromodels.models.{module}"), cls_name)
    out = cls()(text=MM_TEXTS, images=_rgb_shape((64, 48)))
    ids_key = "input_ids" if "input_ids" in out else "token_ids"
    pixel_key = "images" if "images" in out else "pixel_values"
    record = {
        "ids": [[int(i) for i in row] for row in _as_numpy(out[ids_key])],
        "pixels": snapshot_util.stats(_as_numpy(out[pixel_key])),
    }
    snapshot_util.check("processor", name, record)
