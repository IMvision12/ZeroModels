from __future__ import annotations

import importlib

import numpy as np
import pytest

from tests.fixtures.model_repos import build_from_repo

transformers = pytest.importorskip("transformers")

from transformers import AutoTokenizer

TEXTS = [
    "a quick brown fox jumps over the lazy dog",
    "Hello, World!",
    "tokenization 123 parity",
]
PAIRS = [
    "and a slow green turtle",
    "Goodbye.",
    "the lazy dog sleeps",
]


def _np(x):
    if hasattr(x, "cpu"):  # a torch tensor may live on GPU; move it to host first
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _to_rows(out, pad_id=None):
    """Normalize ANY zeromodels / HF tokenizer output to a list of per-row
    real-token id lists, regardless of the output contract:

    * dict ``input_ids`` / ``token_ids`` (+ ``attention_mask`` / ``padding_mask``),
    * dict with a ragged list-of-lists ``input_ids`` (no padding),
    * a ``(input_ids, attention_mask)`` tuple (sam3).

    Padding is stripped via the mask when present, else via ``pad_id``.
    """
    if isinstance(out, tuple):
        ids, mask = _np(out[0]), _np(out[1])
        if ids.ndim == 1:
            ids, mask = ids[None], mask[None]
        return [[int(t) for t, m in zip(r, mk) if m] for r, mk in zip(ids, mask)]

    ids = out["input_ids"] if "input_ids" in out else out["token_ids"]
    mask = out.get("attention_mask", out.get("padding_mask"))

    if (
        isinstance(ids, list)
        and ids
        and isinstance(ids[0], (list, tuple))
        and mask is None
    ):
        return [[int(t) for t in r] for r in ids]  # ragged, already real tokens

    arr = _np(ids)
    if arr.ndim == 1:
        arr = arr[None]
    rows = [[int(t) for t in r] for r in arr]
    if mask is not None:
        m = _np(mask)
        if m.ndim == 1:
            m = m[None]
        rows = [[t for t, mm in zip(r, row_m) if mm] for r, row_m in zip(rows, m)]
    elif pad_id is not None:
        rows = [[t for t in r if t != pad_id] for r in rows]
    return rows


def _hf_rows(hf, add_special):
    return [
        [int(x) for x in hf(t, add_special_tokens=add_special)["input_ids"]]
        for t in TEXTS
    ]


def _assert_rows(name, ours_rows, hf_rows):
    assert len(ours_rows) == len(hf_rows), f"{name}: row count mismatch"
    for i, (o, h) in enumerate(zip(ours_rows, hf_rows)):
        assert o == h, (
            f"{name}: text[{i}] ids differ from HF\n  ours ({len(o)}): {o}\n"
            f"  hf   ({len(h)}): {h}"
        )


def _build_legs(module, cls_name, repo):
    """Both zeromodels construction paths, compared against HF independently:
    ``native`` = ``cls()`` (the shipped release tokenizer.json / class
    defaults, i.e. what ``from_weights(variant)`` uses) and ``from_hf`` =
    ``cls.from_hf(repo)`` (files pulled from the HF repo). A leg that fails to
    BUILD is recorded (a release asset or repo file may legitimately be
    absent, e.g. deberta repos publish no tokenizer.json); only a built leg
    with mismatched ids fails the test."""
    cls = getattr(importlib.import_module(module), cls_name)
    legs = []
    try:
        native = build_from_repo(cls, cls_name)
        if native is None:
            raise ValueError(f"no zeromodels repo for {cls_name} in model_repos.json")
        legs.append(("native", native))
    except Exception as e:
        legs.append(("native", e))
    if repo and hasattr(cls, "from_hf"):
        try:
            legs.append(("from_hf", cls.from_hf(repo)))
        except Exception as e:
            legs.append(("from_hf", e))
    return legs


# name -> (submodule, class, hf_repo | None=use ours.hf_id, add_special, pad_attr)
SPECS = {
    "bert": (
        "zeromodels.models.bert.bert_tokenizer",
        "BertTokenizer",
        "bert-base-uncased",
        True,
        None,
    ),
    "clip": (
        "zeromodels.models.clip.clip_tokenizer",
        "CLIPTokenizer",
        "openai/clip-vit-base-patch16",
        True,
        None,
    ),
    "deberta": (
        "zeromodels.models.deberta.deberta_tokenizer",
        "DebertaTokenizer",
        "microsoft/deberta-base",
        True,
        None,
    ),
    "deberta_v2": (
        "zeromodels.models.deberta_v2.deberta_v2_tokenizer",
        "DebertaV2Tokenizer",
        "microsoft/deberta-v2-xlarge",
        True,
        None,
    ),
    "deberta_v3": (
        "zeromodels.models.deberta_v3.deberta_v3_tokenizer",
        "DebertaV3Tokenizer",
        "microsoft/deberta-v3-base",
        True,
        None,
    ),
    "gpt": (
        "zeromodels.models.gpt.gpt_tokenizer",
        "GptTokenizer",
        "openai-community/openai-gpt",
        False,
        None,
    ),
    "gpt2": (
        "zeromodels.models.gpt2.gpt2_tokenizer",
        "GPT2Tokenizer",
        "openai-community/gpt2",
        False,
        None,
    ),
    "gpt_oss": (
        "zeromodels.models.gpt_oss.gpt_oss_tokenizer",
        "GptOssTokenizer",
        None,
        False,
        None,
    ),
    "granite_speech": (
        "zeromodels.models.granite_speech.granite_speech_tokenizer",
        "GraniteSpeechTokenizer",
        "ibm-granite/granite-speech-3.3-2b",
        False,
        None,
    ),
    "granite_speech_plus": (
        "zeromodels.models.granite_speech_plus.granite_speech_plus_tokenizer",
        "GraniteSpeechPlusTokenizer",
        "ibm-granite/granite-speech-4.1-2b-plus",
        False,
        None,
    ),
    "metaclip2": (
        "zeromodels.models.metaclip2.metaclip2_tokenizer",
        "MetaClip2Tokenizer",
        "facebook/metaclip-2-worldwide-huge-378",
        True,
        None,
    ),
    "metaclip2_mt5": (
        "zeromodels.models.metaclip2.metaclip2_mt5_tokenizer",
        "MetaClip2Mt5Tokenizer",
        "google/mt5-base",
        True,
        None,
    ),
    "moonshine": (
        "zeromodels.models.moonshine.moonshine_tokenizer",
        "MoonshineTokenizer",
        "UsefulSensors/moonshine-tiny",
        False,
        None,
    ),
    "qwen2": (
        "zeromodels.models.qwen2.qwen2_tokenizer",
        "Qwen2Tokenizer",
        None,
        False,
        None,
    ),
    "qwen2_vl": (
        "zeromodels.models.qwen2_vl.qwen2_vl_tokenizer",
        "Qwen2VLTokenizer",
        None,
        False,
        None,
    ),
    "qwen3": (
        "zeromodels.models.qwen3.qwen3_tokenizer",
        "Qwen3Tokenizer",
        None,
        False,
        None,
    ),
    "qwen3_5": (
        "zeromodels.models.qwen3_5.qwen3_5_tokenizer",
        "Qwen3_5Tokenizer",
        None,
        False,
        None,
    ),
    "roberta": (
        "zeromodels.models.roberta.roberta_tokenizer",
        "RobertaTokenizer",
        "roberta-base",
        True,
        None,
    ),
    "siglip": (
        "zeromodels.models.siglip.siglip_tokenizer",
        "SigLIPTokenizer",
        "google/siglip-base-patch16-224",
        True,
        "pad_token_id",
    ),
    "siglip2": (
        "zeromodels.models.siglip2.siglip2_tokenizer",
        "SigLIP2Tokenizer",
        "google/siglip2-base-patch16-224",
        True,
        "pad_token_id",
    ),
    "speech2text": (
        "zeromodels.models.speech2text.speech2text_tokenizer",
        "Speech2TextTokenizer",
        "facebook/s2t-small-librispeech-asr",
        True,
        "pad_token_id",
    ),
    "whisper": (
        "zeromodels.models.whisper.whisper_tokenizer",
        "WhisperTokenizer",
        "openai/whisper-tiny",
        False,
        None,
    ),
    "xlm_roberta": (
        "zeromodels.models.xlm_roberta.xlm_roberta_tokenizer",
        "XLMRobertaTokenizer",
        "xlm-roberta-base",
        True,
        None,
    ),
}


@pytest.mark.parametrize("name", list(SPECS.keys()))
def test_tokenizer_hf_parity(name):
    if name == "metaclip2_mt5":
        pytest.skip("MetaCLIP2 mT5 text tokenizer: HF source repo not pinned")
    module, cls_name, repo, add_special, pad_attr = SPECS[name]
    legs = _build_legs(module, cls_name, repo)
    built = [(leg, tok) for leg, tok in legs if not isinstance(tok, Exception)]
    if not built:
        notes = "; ".join(f"{leg}: {type(e).__name__}: {e}" for leg, e in legs)
        pytest.skip(f"cannot construct {name} on any path ({notes})")

    hf_repo = repo or getattr(built[0][1], "hf_id", None)
    if hf_repo is None:
        pytest.skip(f"{name}: no HF repo to compare against")
    try:
        hf = AutoTokenizer.from_pretrained(hf_repo)
    except Exception as e:
        pytest.skip(f"{name}: HF tokenizer for {hf_repo!r} unavailable: {e}")

    for leg, ours_tok in built:
        if name == "siglip":
            # SigLIP's pad id == eos id and there's no attention mask, so
            # stripping pads by value would drop the real trailing eos. Compare
            # full padded arrays instead (HF padded to the same length).
            ours_ids = _np(ours_tok(TEXTS)["input_ids"])
            hf_ids = _np(
                hf(
                    TEXTS,
                    padding="max_length",
                    max_length=ours_ids.shape[1],
                    add_special_tokens=add_special,
                    return_tensors="np",
                )["input_ids"]
            )
            assert ours_ids.shape == hf_ids.shape and np.array_equal(
                ours_ids, hf_ids
            ), f"{name}[{leg}]: padded input_ids differ from HF"
            continue

        pad_id = getattr(ours_tok, pad_attr) if pad_attr else None
        ours_rows = _to_rows(ours_tok(TEXTS), pad_id)
        hf_rows = _hf_rows(hf, add_special)
        _assert_rows(f"{name}[{leg}]", ours_rows, hf_rows)


def test_sam3_clip_tokenizer_vs_clip():
    """SAM3's CLIP text tokenizer returns a ``(input_ids, attention_mask)`` tuple
    via ``encode`` (no HF ``AutoTokenizer`` of its own); compare its real tokens
    against the OpenAI CLIP tokenizer it mirrors."""
    from zeromodels.models.sam3.sam3_clip_tokenizer import SAM3CLIPTokenizer

    try:
        ours = SAM3CLIPTokenizer.from_hf("openai/clip-vit-base-patch16")
    except Exception as e:
        pytest.skip(f"cannot construct sam3_clip: {e}")
    hf = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    ours_rows = _to_rows(ours.encode(TEXTS))
    hf_rows = _hf_rows(hf, add_special=True)
    _assert_rows("sam3_clip", ours_rows, hf_rows)


def test_bert_token_type_ids_pairs():
    """Exercise ``token_type_ids`` (0 for segment A, 1 for segment B) on a
    BERT text-pair batch, vs HF."""
    from zeromodels.models.bert.bert_tokenizer import BertTokenizer

    try:
        tok = BertTokenizer.from_hf("bert-base-uncased")
    except Exception as e:
        pytest.skip(f"cannot construct bert: {e}")
    ours = tok(TEXTS, text_pair=PAIRS)
    hf = AutoTokenizer.from_pretrained("bert-base-uncased")(
        TEXTS, PAIRS, padding=True, return_tensors="np"
    )
    o_mask = _np(ours["attention_mask"]).astype(bool)
    o_types, h_types = _np(ours["token_type_ids"]), _np(hf["token_type_ids"])
    assert np.array_equal(o_types[o_mask], h_types[o_mask]), (
        "bert pairs: token_type_ids differ"
    )
    assert int(o_types[o_mask].max()) == 1, "expected a second segment (type id 1)"


# ---------------------------------------------------------------------------
# Snapshots + semantics, over EVERY exported tokenizer.
#
# The parity table above is hand-maintained and covers 23 of the 60 tokenizers;
# the other 37 (every LLM: gemma, llama, mistral, qwen-moe, deepseek, glm,
# kimi ...) had no test at all. These enumerate the package instead, so
# a new tokenizer is covered the day it lands rather than when someone
# remembers to add a SPECS row.
#
# Most LLM tokenizers need an `hf_id`, which their own config already publishes
# per variant, so it is resolved from there rather than duplicated here. A
# tokenizer that cannot be built (gated repo, missing asset) skips.
# ---------------------------------------------------------------------------

import inspect  # noqa: E402

from tests.fixtures import snapshot_util  # noqa: E402

SNAPSHOT_TEXTS = [
    "a quick brown fox jumps over the lazy dog",
    "Hello, World!",
    "tokenization 123 parity",
    "",
]


def _all_tokenizers():
    import zeromodels.models as models

    found = {}
    for family in sorted(n for n in dir(models) if not n.startswith("_")):
        package = getattr(models, family)
        for name in getattr(package, "__all__", []):
            obj = getattr(package, name, None)
            if inspect.isclass(obj) and name.endswith("Tokenizer"):
                found[name] = (obj, family)
    return found


ALL_TOKENIZERS = _all_tokenizers()


def _family_hf_id(family):
    """First hf_id any variant of this family publishes."""
    try:
        config = importlib.import_module(f"zeromodels.models.{family}.{family}_config")
    except Exception:
        return None
    for attr in dir(config):
        value = getattr(config, attr)
        if not isinstance(value, dict):
            continue
        for entry in value.values():
            if isinstance(entry, dict) and entry.get("hf_id"):
                return entry["hf_id"]
    return None


def _any_tokenizer(name):
    """Build ``name``, or skip with the reason it cannot be tested as text.

    Not every exported ``*Tokenizer`` encodes text: OneFormer's takes a task
    name (``call(task="panoptic")``) and returns ``task_inputs``. Those skip
    rather than recording a failure as if it were the expected output.
    """
    cls, family = ALL_TOKENIZERS[name]
    try:
        tok = build_from_repo(cls, name)
    except Exception as e:
        pytest.skip(
            f"{name}: cannot build from model_repos.json repo ({type(e).__name__})"
        )
    if tok is None:
        # Not in model_repos.json: fall back to the family config's hf_id.
        hf_id = _family_hf_id(family)
        if not hf_id or "hf_id" not in inspect.signature(cls.__init__).parameters:
            pytest.skip(f"{name}: no repo in model_repos.json or config to build from")
        try:
            tok = cls(hf_id=hf_id)
        except Exception as e:
            pytest.skip(f"{name}: cannot construct via {hf_id!r} ({type(e).__name__})")
    try:
        _to_rows(tok(["probe text"]))
    except Exception as e:
        pytest.skip(f"{name}: does not encode plain text ({type(e).__name__}: {e})")
    return tok


@pytest.mark.parametrize("name", sorted(ALL_TOKENIZERS))
def test_tokenizer_snapshot(name):
    """Pin the exact ids each tokenizer produces for fixed texts.

    Needs no HF reference at compare time, so it still guards the ids when a
    repo is gated, renamed, or simply offline.
    """
    tok = _any_tokenizer(name)
    record = {
        (text or "<empty>"): {"ids": [int(i) for i in _to_rows(tok([text]))[0]]}
        for text in SNAPSHOT_TEXTS
    }
    snapshot_util.check("tokenizer", name, record)


@pytest.mark.parametrize("name", sorted(ALL_TOKENIZERS))
def test_tokenizer_batch_matches_individual(name):
    """A batch must encode each text the same as encoding it alone.

    Padding is the only allowed difference: if batching moves the real tokens,
    every batched inference is quietly wrong.
    """
    tok = _any_tokenizer(name)
    pad_id = getattr(tok, "pad_token_id", None)
    batch = _to_rows(tok(TEXTS), pad_id=pad_id)
    for text, batched in zip(TEXTS, batch):
        alone = _to_rows(tok([text]), pad_id=pad_id)[0]
        assert list(batched) == list(alone), (
            f"{name}: {text!r} encodes differently in a batch than alone"
        )


@pytest.mark.parametrize("name", sorted(ALL_TOKENIZERS))
def test_tokenizer_mask_marks_real_tokens(name):
    """The mask must be 1 on real tokens and 0 on padding, contiguously.

    A mask that disagrees with the ids lets padding into attention, which is
    silent: shapes still line up and outputs are merely wrong.
    """
    tok = _any_tokenizer(name)
    out = tok(TEXTS)
    if not isinstance(out, dict):
        pytest.skip(f"{name}: no dict output")
    mask_key = next((k for k in ("attention_mask", "padding_mask") if k in out), None)
    if mask_key is None:
        pytest.skip(f"{name}: emits no mask")
    ids_key = "input_ids" if "input_ids" in out else "token_ids"
    ids = _np(out[ids_key])
    mask = _np(out[mask_key]).astype(int)
    lengths = [len(_to_rows(tok([t]))[0]) for t in TEXTS]
    for row_mask, length in zip(mask, lengths):
        kept = int(row_mask.sum())
        assert kept == length, (
            f"{name}: mask keeps {kept} tokens, text encodes to {length}"
        )
        assert list(row_mask[:length]) == [1] * length, (
            f"{name}: mask not 1 on real tokens"
        )
        assert not any(row_mask[length:]), f"{name}: mask not 0 on padding"
    assert ids.shape == mask.shape, f"{name}: ids {ids.shape} vs mask {mask.shape}"


@pytest.mark.parametrize("name", sorted(ALL_TOKENIZERS))
def test_tokenizer_decode_roundtrip(name):
    """decode(encode(x)) recovers the text, ignoring case and special tokens.

    Also pins the contract `BaseTokenizer` declares: decode returns a str, and
    batch_decode a flat list of str (metaclip2 returned lists of lists).
    """
    tok = _any_tokenizer(name)
    text = "a quick brown fox jumps over the lazy dog"
    rows = _to_rows(tok([text]))
    try:
        back = tok.decode(rows[0])
    except Exception as e:
        pytest.skip(f"{name}: decode unavailable ({type(e).__name__}: {e})")
    assert isinstance(back, str), (
        f"{name}: decode returned {type(back).__name__}, not str"
    )
    normalized = "".join(c for c in back.lower() if c.isalnum())
    expected = "".join(c for c in text.lower() if c.isalnum())
    assert expected in normalized, f"{name}: decode lost the text: {back!r}"

    batch = tok.batch_decode([rows[0], rows[0]])
    assert all(isinstance(b, str) for b in batch), (
        f"{name}: batch_decode returned {[type(b).__name__ for b in batch]}, not str"
    )
