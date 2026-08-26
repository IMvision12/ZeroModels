"""Golden-file (snapshot) helpers for the processor suites.

A snapshot pins what a component *currently* produces for a fixed input, so an
unintended change to preprocessing shows up as a diff instead of silently
altering everyone's pixels or token ids. This complements the vs-HF parity
tests: parity says "we match the reference on the cases we thought to try",
snapshots say "nothing moved since last time" across every component at once.

Regenerate after an intentional change:

    ZEROMODELS_UPDATE_SNAPSHOTS=1 pytest tests/processors -k snapshot

and review the resulting diff before committing it: an unexplained change here
is the bug, not the test.
"""

from __future__ import annotations

import json
import os
import pathlib

SNAPSHOT_DIR = pathlib.Path(__file__).parent
# Above float32 noise and above the ~1e-5 backend jitter seen in resize kernels,
# but far below anything a real preprocessing change would produce.
TOLERANCE = 1e-4


def updating() -> bool:
    return os.environ.get("ZEROMODELS_UPDATE_SNAPSHOTS") == "1"


def path_for(name: str) -> pathlib.Path:
    return SNAPSHOT_DIR / f"{name}_snapshots.json"


def load(name: str) -> dict:
    path = path_for(name)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save(name: str, data: dict) -> None:
    path_for(name).write_text(
        json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


def compare(expected, actual, where: str) -> list:
    """Structural diff of two snapshot records, as a list of human-readable lines."""
    problems = []
    if type(expected) is not type(actual) and not (
        isinstance(expected, (int, float)) and isinstance(actual, (int, float))
    ):
        return [f"{where}: type {type(expected).__name__} -> {type(actual).__name__}"]
    if isinstance(expected, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                problems.append(f"{where}.{key}: new key {actual[key]!r}")
            elif key not in actual:
                problems.append(
                    f"{where}.{key}: key disappeared (was {expected[key]!r})"
                )
            else:
                problems += compare(expected[key], actual[key], f"{where}.{key}")
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            problems.append(f"{where}: length {len(expected)} -> {len(actual)}")
        else:
            for i, (e, a) in enumerate(zip(expected, actual)):
                problems += compare(e, a, f"{where}[{i}]")
    elif isinstance(expected, bool) or expected is None:
        if expected != actual:
            problems.append(f"{where}: {expected!r} -> {actual!r}")
    elif isinstance(expected, (int, float)):
        if abs(float(expected) - float(actual)) > TOLERANCE:
            problems.append(
                f"{where}: {expected} -> {actual} (d={abs(expected - actual):.2e})"
            )
    elif expected != actual:
        problems.append(f"{where}: {expected!r} -> {actual!r}")
    return problems


def check(name: str, key: str, record: dict) -> None:
    """Assert ``record`` matches the stored snapshot for ``key``.

    In update mode the record is merged into the golden file instead.
    """
    import pytest

    data = load(name)
    if updating():
        data[key] = record
        save(name, data)
        pytest.skip(f"snapshot updated: {name}/{key}")
    if key not in data:
        pytest.fail(
            f"no snapshot for {name}/{key}. Generate it with "
            f"ZEROMODELS_UPDATE_SNAPSHOTS=1 pytest tests/processors -k snapshot"
        )
    problems = compare(data[key], record, key)
    assert not problems, "snapshot mismatch:\n  " + "\n  ".join(problems)


def stats(array) -> dict:
    """Shape + distribution summary of a numeric array.

    Summarised rather than stored elementwise: the point is to catch a changed
    resize / normalization / tiling, which moves these, while keeping the golden
    file reviewable in a diff.
    """
    import numpy as np

    a = np.asarray(array, dtype="float64")
    if a.size == 0:
        return {"shape": list(a.shape), "empty": True}
    return {
        "shape": list(a.shape),
        "mean": round(float(a.mean()), 5),
        "std": round(float(a.std()), 5),
        "min": round(float(a.min()), 5),
        "max": round(float(a.max()), 5),
    }
