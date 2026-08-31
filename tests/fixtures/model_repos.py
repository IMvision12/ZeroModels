"""Canonical repo per tokenizer / processor for tests.

Models no longer bake a default variant: constructing a tokenizer or processor
now requires an explicit ``variant`` / ``hf_id`` / ``tokenizer_file`` (mirroring
``AutoTokenizer.from_pretrained``). ``model_repos.json`` records the repo each
class used to default to, so the test suites can still build every component from
one explicit source instead of a bare ``cls()``.
"""

from __future__ import annotations

import inspect
import json
import pathlib

_REPOS_PATH = pathlib.Path(__file__).parent / "model_repos.json"
MODEL_REPOS: dict[str, str] = json.loads(_REPOS_PATH.read_text(encoding="utf-8"))


def repo_for(name: str) -> str | None:
    """The canonical repo for a tokenizer/processor class name, or None."""
    return MODEL_REPOS.get(name)


def build_from_repo(cls, name: str, **kwargs):
    """Construct ``cls`` from its canonical repo (``model_repos.json``).

    Dispatches on the constructor signature: ``hf_id=<repo>`` when the class takes
    a full Hub repo, else ``variant=<repo-without-'zeromodels/'>`` for the
    zeromodels-hosted variant classes. Returns None if there is no known repo (the
    caller then skips), never a bare ``cls()`` (which now raises by design).
    """
    repo = MODEL_REPOS.get(name)
    if repo is None:
        return None
    params = inspect.signature(cls.__init__).parameters
    if "hf_id" in params:
        return cls(hf_id=repo, **kwargs)
    if "variant" in params:
        variant = repo.split("/", 1)[1] if repo.startswith("zeromodels/") else repo
        return cls(variant=variant, **kwargs)
    return None
