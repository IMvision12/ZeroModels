import os

# Backend + tokenizers-version shim before importing keras / zeromodels.
os.environ.setdefault("KERAS_BACKEND", "torch")

import importlib.metadata as _meta  # noqa: E402

_orig_version = _meta.version
_meta.version = lambda name: "0.23.0" if name == "tokenizers" else _orig_version(name)

import importlib  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

import zeromodels  # noqa: E402,F401  (registers every model + processor)
from zeromodels.conversion.zm_config import write_zm_preprocessor  # noqa: E402

# ---------------------------------------------------------------------------
# CONFIG. DRY_RUN=True lists what needs a zm_preprocessor (the check). Flip to
# False and export a write token (HF_TOKEN) to actually upload.
# ---------------------------------------------------------------------------
ORG = "zeromodels"
DRY_RUN = True


def find_image_size(cfg, depth=0):
    """Search a zm_config for an ``image_size`` (top level or nested)."""
    if not isinstance(cfg, dict) or depth > 2:
        return None
    if isinstance(cfg.get("image_size"), int):
        return cfg["image_size"]
    for value in cfg.values():
        found = find_image_size(value, depth + 1)
        if found is not None:
            return found
    return None


def processor_class_for(cfg):
    """Map a repo's zm_config to its ``<Prefix>ImageProcessor`` class, or None.

    A classification backbone's model_class is ``<Prefix>ImageClassify`` (or
    ``<Prefix>Model``); its processor lives in the same module. Repos whose class
    has no matching ``*ImageProcessor`` (LLMs, most VLM / detection / seg models,
    DINO's non-standard names) return None and are skipped.
    """
    module = cfg.get("model_module")
    model_class = cfg.get("model_class")
    if not module or not model_class:
        return None
    prefix = model_class
    for suffix in ("ImageClassify", "Model"):
        if prefix.endswith(suffix):
            prefix = prefix[: -len(suffix)]
            break
    try:
        return getattr(importlib.import_module(module), prefix + "ImageProcessor")
    except (ImportError, AttributeError):
        return None


def main():
    token = os.environ.get("HF_TOKEN")
    if not DRY_RUN and not token:
        raise SystemExit("Set HF_TOKEN (a write token) to upload.")

    api = HfApi(token=token)
    if not DRY_RUN:
        print("Authenticated as:", api.whoami().get("name"))

    print(f"Scanning {ORG}/* repos for a missing zm_preprocessor.json ...")
    todo, skipped = [], []
    for repo in api.list_models(author=ORG):
        try:
            files = api.list_repo_files(repo.id)
        except Exception:
            continue
        if "zm_config.json" not in files or "zm_preprocessor.json" in files:
            continue
        try:
            cfg = json.load(
                open(hf_hub_download(repo.id, "zm_config.json"), encoding="utf-8")
            )
        except Exception:
            continue
        proc_cls = processor_class_for(cfg)
        if proc_cls is None:
            skipped.append((repo.id, cfg.get("model_class")))
        else:
            todo.append((repo.id, proc_cls, cfg))

    print(
        f"\n{len(todo)} classification-backbone repos need a zm_preprocessor; "
        f"{len(skipped)} non-backbone repos skipped (no matching processor)."
    )
    print("=" * 64)

    done = 0
    for repo_id, proc_cls, cfg in todo:
        size = find_image_size(cfg)
        processor = proc_cls(size=size) if size else proc_cls()
        with tempfile.TemporaryDirectory() as workdir:
            write_zm_preprocessor(workdir, processor, repo_id.split("/")[-1])
            if DRY_RUN:
                print(
                    f"[dry-run] {repo_id:<45} <- {proc_cls.__name__} "
                    f"(size={processor.size['height']}, "
                    f"mean={processor.image_mean})"
                )
            else:
                api.upload_file(
                    path_or_fileobj=os.path.join(workdir, "zm_preprocessor.json"),
                    path_in_repo="zm_preprocessor.json",
                    repo_id=repo_id,
                )
                print(f"uploaded zm_preprocessor.json -> {repo_id}")
        done += 1

    verb = "would be uploaded" if DRY_RUN else "uploaded"
    print("=" * 64)
    print(f"Done. {done} zm_preprocessor.json {verb}.")
    if skipped:
        print(f"\nSkipped {len(skipped)} repos with no matching processor, e.g.:")
        for repo_id, model_class in skipped[:10]:
            print(f"  {repo_id}  ({model_class})")


if __name__ == "__main__":
    main()
