import hashlib
import json
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


def validate_url(url: str) -> bool:
    """Validate if the provided URL is well-formed.

    Args:
        url: URL string to validate

    Returns:
        bool: True if URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def _parse_hf_resolve(url: str):
    """Parse a Hugging Face ``resolve`` URL into ``(repo_id, revision, filename)``.

    Returns ``None`` for anything that is not a
    ``https://huggingface.co/<org>/<repo>/resolve/<rev>/<file>`` URL, so callers
    fall back to a plain streamed download.
    """
    parsed = urlparse(url)
    if parsed.netloc not in ("huggingface.co", "www.huggingface.co"):
        return None
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 5 or parts[2] != "resolve":
        return None
    return f"{parts[0]}/{parts[1]}", parts[3], "/".join(parts[4:])


def download_file(
    file_url: str, cache_dir: Optional[str] = None, force_download: bool = False
) -> str:
    """Download a single file from the specified URL and return its local path.

    A Hugging Face ``resolve`` URL is fetched through the HF cache via
    ``hf_hub_download`` (resume, Xet, ``hf_transfer``, and reuse across runs, the
    way ``transformers`` / ``keras_hub`` do it). Any other URL is streamed to
    ``~/.downloads`` with a tqdm progress bar (via ``huggingface_hub``'s
    ``http_get``). An ``HF_TOKEN`` in the environment is used for gated / private
    repos in both paths.

    Args:
        file_url: URL to download file from
        cache_dir: Directory to cache non-HF files (default: ~/.downloads).
            Ignored for Hugging Face URLs, which use the shared HF cache.
        force_download: Force download even if the file is already cached
    Returns:
        str: Path to the downloaded file
    Raises:
        ValueError: For invalid inputs
        Exception: For download failures
    """
    if not file_url:
        raise ValueError("file_url cannot be empty")
    if not validate_url(file_url):
        raise ValueError(f"Invalid URL format: {file_url}")

    hf = _parse_hf_resolve(file_url)
    if hf is not None:
        from huggingface_hub import hf_hub_download

        repo_id, revision, filename = hf
        return hf_hub_download(
            repo_id,
            filename,
            revision=revision,
            token=os.environ.get("HF_TOKEN"),
            force_download=force_download,
        )

    cache_dir = Path(cache_dir or os.path.expanduser("~/.downloads"))

    file_name = os.path.basename(file_url)
    url_dir = file_url.rsplit("/", 1)[0]
    subdir = hashlib.sha1(url_dir.encode("utf-8")).hexdigest()[:16]
    target_dir = cache_dir / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    local_file = target_dir / file_name

    if local_file.exists() and not force_download:
        print(f"Found cached file at {local_file}")
        return str(local_file)

    headers = {"User-Agent": "zeromodels"}
    token = os.environ.get("HF_TOKEN")
    if token and "huggingface.co" in file_url:
        headers["Authorization"] = f"Bearer {token}"

    # Stream to a sibling ``.incomplete`` file, then atomically move it into
    # place, so an interrupted download never leaves a truncated file that a
    # later call would treat as cached.
    tmp_file = target_dir / f"{file_name}.incomplete"
    try:
        from huggingface_hub.file_download import http_get

        with open(tmp_file, "wb") as f:
            http_get(file_url, f, headers=headers)
        os.replace(tmp_file, local_file)
        return str(local_file)
    except Exception as e:
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass
        print(f"Failed to download file: {str(e)}")
        raise


def _shard_files(index: dict) -> list:
    """Collect the shard filenames from a native Keras ``weight_map``.

    ``weight_map`` values are a single shard filename (older Keras) or a list of
    shard filenames per weight group (Keras >= 3.14).
    """
    if "weight_map" not in index:
        raise ValueError("Sharded weights index must contain 'weight_map'.")
    shards = set()
    for value in index["weight_map"].values():
        if isinstance(value, list):
            shards.update(value)
        else:
            shards.add(value)
    return sorted(shards)


def download_weights(weights_url: str) -> str:
    """Download release weights and return the local path to the primary file.

    Handles a single ``.weights.h5`` or a sharded ``.weights.json`` index (plus
    every shard in its ``weight_map``). For a Hugging Face repo the shards are
    fetched **concurrently** through the HF cache via ``snapshot_download`` (the
    index sits next to them in the snapshot dir); for any other URL each shard is
    streamed with :func:`download_file` into the same cache directory as the
    index. Returns the ``.weights.json`` path for a sharded checkpoint, else the
    ``.weights.h5`` path.
    """
    if not weights_url.lower().endswith(".json"):
        return download_file(weights_url)

    index_path = download_file(weights_url)
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    shard_files = _shard_files(index)

    hf = _parse_hf_resolve(weights_url)
    if hf is not None:
        from huggingface_hub import snapshot_download

        repo_id, revision, _ = hf
        # Concurrent, resumable (Xet / hf_transfer aware) fetch into the same
        # snapshot dir the index lives in.
        snapshot_download(
            repo_id,
            revision=revision,
            token=os.environ.get("HF_TOKEN"),
            allow_patterns=shard_files,
        )
    else:
        base_url = "/".join(weights_url.split("/")[:-1])
        for shard_file in shard_files:
            download_file(f"{base_url}/{shard_file}")
    return index_path
