from __future__ import annotations

import glob
import io
import os
from dataclasses import dataclass, fields
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

import numpy as np

from zeromodels.utils.image_util import load_image

VideoInput = Union[
    str,
    bytes,
    bytearray,
    os.PathLike,
    np.ndarray,
    Sequence[Any],
]

_FRAME_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class VideoMetadata:
    """Metadata about a decoded video.

    Mirrors :class:`transformers.video_utils.VideoMetadata` so that downstream
    code can consume either interchangeably.
    """

    total_num_frames: int
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None
    video_backend: Optional[str] = None
    frames_indices: Optional[List[int]] = None

    def __iter__(self):
        return (f.name for f in fields(self))

    def __len__(self):
        return len(fields(self))

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    @property
    def timestamps(self) -> List[float]:
        """Per-sampled-frame timestamps, in seconds."""
        if self.fps is None or self.frames_indices is None:
            raise ValueError(
                "Cannot infer `timestamps` when `fps` or `frames_indices` is None."
            )
        return [idx / self.fps for idx in self.frames_indices]

    @property
    def sampled_fps(self) -> float:
        """Effective FPS of the returned frame stack."""
        if (
            self.frames_indices is None
            or self.total_num_frames is None
            or self.fps is None
        ):
            return self.fps or 24.0
        return len(self.frames_indices) / self.total_num_frames * self.fps

    def update(self, dictionary):
        for key, value in dictionary.items():
            if hasattr(self, key):
                setattr(self, key, value)


def default_sample_indices_fn(
    metadata: VideoMetadata,
    num_frames: Optional[int] = None,
    fps: Optional[float] = None,
    **kwargs,
) -> np.ndarray:
    """Compute frame indices using the reference default uniform-sampling logic.

    When ``num_frames`` is set, returns ``np.arange(0, total, total/num_frames)``
    cast to ``int``. When ``fps`` is set instead, the target count is derived
    from ``metadata.fps``. When neither is set, every frame is returned.

    Note: matches the reference byte-for-byte, including the float-step arange quirk
    that can occasionally yield one extra index for certain ratios.
    """
    total_num_frames = metadata.total_num_frames
    video_fps = metadata.fps

    if num_frames is None and fps is not None:
        if video_fps is None or video_fps <= 0:
            raise ValueError(
                "Cannot resample by `fps` because the source video has no known FPS."
            )
        num_frames = int(total_num_frames / video_fps * fps)
        if num_frames > total_num_frames:
            raise ValueError(
                f"When loading the video with fps={fps}, we computed "
                f"num_frames={num_frames} which exceeds "
                f"total_num_frames={total_num_frames}. Check fps or video metadata."
            )

    if num_frames is not None:
        indices = np.arange(
            0, total_num_frames, total_num_frames / num_frames, dtype=int
        )
    else:
        indices = np.arange(0, total_num_frames, dtype=int)
    return indices


def load_video(
    video: VideoInput,
    num_frames: Optional[int] = None,
    fps: Optional[float] = None,
    backend: str = "pyav",
    sample_indices_fn: Optional[Callable] = None,
    **kwargs,
) -> Tuple[np.ndarray, VideoMetadata]:
    """Load a video into a numpy array with reference-compatible semantics.

    Args:
        video: A local file path, an ``http(s)://`` URL, raw video bytes,
            a pre-decoded frame stack (``(N, H, W, 3)`` ndarray), or: as a
            zeromodels extension: a directory containing pre-extracted frame
            images.
        num_frames: Number of frames to sample uniformly. Mutually exclusive
            with ``fps`` unless ``sample_indices_fn`` is given.
        fps: Target frames per second; implies uniform resampling from the
            source video's fps.
        backend: One of ``"pyav"``, ``"opencv"``, ``"decord"``. Defaults to
            ``"pyav"`` to match the reference. Missing backend libraries raise
            ``ImportError`` at call time.
        sample_indices_fn: Optional callable ``fn(metadata, **kwargs) ->
            indices``. Overrides ``num_frames`` / ``fps`` when supplied.
        **kwargs: Forwarded to ``sample_indices_fn``.

    Returns:
        ``(frames, metadata)`` where ``frames`` is a ``(N, H, W, 3)`` uint8
        RGB numpy array and ``metadata`` is a :class:`VideoMetadata`
        instance describing the decoded stream (with
        ``metadata.frames_indices`` filled in).
    """
    if fps is not None and num_frames is not None and sample_indices_fn is None:
        raise ValueError(
            "`num_frames`, `fps`, and `sample_indices_fn` are mutually exclusive; "
            "please pass only one."
        )

    if sample_indices_fn is None:

        def sample_indices_fn(metadata, **fn_kwargs):  # type: ignore[misc]
            return default_sample_indices_fn(
                metadata, num_frames=num_frames, fps=fps, **fn_kwargs
            )

    # Pre-decoded frame stack passthrough (ndarray or list of frames).
    if isinstance(video, np.ndarray) or (
        not isinstance(video, (str, bytes, bytearray, os.PathLike))
        and hasattr(video, "__len__")
    ):
        return passthrough_frames(video)

    if isinstance(video, os.PathLike):
        video = os.fspath(video)

    if isinstance(video, (bytes, bytearray)):
        if backend == "opencv":
            raise ValueError(
                "OpenCV backend cannot read from raw bytes. Use "
                "backend='pyav' or backend='decord'."
            )
        file_obj: Any = io.BytesIO(bytes(video))
    elif isinstance(video, str):
        if os.path.isdir(video):
            return load_frame_directory(video, sample_indices_fn, **kwargs)
        parsed = urlparse(video)
        if parsed.scheme in ("http", "https"):
            if backend == "opencv":
                raise ValueError(
                    "OpenCV backend cannot read URLs directly. Use "
                    "backend='pyav' or backend='decord', or download the "
                    "file to disk first."
                )
            file_obj = download_url(video)
        elif os.path.isfile(video):
            file_obj = video
        else:
            raise FileNotFoundError(
                f"Video path does not exist or is not a file: {video!r}"
            )
    else:
        raise TypeError(f"Unsupported video input type: {type(video).__name__}.")

    decoder = VIDEO_DECODERS.get(backend)
    if decoder is None:
        raise ValueError(
            f"Unknown backend {backend!r}. Must be one of {sorted(VIDEO_DECODERS)}."
        )

    return decoder(file_obj, sample_indices_fn, **kwargs)


def passthrough_frames(video) -> Tuple[np.ndarray, VideoMetadata]:
    frames = np.asarray(video)
    if frames.ndim != 4 or frames.shape[-1] not in (1, 3, 4):
        raise ValueError(
            f"Passthrough frames must be shaped (N, H, W, C); got {frames.shape}."
        )
    n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    metadata = VideoMetadata(
        total_num_frames=int(n),
        video_backend="passthrough",
        height=int(h),
        width=int(w),
        frames_indices=list(range(int(n))),
    )
    return frames, metadata


def download_url(url: str) -> io.BytesIO:
    import urllib.request

    with urllib.request.urlopen(url) as response:
        return io.BytesIO(response.read())


def load_frame_directory(
    path: str, sample_indices_fn: Callable, **kwargs
) -> Tuple[np.ndarray, VideoMetadata]:
    files: List[str] = []
    for ext in _FRAME_SUFFIXES:
        files.extend(glob.glob(os.path.join(path, f"*{ext}")))
        files.extend(glob.glob(os.path.join(path, f"*{ext.upper()}")))
    files = sorted(set(files))
    if not files:
        raise ValueError(f"No image frames found in directory: {path}")

    first = load_image(files[0])
    h, w = first.shape[:2]

    metadata = VideoMetadata(
        total_num_frames=len(files),
        fps=None,
        duration=None,
        video_backend="directory",
        height=int(h),
        width=int(w),
    )
    indices = np.asarray(sample_indices_fn(metadata=metadata, **kwargs), dtype=int)
    indices = np.clip(indices, 0, len(files) - 1)
    frames = [load_image(files[int(i)]) for i in indices]
    metadata.frames_indices = [int(i) for i in indices]
    return np.stack(frames), metadata


def read_video_opencv(
    video_path, sample_indices_fn: Callable, **kwargs
) -> Tuple[np.ndarray, VideoMetadata]:
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "The `opencv` backend requires `opencv-python`. Install with "
            "`pip install opencv-python`."
        ) from e

    if not isinstance(video_path, str):
        raise ValueError(
            "OpenCV backend requires a local file path, not a file-like object."
        )

    cap = cv2.VideoCapture(video_path)
    total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    duration = (total_num_frames / video_fps) if video_fps else 0.0

    metadata = VideoMetadata(
        total_num_frames=int(total_num_frames),
        fps=float(video_fps),
        duration=float(duration),
        video_backend="opencv",
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )

    indices = np.asarray(sample_indices_fn(metadata=metadata, **kwargs), dtype=int)
    want = {int(i) for i in indices}

    frames: List[np.ndarray] = []
    idx = 0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if idx in want:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
            if idx >= total_num_frames:
                break
    finally:
        cap.release()

    metadata.frames_indices = [int(i) for i in indices]
    if not frames:
        raise ValueError(f"No frames decoded from video: {video_path}")
    return np.stack(frames), metadata


def read_video_pyav(
    file_obj, sample_indices_fn: Callable, **kwargs
) -> Tuple[np.ndarray, VideoMetadata]:
    try:
        import av
    except ImportError as e:
        raise ImportError(
            "The `pyav` backend requires `av`. Install with `pip install av`."
        ) from e

    container = av.open(file_obj)
    stream = container.streams.video[0]
    total_num_frames = int(stream.frames)
    video_fps = float(stream.average_rate) if stream.average_rate else 0.0
    duration = float(stream.duration * stream.time_base) if stream.duration else 0.0

    metadata = VideoMetadata(
        total_num_frames=total_num_frames,
        fps=video_fps,
        duration=duration,
        video_backend="pyav",
        height=int(stream.codec_context.height),
        width=int(stream.codec_context.width),
    )

    indices = np.asarray(sample_indices_fn(metadata=metadata, **kwargs), dtype=int)
    want = {int(i) for i in indices}

    frames: List[np.ndarray] = []
    for idx, frame in enumerate(container.decode(video=0)):
        if idx in want:
            frames.append(frame.to_ndarray(format="rgb24"))
        if total_num_frames and idx + 1 >= total_num_frames:
            break
    container.close()

    metadata.frames_indices = [int(i) for i in indices]
    if not frames:
        raise ValueError("No frames decoded from video with `pyav` backend.")
    return np.stack(frames), metadata


def read_video_decord(
    file_obj, sample_indices_fn: Callable, **kwargs
) -> Tuple[np.ndarray, VideoMetadata]:
    try:
        import decord
    except ImportError as e:
        raise ImportError(
            "The `decord` backend requires `decord`. Install with `pip install decord`."
        ) from e

    reader = decord.VideoReader(file_obj)
    total_num_frames = len(reader)
    video_fps = float(reader.get_avg_fps())
    duration = (total_num_frames / video_fps) if video_fps else 0.0
    first = reader[0].asnumpy()
    h, w = first.shape[:2]

    metadata = VideoMetadata(
        total_num_frames=int(total_num_frames),
        fps=float(video_fps),
        duration=float(duration),
        video_backend="decord",
        height=int(h),
        width=int(w),
    )

    indices = np.asarray(sample_indices_fn(metadata=metadata, **kwargs), dtype=int)
    frames = reader.get_batch([int(i) for i in indices]).asnumpy()
    metadata.frames_indices = [int(i) for i in indices]
    return frames, metadata


VIDEO_DECODERS = {
    "opencv": read_video_opencv,
    "pyav": read_video_pyav,
    "decord": read_video_decord,
}


def sample_frames(
    num_total: int,
    num_samples: int,
    mode: str = "uniform",
    seed: Optional[int] = None,
) -> List[int]:
    """Return frame indices sampled from ``[0, num_total)``.

    Convenience helper for ad-hoc use when you already hold a decoded frame
    array. For videos on disk, prefer :func:`load_video` with ``num_frames``
    so the sampling matches the reference's semantics.
    """
    if num_total <= 0:
        raise ValueError(f"num_total must be positive, got {num_total}.")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")
    if num_samples >= num_total:
        return list(range(num_total))

    if mode == "uniform":
        return [int(i) for i in np.linspace(0, num_total - 1, num_samples)]
    if mode == "random":
        rng = np.random.default_rng(seed)
        picks = rng.choice(num_total, size=num_samples, replace=False)
        return sorted(int(i) for i in picks)

    raise ValueError(f"Unknown sample mode: {mode!r}. Use 'uniform' or 'random'.")
