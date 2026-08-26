# Video utils

Decoding a video and choosing which frames to keep. This is the input path for the
video-capable VLMs such as [Qwen2.5-VL](qwen2_5_vl.md).

```python
from zeromodels.utils import (
    load_video,
    sample_frames,
    default_sample_indices_fn,
    VideoMetadata,
    VIDEO_DECODERS,
)
```

## load_video

```python
load_video(video, num_frames=None, fps=None, backend="pyav",
           sample_indices_fn=None, **kwargs) -> Tuple[np.ndarray, VideoMetadata]
```

Decode a video and sample frames from it. Returns `(frames, metadata)`, where `frames` is
`(T, H, W, 3)` uint8 and `metadata` is a [`VideoMetadata`](#videometadata).

**Parameters**

- **video**: a local path, a URL, raw `bytes`, a directory of frame images, a NumPy array, or a sequence of frames. Arrays and sequences pass straight through without decoding.
- **num_frames** (`int`, *optional*): sample exactly this many frames, spread uniformly across the whole video.
- **fps** (`float`, *optional*): sample at this rate instead of a fixed count. Passing both `num_frames` and `fps` raises.
- **backend** (`str`, *optional*, defaults to `"pyav"`): decoder to use, one of `VIDEO_DECODERS`.
- **sample_indices_fn** (`callable`, *optional*): full control over which frames are taken. Receives the `VideoMetadata` and returns an array of indices.
- **kwargs**: forwarded to `sample_indices_fn`.

```python
from zeromodels.utils import load_video

frames, meta = load_video("clip.mp4", num_frames=16)
print(frames.shape, meta.total_num_frames, meta.fps)
```

`num_frames` and `fps` answer different questions. A fixed **count** gives every video the
same token budget regardless of length, which is what a VLM's context window wants. A fixed
**rate** keeps motion at a consistent speed, so a 10 s clip and a 60 s clip produce very
different frame counts. Most VLM pipelines want `num_frames`.

## Decoders

Dispatch goes through a dict, so the dependency is only needed for the backend you actually
name:

```python
from zeromodels.utils import VIDEO_DECODERS

print(list(VIDEO_DECODERS))
```

```
['opencv', 'pyav', 'decord']
```

| Backend | Package | Notes |
|---|---|---|
| `pyav` | `av` | The default. Widest container and codec coverage, correct on variable-frame-rate files. |
| `opencv` | `opencv-python` | Convenient if OpenCV is already a dependency. Seeking can be approximate on some containers. |
| `decord` | `decord` | Fastest for dense random-access sampling. |

Paths, URLs, and raw bytes are routed to the chosen decoder. Two inputs bypass it entirely:
a **directory** is read as an ordered set of frame images, and an **array or sequence** is
passed through as already-decoded frames.

## VideoMetadata

```python
VideoMetadata(
    total_num_frames,
    fps=None,
    width=None,
    height=None,
    duration=None,
    video_backend=None,
    frames_indices=None,
)
```

What the decoder learned about the source, returned alongside the frames.

- **total_num_frames**: frames in the *source*, not in what you got back.
- **fps**, **duration**, **width**, **height**: as reported by the container.
- **video_backend**: which decoder produced this.
- **frames_indices**: which source frames were actually kept.

`frames_indices` is the field that matters downstream. Models with temporal position
encodings, Qwen-VL's M-RoPE among them, need to know where each sampled frame sat on the
real timeline; without it, 16 frames from a 3-second clip and 16 from a 3-minute one look
identical to the model.

## sample_frames

```python
sample_frames(num_total, num_samples, mode="uniform", seed=None) -> List[int]
```

Pick `num_samples` indices out of `num_total`, standalone from any decoding.

```python
from zeromodels.utils import sample_frames

print(sample_frames(100, 8))
```

```
[0, 14, 28, 42, 56, 70, 84, 99]
```

`mode` is `"uniform"` (`linspace` over `[0, num_total - 1]`, so both endpoints are included)
or `"random"`; pass a `seed` to make the random mode reproducible. Asking for at least as
many samples as there are frames returns all of them rather than raising.

**This is not the same policy `load_video(num_frames=...)` uses.** That path goes through
`default_sample_indices_fn`, which does `arange(0, total, total / num_frames)`: it starts at
0 but does not reach the last frame. `sample_frames` is the convenience helper for frames
you already hold in an array; for video on disk prefer `load_video`, whose sampling matches
the reference implementation the VLM checkpoints were trained against.

## default_sample_indices_fn

```python
default_sample_indices_fn(metadata, num_frames=None, fps=None, **kwargs) -> np.ndarray
```

The policy `load_video` uses when you do not pass your own: honor `num_frames` if given,
otherwise derive a count from `fps` against the source's own fps, otherwise keep every
frame. Resampling by `fps` needs the container to report its frame rate, and raises if it
does not.

Wrap it when you want the default behavior with one adjustment, such as skipping a lead-in:

```python
from zeromodels.utils import default_sample_indices_fn, load_video


def skip_intro(metadata, **kwargs):
    indices = default_sample_indices_fn(metadata, num_frames=16, **kwargs)
    return indices[indices > int(metadata.fps * 5)]


frames, meta = load_video("clip.mp4", sample_indices_fn=skip_intro)
```

**A custom `sample_indices_fn` takes over completely: `num_frames` and `fps` are no longer
consulted.** Passing `num_frames=16` alongside your own function does not raise, it is just
silently ignored, which is why the count is set inside the function above. Only `**kwargs`
reach it.

See also [Image utils](utils_image.md), [Visualization utils](utils_visualization.md), and
[Utilities](utils.md).
