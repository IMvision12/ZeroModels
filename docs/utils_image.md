# Image utils

Getting an image from wherever it lives into the `(H, W, 3)` uint8 array the processors
expect, and resolving the shape and data format conventions around it.

```python
from zeromodels.utils import load_image, get_data_format, standardize_input_shape
```

## load_image

```python
load_image(image) -> np.ndarray
```

One entry point for every image source, returning an `(H, W, 3)` **uint8 RGB** array.

**Parameters**

- **image**: a local path, an `http(s)://` URL, raw encoded `bytes`, a `PIL.Image.Image`, or a NumPy array.

```python
from zeromodels.utils import load_image

image = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")
print(image.shape, image.dtype)
```

```
(480, 640, 3) uint8
```

Arrays are normalized rather than rejected:

| Input | What happens |
|---|---|
| 2D `(H, W)` grayscale | Broadcast across 3 channels |
| `(H, W, 4)` RGBA | Truncated to RGB, alpha dropped |
| float array in `[0, 1]` | Scaled to `[0, 255]` uint8 |
| any other dtype | Cast to uint8 |
| `PIL.Image` in any mode | Converted to RGB |

An array is **assumed to already be HWC RGB**. It is the one input `load_image` cannot
sanity-check for you, so a channels-first tensor or a BGR array from OpenCV passes straight
through, wrong. Convert before handing it over.

Anything else raises: a missing path gives `FileNotFoundError`, an unsupported type gives
`TypeError`, and a 3-channel-less array gives `ValueError`. It never returns a silently
wrong shape.

## get_data_format

```python
get_data_format(data_format=None) -> str
```

Resolve `None` to `keras.config.image_data_format()`, and validate anything else. Use it
wherever a layer or processor takes an optional `data_format`.

```python
from zeromodels.utils import get_data_format

print(get_data_format())
print(get_data_format("channels_first"))
```

```
channels_last
channels_first
```

## standardize_input_shape

```python
standardize_input_shape(image_size, data_format=None) -> Tuple[int, int, int]
```

Turn a flexible `image_size` into the canonical 3-tuple for the active data format. This is
what lets every functional vision model accept `image_size=512`, `(512, 512)`, or
`(512, 512, 3)` interchangeably.

```python
from zeromodels.utils import standardize_input_shape

print(standardize_input_shape(224))
print(standardize_input_shape((512, 384)))
print(standardize_input_shape(224, data_format="channels_first"))
```

```
(224, 224, 3)
(512, 384, 3)
(3, 224, 224)
```

- `int N` becomes `(N, N, 3)` under `channels_last`, `(3, N, N)` under `channels_first`.
- `(H, W)` gains the channel dimension in the right position.
- An explicit 3-tuple is checked against the active format.

That last case **raises rather than transposing**. Passing `(3, 224, 224)` while
`channels_last` is active is far more likely to be a mistake than an instruction, and a
silent transpose would surface much later as a shape error inside the model:

```
ValueError: image_size (3, 224, 224) does not match data_format 'channels_last'.
Expected (H, W, C) with C in {1, 3, 4}; for channels_first inputs call
keras.config.set_image_data_format('channels_first').
```

## What is not here

Pixel-level operations, resize, crop, pad, rescale, normalize, live on `BaseImageProcessor`
so every image model shares one backend-agnostic implementation. See
[Main Classes](main_classes.md#baseimageprocessor).

One in-graph helper does live in `image_util`, used by the classification backbones for
their `include_normalization` path:

```python
from zeromodels.utils.image_util import normalize_image_for_classify_models

x = normalize_image_for_classify_models(x, mode="imagenet")
```

It takes `[0, 255]`, divides by 255, and applies a named preset: `"imagenet"`,
`"inception"`, `"dpn"`, `"clip"`, `"zero_to_one"`, or `"minus_one_to_one"`. Being pure
`keras.ops`, it is symbolic-safe and can sit inside a functional graph, unlike the eager
normalization on the processor side.

See also [Video utils](utils_video.md), [Visualization utils](utils_visualization.md), and
[Utilities](utils.md).
