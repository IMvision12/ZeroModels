# PVTv2

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/pvt-v2-&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/pvt-v2-b0")</code>.
</div>

PVTv2 is a hierarchical vision transformer: four stages that halve the spatial resolution
and grow the channel width, so a single backbone produces a feature pyramid the way a CNN
does. It improves on [PVT](pvt.md) in three ways: an **overlapping** convolutional patch
embedding, **spatial-reduction attention** that shrinks the key/value sequence with a
strided convolution (so attention stays affordable at high resolution), and a
**convolutional feed-forward network** (a 3x3 depthwise conv between the two dense layers)
that removes the need for any position embedding. Dropping position embeddings is what lets
it run at arbitrary input resolution with no interpolation.

The `b2_linear` variant swaps spatial-reduction attention for **linear attention**: instead
of a strided conv, it average-pools every stage to a fixed 7x7 grid, so the key/value length
is constant regardless of input size and the cost is linear in the number of tokens.

**Paper**: [PVTv2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/abs/2106.13797)

For the first-generation model (non-overlapping patches, learned position embeddings), see
[PVT](pvt.md).

## API

### PvtV2ImageClassify

```python
PvtV2ImageClassify(
    hidden_sizes=(32, 64, 160, 256),
    depths=(2, 2, 2, 2),
    num_attention_heads=(1, 2, 5, 8),
    sr_ratios=(8, 4, 2, 1),
    mlp_ratios=(8, 8, 4, 4),
    linear_attention=False,
    image_size=224,
    include_normalization=True,
    normalization_mode="imagenet",
    num_classes=1000,
    classifier_activation="linear",
    name="PvtV2ImageClassify",
)
```

The classifier: the backbone, a global average pool over the last stage, and one dense head.
`include_normalization=True` means the model takes **raw `[0, 255]` pixels** and applies
ImageNet mean/std internally, so there is no separate image processor to construct.

**Parameters**

- **hidden_sizes** / **depths** / **num_attention_heads** / **sr_ratios** / **mlp_ratios** (`tuple`): per-stage width, block count, heads, spatial-reduction ratio, and FFN expansion. `from_weights` fills these from the variant config.
- **linear_attention** (`bool`, *optional*, defaults to `False`): use the fixed-7x7 pooled linear-attention variant (`b2_linear`).
- **image_size** (`int`, *optional*, defaults to `224`): resolution the model is built for.
- **include_normalization** (`bool`, *optional*, defaults to `True`): bake ImageNet normalization into the graph.
- **num_classes** (`int`, *optional*, defaults to `1000`): classifier outputs.

**Call** `model(pixel_values, training=False)`. **Returns** class logits of shape `(B, num_classes)`.

### PvtV2Model

The backbone alone. With `as_backbone=True` it returns the four stage feature maps
(the pyramid) instead of just the last one, for detection or segmentation necks.

```python
PvtV2Model(as_backbone=False, hidden_sizes=(32, 64, 160, 256), ..., include_normalization=True)
```

### PvtV2Config

Typed config (`model_type="pvt_v2"`) holding the fields above; serialized into each Hub
repo's `zm_config.json`.

## Model Variants

For `PvtV2ImageClassify.from_weights("zeromodels/<variant>")`:

| Variant id         | Params | ImageNet-1k top-1 | Notes                     |
|--------------------|-------:|------------------:|---------------------------|
| `pvt-v2-b0`        |  3.7M  |             70.5% |                           |
| `pvt-v2-b1`        | 14.0M  |             78.7% |                           |
| `pvt-v2-b2`        | 25.4M  |             82.0% |                           |
| `pvt-v2-b2-linear` | 22.6M  |             82.1% | linear (pooled) attention |
| `pvt-v2-b3`        | 45.2M  |             83.1% |                           |
| `pvt-v2-b4`        | 62.6M  |             83.6% |                           |
| `pvt-v2-b5`        | 82.0M  |             83.8% |                           |

Reported top-1 is from the paper. All variants are 224x224, 1000 classes.

## Basic Usage

```python
import keras
import numpy as np
from PIL import Image
from zeromodels.models.pvt_v2 import PvtV2ImageClassify

model = PvtV2ImageClassify.from_weights("zeromodels/pvt-v2-b2")

image = Image.open("assets/data/hf_cat_2.jpg").convert("RGB").resize((224, 224))
pixels = np.asarray(image, "float32")[None]  # (1, 224, 224, 3), raw [0, 255]

logits = model(pixels, training=False)
top5 = np.argsort(keras.ops.convert_to_numpy(logits)[0])[-5:][::-1]
print("top-5 ImageNet-1k class ids:", top5.tolist())
```

Normalization is inside the model, so pass raw pixels. Map the class ids to the
[ImageNet-1k label list](https://huggingface.co/datasets/imagenet-1k) to read names.

## Feature Pyramid

For detection / segmentation, take the four stage outputs:

```python
from zeromodels.models.pvt_v2 import PvtV2Model

backbone = PvtV2Model.from_weights("zeromodels/pvt-v2-b2", as_backbone=True)
feats = backbone(np.zeros((1, 224, 224, 3), "float32"), training=False)
print([tuple(f.shape) for f in feats])
# [(1, 56, 56, 64), (1, 28, 28, 128), (1, 14, 14, 320), (1, 7, 7, 512)]
```

The strides are 4, 8, 16, 32, matching a standard CNN backbone.

## Variable Input Resolution

PVTv2 has **no position embeddings**, so any resolution works with no interpolation: build
the model at the size you want.

```python
model = PvtV2ImageClassify.from_weights("zeromodels/pvt-v2-b2", image_size=384)
logits = model(np.zeros((1, 384, 384, 3), "float32"), training=False)
```

## Data Format

**The model supports both `channels_last` and `channels_first`, and the two are
bit-exact.** A model reads `keras.config.image_data_format() `when it is **constructed**
(there is no `data_format` argument); set the format before building.

```python
import keras
keras.config.set_image_data_format("channels_first")
model = PvtV2ImageClassify.from_weights("zeromodels/pvt-v2-b0")  # expects (B, 3, H, W)
```

## Loading Fine-tuned and Community Weights

Any Hugging Face repo whose `model_type` is `"pvt_v2"` (for example the original
`OpenGVLab/pvt_v2_*` checkpoints) loads with the `hf:` prefix, converting on the fly:

```python
from zeromodels.models.pvt_v2 import PvtV2ImageClassify

model = PvtV2ImageClassify.from_weights("hf:OpenGVLab/pvt_v2_b2")
model = PvtV2ImageClassify.from_weights("hf:<user>/pvt-v2-finetuned-on-my-data")

# Architecture only, randomly initialized
model = PvtV2ImageClassify.from_weights("zeromodels/pvt-v2-b2", load_weights=False)
```

`PvtV2Model` accepts `hf:` the same way.
