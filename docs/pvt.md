# PVT

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/pvt-&lt;variant&gt;-224</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/pvt-tiny-224")</code>.
</div>

PVT (Pyramid Vision Transformer) is a hierarchical vision transformer: four stages that
halve the spatial resolution and grow the channel width, so a single backbone produces a
CNN-style feature pyramid usable for classification and dense prediction. Each stage is a
**non-overlapping** convolutional patch embedding with a **learned position embedding**,
**spatial-reduction attention** (the key/value tokens are shrunk by a strided convolution so
attention stays affordable at high resolution), and a standard two-dense feed-forward
network. The last stage prepends a class token, and the classifier reads it.

**Paper**: [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122)

For the second-generation model (overlapping patches, no position embeddings, convolutional
FFN, and a linear-attention option), see [PVTv2](pvt_v2.md).

## API

### PvtImageClassify

```python
PvtImageClassify(
    hidden_sizes=(64, 128, 320, 512),
    depths=(2, 2, 2, 2),
    num_attention_heads=(1, 2, 5, 8),
    sr_ratios=(8, 4, 2, 1),
    mlp_ratios=(8, 8, 4, 4),
    image_size=224,
    include_normalization=True,
    normalization_mode="imagenet",
    num_classes=1000,
    classifier_activation="linear",
    name="PvtImageClassify",
)
```

The classifier: the backbone plus a dense head over the last stage's class token.
`include_normalization=True` means the model takes **raw `[0, 255]` pixels** and applies
ImageNet mean/std internally, so there is no separate image processor to construct.

**Parameters**

- **hidden_sizes** / **depths** / **num_attention_heads** / **sr_ratios** / **mlp_ratios** (`tuple`): per-stage width, block count, heads, spatial-reduction ratio, and FFN expansion. The variants differ only in `depths`; `from_weights` fills these from the variant config.
- **image_size** (`int`, *optional*, defaults to `224`): resolution the model is built for. The learned position embeddings are interpolated to this grid (see [Variable Input Resolution](#variable-input-resolution)).
- **include_normalization** (`bool`, *optional*, defaults to `True`): bake ImageNet normalization into the graph.
- **num_classes** (`int`, *optional*, defaults to `1000`): classifier outputs.

**Call** `model(pixel_values, training=False)`. **Returns** class logits of shape `(B, num_classes)`.

### PvtModel

The backbone alone. With `as_backbone=True` it returns the four stage feature maps
(the pyramid, class token dropped) instead of just the last one, for detection or
segmentation necks.

```python
PvtModel(as_backbone=False, hidden_sizes=(64, 128, 320, 512), ..., include_normalization=True)
```

### PvtConfig

Typed config (`model_type="pvt"`) holding the fields above; serialized into each Hub repo's
`zm_config.json`.

## Model Variants

For `PvtImageClassify.from_weights("zeromodels/<variant>")`. Every variant shares the widths
`(64, 128, 320, 512)` and differs only in depth:

| Variant id        | Depths        | Params | ImageNet-1k top-1 |
|-------------------|---------------|-------:|------------------:|
| `pvt-tiny-224`    | (2, 2, 2, 2)  | 13.2M  |             75.1% |
| `pvt-small-224`   | (3, 4, 6, 3)  | 24.5M  |             79.8% |
| `pvt-medium-224`  | (3, 4, 18, 3) | 44.2M  |             81.2% |
| `pvt-large-224`   | (3, 8, 27, 3) | 61.4M  |             81.7% |

Reported top-1 is from the paper. All variants are 224x224, 1000 classes.

## Basic Usage

```python
import keras
import numpy as np
from PIL import Image
from zeromodels.models.pvt import PvtImageClassify

model = PvtImageClassify.from_weights("zeromodels/pvt-tiny-224")

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
from zeromodels.models.pvt import PvtModel

backbone = PvtModel.from_weights("zeromodels/pvt-tiny-224", as_backbone=True)
feats = backbone(np.zeros((1, 224, 224, 3), "float32"), training=False)
print([tuple(f.shape) for f in feats])
# [(1, 56, 56, 64), (1, 28, 28, 128), (1, 14, 14, 320), (1, 7, 7, 512)]
```

The strides are 4, 8, 16, 32, matching a standard CNN backbone.

## Variable Input Resolution

Unlike [PVTv2](pvt_v2.md), PVT v1 has **learned position embeddings**, so a non-224 input
needs them resized. Build the model at the target size and `from_weights` bilinearly
interpolates each stage's position embedding from its trained 224 grid to the new grid at
load time.

```python
model = PvtImageClassify.from_weights("zeromodels/pvt-tiny-224", image_size=384)
logits = model(np.zeros((1, 384, 384, 3), "float32"), training=False)
```

## Data Format

**The model supports both `channels_last` and `channels_first`, and the two are
bit-exact.** A model reads `keras.config.image_data_format()` when it is **constructed**
(there is no `data_format` argument); set the format before building.

```python
import keras

keras.config.set_image_data_format("channels_first")
model = PvtImageClassify.from_weights("zeromodels/pvt-tiny-224")  # expects (B, 3, H, W)
```

## Loading Fine-tuned and Community Weights

Any Hugging Face repo whose `model_type` is `"pvt"` (for example the original
`Zetatech/pvt-*-224` checkpoints) loads with the `hf:` prefix, converting on the fly:

```python
from zeromodels.models.pvt import PvtImageClassify

model = PvtImageClassify.from_weights("hf:Zetatech/pvt-tiny-224")
model = PvtImageClassify.from_weights("hf:<user>/pvt-finetuned-on-my-data")

# Architecture only, randomly initialized
model = PvtImageClassify.from_weights("zeromodels/pvt-tiny-224", load_weights=False)
```

`PvtModel` accepts `hf:` the same way.
