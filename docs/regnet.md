# RegNet

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/regnet-&lt;variant&gt;</a>
(12 <b>X</b> + 12 <b>Y</b> variants; each repo carries <code>zm_config.json</code> +
<code>model.weights.h5</code>). Load with
<code>from_weights("zeromodels/regnet-y-040")</code>.
</div>

RegNet (Designing Network Design Spaces) is a family of ConvNets whose per-stage widths and
depths follow a simple **quantized-linear rule** found by searching design spaces rather than
tuning individual architectures. It is a 3x3 stride-2 stem followed by **four stages** of
residual blocks; each block is a `1x1 -> 3x3 grouped -> 1x1` bottleneck, and the **RegNet-Y**
variant inserts a **Squeeze-and-Excitation** module. The 3x3 convolution is split into
`out_channels // groups_width` groups. Width grows and resolution halves each stage, so a
single backbone yields a standard CNN feature pyramid usable for classification and dense
prediction.

**Paper**: [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)

RegNet comes in two families: **X** (plain bottleneck) and **Y** (+ Squeeze-and-Excitation,
the stronger and more common one).

## API

### RegNetImageClassify

```python
RegNetImageClassify(
    embedding_size=32,
    hidden_sizes=(128, 192, 512, 1088),
    depths=(2, 6, 12, 2),
    groups_width=64,
    layer_type="y",
    downsample_in_first_stage=True,
    image_size=224,
    include_normalization=True,
    normalization_mode="imagenet",
    num_classes=1000,
    classifier_activation="linear",
    name="RegNetImageClassify",
)
```

The classifier: the backbone plus a GlobalAveragePooling2D + dense head.
`include_normalization=True` means the model takes **raw `[0, 255]` pixels** and applies
ImageNet mean/std internally, so there is no separate image processor to construct.

**Parameters**

- **embedding_size** (`int`): output width of the 3x3 stride-2 stem.
- **hidden_sizes** / **depths** (`tuple`): per-stage output width and block count.
- **groups_width** (`int`): channels per group of the 3x3 grouped convolution (a block's group count is `out_channels // groups_width`).
- **layer_type** (`str`): `"y"` (adds Squeeze-and-Excitation) or `"x"`.
- **downsample_in_first_stage** (`bool`): whether the first stage downsamples. `True` for the standard checkpoints (RegNet has no pooling stem).
- **image_size** (`int`, *optional*, defaults to `224`): resolution the model is built for.
- **include_normalization** (`bool`, *optional*, defaults to `True`): bake ImageNet normalization into the graph.
- **num_classes** (`int`, *optional*, defaults to `1000`): classifier outputs.

`from_weights` fills the architectural fields from the variant's config, so you normally pass
only the repo id.

**Call** `model(pixel_values, training=False)`. **Returns** class logits of shape
`(B, num_classes)`.

### RegNetModel

The backbone alone. With `as_backbone=True` it returns the four stage feature maps
(the pyramid) instead of just the last one, for detection or segmentation necks.

```python
RegNetModel(as_backbone=False, layer_type="y", groups_width=64, ..., include_normalization=True)
```

### RegNetConfig

Typed config (`model_type="regnet"`) holding the fields above; serialized into each Hub repo's
`zm_config.json`.

## Model Variants

For `RegNetImageClassify.from_weights("zeromodels/regnet-<variant>")`. The number is the
model's compute in units of 0.1 GFLOPs (`002` = 0.2 GF ... `320` = 32 GF).

| Family                 | Variants                                                                 |
|------------------------|--------------------------------------------------------------------------|
| **RegNet-X** (plain)   | `regnet-x-{002,004,006,008,016,032,040,064,080,120,160,320}`             |
| **RegNet-Y** (+ SE)    | `regnet-y-{002,004,006,008,016,032,040,064,080,120,160,320}`             |

At matched compute the Y family (with Squeeze-and-Excitation) is generally stronger; e.g.
`regnet-y-320` reaches ~80.9% ImageNet-1k top-1. All are 224x224, 1000 classes. The larger
self-supervised `facebook/regnet-y-{320,640,1280,10b}-seer` checkpoints are not mirrored here
but load on the fly with the `hf:` prefix (see [below](#loading-fine-tuned-and-community-weights)).

## Basic Usage

```python
import keras
import numpy as np
from PIL import Image
from zeromodels.models.regnet import RegNetImageClassify

model = RegNetImageClassify.from_weights("zeromodels/regnet-y-040")

image = Image.open("assets/data/coco_bear.jpg").convert("RGB").resize((224, 224))
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
from zeromodels.models.regnet import RegNetModel

backbone = RegNetModel.from_weights("zeromodels/regnet-y-040", as_backbone=True)
feats = backbone(np.zeros((1, 224, 224, 3), "float32"), training=False)
print([tuple(f.shape) for f in feats])
# spatial 56 / 28 / 14 / 7 (strides 4, 8, 16, 32); channels are the variant's hidden_sizes
```

The strides are 4, 8, 16, 32, matching a standard CNN backbone. RegNet is fully
convolutional (no learned position embeddings), so any input resolution works with no
weight interpolation: build the model at the target `image_size`.

## Data Format

**The model supports both `channels_last` and `channels_first`, and the two are
bit-exact.** A model reads `keras.config.image_data_format()` when it is **constructed**
(there is no `data_format` argument); set the format before building.

```python
import keras

keras.config.set_image_data_format("channels_first")
model = RegNetImageClassify.from_weights(
    "zeromodels/regnet-y-040"
)  # expects (B, 3, H, W)
```

## Loading Fine-tuned and Community Weights

The `zeromodels/regnet-*` repos above are pre-converted. Any **other** Hugging Face repo whose
`model_type` is `"regnet"` (the upstream `facebook/regnet-*` and `-seer` checkpoints, or any
fine-tune) loads with the `hf:` prefix, converting on the fly:

```python
from zeromodels.models.regnet import RegNetImageClassify

model = RegNetImageClassify.from_weights("hf:facebook/regnet-x-320")
model = RegNetImageClassify.from_weights("hf:<user>/regnet-finetuned-on-my-data")

# Architecture only, randomly initialized
model = RegNetImageClassify.from_weights("hf:facebook/regnet-y-040", load_weights=False)
```

`RegNetModel` accepts `hf:` the same way.
