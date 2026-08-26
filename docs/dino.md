# DINO

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

DINO is self-supervised: it trains a ViT with no labels, by matching the outputs of a
student and a teacher network across different crops of the same image. The surprise in
the paper was that the resulting features are *semantic* for free. The [CLS] token's
attention lands on objects, and the patch features cluster into parts, without a single
annotation.

These are **backbones**, not task models. They take an image and return features; what you
do with them, classification, segmentation, retrieval, is up to a head you add. The
figures below make the learned structure visible by running PCA on the patch features and
mapping the top three components to RGB.

**Paper**: [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)

## API

### DinoViTModel

```python
DinoViTModel(
    as_backbone=False,
    patch_size=16,
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4.0,
    qkv_bias=True,
    qk_norm=False,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    include_normalization=True,
    normalization_mode="imagenet",
    image_size=224,
    input_tensor=None,
    name="DinoViTModel",
)
```

The DINO Vision Transformer. **This is the main backbone class.**

**Parameters**

- **as_backbone** (`bool`, *optional*, defaults to `False`): return a list of intermediate feature maps (embedding + one per block) instead of the final token sequence.
- **patch_size** (`int`, *optional*, defaults to `16`): pixels per patch. `8` for the `*8` variants. Filled in by `from_weights`.
- **embed_dim** / **depth** / **num_heads** (`int`, *optional*): transformer width, blocks, and heads. Set from the variant config.
- **mlp_ratio** / **qkv_bias** / **qk_norm** / **drop_rate** / **attn_drop_rate**: standard ViT block knobs.
- **include_normalization** (`bool`, *optional*, defaults to `True`): normalize inside the model, so you feed raw `[0, 255]` pixels.
- **normalization_mode** (`str`, *optional*, defaults to `"imagenet"`): which mean/std to use when normalizing.
- **image_size** (`int` or `tuple`, *optional*, defaults to `224`): input resolution the model is built for.
- **input_tensor** (`dict`, *optional*): pre-existing input tensors to build on.
- **name** (`str`, *optional*, defaults to `"DinoViTModel"`): model name.

**Call** `model(pixel_values, training=False)` with raw `[0, 255]` pixels. **Returns** the
token sequence `(B, 1 + num_patches, embed_dim)`, the leading token being `[CLS]`. With
`as_backbone=True`, a list of `depth + 1` such tensors.

### DinoResNetModel

```python
DinoResNetModel(
    as_backbone=False,
    depths=None,
    filters=None,
    include_normalization=True,
    normalization_mode="imagenet",
    image_size=224,
    input_tensor=None,
    name="DinoResNetModel",
)
```

The DINO ResNet-50 backbone, for a convolutional alternative. **Returns** the final
`(B, 7, 7, 2048)` feature map under `channels_last`, or with `as_backbone=True` the four
stage maps (`(B, 56, 56, 256)` through `(B, 7, 7, 2048)`).

## Preprocessing

`DinoImageProcessor` handles both DINO families, keyed on `model_type` (the same tag the
model config carries). Loading it with `from_weights("zeromodels/<variant>")` reads the
recipe from the repo's `zm_preprocessor.json`, so the right one comes back automatically;
`DinoImageProcessor()` with no arguments is the ViT default.

- **ViT** (`facebook/dino-*`, `model_type="dino_vit"`, the default) matches transformers'
  `ViTImageProcessor`: a square resize to 224 (bilinear, through PIL on the raw uint8
  image), rescale to `[0, 1]`, and ImageNet-standard normalization. No center crop.
- **ResNet-50** (`model_type="dino_resnet"`) matches the torch.hub `facebookresearch/dino`
  eval transform: an aspect-preserving shortest-edge resize to 256 (bicubic), a center
  crop to 224, then the same rescale and normalization.

Because the processor already normalizes, load the model with `include_normalization=False`:

```python
from zeromodels.models.dino import DinoViTModel, DinoImageProcessor

model = DinoViTModel.from_weights("zeromodels/dino-vitb16", include_normalization=False)
processor = DinoImageProcessor.from_weights("zeromodels/dino-vitb16")

pixel_values = processor("bear.jpg")["pixel_values"]  # (1, 224, 224, 3), normalized
tokens = model(pixel_values, training=False)
```

`dino-resnet50` loads the same way and needs no extra arguments: its Hub
`zm_preprocessor.json` sets `model_type="dino_resnet"`, so the crop recipe is restored for
you.

```python
from zeromodels.models.dino import DinoResNetModel, DinoImageProcessor

model = DinoResNetModel.from_weights(
    "zeromodels/dino-resnet50", include_normalization=False
)
processor = DinoImageProcessor.from_weights("zeromodels/dino-resnet50")

pixel_values = processor("bear.jpg")["pixel_values"]  # (1, 224, 224, 3), normalized
features = model(pixel_values, training=False)  # (1, 7, 7, 2048)
```

**Built-in normalization**: the models default to `include_normalization=True`, so you can
instead feed **raw `[0, 255]` pixels** (resized to the model's `image_size`) and the
ImageNet normalization happens inside. Pass `include_normalization=False` if you have
already normalized.

## Model Variants

| Variant id | Backbone | Patch | Params |
|---|---|---|---:|
| `dino-vits16` | ViT-S | 16 | ~21 M |
| `dino-vits8` | ViT-S | 8 | ~21 M |
| `dino-vitb16` | ViT-B | 16 | ~85 M |
| `dino-vitb8` | ViT-B | 8 | ~85 M |
| `dino-resnet50` | ResNet-50 | n/a | ~23 M |

The `*8` variants use an 8-pixel patch, so four times as many tokens and a much finer
feature map, at a higher compute cost.

## Basic Usage: Feature Extraction

<img src="../assets/dino_pca_output.jpg" alt="DINO ViT-B/16: a bear beside the PCA of its patch features" width="360">

Run the backbone, drop the `[CLS]` token, and PCA the patch features to three components
for an RGB view of what the model sees. The bear's head and body separate cleanly from
the grass, with no supervision anywhere in the pipeline.

```python
import keras
import numpy as np
import torch
from PIL import Image
from zeromodels.models.dino import DinoImageProcessor, DinoViTModel

size, patch = 896, 16
model = DinoViTModel.from_weights(
    "zeromodels/dino-vitb16", image_size=size, include_normalization=False
)
processor = DinoImageProcessor.from_weights(
    "zeromodels/dino-vitb16", image_resolution=size
)

x = processor("assets/data/coco_bear.jpg")["pixel_values"]  # (1, 896, 896, 3)

with torch.no_grad():
    tokens = model(x, training=False)
tokens = np.asarray(keras.ops.convert_to_numpy(tokens))[0]
print(tokens.shape)  # (1 + num_patches, embed_dim)

# PCA the patch tokens (drop the CLS token) to RGB.
grid = size // patch
patches = tokens[1:].reshape(grid * grid, -1).astype("float64")
patches -= patches.mean(0, keepdims=True)
proj = patches @ np.linalg.svd(patches, full_matrices=False)[2][:3].T
proj = proj.reshape(grid, grid, 3)
lo, hi = proj.min((0, 1)), proj.max((0, 1))
proj = (proj - lo) / (hi - lo + 1e-8)

vis = Image.fromarray((proj * 255).astype("uint8")).resize((size, size), Image.BILINEAR)
vis.save("assets/dino_pca.jpg")
```

```
(3137, 768)
```

`3137 = 1 + 56 * 56`: one `[CLS]` token plus a 56x56 patch grid at 896/16. The `[CLS]`
token, `tokens[0]`, is the image-level embedding you would feed a classification head; the
patches are the dense features the PCA above visualizes.

> Use `torch.no_grad()` on the torch backend. These are pure forward passes; autograd
> would retain every intermediate for nothing.

### Batch Processing Multiple Images

Stack images that share a size into one batch:

<img src="../assets/dino_pca_batch_output.jpg" alt="DINO ViT-B/16 on two elephants and a horse jumper, each beside its feature PCA" width="440">

```python
import keras
import numpy as np
import torch
from zeromodels.models.dino import DinoImageProcessor, DinoViTModel

size = 896
model = DinoViTModel.from_weights(
    "zeromodels/dino-vitb16", image_size=size, include_normalization=False
)
processor = DinoImageProcessor.from_weights(
    "zeromodels/dino-vitb16", image_resolution=size
)

paths = ["assets/data/coco_elephants.jpg", "assets/data/coco_horse_jump.jpg"]
batch = processor(paths)["pixel_values"]  # (2, 896, 896, 3)

with torch.no_grad():
    tokens = model(batch, training=False)
print(np.asarray(keras.ops.convert_to_numpy(tokens)).shape)  # (2, 3137, 768)
```

```
(2, 3137, 768)
```

Each row of the figure is an image beside its own feature PCA: the elephants separate from
the scrub, the horse and rider from the trees behind.

## Intermediate Features

`as_backbone=True` returns the embedding plus one feature map per block, for feeding a
DPT-style neck or an FPN:

```python
model = DinoViTModel.from_weights(
    "zeromodels/dino-vitb16", as_backbone=True, image_size=size
)
features = model(x, training=False)  # x from above, at 896
print(len(features), features[-1].shape)  # 13  (1, 3137, 768)
```

`DinoResNetModel(as_backbone=True)` gives the four convolutional stage maps instead.

## Data Format

The ViT works in token space, so it is layout-agnostic. `DinoResNetModel` reads
`keras.config.image_data_format()` when it is **constructed** and returns
`channels_last` `(B, 7, 7, 2048)` or `channels_first` `(B, 2048, 7, 7)` accordingly.

```python
import keras

keras.config.set_image_data_format("channels_first")
model = DinoResNetModel.from_weights(
    "zeromodels/dino-resnet50"
)  # output (B, 2048, 7, 7)
```

## Input Resolution

Any size that is a **multiple of the patch size** works: the learned position embeddings
are bilinearly interpolated to the requested patch grid at load time, so the pretrained
weights stay valid. The figures here use `image_size=896` for a finer map than the
default 224 gives.

## Loading Fine-tuned and Community Weights

Any Hugging Face repo whose `model_type` is `"vit"` in the DINO layout loads with the
`hf:` prefix.

```python
from zeromodels.models.dino import DinoViTModel

model = DinoViTModel.from_weights("hf:facebook/dino-vits16")
model = DinoViTModel.from_weights("hf:<user>/dino-finetuned")

# Architecture only, randomly initialized
model = DinoViTModel.from_weights("zeromodels/dino-vitb16", load_weights=False)
```

See also [DINOv2](dinov2.md), which adds register-free dense features and layer scale, and
[DINOv3](dinov3.md), which adds register tokens and rotary position embeddings.
