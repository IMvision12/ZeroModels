# DINOv2

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>kf_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

DINOv2 keeps [DINO](dino.md)'s self-supervised recipe and scales it: a curated 142 M-image
dataset, a larger teacher, and a training objective (iBOT plus DINO) that supervises
individual patches, not just the image-level `[CLS]` token. The result is **dense
features good enough to use raw**: run a linear probe on the patch tokens and you get
competitive depth, segmentation, and correspondence, with the backbone frozen.

Like DINO these are backbones, not task models. The figures below PCA the patch features
to three RGB components, the visualization the DINOv2 paper made famous: the object pops
out of the background and its parts take on consistent colours.

**Paper**: [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)

## API

### DinoV2Model

```python
DinoV2Model(
    as_backbone=False,
    patch_size=14,
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4.0,
    qkv_bias=True,
    qk_norm=False,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    layer_scale_init=1.0,
    include_normalization=True,
    normalization_mode="imagenet",
    image_size=224,
    input_tensor=None,
    name="DinoV2Model",
)
```

The DINOv2 Vision Transformer. **This is the backbone class.**

**Parameters**

- **as_backbone** (`bool`, *optional*, defaults to `False`): return a list of intermediate feature maps (embedding + one per block) instead of the final token sequence.
- **patch_size** (`int`, *optional*, defaults to `14`): pixels per patch. DINOv2 uses 14, so a 224 input is a 16x16 grid.
- **embed_dim** / **depth** / **num_heads** (`int`, *optional*): transformer width, blocks, and heads. Filled in by `from_weights` from the variant config.
- **layer_scale_init** (`float`, *optional*, defaults to `1.0`): initial LayerScale value, DINOv2's per-branch learned scaling.
- **include_normalization** (`bool`, *optional*, defaults to `True`): normalize inside the model, so you feed raw `[0, 255]` pixels.
- **normalization_mode** (`str`, *optional*, defaults to `"imagenet"`): which mean/std to use when normalizing.
- **image_size** (`int` or `tuple`, *optional*, defaults to `224`): input resolution the model is built for.
- **input_tensor** (`dict`, *optional*): pre-existing input tensors to build on.
- **name** (`str`, *optional*, defaults to `"DinoV2Model"`): model name.

**Call** `model(pixel_values, training=False)` with raw `[0, 255]` pixels. **Returns** the
token sequence `(B, 1 + num_patches, embed_dim)`, the leading token being `[CLS]`. With
`as_backbone=True`, a list of `depth + 1` such tensors.

## Preprocessing

`DinoV2ImageProcessor.from_weights("zeromodels/<variant>")` reads its settings from the
repo's `kf_preprocessor.json`; `DinoV2ImageProcessor()` with no arguments gives the same
defaults. Two matching options:

- **`DinoV2ImageProcessor`** (matches transformers' `BitImageProcessor` for
  `facebook/dinov2-*`): an aspect-preserving shortest-edge resize to 256 (bicubic, through
  PIL on the raw uint8 image), a center crop to 224, rescale to `[0, 1]`, and
  ImageNet-standard normalization. Because it already normalizes, load the model with
  `include_normalization=False`:

  ```python
  from zeromodels.models.dino_v2 import DinoV2Model, DinoV2ImageProcessor

  model = DinoV2Model.from_weights(
      "zeromodels/dinov2-base", include_normalization=False
  )
  processor = DinoV2ImageProcessor.from_weights("zeromodels/dinov2-base")

  pixel_values = processor("bear.jpg")["pixel_values"]  # (1, 224, 224, 3), normalized
  tokens = model(pixel_values, training=False)
  ```

- **Built-in normalization**: `DinoV2Model` defaults to `include_normalization=True`, so
  you can instead feed **raw `[0, 255]` pixels** (resized to the model's `image_size`) and
  the ImageNet normalization happens inside. Pass `include_normalization=False` if you have
  already normalized.

## Model Variants

| Variant id | Backbone | Patch | Params |
|---|---|---|---:|
| `dinov2-small` | ViT-S | 14 | ~21 M |
| `dinov2-base` | ViT-B | 14 | ~86 M |
| `dinov2-large` | ViT-L | 14 | ~300 M |

## Basic Usage: Feature Extraction

<img src="../assets/dinov2_pca_output.jpg" alt="DINOv2 ViT-g/14 (giant): a motorcyclist beside the PCA of its patch features" width="500">

Run the backbone, drop the `[CLS]` token, and PCA the patch features to three components.
The rider and machine take one colour, the grass and road two others, all recovered with
no labels.

> The giant is ~1.1B parameters (a ~4.5 GB download). If memory is tight, load it in half
> precision with `load_dtype="bfloat16"`; the features are effectively unchanged.

```python
import keras
import numpy as np
import torch
from PIL import Image
from zeromodels.models.dino_v2 import DinoV2ImageProcessor, DinoV2Model

size, patch = 896, 14
model = DinoV2Model.from_weights(
    "zeromodels/dinov2-giant", image_size=size, include_normalization=False
)
processor = DinoV2ImageProcessor.from_weights(
    "zeromodels/dinov2-giant", resize_size=1024, crop_size=size
)

x = processor("assets/data/coco_motorcycle.jpg")["pixel_values"]  # (1, 896, 896, 3)

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
vis.save("assets/dinov2_pca.jpg")
```

```
(4097, 1536)
```

`4097 = 1 + 64 * 64`: one `[CLS]` token plus a 64x64 patch grid at 896/14. DINOv2's whole
selling point is that those patch features are good enough to use directly, so a linear
layer on `tokens[1:]` is a real segmentation or depth head.

> Use `torch.no_grad()` on the torch backend. These are pure forward passes; autograd
> would retain every intermediate for nothing.

### Batch Processing Multiple Images

Stack images that share a size into one batch:

<img src="../assets/dinov2_pca_batch_output.jpg" alt="DINOv2 ViT-g/14 (giant) on two cats and a busy street, each beside its feature PCA" width="440">

```python
import keras
import numpy as np
import torch
from zeromodels.models.dino_v2 import DinoV2ImageProcessor, DinoV2Model

size = 896
model = DinoV2Model.from_weights(
    "zeromodels/dinov2-giant", image_size=size, include_normalization=False
)
processor = DinoV2ImageProcessor.from_weights(
    "zeromodels/dinov2-giant", resize_size=1024, crop_size=size
)

paths = ["assets/data/coco_cats.jpg", "assets/data/coco_bicycles.jpg"]
batch = processor(paths)["pixel_values"]  # (2, 896, 896, 3)

with torch.no_grad():
    tokens = model(batch, training=False)
print(np.asarray(keras.ops.convert_to_numpy(tokens)).shape)  # (2, 4097, 1536)
```

```
(2, 4097, 1536)
```

The two cats separate from the blanket as one region each; even the cluttered street
resolves the riders, bikes, and shopfront into distinct patches.

## Intermediate Features

`as_backbone=True` returns the embedding plus one feature map per block, for feeding a
DPT-style neck or a segmentation head, which is exactly how DINOv2 is used for dense
prediction:

```python
model = DinoV2Model.from_weights(
    "zeromodels/dinov2-giant", as_backbone=True, image_size=size
)
features = model(x, training=False)  # x from above, at 896
print(len(features), features[-1].shape)  # 41  (1, 4097, 1536)
```

## Data Format

The ViT works in token space, so it is layout-agnostic: it returns `(B, tokens, dim)`
regardless of `keras.config.image_data_format()`.

## Input Resolution

Any size that is a **multiple of the patch size, 14**, works: the learned position
embeddings are bilinearly interpolated to the requested patch grid at load time, so the
pretrained weights stay valid. The figures here use `image_size=896` for a finer map than
the default 224 gives.

> Interpolation runs in the Keras `.weights.h5` reader, which the Hub checkpoint path uses. The
> `hf:` path assigns the checkpoint tensor directly and so requires an exact match; ask
> for a non-default size over `hf:` and it raises a position-embedding shape mismatch.

## Loading Fine-tuned and Community Weights

Any Hugging Face repo whose `model_type` is `"dinov2"` loads with the `hf:` prefix.

```python
from zeromodels.models.dino_v2 import DinoV2Model

model = DinoV2Model.from_weights("hf:facebook/dinov2-small")
model = DinoV2Model.from_weights("hf:<user>/dinov2-finetuned")

# Architecture only, randomly initialized
model = DinoV2Model.from_weights("zeromodels/dinov2-giant", load_weights=False)
```

See also [DINO](dino.md), the original, and [DINOv3](dinov3.md), which adds register
tokens and rotary position embeddings.
