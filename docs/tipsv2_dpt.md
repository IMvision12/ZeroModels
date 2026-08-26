# TIPSv2-DPT

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code> +
<code>zm_preprocessor.json</code>). Load with
<code>from_weights("zeromodels/&lt;variant&gt;")</code>. All three task classes
load from the <b>same</b> single-weight repo.
</div>

TIPSv2-DPT stacks **DPT (Dense Prediction Transformer)** heads on the
[TIPSv2](tipsv2.md) vision backbone. Hidden states are captured at four stages,
reassembled into a pyramid, fused RefineNet-style, and decoded into dense maps. A
single checkpoint carries **both** a depth head (256-bin soft-argmax regression)
and a semantic-segmentation head, so one repo serves three task classes:

- **`Tipsv2DptDensePredict`** - depth + segmentation in one forward pass.
- **`Tipsv2DptDepthEstimation`** - depth only.
- **`Tipsv2DptSemanticSegment`** - segmentation only.

**Paper**: [TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment](https://huggingface.co/papers/2604.12012)

## API

The three classes share one constructor signature (the tiny defaults below are for
`so400m14-dpt`); pick the class for the outputs you want.

### Tipsv2DptDensePredict

```python
Tipsv2DptDensePredict(
    image_size=448,
    patch_size=14,
    num_register_tokens=1,
    vision_hidden_dim=1152,
    vision_num_layers=27,
    vision_num_heads=16,
    vision_mlp_ratio=3.736111111111111,
    vision_use_swiglu_ffn=False,
    vision_layerscale_value=1.0,
    vision_layer_norm_eps=1e-06,
    out_indices=None,
    neck_hidden_sizes=None,
    reassemble_factors=None,
    fusion_hidden_size=256,
    num_depth_bins=256,
    min_depth=0.001,
    max_depth=10.0,
    depth_decoder_activation="relu",
    num_labels=150,
    input_tensor=None,
    name=None,
)
```

Depth + segmentation. Output dict: `predicted_depth` `(B, H', W')` in meters and
`segmentation_logits` `(B, H', W', num_labels)`, both at the DPT feature resolution.

**Parameters**

- **image_size** (`int`, *optional*, defaults to `448`): input image spec. An `int` builds an `N x N x 3` input, a 2-tuple `(H, W)` assumes 3 channels, and a 3-tuple follows the active `keras.config.image_data_format()`.
- **patch_size** (`int`, *optional*, defaults to `14`): ViT patch size.
- **num_register_tokens** (`int`, *optional*, defaults to `1`): number of register tokens in the backbone.
- **vision_hidden_dim** (`int`, *optional*, defaults to `1152`): backbone hidden dimension.
- **vision_num_layers** (`int`, *optional*, defaults to `27`): backbone depth.
- **vision_num_heads** (`int`, *optional*, defaults to `16`): backbone attention heads.
- **vision_mlp_ratio** (`float`, *optional*, defaults to `3.7361`): backbone MLP inner-width multiplier.
- **vision_use_swiglu_ffn** (`bool`, *optional*, defaults to `False`): use the SwiGLU feed-forward backbone (the `g14-dpt` variant).
- **vision_layerscale_value** (`float`, *optional*, defaults to `1.0`): backbone LayerScale init.
- **vision_layer_norm_eps** (`float`, *optional*, defaults to `1e-06`): backbone LayerNorm epsilon.
- **out_indices** (`NoneType`, *optional*, defaults to `None`): 1-indexed backbone stages tapped for the neck; `None` uses the per-variant default.
- **neck_hidden_sizes** (`NoneType`, *optional*, defaults to `None`): per-stage reassemble channel widths; `None` uses the per-variant default.
- **reassemble_factors** (`NoneType`, *optional*, defaults to `None`): per-stage spatial resample factors; `None` uses the per-variant default.
- **fusion_hidden_size** (`int`, *optional*, defaults to `256`): RefineNet fusion channel width.
- **num_depth_bins** (`int`, *optional*, defaults to `256`): soft-argmax depth bins.
- **min_depth** (`float`, *optional*, defaults to `0.001`): minimum depth (meters).
- **max_depth** (`float`, *optional*, defaults to `10.0`): maximum depth (meters).
- **depth_decoder_activation** (`str`, *optional*, defaults to `'relu'`): depth decoder activation.
- **num_labels** (`int`, *optional*, defaults to `150`): segmentation classes (ADE20K = 150).
- **input_tensor** (`NoneType`, *optional*, defaults to `None`): pre-existing input tensors to build on.
- **name** (`NoneType`, *optional*, defaults to `None`): model name.

### Tipsv2DptDepthEstimation

Same signature as `Tipsv2DptDensePredict`. Output dict: `predicted_depth`
`(B, H', W')`.

### Tipsv2DptSemanticSegment

Same signature as `Tipsv2DptDensePredict`. Output dict: `segmentation_logits`
`(B, H', W', num_labels)`.

> All three classes load from the **same** repo. `from_weights` warm-starts the
> single-task heads from the combined checkpoint, so you only download one weight
> file regardless of which class you use.

## Preprocessing

### Tipsv2DptImageProcessor

```python
Tipsv2DptImageProcessor(
    image_resolution=448,
    resample="bilinear",
    do_normalize=False,
    do_resize=True,
    mean=(0.0, 0.0, 0.0),
    std=(1.0, 1.0, 1.0),
    data_format=None,
)
```

Resize to `image_resolution` and rescale to `[0, 1]` (no mean/std normalization),
matching the TIPSv2 backbone's expected inputs.

**Parameters**

- **image_resolution** (`int`, *optional*, defaults to `448`): target square resolution.
- **resample** (`str`, *optional*, defaults to `'bilinear'`): resize interpolation.
- **do_normalize** (`bool`, *optional*, defaults to `False`): apply mean/std normalization (off for TIPSv2).
- **do_resize** (`bool`, *optional*, defaults to `True`): resize the image.
- **mean** (`tuple`, *optional*, defaults to `(0.0, 0.0, 0.0)`): per-channel normalization mean.
- **std** (`tuple`, *optional*, defaults to `(1.0, 1.0, 1.0)`): per-channel normalization std.
- **data_format** (`NoneType`, *optional*, defaults to `None`): `"channels_last"`. Defaults to `keras.config.image_data_format()`.

## Model Variants

Load any of these with `from_weights("zeromodels/<variant id>")`.

| Variant id | Image size | Patch | FFN | Weights |
|---|---:|---:|---|---|
| `tipsv2-b14-dpt` | 448 | 14 | MLP | hub |
| `tipsv2-l14-dpt` | 448 | 14 | MLP | hub |
| `tipsv2-so400m14-dpt` | 448 | 14 | MLP | hub |
| `tipsv2-g14-dpt` | 448 | 14 | SwiGLU | hub |

The backbone size (`b/l/so400m/g`), `out_indices`, `neck_hidden_sizes` and
`reassemble_factors` are all read from each repo's config; you never pass them by
hand when loading.

## Basic Usage: Depth + Segmentation

<img src="../assets/data/coco_bear.jpg" alt="A bear on grass" width="380">

```python
import keras
from zeromodels.models.tipsv2_dpt import (
    Tipsv2DptDensePredict,
    Tipsv2DptImageProcessor,
)

model = Tipsv2DptDensePredict.from_weights("zeromodels/tipsv2-b14-dpt")
proc = Tipsv2DptImageProcessor(image_resolution=model.image_size)

pixel_values = proc("assets/data/coco_bear.jpg")["pixel_values"]
out = model(pixel_values)

depth = keras.ops.convert_to_numpy(out["predicted_depth"])  # (1, 256, 256)
seg = keras.ops.convert_to_numpy(out["segmentation_logits"])  # (1, 256, 256, 150)
print("depth", depth.shape, f"[{depth.min():.3f}, {depth.max():.3f}] m")
print("seg  ", seg.shape)
```

```
depth (1, 256, 256) [0.746, 2.730] m
seg   (1, 256, 256, 150)
```

Outputs are at the DPT feature resolution (256x256); upsample to the original image
size for visualization.

## Single-Task Classes

Both single-task classes load from the same repo and return one output each:

```python
from zeromodels.models.tipsv2_dpt import (
    Tipsv2DptDepthEstimation,
    Tipsv2DptSemanticSegment,
)

depth_model = Tipsv2DptDepthEstimation.from_weights("zeromodels/tipsv2-b14-dpt")
seg_model = Tipsv2DptSemanticSegment.from_weights("zeromodels/tipsv2-b14-dpt")

depth = depth_model(pixel_values)["predicted_depth"]
seg = seg_model(pixel_values)["segmentation_logits"]
```

## Data Format

TIPSv2-DPT is **`channels_last` only** - the reassemble stage reshapes tokens into a
`channels_last` grid, so `channels_first` is not supported. Leave
`keras.config.image_data_format()` at its default (`"channels_last"`), which is what
the image processor and model both expect.

## Loading Upstream Weights

Every class also accepts the `hf:` prefix to convert an upstream
`google/tipsv2-*-dpt` checkpoint on the fly:

```python
from zeromodels.models.tipsv2_dpt import Tipsv2DptDensePredict

model = Tipsv2DptDensePredict.from_weights("hf:google/tipsv2-b14-dpt")
```

No shape arguments are needed. The architecture is read from the repo's
`config.json` and mapped onto the constructor. Loading `hf:google/tipsv2-b14-dpt`
and the `zeromodels/tipsv2-b14-dpt` Hub variant produces identical outputs, since
they are the same checkpoint by two routes.
