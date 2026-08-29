# Table Transformer

<div class="kf-note kf-note--weights">
<b>Weights:</b> Table Transformer loads Microsoft's PubTables-1M checkpoints on the fly with
the <code>hf:</code> prefix, e.g.
<code>from_weights("hf:microsoft/table-transformer-detection")</code>. The architecture is read
from the repo's <code>config.json</code>, so no shape arguments are needed.
</div>

Table Transformer (TATR) applies the DETR detection recipe to tables. It is the same
end-to-end set-prediction model: a ResNet backbone produces a feature map, a transformer
encoder-decoder attends over it with a fixed set of learned object queries, and each query
emits one class and one box, with no anchors and no non-maximum suppression. It ships in two
tasks, both the same architecture with a different query count and label set:

- **Table detection** finds tables in a page image (classes: table, table rotated).
- **Table-structure recognition** decomposes a cropped table into its cells (classes: table,
  column, row, column header, projected row header, spanning cell).

Two details separate it from DETR: the encoder and decoder layers are **pre-normalized** (the
LayerNorm sits before each attention / feed-forward sub-layer) with an extra final encoder
LayerNorm, and the backbone is a **ResNet-18** (so the 1x1 input projection reduces 512
channels rather than 2048).

**Paper**: [PubTables-1M: Towards comprehensive table extraction from unstructured
documents](https://arxiv.org/abs/2110.00061)

## API

### TableTransformerDetect

```python
TableTransformerDetect(
    hidden_dim=256,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout_rate=0.1,
    num_queries=15,
    num_classes=3,
    image_size=800,
    input_tensor=None,
    name="TableTransformerDetect",
)
```

The detection model: backbone, transformer, and the class and box heads. **This one class
serves both tasks**; only `num_queries` and `num_classes` differ between the checkpoints, and
`from_weights` fills them in from the repo config.

**Parameters**

- **hidden_dim** (`int`, *optional*, defaults to `256`): transformer width, the `d_model` of the HF config.
- **num_heads** (`int`, *optional*, defaults to `8`): attention heads.
- **num_encoder_layers** (`int`, *optional*, defaults to `6`): encoder depth.
- **num_decoder_layers** (`int`, *optional*, defaults to `6`): decoder depth.
- **dim_feedforward** (`int`, *optional*, defaults to `2048`): FFN inner dimension.
- **dropout_rate** (`float`, *optional*, defaults to `0.1`): dropout, active only during training.
- **num_queries** (`int`, *optional*, defaults to `15`): learned object queries, the hard ceiling on detections per image. Detection uses 15, structure recognition 125.
- **num_classes** (`int`, *optional*, defaults to `3`): object classes plus the "no object" class (detection: 2 + 1; structure: 6 + 1).
- **image_size** (`int`, *optional*, defaults to `800`): input resolution the model is built for.
- **input_tensor** (`dict`, *optional*): pre-existing input tensor to build on.
- **name** (`str`, *optional*, defaults to `"TableTransformerDetect"`): model name.

**Call** `model(pixel_values, training=False)`. **Returns** a `dict`:

- **logits** (`(B, num_queries, num_classes)`): per-query class logits.
- **pred_boxes** (`(B, num_queries, 4)`): normalized `(cx, cy, w, h)` in `[0, 1]`.

Raw output is one prediction per query, most of them the "no object" class. Run it through
`post_process_object_detection` to get scored, pixel-space boxes.

### TableTransformerModel

```python
TableTransformerModel(
    hidden_dim=256,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout_rate=0.1,
    num_queries=15,
    image_size=800,
    input_tensor=None,
    name="TableTransformerModel",
)
```

The backbone and transformer without detection heads, ending at the decoder hidden states.
Use it when you want Table Transformer features to attach your own head to.

**Parameters** are identical to [TableTransformerDetect](#tabletransformerdetect), minus
**num_classes**, and **name** defaults to `"TableTransformerModel"`.

**Returns** the decoder's last hidden state, `(B, num_queries, hidden_dim)`.

## Preprocessing

### TableTransformerImageProcessor

```python
TableTransformerImageProcessor(
    size=None,
    resample="bilinear",
    do_rescale=True,
    rescale_factor=1 / 255,
    do_normalize=True,
    image_mean=None,
    image_std=None,
    return_tensor=True,
    data_format=None,
)
```

Resizes to a fixed square, rescales to `[0, 1]`, and normalizes with ImageNet statistics
(the same preprocessing the Table Transformer checkpoints ship with).

**Parameters**

- **size** (`dict`, *optional*, defaults to `{"height": 800, "width": 800}`): target size.
- **resample** (`str`, *optional*, defaults to `"bilinear"`): resize interpolation.
- **do_rescale** (`bool`, *optional*, defaults to `True`): scale pixels to `[0, 1]`.
- **rescale_factor** (`float`, *optional*, defaults to `1/255`): the rescaling factor.
- **do_normalize** (`bool`, *optional*, defaults to `True`): apply mean/std normalization.
- **image_mean** (`tuple`, *optional*, defaults to `(0.485, 0.456, 0.406)`): per-channel mean.
- **image_std** (`tuple`, *optional*, defaults to `(0.229, 0.224, 0.225)`): per-channel std.
- **return_tensor** (`bool`, *optional*, defaults to `True`): return backend tensors rather than numpy.
- **data_format** (`str`, *optional*): `"channels_last"` or `"channels_first"`. Defaults to `keras.config.image_data_format()`.

**Call** `processor(image)` with a path, a PIL image, an array, or a **list** of any mix of
those. **Returns** a `dict` with **pixel_values** (`(B, H, W, 3)`).

**post_process_object_detection**

```python
processor.post_process_object_detection(
    outputs, threshold=0.7, target_sizes=None, label_names=None
)
```

Softmaxes the logits, drops the "no object" class, keeps whatever clears `threshold`, and
converts boxes to pixel-space `(x0, y0, x1, y1)` scaled to `target_sizes`.

- **outputs**: the `dict` returned by the model.
- **threshold** (`float`, *optional*, defaults to `0.7`): minimum class probability.
- **target_sizes** (`list` of `(height, width)`, *optional*): original image sizes, one per batch element.
- **label_names** (`list` of `str`, *optional*): class names. Defaults to the six structure-recognition labels; pass `TABLE_DETECTION_LABELS` for the detection model.

**Returns** a list with one `dict` per image, each holding **scores**, **labels**,
**label_names**, and **boxes** (`(x0, y0, x1, y1)` in pixels).

The module also exports the two label tuples for convenience:

```python
from zeromodels.models.table_transformer.table_transformer_image_processor import (
    TABLE_DETECTION_LABELS,   # ("table", "table rotated")
    TABLE_STRUCTURE_LABELS,   # ("table", "table column", "table row", ...)
)
```

## Model Variants

All variants are ResNet-18 backboned. Load any of them with
`TableTransformerDetect.from_weights("hf:<repo>")`.

| Task                   | HF repo                                                        | Queries | Classes |
|------------------------|---------------------------------------------------------------|--------:|--------:|
| Table detection        | `microsoft/table-transformer-detection`                       | 15      | 2 + 1   |
| Structure recognition  | `microsoft/table-transformer-structure-recognition`           | 125     | 6 + 1   |
| Structure v1.1 (all)   | `microsoft/table-transformer-structure-recognition-v1.1-all`  | 125     | 6 + 1   |
| Structure v1.1 (fin)   | `microsoft/table-transformer-structure-recognition-v1.1-fin`  | 125     | 6 + 1   |
| Structure v1.1 (pub)   | `microsoft/table-transformer-structure-recognition-v1.1-pub`  | 125     | 6 + 1   |

The v1.1 checkpoints store the backbone under the native Hugging Face ResNet naming rather
than the timm-style layout of the two originals; the loader handles both automatically.

## Basic Usage: Table Detection

```python
from PIL import Image
from zeromodels.models.table_transformer import (
    TableTransformerDetect,
    TableTransformerImageProcessor,
)
from zeromodels.models.table_transformer.table_transformer_image_processor import (
    TABLE_DETECTION_LABELS,
)

model = TableTransformerDetect.from_weights("hf:microsoft/table-transformer-detection")
processor = TableTransformerImageProcessor()

image = Image.open("page.jpg").convert("RGB")
inputs = processor(image)

output = model(inputs["pixel_values"], training=False)
# output["logits"]:     (1, 15, 3)
# output["pred_boxes"]: (1, 15, 4)

results = processor.post_process_object_detection(
    output,
    threshold=0.9,
    target_sizes=[(image.height, image.width)],
    label_names=TABLE_DETECTION_LABELS,
)[0]

for score, name, box in sorted(
    zip(results["scores"], results["label_names"], results["boxes"]),
    key=lambda d: -float(d[0]),
):
    print(f"{name:14s} {float(score):.3f}  {[round(float(v)) for v in box]}")
```

Each kept detection is a table region in pixel coordinates. Crop the page to those boxes to
get the table images for the next stage.

## Table-Structure Recognition

Structure recognition takes a **cropped table** image and returns its rows, columns, and
header / spanning cells. Same call, a different checkpoint, and the default label set already
matches:

```python
from PIL import Image
from zeromodels.models.table_transformer import (
    TableTransformerDetect,
    TableTransformerImageProcessor,
)

model = TableTransformerDetect.from_weights(
    "hf:microsoft/table-transformer-structure-recognition-v1.1-all"
)
processor = TableTransformerImageProcessor()

table = Image.open("cropped_table.jpg").convert("RGB")
output = model(processor(table)["pixel_values"], training=False)
# output["logits"]:     (1, 125, 7)
# output["pred_boxes"]: (1, 125, 4)

cells = processor.post_process_object_detection(
    output, threshold=0.6, target_sizes=[(table.height, table.width)]
)[0]
# cells["label_names"] are drawn from the six structure classes:
# table, table column, table row, table column header,
# table projected row header, table spanning cell
```

Intersecting the predicted rows and columns reconstructs the cell grid; the header and
spanning-cell predictions then tag which cells are headers or span multiple rows/columns.

## Data Format

**Both the model and the processor support `channels_last` and `channels_first`.** They pick
the format differently:

| | How it picks the format |
|---|---|
| Processor | A `data_format` kwarg, per instance. `None` (the default) resolves to `keras.config.image_data_format()`. |
| Model | Reads `keras.config.image_data_format()` when it is **constructed**. There is no `data_format` argument. |

Set the global format before constructing the model, and both sides agree:

```python
import keras

keras.config.set_image_data_format("channels_first")
model = TableTransformerDetect.from_weights("hf:microsoft/table-transformer-detection")
# now expects (B, 3, H, W)
```

Detections are identical under either layout; only the tensor shape changes. The
post-processor emits `xyxy` pixel boxes and class indices, which have no channel axis, so it
takes no `data_format` kwarg.

## Loading Fine-tuned and Community Weights

Any Hugging Face repo whose `model_type` is `"table-transformer"` loads with the `hf:`
prefix, including the official `microsoft/table-transformer-*` checkpoints and arbitrary
fine-tunes:

```python
from zeromodels.models.table_transformer import TableTransformerDetect

model = TableTransformerDetect.from_weights("hf:<user>/my-table-transformer-finetune")

# Architecture only, randomly initialized
model = TableTransformerDetect.from_weights(
    "hf:microsoft/table-transformer-detection", load_weights=False
)
```

No shape arguments are needed. The architecture is read from the repo's `config.json` and
mapped onto the constructor: `d_model`, `encoder_attention_heads`, `encoder_layers`,
`decoder_layers`, `encoder_ffn_dim`, `num_queries`, and the label count. `TableTransformerModel`
accepts `hf:` the same way, warm-starting the backbone and transformer from the detector's
weights.
