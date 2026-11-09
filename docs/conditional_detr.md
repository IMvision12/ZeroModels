# Conditional DETR

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Conditional DETR keeps DETR's recipe (a ResNet backbone, a transformer encoder over the
stride-32 feature map, learned object queries, set prediction) but changes how the
decoder's cross-attention finds objects. DETR's queries have to learn content and
location in one embedding, which is why it needs around 500 training epochs.

Conditional DETR splits them. Each query predicts a 2D reference point, and its
cross-attention query becomes the concatenation of a content part and a *spatial* part:
the sine embedding of that point, scaled by a transformation computed from the decoder
content. Keys are split the same way, so every head attends to a band around the object
(an extremity, or the inside of the box). Boxes are then predicted as offsets from the
reference point. The result trains 6.7x to 10x faster than DETR for the same accuracy.

**Paper**: [Conditional DETR for Fast Training Convergence](https://arxiv.org/abs/2108.06152)

## API

### ConditionalDETRDetect

```python
ConditionalDETRDetect(
    backbone_variant="ResNet50",
    hidden_dim=256,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout_rate=0.1,
    num_queries=300,
    num_classes=91,
    image_size=None,
    input_tensor=None,
    name="ConditionalDETRDetect",
)
```

The detection model: backbone, transformer, and the class and box heads. **This is the
class for object detection.**

**Parameters**

- **backbone_variant** (`str`, *optional*, defaults to `"ResNet50"`): CNN backbone, `"ResNet50"` or `"ResNet101"`.
- **hidden_dim** (`int`, *optional*, defaults to `256`): transformer width, the `d_model` of the HF config.
- **num_heads** (`int`, *optional*, defaults to `8`): attention heads.
- **num_encoder_layers** (`int`, *optional*, defaults to `6`): encoder depth.
- **num_decoder_layers** (`int`, *optional*, defaults to `6`): decoder depth.
- **dim_feedforward** (`int`, *optional*, defaults to `2048`): FFN inner dimension.
- **dropout_rate** (`float`, *optional*, defaults to `0.1`): dropout, active only during training.
- **num_queries** (`int`, *optional*, defaults to `300`): object queries, the ceiling on detections per image.
- **num_classes** (`int`, *optional*, defaults to `91`): COCO's 91 classes. Scores are per-class sigmoids, so there is no "no object" class.
- **image_size** (`int` or `tuple`, *optional*, defaults to `None`): `None` builds a dynamic input that takes the processor's aspect-preserving resize. An int or `(H, W)` fixes the input size.
- **input_tensor** (*optional*): pre-existing input tensor to build on.
- **name** (`str`, *optional*, defaults to `"ConditionalDETRDetect"`): model name.

**Call** `model(pixel_values, training=False)`. **Returns** a `dict`:

- **logits** (`(B, num_queries, num_classes)`): per-query class logits (apply a sigmoid for scores).
- **pred_boxes** (`(B, num_queries, 4)`): normalized `(cx, cy, w, h)` in `[0, 1]`.
- **last_hidden_state** (`(B, num_queries, hidden_dim)`): decoder output.

### ConditionalDetrModel

```python
ConditionalDetrModel(
    backbone_variant="ResNet50",
    hidden_dim=256,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout_rate=0.1,
    num_queries=300,
    image_size=None,
    input_tensor=None,
    name="ConditionalDetrModel",
)
```

The backbone and transformer without detection heads. **Parameters** match
[ConditionalDETRDetect](#conditionaldetrdetect), minus **num_classes**. **Returns** a
`dict` with **last_hidden_state** `(B, num_queries, hidden_dim)` and
**reference_points** `(B, num_queries, 2)`, the per-query `(x, y)` reference point.
It loads from the same repos as the detector.

## Preprocessing

### ConditionalDETRImageProcessor

```python
ConditionalDETRImageProcessor(
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

The same preprocessing as [DETR](detr.md#detrimageprocessor): an aspect-preserving
resize (short side 800, long side at most 1333, the reference default), rescale to
`[0, 1]`, ImageNet normalization. A batch of different sizes is zero-padded to the
largest image. Single images are exact; padded batches are a close approximation,
since the model takes no pixel mask.

**post_process_object_detection**

```python
processor.post_process_object_detection(
    outputs, threshold=0.5, target_sizes=None, label_names=None, top_k=100
)
```

Takes the `top_k` highest sigmoid scores over every (query, class) pair, as the
reference does, converts their boxes to pixel-space `(x0, y0, x1, y1)` scaled to
`target_sizes`, and keeps those above `threshold`. **Returns** one `dict` per image with
**scores**, **labels**, **label_names** and **boxes**, sorted by score.

## Model Variants

| Variant id                   | Backbone  | Params | HF original                            |
|------------------------------|-----------|-------:|----------------------------------------|
| `conditional-detr-resnet-50` | ResNet-50 | ~43.5 M | `microsoft/conditional-detr-resnet-50` |

## Basic Usage: Object Detection

```python
from PIL import Image
from zeromodels.models.conditional_detr import (
    ConditionalDETRDetect,
    ConditionalDETRImageProcessor,
)

repo = "zeromodels/conditional-detr-resnet-50"
model = ConditionalDETRDetect.from_weights(repo)
processor = ConditionalDETRImageProcessor.from_weights(repo)

image = Image.open("assets/data/coco_cats.jpg").convert("RGB")
inputs = processor(image)  # (1, 800, 1066, 3)

output = model(inputs["pixel_values"], training=False)
# output["logits"]:     (1, 300, 91)
# output["pred_boxes"]: (1, 300, 4)

results = processor.post_process_object_detection(
    output, threshold=0.5, target_sizes=[(image.height, image.width)]
)[0]
for score, name, box in zip(
    results["scores"], results["label_names"], results["boxes"]
):
    print(f"{name:8s} {float(score):.3f}  {[round(float(v), 1) for v in box]}")
```

```
DETECTIONS_PLACEHOLDER
```

## Data Format

Models read `keras.config.image_data_format()` when they are constructed; the processor
takes a `data_format` argument (defaulting to the same global setting). Set the format
once before building the model and both sides agree. Detections are identical under
either layout.

## Loading from the original checkpoints

Any Hugging Face repo whose `model_type` is `"conditional_detr"` converts on the fly with
the `hf:` prefix, including fine-tunes:

```python
model = ConditionalDETRDetect.from_weights("hf:microsoft/conditional-detr-resnet-50")
model = ConditionalDETRDetect.from_weights("hf:<user>/conditional-detr-finetune")
processor = ConditionalDETRImageProcessor.from_weights(
    "hf:microsoft/conditional-detr-resnet-50"
)
```

The architecture is read from the repo's `config.json`. DC5 checkpoints (a dilated last
backbone stage) are not supported.
