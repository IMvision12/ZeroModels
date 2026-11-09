# Deformable DETR

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Deformable DETR fixes DETR's two weak spots, slow convergence and poor small-object
accuracy, by replacing dense attention with **multi-scale deformable attention**. Every
query attends to only a handful of points (4 per head per level), sampled at learned
offsets around a reference point and mixed with learned weights. That makes attention
cost linear in the number of pixels, so the encoder can afford a feature pyramid: the
ResNet stride-8, 16 and 32 stages plus an extra stride-64 level, all projected to 256
channels.

The decoder refines 300 object queries, each attending around its own reference point.
Two optional upgrades ship as separate checkpoints:

- **Iterative box refinement**: every decoder layer refines the boxes of the previous
  one with its own box head.
- **Two-stage**: the encoder output itself proposes regions, and the top 300 initialize
  the decoder queries.

The model reaches better accuracy than DETR (especially on small objects) with about
10x fewer training epochs.

**Paper**: [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159)

## API

### DeformableDETRDetect

```python
DeformableDETRDetect(
    backbone_variant="ResNet50",
    d_model=256,
    encoder_layers=6,
    decoder_layers=6,
    encoder_ffn_dim=1024,
    decoder_ffn_dim=1024,
    encoder_attention_heads=8,
    decoder_attention_heads=8,
    num_feature_levels=4,
    encoder_n_points=4,
    decoder_n_points=4,
    num_queries=300,
    with_box_refine=False,
    two_stage=False,
    dropout_rate=0.1,
    num_classes=91,
    image_size=None,
    input_tensor=None,
    name=None,
)
```

The detection model. **This is the class for object detection**, for every variant
below: the repo's `zm_config.json` sets `num_feature_levels`, `with_box_refine` and
`two_stage`.

**Parameters**

- **backbone_variant** (`str`, *optional*, defaults to `"ResNet50"`): `"ResNet50"` or `"ResNet101"`.
- **d_model** (`int`, *optional*, defaults to `256`): transformer width.
- **encoder_layers** / **decoder_layers** (`int`, *optional*, default `6`): depths.
- **encoder_ffn_dim** / **decoder_ffn_dim** (`int`, *optional*, default `1024`): FFN widths.
- **encoder_attention_heads** / **decoder_attention_heads** (`int`, *optional*, default `8`): heads.
- **num_feature_levels** (`int`, *optional*, defaults to `4`): `4` for the multi-scale pyramid, `1` for the stride-32 map only.
- **encoder_n_points** / **decoder_n_points** (`int`, *optional*, default `4`): sampling points per head per level.
- **num_queries** (`int`, *optional*, defaults to `300`): object queries (two-stage: proposals kept).
- **with_box_refine** (`bool`, *optional*, defaults to `False`): iterative box refinement.
- **two_stage** (`bool`, *optional*, defaults to `False`): encoder proposals initialize the queries. Requires `with_box_refine`.
- **dropout_rate** (`float`, *optional*, defaults to `0.1`): dropout, active only during training.
- **num_classes** (`int`, *optional*, defaults to `91`): sigmoid-scored classes, no "no object" class.
- **image_size** (`int` or `tuple`, *optional*, defaults to `None`): `None` builds a dynamic input; any resolution works.
- **input_tensor** (*optional*): pre-existing input tensor to build on.
- **name** (`str`, *optional*): model name.

**Call** `model(pixel_values, training=False)`. **Returns** a `dict` with **logits**
`(B, num_queries, num_classes)`, **pred_boxes** `(B, num_queries, 4)` (normalized
`(cx, cy, w, h)`) and **last_hidden_state** `(B, num_queries, d_model)`.

The ResNet backbone and the per-level projections are an ordinary functional graph;
the deformable transformer runs on the concrete feature-map sizes of each call, which
is what lets one model accept any input resolution.

### DeformableDetrModel

The same model without the detector's heads. **Parameters** match
[DeformableDETRDetect](#deformabledetrdetect). **Returns** **last_hidden_state**
`(B, num_queries, d_model)` and **intermediate_hidden_states**
`(B, decoder_layers, num_queries, d_model)`. It loads from the same repos as the
detector.

## Preprocessing

### DeformableDETRImageProcessor

The same preprocessing as [DETR](detr.md#detrimageprocessor): an aspect-preserving
resize (short side 800, long side at most 1333), rescale to `[0, 1]` and ImageNet
normalization. Its **post_process_object_detection** takes the top-100 sigmoid scores
over every (query, class) pair, as the reference does (see
[Conditional DETR](conditional_detr.md#conditionaldetrimageprocessor)).

## Model Variants

| Variant id                                  | Levels | Box refine | Two-stage | Params  | HF original                                           |
|---------------------------------------------|:------:|:----------:|:---------:|--------:|-------------------------------------------------------|
| `deformable-detr`                           | 4      |            |           | ~40.2 M | `SenseTime/deformable-detr`                           |
| `deformable-detr-single-scale`              | 1      |            |           | ~34.2 M | `SenseTime/deformable-detr-single-scale`              |
| `deformable-detr-with-box-refine`           | 4      | yes        |           | ~41.0 M | `SenseTime/deformable-detr-with-box-refine`           |
| `deformable-detr-with-box-refine-two-stage` | 4      | yes        | yes       | ~41.3 M | `SenseTime/deformable-detr-with-box-refine-two-stage` |

## Basic Usage: Object Detection

```python
from PIL import Image
from zeromodels.models.deformable_detr import (
    DeformableDETRDetect,
    DeformableDETRImageProcessor,
)

repo = "zeromodels/deformable-detr-with-box-refine-two-stage"
model = DeformableDETRDetect.from_weights(repo)
processor = DeformableDETRImageProcessor.from_weights(repo)

image = Image.open("assets/data/coco_cats.jpg").convert("RGB")
output = model(processor(image)["pixel_values"], training=False)
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

## Loading from the original checkpoints

Any Hugging Face repo whose `model_type` is `"deformable_detr"` converts on the fly with
the `hf:` prefix:

```python
model = DeformableDETRDetect.from_weights("hf:SenseTime/deformable-detr")
model = DeformableDETRDetect.from_weights("hf:<user>/deformable-detr-finetune")
```

The architecture (levels, box refinement, two-stage) is read from the repo's
`config.json`. DC5 checkpoints (`SenseTime/deformable-detr-single-scale-dc5`, a dilated
last backbone stage) are not supported.
