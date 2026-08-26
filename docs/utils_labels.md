# Class labels

Models predict integer ids. These lists turn them back into names.

```python
from zeromodels.utils.labels_util import COCO_80_CLASSES, ADE20K_150_CLASSES

print(len(COCO_80_CLASSES), COCO_80_CLASSES[0])
print(len(ADE20K_150_CLASSES), ADE20K_150_CLASSES[0])
```

```
80 person
150 wall
```

They live in `zeromodels.utils.labels_util` and are imported by path, not re-exported from
`zeromodels.utils`. All are plain tuples, so they are cheap to import and safe to share.

| Constant | Size | Used by |
|---|---|---|
| `COCO_91_CLASSES` | 91 | [DETR](detr.md), the DETR-family detectors, which keep COCO's original sparse ids |
| `COCO_80_CLASSES` | 80 | Detectors trained on the contiguous 80-class remap |
| `COCO_PANOPTIC_133_CLASSES` | 133 | [Mask2Former](mask2former.md), [OneFormer](oneformer.md), [EoMT](eomt.md) panoptic variants |
| `COCO_PANOPTIC_THING_IDS` / `COCO_PANOPTIC_STUFF_IDS` | 80 / 53 | Splitting a panoptic result into countable objects and background stuff |
| `ADE20K_150_CLASSES` | 150 | [SegFormer](segformer.md), [MaskFormer](maskformer.md), the ADE20K semantic variants |
| `CITYSCAPES_19_CLASSES` | 19 | The Cityscapes-trained segmentation variants |
| `PASCAL_VOC_CLASSES` | 21 | [DeepLabV3](deeplabv3.md) and other VOC variants, index 0 being background |

## Usage

Most of the time you pass one straight into the post-processor rather than indexing it
yourself, and read the names back out:

```python
result = processor.post_process_object_detection(
    outputs, threshold=0.9, target_sizes=[(h, w)], label_names=COCO_80_CLASSES
)[0]
print(result["label_names"])
```

The detection post-processor defaults to `COCO_91_CLASSES`, so a model with a 91-class head
needs nothing at all. Segmentation takes the list the same way, and
[`plot_segmentation`](utils_visualization.md#plot_segmentation) accepts one as
`class_names` for its legend.

## The 91 vs 80 trap

The two COCO lists are the single most common source of confidently wrong labels, because
both produce plausible names for the same id. Original COCO ids run to 90 with gaps where
categories were dropped; `COCO_91_CLASSES` preserves those gaps as `"N/A"` entries, 11 of
them, while `COCO_80_CLASSES` closes them up.

```python
print(COCO_91_CLASSES[11:14])
print(COCO_80_CLASSES[11:14])
```

```
('fire hydrant', 'N/A', 'stop sign')
('stop sign', 'parking meter', 'bench')
```

Index 0 is another giveaway: `"N/A"` in the 91-class list (ids start at 1), `"person"` in
the 80-class one. The offset starts at zero and grows with each gap passed, so early classes
look correct and later ones are quietly shifted, which is exactly the failure that survives
a quick eyeball check. Match the list to the head's output width before trusting any of it.

## Panoptic

`COCO_PANOPTIC_133_CLASSES` carries its group in the string, which keeps the legend readable
when things and stuff sit side by side:

```python
print(COCO_PANOPTIC_133_CLASSES[0])
print(COCO_PANOPTIC_133_CLASSES[80])
```

```
things: person
stuff: banner
```

The first 80 entries are things, the remaining 53 are stuff, and the two id tuples give you
that split without parsing strings:

```python
from zeromodels.utils.labels_util import (
    COCO_PANOPTIC_THING_IDS,
    COCO_PANOPTIC_STUFF_IDS,
)

print(COCO_PANOPTIC_THING_IDS[:5])
print(COCO_PANOPTIC_STUFF_IDS[:5])
```

```
(0, 1, 2, 3, 4)
(80, 81, 82, 83, 84)
```

They do not overlap and together cover all 133 ids. Use them to count instances (things
only) or to build a background mask (stuff only) from a panoptic result.

See also [Visualization utils](utils_visualization.md) and [Utilities](utils.md).
