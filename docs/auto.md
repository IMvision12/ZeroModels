# Auto Classes

The `AutoZM*` loaders pick the right class for you. Point one at a repo and it reads the
repo's `model_type`, looks it up, and loads the matching model, config, or preprocessor,
so you never have to remember whether a checkpoint is a `BertModel` or a `DETRDetect`:

```python
from zeromodels import AutoZModel, AutoZMTokenizer

model = AutoZModel.from_weights("zeromodels/bert-base-uncased")  # -> BertModel
tok = AutoZMTokenizer.from_weights("zeromodels/bert-base-uncased")
```

This is the `transformers` `AutoModel` idea, ZeroModels flavored. The classes are named
`AutoZM*` (`AutoZModel` for the backbone) on purpose: the `ZM` prefix keeps them from
colliding with `transformers`' own `AutoModel` / `AutoTokenizer` / ... when both libraries
are imported in the same program.

## How it resolves

Every hosted checkpoint records a `model_type`, and that string is the whole key:

- a **ZeroModels repo** records it in `zm_config.json` (the typed config's `model_type`,
  distinct per family, e.g. `deberta_v3`);
- a **raw `transformers` / `timm` repo** records it in `config.json` (the upstream
  `model_type`).

`from_weights` reads that string and dispatches to the matching class's own
`from_weights`, so the same call works for both sources:

```python
from zeromodels.auto import AutoZMDetect

AutoZMDetect.from_weights("zeromodels/detr-resnet-50")  # ZeroModels repo -> DETRDetect
AutoZMDetect.from_weights("hf:facebook/detr-resnet-50")  # raw HF repo     -> DETRDetect
```

Resolution is by `model_type` only, never by reading a stored class name, so a community
fine-tune of the same architecture loads exactly like the official weights.

!!! note
    An `AutoZM*` needs a **repo** to read the `model_type` from, either a `zeromodels/…`
    id or an `hf:org/repo` id. A bare release variant (`"resnet50_a1_in1k"`) has no repo to
    inspect, so pass it to the concrete class instead
    (`ResNetImageClassify.from_weights("resnet50_a1_in1k")`).

## Task classes

Like `transformers`' `AutoModelForImageClassification` / `AutoModelForObjectDetection` /
…, there is one Auto per task. `AutoZModel` loads the bare backbone; the task classes load
the model **with its head**. Each is named after the ZeroModels task suffix:

| Auto class | Loads | Example |
|---|---|---|
| `AutoZModel` | the backbone (`*Model`) | `bert -> BertModel` |
| `AutoZMImageClassify` | image classifiers | `resnet -> ResNetImageClassify` |
| `AutoZMDetect` | object detectors | `detr -> DETRDetect` |
| `AutoZMSemanticSegment` | semantic segmentation | `segformer -> SegFormerSemanticSegment` |
| `AutoZMUniversalSegment` | universal / panoptic segmentation | `oneformer -> OneFormerUniversalSegment` |
| `AutoZMDepthEstimation` | depth estimation | `depth_anything -> DepthAnythingV2DepthEstimation` |
| `AutoZMTextGenerate` | causal LMs | `llama -> Llama2TextGenerate` |
| `AutoZMConditionalGenerate` | multimodal / seq2seq generators | `qwen2_vl -> Qwen2VLConditionalGenerate` |
| `AutoZMMaskedLM` | masked-LM heads | `bert -> BertMaskedLM` |
| `AutoZMSequenceClassify` / `AutoZMTokenClassify` / `AutoZMQnA` / `AutoZMMultipleChoice` | text task heads | `bert -> BertSequenceClassify` |
| `AutoZMZeroShotClassify` | contrastive zero-shot | `clip -> CLIPZeroShotClassify` |

The backbone loader and the task loaders read the same repo; you pick the Auto for the
task you want:

```python
from zeromodels.auto import AutoZMImageClassify, AutoZMSemanticSegment

clf = AutoZMImageClassify.from_weights("zeromodels/resnet-50")
seg = AutoZMSemanticSegment.from_weights("zeromodels/segformer_b0_ade_512")
```

The full set (26 task classes, e.g. `AutoZMInstanceSegment`, `AutoZMPanopticSegment`,
`AutoZMAudioClassify`, `AutoZMTextModel`, `AutoZMVisionModel`, …) is exported from
`zeromodels.auto`.

## Config and preprocessors

The same detection drives the config and preprocessor loaders, so one repo id loads the
whole set:

```python
from zeromodels import (
    AutoZModel,
    AutoZMConfig,
    AutoZMTokenizer,
    AutoZMProcessor,
    AutoZMImageProcessor,
)

repo = "zeromodels/clip_vit_base_16"
model = AutoZModel.from_weights(repo)  # -> CLIPModel
cfg = AutoZMConfig.from_weights(repo)  # -> CLIPConfig (built from zm_config.json)
proc = AutoZMProcessor.from_weights(repo)  # -> CLIPProcessor
```

| Auto class | Returns |
|---|---|
| `AutoZMConfig` | the typed `BaseConfig` for the repo (built from its `zm_config.json`) |
| `AutoZMTokenizer` | the family's tokenizer |
| `AutoZMProcessor` | the family's processor (image + text, audio + text, …) |
| `AutoZMImageProcessor` | the family's image processor |

## Inspecting and overriding the mapping

Each Auto exposes its live `model_type -> class` table, which is handy for discovery:

```python
AutoZModel.mapping()["bert"]  # <class 'BertModel'>
sorted(AutoZMDetect.mapping())  # every detector's model_type
```

The tables themselves are the explicit, hand-maintained mappings in
`zeromodels/auto/auto_mapping_names.py` (the `transformers` `MODEL_MAPPING_NAMES` pattern).
To point a `model_type` at a different class for the current session, use `register`:

```python
from zeromodels.auto import AutoZMDetect

AutoZMDetect.register("my_detector", MyDetectClass)
```

## Ambiguous and versioned checkpoints

A ZeroModels repo always resolves cleanly, because its `zm_config.json` records a
version-specific `model_type` (`deberta_v2` vs `deberta_v3`, `siglip` vs `siglip2`). A raw
`hf:` repo can be ambiguous when several ZeroModels classes share one upstream
`model_type`, for example HF `deberta-v2`, which both DeBERTa v2 and v3 use. In that case
the Auto refuses to guess and names the candidates:

```python
AutoZModel.from_weights("hf:microsoft/deberta-v3-base")
# ValueError: model_type 'deberta-v2' is ambiguous (loads to any of
# ['DebertaV2Model', 'DebertaV3Model']); resolve it by loading the concrete class directly.
```

A handful of families genuinely share one `model_type` (Llama v1/v2 are both `llama`;
Depth Anything v1/v2 are both `depth_anything`). The table maps these to the **newer**
class by default (`Llama2Model`, `DepthAnythingV2Model`); load the older one through its
own class or `register` an override.
