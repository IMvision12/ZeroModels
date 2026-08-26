# Classification Backbones

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>kf_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Every classification architecture in `zeromodels.models.<arch>` exposes **two classes** that share the same architectural parameters and the same set of pretrained variants:

| Class        | What it returns                                                     | Typical use                                    |
|--------------|---------------------------------------------------------------------|------------------------------------------------|
| `XModel`     | The last layer output **before** the classifier head                | Feature extractor / transfer learning          |
| `XImageClassify`  | Class logits - `XModel` plus the original architecture's head       | Drop-in classification                         |

`XImageClassify` composes `XModel` internally and attaches the per-architecture head (CLS-token linear for ViT-family; GAP + Dense for CNNs; LayerNorm + mean-pool + Dense for hierarchical Transformers; etc.). You don't need to know which head pattern your architecture uses - `XImageClassify` already wires the correct one.

## Quick start

```python
from zeromodels.models.resnet import ResNetImageClassify, ResNetModel

# Full classifier - 1000-class logits
classifier = ResNetImageClassify.from_weights("zeromodels/resnet50_a1_in1k")
logits = classifier(images)  # (B, 1000)

# Just the backbone - last-stage feature map, no head
backbone = ResNetModel.from_weights("zeromodels/resnet50_a1_in1k")
feature_map = backbone(images)  # (B, H/32, W/32, 2048)
```

The same pattern works for every classification arch - swap `ResNet` for `CaiT`, `ViT`, `ConvNeXt`, `EfficientNet`, `Swin`, `MobileNetV3`, etc.

## Data Format

Every backbone reads `keras.config.image_data_format()` when it is constructed and accepts
both `channels_last` and `channels_first` input, returning its output in the same layout you
feed it.

`Swin`, `SwinV2`, and `MobileNetV4` run their spatial core in `channels_last`: a
`channels_first` image is transposed to `channels_last` at the model's entry (the "door"),
the window attention and convolutions run in `channels_last`, and the spatial output is
transposed back to `channels_first` to match the input. The result is identical to passing
`channels_last` directly. The other convolutional backbones (`ResNet`, `ConvNeXt`,
`MobileNetV2`/`V3`, and similar) run their convolutions natively in the requested layout, and
the ViT-family backbones work in token space, so they are layout-agnostic.

## `as_backbone=True` - multi-scale features

Pass `as_backbone=True` to `XModel` (not `XImageClassify`) to get a **list of per-stage feature maps** instead of a single tensor. This is what you'd hook an FPN / segmentation neck / detection head onto.

```python
from zeromodels.models.resnet import ResNetModel

backbone = ResNetModel.from_weights("zeromodels/resnet50_a1_in1k", as_backbone=True)
features = backbone(images)
# features is a list of 4 tensors at strides [4, 8, 16, 32]
for i, f in enumerate(features):
    print(f"stage {i}: {f.shape}")
```

The number of stages and their semantics depend on the architecture:

| Family                                              | Stages when `as_backbone=True`                                  |
|-----------------------------------------------------|-----------------------------------------------------------------|
| ResNet / ResNetV2 / Res2Net / ResNeXt / SENet       | 4 (one per residual stage)                                      |
| ConvNeXt / ConvNeXtV2                               | 4                                                               |
| EfficientNet / EfficientNet-Lite / EfficientNetV2   | 5 (at stride-2 boundaries; head conv excluded)                  |
| EfficientFormer                                     | 4                                                               |
| MobileNetV2 / MobileNetV3                           | 5 (head conv excluded)                                          |
| MobileNetV4                                         | 5 (at stride-2 boundaries)                                      |
| MobileViT / MobileViTV2                             | 5 (S0–S4)                                                       |
| DenseNet                                            | 4 (post-transition + final BN/ReLU)                             |
| VGG                                                 | 5 (post-pool stages)                                            |
| Xception                                            | 3 (entry / middle / exit flow)                                  |
| ConvMixer                                           | 1 (no natural multi-scale hierarchy: singleton list)           |
| Swin / SwinV2 / PoolFormer / NextViT / MaxViT       | 4 (native hierarchical pyramid)                                 |
| InceptionV3 / InceptionV4 / InceptionNext           | 4 (at major reduction boundaries)                               |
| InceptionResNetV2                                   | 3 (post each Inception-ResNet group; `conv2d_7b` head excluded) |
| MiT                                                 | 4 (segmentation-style hierarchical features)                    |
| ViT / DeiT / FlexiViT / CaiT                        | `depth + 1` (per-transformer-block + final LN)                  |
| PiT                                                 | 5 (stem + per-stage + final norm)                               |
| MLP-Mixer / ResMLP                                  | per-block (typically ~12; final norm excluded)                  |

**Head conv exclusion.** For architectures that have a 1×1 "head conv" at the end of the backbone-feature function (EfficientNet / MobileNet families, NextViT's trailing BN, InceptionResNetV2's `conv2d_7b`), the head conv is **only** applied when `as_backbone=False`. When you ask for stages, you get the last MBConv/stage output before that head conv, which is usually what you want for downstream tasks.

## `from_weights`: variants and HF checkpoints

Both classes use the same variant registry, so any string you can pass to one works on the other:

```python
ResNetImageClassify.from_weights("zeromodels/resnet50_a1_in1k")
ResNetModel.from_weights("zeromodels/resnet50_a1_in1k")  # same weights, no Dense head
ResNetModel.from_weights(
    "hf:timm/resnet50.a1_in1k"
)  # any timm variant via on-the-fly conversion
```

Official zeromodels checkpoints load by Hub repo id (`kf_config.json` +
`model.weights.h5`). Under the hood `XModel.from_hub_repo` warm-starts from
`XImageClassify`'s weight file and `copy_weights_by_path_suffix` picks the backbone
subset (the classifier `Dense` is dropped).

## Custom heads / transfer learning

Because `XModel` and `XImageClassify` share architecture parameters and layer names, you can chain them in any of three idiomatic ways:

```python
import keras
from keras import layers
from zeromodels.models.convnext import ConvNeXtModel

backbone = ConvNeXtModel.from_weights("zeromodels/convnext_base_fb_in22k_ft_in1k")

# Option 1: Sequential with a custom head
model = keras.Sequential(
    [
        backbone,
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, name="predictions"),
    ]
)

# Option 2: Functional API with intermediate fan-out
inputs = backbone.input
features = backbone.output  # (B, H/32, W/32, 1024)
pooled = layers.GlobalAveragePooling2D()(features)
logits = layers.Dense(num_classes)(pooled)
model = keras.Model(inputs, logits)

# Option 3: Reach into the per-stage outputs for FPN / segmentation
multi = ConvNeXtModel.from_weights(
    "zeromodels/convnext_base_fb_in22k_ft_in1k", as_backbone=True
)
c2, c3, c4, c5 = multi.output  # 4 stages
# ...feed c2..c5 into an FPN
```

## Fine-tuning with a different number of classes

Two equivalent paths, picked based on whether you want safety or one-liner ergonomics.

### Path A: `XModel` + your own head (strict)

Recommended when you want explicit control or are unsure the variant is correct. The class type (`XModel`) guarantees you can't accidentally load a wrong-shape head; everything that loads must be a real backbone weight.

```python
import keras
from keras import layers
from zeromodels.models.resnet import ResNetModel

backbone = ResNetModel.from_weights(
    "zeromodels/resnet50_a1_in1k"
)  # strict load, no skip
classifier = keras.Sequential(
    [
        backbone,
        layers.GlobalAveragePooling2D(),
        layers.Dense(10, activation="softmax"),  # new head, randomly init
    ]
)
```

### Path B: `XImageClassify` + `skip_mismatch=True` (convenient)

One line. Loads matching backbone weights and silently re-initializes any layer whose shape doesn't match (typically just the classifier `Dense`).

```python
from zeromodels.models.resnet import ResNetImageClassify

model = ResNetImageClassify.from_weights(
    "zeromodels/resnet50_a1_in1k",
    num_classes=10,
    skip_mismatch=True,  # head Dense reshaped (1000→10), reset to random init
)
```

**Trade-off:** `skip_mismatch=True` is shape-based, not name-based. If you point it at a wrong variant or a corrupt file, it will *quietly* skip more than the head and leave parts of the backbone randomly initialized. Keras emits `warnings.warn` per skipped layer, but warnings are easy to miss: especially in notebook stderr streams. For sensitive training runs, prefer **Path A**.

**Scope:** `skip_mismatch=True` applies to the Hub Keras weight path (`.h5` / `.json` from
`zeromodels/<variant>`). The `hf:` prefix goes through hand-mapped `transfer_from_*`
functions that ignore the flag.

### Feature extractor (frozen backbone)

Either path supports `trainable = False`:

```python
backbone = ResNetModel.from_weights("zeromodels/resnet50_a1_in1k")
backbone.trainable = False  # whole backbone frozen

model = keras.Sequential(
    [
        backbone,
        layers.GlobalAveragePooling2D(),
        layers.Dense(10, activation="softmax"),
    ]
)
```

Or partial:

```python
backbone = ResNetModel.from_weights("zeromodels/resnet50_a1_in1k")
for layer in backbone.layers[:-30]:  # freeze all but last 30 layers
    layer.trainable = False
```

## Subclass relationships

A few architectures are thin subclasses of a parent that swap the variant registry only:

| Subclass            | Parent      | Differs in                       |
|---------------------|-------------|----------------------------------|
| `DeiT`, `FlexiViT`  | `ViT`       | weights only                     |
| `ConvNeXtV2`        | `ConvNeXt`  | weights + GRN block enabled      |
| `ResNeXt`, `SENet`  | `ResNet`    | block fn + arch defaults         |

Both `XModel` and `XImageClassify` are subclassed so that the subclass's variants resolve correctly. The composition rule still holds: `DeiTImageClassify` composes `DeiTModel`, not `ViTModel`.

## When to use which

- **Inference on ImageNet-style 1000-class tasks** → `XImageClassify.from_weights(...)`.
- **Transfer learning** (fine-tune with a new head on a new dataset) → `XModel.from_weights(...)` plus your own head.
- **Segmentation, detection, FPN, anything multi-scale** → `XModel.from_weights(..., as_backbone=True)`.
- **Saving to disk**: both classes are `keras.saving.register_keras_serializable` decorated, so `model.save("foo.keras")` and `keras.models.load_model("foo.keras")` round-trip cleanly.
