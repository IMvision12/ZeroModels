# TIPSv2

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code> +
<code>tokenizer.json</code> + <code>zm_preprocessor.json</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>. The full model
and both towers load from the same repo.
</div>

TIPSv2 (Text-Image Pre-training with Spatial awareness) is a family of contrastive
vision-language dual encoders from Google DeepMind. The vision tower is a
**DINOv2-style ViT with register tokens**; the text tower is a bidirectional
transformer with fixed sinusoidal positions and a masked-mean pooled output. The
two towers are aligned with a temperature-scaled softmax contrastive objective,
with a focus on **dense patch-to-text alignment** (the property that also makes the
[TIPSv2-DPT](tipsv2_dpt.md) depth / segmentation heads work well).

There is no learned projection on either tower: the pooled vision and text features
share one dimension and are compared directly, so `embed_dim == hidden_dim`.

**Paper**: [TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment](https://huggingface.co/papers/2604.12012)

## API

### Tipsv2Model

```python
Tipsv2Model(
    image_size=448,
    patch_size=14,
    num_register_tokens=1,
    vision_hidden_dim=768,
    vision_num_layers=12,
    vision_num_heads=12,
    vision_mlp_ratio=4.0,
    vision_use_swiglu_ffn=False,
    vision_layerscale_value=1.0,
    vision_layer_norm_eps=1e-06,
    resize_mode="bilinear",
    vocab_size=32000,
    max_seq_len=64,
    embed_dim=768,
    text_hidden_dim=768,
    text_num_layers=12,
    text_num_heads=12,
    text_mlp_dim=3072,
    text_hidden_act="relu",
    text_layer_norm_eps=1e-05,
    text_scale_sqrt_depth=True,
    text_pooling_epsilon=1e-08,
    temperature_init_value=0.005065968260169029,
    input_tensor=None,
    name="Tipsv2Model",
)
```

Full TIPSv2 dual encoder with the contrastive head. Inputs are a dict
`{"images", "token_ids", "padding_mask"}`; outputs are `image_embeddings`,
`text_embeddings` (both L2-normalized), `logits_per_image` and `logits_per_text`.

**Parameters**

- **image_size** (`int`, *optional*, defaults to `448`): input image spec. An `int` builds an `N x N x 3` input, a 2-tuple `(H, W)` assumes 3 channels, and a 3-tuple follows the active `keras.config.image_data_format()`.
- **patch_size** (`int`, *optional*, defaults to `14`): ViT patch size.
- **num_register_tokens** (`int`, *optional*, defaults to `1`): number of register tokens inserted after the position embedding.
- **vision_hidden_dim** (`int`, *optional*, defaults to `768`): ViT hidden dimension.
- **vision_num_layers** (`int`, *optional*, defaults to `12`): ViT encoder depth.
- **vision_num_heads** (`int`, *optional*, defaults to `12`): ViT attention heads.
- **vision_mlp_ratio** (`float`, *optional*, defaults to `4.0`): MLP inner-width multiplier in the vision blocks (a float; `so400m14` uses `3.7361`).
- **vision_use_swiglu_ffn** (`bool`, *optional*, defaults to `False`): use a SwiGLU feed-forward instead of a standard MLP (the `g14` backbone).
- **vision_layerscale_value** (`float`, *optional*, defaults to `1.0`): LayerScale init value.
- **vision_layer_norm_eps** (`float`, *optional*, defaults to `1e-06`): vision LayerNorm epsilon.
- **resize_mode** (`str`, *optional*, defaults to `'bilinear'`): position-embedding interpolation mode when the grid differs from the checkpoint.
- **vocab_size** (`int`, *optional*, defaults to `32000`): tokenizer vocabulary size.
- **max_seq_len** (`int`, *optional*, defaults to `64`): text input length.
- **embed_dim** (`int`, *optional*, defaults to `768`): shared joint embedding dimension (equals the tower hidden dims).
- **text_hidden_dim** (`int`, *optional*, defaults to `768`): text encoder hidden dimension.
- **text_num_layers** (`int`, *optional*, defaults to `12`): text encoder depth.
- **text_num_heads** (`int`, *optional*, defaults to `12`): text encoder attention heads.
- **text_mlp_dim** (`int`, *optional*, defaults to `3072`): MLP inner width in the text blocks.
- **text_hidden_act** (`str`, *optional*, defaults to `'relu'`): text MLP activation.
- **text_layer_norm_eps** (`float`, *optional*, defaults to `1e-05`): text LayerNorm epsilon.
- **text_scale_sqrt_depth** (`bool`, *optional*, defaults to `True`): scale token embeddings by `sqrt(hidden_dim)` before adding positions.
- **text_pooling_epsilon** (`float`, *optional*, defaults to `1e-08`): epsilon in the masked-mean pooling denominator.
- **temperature_init_value** (`float`, *optional*, defaults to `0.00506...`): constant temperature dividing the logits (not a learned weight in the checkpoint).
- **input_tensor** (`NoneType`, *optional*, defaults to `None`): pre-existing input tensors to build on.
- **name** (`str`, *optional*, defaults to `'Tipsv2Model'`): model name.

### Tipsv2VisionModel

```python
Tipsv2VisionModel(
    image_size=448,
    patch_size=14,
    num_register_tokens=1,
    vision_hidden_dim=768,
    vision_num_layers=12,
    vision_num_heads=12,
    vision_mlp_ratio=4.0,
    vision_use_swiglu_ffn=False,
    vision_layerscale_value=1.0,
    vision_layer_norm_eps=1e-06,
    resize_mode="bilinear",
    input_tensor=None,
    name="Tipsv2VisionModel",
)
```

The vision tower alone. Output dict: `last_hidden_state` and `pooler_output` (the
CLS token).

### Tipsv2TextModel

```python
Tipsv2TextModel(
    vocab_size=32000,
    max_seq_len=64,
    text_hidden_dim=768,
    text_num_layers=12,
    text_num_heads=12,
    text_mlp_dim=3072,
    text_hidden_act="relu",
    text_layer_norm_eps=1e-05,
    text_scale_sqrt_depth=True,
    text_pooling_epsilon=1e-08,
    input_tensor=None,
    name="Tipsv2TextModel",
)
```

The text tower alone. Inputs `{"token_ids", "padding_mask"}`; output dict:
`last_hidden_state` and `pooler_output` (masked-mean over real tokens).

> **`Tipsv2Model` gives you both embeddings and logits.** Reach for the towers when
> you only need one modality (e.g. pre-computing an image or text index).

## Preprocessing

### Tipsv2ImageProcessor

```python
Tipsv2ImageProcessor(
    image_resolution=448,
    resample="bilinear",
    do_normalize=False,
    do_resize=True,
    mean=(0.0, 0.0, 0.0),
    std=(1.0, 1.0, 1.0),
    data_format=None,
)
```

Image processor for TIPSv2: resize to `image_resolution` and rescale to `[0, 1]`.
There is **no mean/std normalization** and no center crop.

**Parameters**

- **image_resolution** (`int`, *optional*, defaults to `448`): target square resolution.
- **resample** (`str`, *optional*, defaults to `'bilinear'`): resize interpolation.
- **do_normalize** (`bool`, *optional*, defaults to `False`): apply mean/std normalization (off for TIPSv2).
- **do_resize** (`bool`, *optional*, defaults to `True`): resize the image.
- **mean** (`tuple`, *optional*, defaults to `(0.0, 0.0, 0.0)`): per-channel normalization mean.
- **std** (`tuple`, *optional*, defaults to `(1.0, 1.0, 1.0)`): per-channel normalization std.
- **data_format** (`NoneType`, *optional*, defaults to `None`): `"channels_last"` or `"channels_first"`. Defaults to `keras.config.image_data_format()`.

### Tipsv2Tokenizer

```python
Tipsv2Tokenizer(
    variant=None,
    tokenizer_file=None,
    max_seq_len=64,
    unk_token="<unk>",
    pad_token="<pad>",
)
```

TIPSv2 text tokenizer (`tokenizers` Rust backend, byte-fallback BPE with
lowercase + Metaspace normalization). `call` returns fixed-length `input_ids`
padded with `<pad>` (id 0) plus an `attention_mask`.

**Parameters**

- **variant** (`NoneType`, *optional*, defaults to `None`): variant key, used to fetch the matching tokenizer files.
- **tokenizer_file** (`NoneType`, *optional*, defaults to `None`): explicit `tokenizer.json` path, overriding `variant`.
- **max_seq_len** (`int`, *optional*, defaults to `64`): text input length.
- **unk_token** (`str`, *optional*, defaults to `'<unk>'`): unknown-token string.
- **pad_token** (`str`, *optional*, defaults to `'<pad>'`): padding token string.

### Tipsv2Processor

```python
Tipsv2Processor(
    image_resolution=448,
    resample="bilinear",
    do_normalize=False,
    do_resize=True,
    variant=None,
    tokenizer_file=None,
    max_seq_len=64,
    unk_token="<unk>",
    pad_token="<pad>",
    tokenizer=None,
    image_processor=None,
)
```

Combined image + text processor. `processor(text=..., images=...)` (or
`image_paths=...`) returns `{"images", "token_ids", "padding_mask"}`, which is
exactly the `Tipsv2Model` input dict.

**Parameters**

- **image_resolution** (`int`, *optional*, defaults to `448`): target square resolution.
- **resample** (`str`, *optional*, defaults to `'bilinear'`): resize interpolation.
- **do_normalize** (`bool`, *optional*, defaults to `False`): apply mean/std normalization (off for TIPSv2).
- **do_resize** (`bool`, *optional*, defaults to `True`): resize the image.
- **variant** (`NoneType`, *optional*, defaults to `None`): variant key, used to fetch the matching tokenizer files.
- **tokenizer_file** (`NoneType`, *optional*, defaults to `None`): explicit `tokenizer.json` path, overriding `variant`.
- **max_seq_len** (`int`, *optional*, defaults to `64`): text input length.
- **unk_token** (`str`, *optional*, defaults to `'<unk>'`): unknown-token string.
- **pad_token** (`str`, *optional*, defaults to `'<pad>'`): padding token string.
- **tokenizer** (`NoneType`, *optional*, defaults to `None`): a pre-built tokenizer, instead of building one.
- **image_processor** (`NoneType`, *optional*, defaults to `None`): a pre-built image processor.

## Model Variants

Load any of these with `from_weights("zeromodels/<variant id>")`.

| Variant id | Image size | Patch | FFN | Weights |
|---|---:|---:|---|---|
| `tipsv2-b14` | 448 | 14 | MLP | hub |
| `tipsv2-l14` | 448 | 14 | MLP | hub |
| `tipsv2-so400m14` | 448 | 14 | MLP | hub |
| `tipsv2-g14` | 448 | 14 | SwiGLU | hub |

`g14` uses the DINOv2-giant SwiGLU feed-forward; the constructor picks it up
automatically from the config (`vision_use_swiglu_ffn=True`).

## Basic Usage: Zero-Shot Classification

<img src="../assets/data/coco_bear.jpg" alt="A bear on grass" width="380">

```python
import keras
from zeromodels.models.tipsv2 import Tipsv2Model, Tipsv2Processor

model = Tipsv2Model.from_weights("zeromodels/tipsv2-b14")
processor = Tipsv2Processor.from_weights("zeromodels/tipsv2-b14")

labels = [
    "a photo of a person skiing",
    "a photo of green apples",
    "a photo of a bear",
    "a photo of a living room",
]
inputs = processor(text=labels, image_paths="assets/data/coco_bear.jpg")
output = model(inputs)

# (1, 4): one image, four class prompts.
probs = keras.ops.convert_to_numpy(
    keras.ops.softmax(output["logits_per_image"], axis=-1)
).squeeze()
for label, p in zip(labels, probs):
    print(f"{p:.6f}  {label}")
```

```
0.000000  a photo of a person skiing
0.000000  a photo of green apples
1.000000  a photo of a bear
0.000000  a photo of a living room
```

The processor returns the model's full input dict, so `model(inputs)` works
directly with no key remapping.

## Towers Only

Each tower loads from the same repo and produces embeddings you can index or cache:

```python
from zeromodels.models.tipsv2 import Tipsv2VisionModel, Tipsv2TextModel

vision = Tipsv2VisionModel.from_weights("zeromodels/tipsv2-b14")
text = Tipsv2TextModel.from_weights("zeromodels/tipsv2-b14")

image_feats = vision(inputs["images"])["pooler_output"]
text_feats = text(
    {"token_ids": inputs["token_ids"], "padding_mask": inputs["padding_mask"]}
)["pooler_output"]
```

## Data Format

**The models and the processors support `channels_last` and `channels_first`.**

Processors take a `data_format` kwarg per instance, where `None` resolves to
`keras.config.image_data_format()`. Models have no such argument and read
`keras.config.image_data_format()` when they are **constructed**. To switch the whole
pipeline, set the global format before building the model:

```python
import keras

keras.config.set_image_data_format("channels_first")
```

`set_image_data_format` is global state. Set it once at the top of a script rather
than toggling it between calls, since already-built models keep the layout they were
constructed with.

## Loading Upstream Weights

Every class also accepts the `hf:` prefix to convert an upstream
`google/tipsv2-*` checkpoint on the fly (the original repos ship a SentencePiece
`tokenizer.model`, which `Tipsv2Tokenizer` rebuilds into a fast `tokenizer.json`):

```python
from zeromodels.models.tipsv2 import Tipsv2Model, Tipsv2Processor

model = Tipsv2Model.from_weights("hf:google/tipsv2-b14")
processor = Tipsv2Processor.from_weights("hf:google/tipsv2-b14")
```

No shape arguments are needed. The architecture is read from the repo's
`config.json` and mapped onto the constructor. Loading `hf:google/tipsv2-b14` and
the `zeromodels/tipsv2-b14` Hub variant produces identical outputs, since they are
the same checkpoint by two routes.
