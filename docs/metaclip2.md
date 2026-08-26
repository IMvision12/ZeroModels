# MetaCLIP 2

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

MetaCLIP 2 is a CLIP-architecture dual encoder trained on a worldwide,
multi-language data distribution rather than an English-filtered one. The
architecture matches OpenAI CLIP, so the layer code is shared; what differs is
the data recipe, a **901629-token XLM-RoBERTa vocabulary** in place of CLIP's
49408-token English BPE, and a set of `worldwide` variants.

Three `mt5` variants use a SentencePiece tokenizer instead, exposed separately
as `MetaClip2Mt5Tokenizer`.

**Paper**: [MetaCLIP 2: A Worldwide Scaling Recipe](https://arxiv.org/abs/2507.22062)

## API

### MetaClip2Model

```python
MetaClip2Model(
    embed_dim=512,
    image_size=224,
    vision_num_layers=12,
    vision_hidden_dim=768,
    vision_patch_size=32,
    vision_num_heads=None,
    max_seq_len=77,
    vocab_size=901629,
    text_hidden_dim=512,
    text_num_heads=8,
    text_num_layers=12,
    vision_mlp_ratio=4.0,
    text_mlp_ratio=4.0,
    hidden_act="gelu",
    eos_token_id=2,
    input_tensor=None,
    name="MetaClip2Model",
)
```

MetaCLIP 2 (multilingual / worldwide) contrastive vision-language model.

**Parameters**

- **embed_dim** (`int`, *optional*, defaults to `512`): shared joint embedding dimension.
- **image_size** (`int`, *optional*, defaults to `224`): input image spec. An `int` builds an `N x N x 3` input, a 2-tuple `(H, W)` assumes 3 channels, and a 3-tuple follows the active `keras.config.image_data_format()`.
- **vision_num_layers** (`int`, *optional*, defaults to `12`): ViT encoder depth.
- **vision_hidden_dim** (`int`, *optional*, defaults to `768`): ViT hidden dimension.
- **vision_patch_size** (`int`, *optional*, defaults to `32`): ViT patch size.
- **vision_num_heads** (`NoneType`, *optional*, defaults to `None`): ViT attention heads.
- **max_seq_len** (`int`, *optional*, defaults to `77`): text input length.
- **vocab_size** (`int`, *optional*, defaults to `901629`): tokenizer vocabulary size.
- **text_hidden_dim** (`int`, *optional*, defaults to `512`): text encoder hidden dimension.
- **text_num_heads** (`int`, *optional*, defaults to `8`): text encoder attention heads.
- **text_num_layers** (`int`, *optional*, defaults to `12`): text encoder depth.
- **vision_mlp_ratio** (`float`, *optional*, defaults to `4.0`): MLP expansion ratio in the vision blocks.
- **text_mlp_ratio** (`float`, *optional*, defaults to `4.0`): MLP expansion ratio in the text blocks.
- **hidden_act** (`str`, *optional*, defaults to `'gelu'`): MLP activation.
- **eos_token_id** (`int`, *optional*, defaults to `2`): end-of-sequence token id.
- **input_tensor** (`NoneType`, *optional*, defaults to `None`): pre-existing input tensors to build on.
- **name** (`str`, *optional*, defaults to `'MetaClip2Model'`): model name.

### MetaClip2ZeroShotClassify

```python
MetaClip2ZeroShotClassify(
    embed_dim=512,
    image_size=224,
    vision_num_layers=12,
    vision_hidden_dim=768,
    vision_patch_size=32,
    vision_num_heads=None,
    max_seq_len=77,
    vocab_size=901629,
    text_hidden_dim=512,
    text_num_heads=8,
    text_num_layers=12,
    vision_mlp_ratio=4.0,
    text_mlp_ratio=4.0,
    hidden_act="gelu",
    eos_token_id=2,
    input_tensor=None,
    name="MetaClip2ZeroShotClassify",
)
```

MetaCLIP 2 + contrastive similarity head for zero-shot classification / retrieval.

**Parameters**

- **embed_dim** (`int`, *optional*, defaults to `512`): shared joint embedding dimension.
- **image_size** (`int`, *optional*, defaults to `224`): input image spec. An `int` builds an `N x N x 3` input, a 2-tuple `(H, W)` assumes 3 channels, and a 3-tuple follows the active `keras.config.image_data_format()`.
- **vision_num_layers** (`int`, *optional*, defaults to `12`): ViT encoder depth.
- **vision_hidden_dim** (`int`, *optional*, defaults to `768`): ViT hidden dimension.
- **vision_patch_size** (`int`, *optional*, defaults to `32`): ViT patch size.
- **vision_num_heads** (`NoneType`, *optional*, defaults to `None`): ViT attention heads.
- **max_seq_len** (`int`, *optional*, defaults to `77`): text input length.
- **vocab_size** (`int`, *optional*, defaults to `901629`): tokenizer vocabulary size.
- **text_hidden_dim** (`int`, *optional*, defaults to `512`): text encoder hidden dimension.
- **text_num_heads** (`int`, *optional*, defaults to `8`): text encoder attention heads.
- **text_num_layers** (`int`, *optional*, defaults to `12`): text encoder depth.
- **vision_mlp_ratio** (`float`, *optional*, defaults to `4.0`): MLP expansion ratio in the vision blocks.
- **text_mlp_ratio** (`float`, *optional*, defaults to `4.0`): MLP expansion ratio in the text blocks.
- **hidden_act** (`str`, *optional*, defaults to `'gelu'`): MLP activation.
- **eos_token_id** (`int`, *optional*, defaults to `2`): end-of-sequence token id.
- **input_tensor** (`NoneType`, *optional*, defaults to `None`): pre-existing input tensors to build on.
- **name** (`str`, *optional*, defaults to `'MetaClip2ZeroShotClassify'`): model name.

### MetaClip2ImageClassify

```python
MetaClip2ImageClassify(
    num_classes=1000,
    image_size=224,
    vision_num_layers=12,
    vision_hidden_dim=768,
    vision_patch_size=16,
    vision_num_heads=None,
    vision_mlp_ratio=4.0,
    hidden_act="gelu",
    input_tensor=None,
    name="MetaClip2ImageClassify",
)
```

MetaCLIP 2 vision encoder + linear image-classification head.

**Parameters**

- **num_classes** (`int`, *optional*, defaults to `1000`): number of output classes.
- **image_size** (`int`, *optional*, defaults to `224`): input image spec. An `int` builds an `N x N x 3` input, a 2-tuple `(H, W)` assumes 3 channels, and a 3-tuple follows the active `keras.config.image_data_format()`.
- **vision_num_layers** (`int`, *optional*, defaults to `12`): ViT encoder depth.
- **vision_hidden_dim** (`int`, *optional*, defaults to `768`): ViT hidden dimension.
- **vision_patch_size** (`int`, *optional*, defaults to `16`): ViT patch size.
- **vision_num_heads** (`NoneType`, *optional*, defaults to `None`): ViT attention heads.
- **vision_mlp_ratio** (`float`, *optional*, defaults to `4.0`): MLP expansion ratio in the vision blocks.
- **hidden_act** (`str`, *optional*, defaults to `'gelu'`): MLP activation.
- **input_tensor** (`NoneType`, *optional*, defaults to `None`): pre-existing input tensors to build on.
- **name** (`str`, *optional*, defaults to `'MetaClip2ImageClassify'`): model name.

### MetaClip2VisionModel

```python
MetaClip2VisionModel(
    image_size=224,
    vision_num_layers=12,
    vision_hidden_dim=768,
    vision_patch_size=32,
    vision_num_heads=None,
    vision_mlp_ratio=4.0,
    hidden_act="gelu",
    input_tensor=None,
    name="MetaClip2VisionModel",
)
```

MetaCLIP 2 vision tower as a standalone model: no text encoder, no projection.

**Parameters**

- **image_size** (`int`, *optional*, defaults to `224`): input image spec. An `int` builds an `N x N x 3` input, a 2-tuple `(H, W)` assumes 3 channels, and a 3-tuple follows the active `keras.config.image_data_format()`.
- **vision_num_layers** (`int`, *optional*, defaults to `12`): ViT encoder depth.
- **vision_hidden_dim** (`int`, *optional*, defaults to `768`): ViT hidden dimension.
- **vision_patch_size** (`int`, *optional*, defaults to `32`): ViT patch size.
- **vision_num_heads** (`NoneType`, *optional*, defaults to `None`): ViT attention heads.
- **vision_mlp_ratio** (`float`, *optional*, defaults to `4.0`): MLP expansion ratio in the vision blocks.
- **hidden_act** (`str`, *optional*, defaults to `'gelu'`): MLP activation.
- **input_tensor** (`NoneType`, *optional*, defaults to `None`): pre-existing input tensors to build on.
- **name** (`str`, *optional*, defaults to `'MetaClip2VisionModel'`): model name.

### MetaClip2TextModel

```python
MetaClip2TextModel(
    max_seq_len=77,
    vocab_size=901629,
    text_hidden_dim=512,
    text_num_heads=8,
    text_num_layers=12,
    text_mlp_ratio=4.0,
    hidden_act="gelu",
    eos_token_id=2,
    input_tensor=None,
    name="MetaClip2TextModel",
)
```

MetaCLIP 2 text tower as a standalone model: no vision encoder, no projection.

**Parameters**

- **max_seq_len** (`int`, *optional*, defaults to `77`): text input length.
- **vocab_size** (`int`, *optional*, defaults to `901629`): tokenizer vocabulary size.
- **text_hidden_dim** (`int`, *optional*, defaults to `512`): text encoder hidden dimension.
- **text_num_heads** (`int`, *optional*, defaults to `8`): text encoder attention heads.
- **text_num_layers** (`int`, *optional*, defaults to `12`): text encoder depth.
- **text_mlp_ratio** (`float`, *optional*, defaults to `4.0`): MLP expansion ratio in the text blocks.
- **hidden_act** (`str`, *optional*, defaults to `'gelu'`): MLP activation.
- **eos_token_id** (`int`, *optional*, defaults to `2`): end-of-sequence token id.
- **input_tensor** (`NoneType`, *optional*, defaults to `None`): pre-existing input tensors to build on.
- **name** (`str`, *optional*, defaults to `'MetaClip2TextModel'`): model name.

> **`MetaClip2Model` gives you embeddings, `MetaClip2ZeroShotClassify` gives you
> logits.**

## Preprocessing

### MetaClip2ImageProcessor

```python
MetaClip2ImageProcessor(
    image_resolution=224,
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
    do_center_crop=True,
    do_normalize=True,
    do_resize=True,
    square_resize=True,
    data_format=None,
)
```

Image processor for MetaCLIP 2: direct square bicubic resize.

Resize geometry is not uniform across variants: most publish a square
`size`, but `metaclip2_worldwide_huge_quickgelu` publishes
`{"shortest_edge": 224}`. Pass `square_resize=False` for that one.

**Parameters**

- **image_resolution** (`int`, *optional*, defaults to `224`): target square resolution.
- **mean** (`tuple`, *optional*, defaults to `(0.48145466, 0.4578275, 0.40821073)`): per-channel normalization mean.
- **std** (`tuple`, *optional*, defaults to `(0.26862954, 0.26130258, 0.27577711)`): per-channel normalization std.
- **do_center_crop** (`bool`, *optional*, defaults to `True`): center-crop after the resize.
- **do_normalize** (`bool`, *optional*, defaults to `True`): apply mean/std normalization.
- **do_resize** (`bool`, *optional*, defaults to `True`): resize before cropping.
- **square_resize** (`bool`, *optional*, defaults to `True`): stretch straight onto the square rather than resizing on the shortest edge.
- **data_format** (`NoneType`, *optional*, defaults to `None`): `"channels_last"` or `"channels_first"`. Defaults to `keras.config.image_data_format()`.

### MetaClip2Tokenizer

```python
MetaClip2Tokenizer(
    variant=None,
    tokenizer_file=None,
    max_seq_len=77,
    bos_token_id=0,
    eos_token_id=2,
    pad_token_id=1,
    unk_token_id=3,
)
```

XLM-RoBERTa tokenizer for MetaCLIP 2 worldwide variants (``tokenizers`` backend).

**Parameters**

- **variant** (`NoneType`, *optional*, defaults to `None`): variant key, used to fetch the matching tokenizer files.
- **tokenizer_file** (`NoneType`, *optional*, defaults to `None`): explicit `tokenizer.json` path, overriding `variant`.
- **max_seq_len** (`int`, *optional*, defaults to `77`): text input length.
- **bos_token_id** (`int`, *optional*, defaults to `0`): begin-of-sequence token id.
- **eos_token_id** (`int`, *optional*, defaults to `2`): end-of-sequence token id.
- **pad_token_id** (`int`, *optional*, defaults to `1`): padding token id.
- **unk_token_id** (`int`, *optional*, defaults to `3`): unknown token id.

### MetaClip2Mt5Tokenizer

```python
MetaClip2Mt5Tokenizer(
    sentencepiece_model_file=None,
    max_seq_len=77,
    eos_token_id=1,
    pad_token_id=1,
    unk_token_id=2,
)
```

SigLIP-style SentencePiece tokenizer for the MetaCLIP 2 mT5 variants.

Only the three `metaclip2_mt5_*` variants use this. It is SentencePiece
based and has no `tokenizer.json`, so it is fetched as `spiece.model`.

**Parameters**

- **sentencepiece_model_file** (`NoneType`, *optional*, defaults to `None`): explicit `spiece.model` path.
- **max_seq_len** (`int`, *optional*, defaults to `77`): text input length.
- **eos_token_id** (`int`, *optional*, defaults to `1`): end-of-sequence token id.
- **pad_token_id** (`int`, *optional*, defaults to `1`): padding token id.
- **unk_token_id** (`int`, *optional*, defaults to `2`): unknown token id.

### MetaClip2Processor

```python
MetaClip2Processor(
    image_resolution=224,
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
    do_center_crop=True,
    do_normalize=True,
    do_resize=True,
    data_format=None,
    variant=None,
    tokenizer_file=None,
    max_seq_len=77,
    tokenizer=None,
    image_processor=None,
)
```

Combined image + text processor for MetaCLIP 2.

**Parameters**

- **image_resolution** (`int`, *optional*, defaults to `224`): target square resolution.
- **mean** (`tuple`, *optional*, defaults to `(0.48145466, 0.4578275, 0.40821073)`): per-channel normalization mean.
- **std** (`tuple`, *optional*, defaults to `(0.26862954, 0.26130258, 0.27577711)`): per-channel normalization std.
- **do_center_crop** (`bool`, *optional*, defaults to `True`): center-crop after the resize.
- **do_normalize** (`bool`, *optional*, defaults to `True`): apply mean/std normalization.
- **do_resize** (`bool`, *optional*, defaults to `True`): resize before cropping.
- **data_format** (`NoneType`, *optional*, defaults to `None`): `"channels_last"` or `"channels_first"`. Defaults to `keras.config.image_data_format()`.
- **variant** (`NoneType`, *optional*, defaults to `None`): variant key, used to fetch the matching tokenizer files.
- **tokenizer_file** (`NoneType`, *optional*, defaults to `None`): explicit `tokenizer.json` path, overriding `variant`.
- **max_seq_len** (`int`, *optional*, defaults to `77`): text input length.
- **tokenizer** (`NoneType`, *optional*, defaults to `None`): a pre-built tokenizer, instead of building one.
- **image_processor** (`NoneType`, *optional*, defaults to `None`): a pre-built image processor.

## Model Variants

Load any of these with `from_weights("zeromodels/<variant id>")`.

| Variant id | Image size | Patch | Weights |
|---|---:|---:|---|
| `metaclip2_worldwide_s16_224` | 224 | 16 | hub |
| `metaclip2_worldwide_s16_384` | 384 | 16 | hub |
| `metaclip2_worldwide_m16_224` | 224 | 16 | hub |
| `metaclip2_worldwide_m16_384` | 384 | 16 | hub |
| `metaclip2_worldwide_b16_224` | 224 | 16 | hub |
| `metaclip2_worldwide_b16_384` | 384 | 16 | hub |
| `metaclip2_worldwide_b32_224` | 224 | 32 | hub |
| `metaclip2_worldwide_b32_384` | 384 | 32 | hub |
| `metaclip2_worldwide_l14_224` | 224 | 14 | on the fly from `facebook/metaclip-2-worldwide-l14` |
| `metaclip2_worldwide_huge_quickgelu` | 224 | 14 | on the fly from `facebook/metaclip-2-worldwide-huge-quickgelu` |
| `metaclip2_worldwide_huge_378` | 378 | 14 | on the fly from `facebook/metaclip-2-worldwide-huge-378` |
| `metaclip2_worldwide_giant_224` | 224 | 14 | on the fly from `facebook/metaclip-2-worldwide-giant` |
| `metaclip2_worldwide_giant_378` | 378 | 14 | on the fly from `facebook/metaclip-2-worldwide-giant-378` |
| `metaclip2_mt5_worldwide_s16_224` | 224 | 16 | hub |
| `metaclip2_mt5_worldwide_m16_224` | 224 | 16 | hub |
| `metaclip2_mt5_worldwide_b32_224` | 224 | 32 | hub |

### Variants without Hub Keras weights

The five variants marked *on the fly* have **no preconverted zeromodels Hub
repo**. Calling `from_weights` on them silently falls back to downloading the
upstream Hugging Face checkpoint and converting it in-process. That still works,
but it differs from the other eleven in ways worth knowing:

- It downloads from the upstream Hub repo, not a zeromodels `.weights.h5`, so it
  depends on that repo staying available and on any gating it may have.
- Conversion runs every time the weights are not already cached, which is slower
  than fetching a prebuilt `.weights.h5`.
- These are the largest variants (l14, huge, giant), so the download is big.

If you want the fast path, prefer one of the eleven Hub Keras variants.

## Basic Usage: Zero-Shot Classification

<img src="../assets/data/coco_teddy_bears.jpg" alt="Two teddy bears" width="380">

```python
import keras
from zeromodels.models.metaclip2 import (
    MetaClip2Processor,
    MetaClip2ZeroShotClassify,
)

processor = MetaClip2Processor.from_weights("zeromodels/metaclip2_worldwide_s16_224")
model = MetaClip2ZeroShotClassify.from_weights("zeromodels/metaclip2_worldwide_s16_224")

labels = [
    "a photo of teddy bears",
    "a photo of a bowl of food",
    "a photo of a skier",
    "a photo of a living room",
]
inputs = processor(text=labels, image_paths="assets/data/coco_teddy_bears.jpg")
output = model(
    {
        "images": inputs["images"],
        "token_ids": inputs["token_ids"],
        "padding_mask": inputs["padding_mask"],
    }
)

# (1, 4): one image, four class prompts.
probs = keras.ops.convert_to_numpy(
    keras.ops.softmax(output["image_logits"], axis=-1)
).squeeze()
for label, p in zip(labels, probs):
    print(f"{p:.6f}  {label}")
```

```
1.000000  a photo of teddy bears
0.000000  a photo of a bowl of food
0.000000  a photo of a skier
0.000000  a photo of a living room
```

Unlike SigLIP, `MetaClip2Processor` already returns `token_ids` and
`padding_mask` under the model's own key names, so no remapping is needed.

## Batch Processing Multiple Images

Pass a list of paths. `image_logits` becomes `(num_images, num_texts)`, one row per
image, and the same label set is scored against each:

<p>
  <img src="../assets/data/coco_teddy_bears.jpg" alt="Two teddy bears" width="300">
  <img src="../assets/data/coco_food_bowl.jpg" alt="A bowl of rice, broccoli and chili" width="300">
</p>

```python
image_paths = ["assets/data/coco_teddy_bears.jpg", "assets/data/coco_food_bowl.jpg"]
labels = [
    "a photo of teddy bears",
    "a photo of a bowl of food",
    "a photo of a skier",
    "a photo of a living room",
]

inputs = processor(text=labels, image_paths=image_paths)
output = model(
    {
        "images": inputs["images"],
        "token_ids": inputs["token_ids"],
        "padding_mask": inputs["padding_mask"],
    }
)

probs = keras.ops.convert_to_numpy(
    keras.ops.softmax(output["image_logits"], axis=-1)
)  # (2, 4)
for path, row in zip(image_paths, probs):
    print(f"\n{path}")
    for label, p in zip(labels, row):
        print(f"  {p:.6f}  {label}")
```

```
assets/data/coco_teddy_bears.jpg
  1.000000  a photo of teddy bears
  0.000000  a photo of a bowl of food
  0.000000  a photo of a skier
  0.000000  a photo of a living room

assets/data/coco_food_bowl.jpg
  0.000014  a photo of teddy bears
  0.999821  a photo of a bowl of food
  0.000113  a photo of a skier
  0.000052  a photo of a living room
```

## Multilingual Prompts

MetaCLIP 2 is trained worldwide, so class prompts do not have to be English.
Scoring the same concept in four languages against one distractor:

```python
labels = [
    "a photo of teddy bears",  # English
    "une photo d'ours en peluche",  # French
    "una foto de ositos de peluche",  # Spanish
    "ein Foto von Teddybären",  # German
    "a photo of a truck",  # English distractor
]
inputs = processor(text=labels, image_paths="assets/data/coco_teddy_bears.jpg")
output = model(
    {
        "images": inputs["images"],
        "token_ids": inputs["token_ids"],
        "padding_mask": inputs["padding_mask"],
    }
)
probs = keras.ops.convert_to_numpy(
    keras.ops.softmax(output["image_logits"], axis=-1)
).squeeze()
```

```
0.855608  a photo of teddy bears
0.038126  une photo d'ours en peluche
0.103371  una foto de ositos de peluche
0.002894  ein Foto von Teddybären
0.000000  a photo of a truck
```

Every language ranks the correct concept far above the distractor, which is the
capability being shown. English still takes most of the mass here, so treat this as
"all four languages work", not "all four are equally strong".

Accents matter. Writing the German prompt as `Teddybaren` instead of
`Teddybären` drops it from `0.002894` to `0.000078`, roughly 37x worse, because the
tokenizer sees a different word.

## Data Format

**Both the models and the processors support `channels_last` and `channels_first`.**

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

## Loading Fine-tuned and Community Weights

You are not limited to the variants above. Any Hugging Face repo whose
`model_type` is `"metaclip_2"` loads directly with the `hf:` prefix, including
community fine-tunes.

```python
from zeromodels.models.metaclip2 import MetaClip2ZeroShotClassify

model = MetaClip2ZeroShotClassify.from_weights("hf:facebook/metaclip-2-worldwide-s16")
model = MetaClip2ZeroShotClassify.from_weights("hf:<user>/my-finetune")
```

No shape arguments are needed. The architecture is read from the repo's `config.json`
and mapped onto the constructor.

All 5 model classes accept `hf:`, as do `MetaClip2ImageProcessor`,
`MetaClip2Tokenizer`, `MetaClip2Mt5Tokenizer`, and `MetaClip2Processor`, so you can
pull the matching preprocessing from the same repo:

```python
processor = MetaClip2Processor.from_weights("hf:facebook/metaclip-2-worldwide-s16")
```

Loading `hf:facebook/metaclip-2-worldwide-s16` and the `metaclip2_worldwide_s16_224`
Hub variant produces identical outputs, since they are the same checkpoint by two
routes.
