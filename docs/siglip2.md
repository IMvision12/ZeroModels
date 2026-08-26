# SigLIP 2

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>kf_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

SigLIP 2 keeps SigLIP's sigmoid loss and adds captioning-based pretraining,
self-distillation and masked prediction, which improves dense features and
localization. The tokenizer is the big practical change: a **256k multilingual
Gemma vocabulary** replaces SigLIP's 32k English one, so prompts in many
languages work without a separate multilingual checkpoint.

The architecture classes are thin subclasses of the SigLIP ones, so the layer
code is shared and only the vocabulary, variants and weights differ.

**Paper**: [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786)

## API

### SigLIP2Model

```python
SigLIP2Model(name="SigLIP2Model")
```

SigLIP 2 dual encoder (no contrastive head).

A thin subclass of `SigLIPModel`: it forwards `*args, **kwargs` and only
changes the default `name`.

**Parameters** are inherited from `SigLIP2Model`.

**Parameters**

- **name** (`str`, *optional*, defaults to `'SigLIP2Model'`): model name.

### SigLIP2ZeroShotClassify

```python
SigLIP2ZeroShotClassify(
    image_size=224,
    patch_size=16,
    vision_hidden_dim=768,
    vision_num_layers=12,
    vision_num_heads=12,
    vision_mlp_dim=3072,
    vocab_size=256000,
    embed_dim=768,
    text_hidden_dim=768,
    text_num_layers=12,
    text_num_heads=12,
    text_mlp_dim=3072,
    max_seq_len=64,
    input_tensor=None,
    name="SigLIP2ZeroShotClassify",
)
```

SigLIP 2 + sigmoid-similarity head for zero-shot classification.

**Parameters**

- **image_size** (`int`, *optional*, defaults to `224`): input image spec. An `int` builds an `N x N x 3` input, a 2-tuple `(H, W)` assumes 3 channels, and a 3-tuple follows the active `keras.config.image_data_format()`.
- **patch_size** (`int`, *optional*, defaults to `16`): ViT patch size.
- **vision_hidden_dim** (`int`, *optional*, defaults to `768`): ViT hidden dimension.
- **vision_num_layers** (`int`, *optional*, defaults to `12`): ViT encoder depth.
- **vision_num_heads** (`int`, *optional*, defaults to `12`): ViT attention heads.
- **vision_mlp_dim** (`int`, *optional*, defaults to `3072`): MLP inner width in the vision blocks.
- **vocab_size** (`int`, *optional*, defaults to `256000`): tokenizer vocabulary size.
- **embed_dim** (`int`, *optional*, defaults to `768`): shared joint embedding dimension.
- **text_hidden_dim** (`int`, *optional*, defaults to `768`): text encoder hidden dimension.
- **text_num_layers** (`int`, *optional*, defaults to `12`): text encoder depth.
- **text_num_heads** (`int`, *optional*, defaults to `12`): text encoder attention heads.
- **text_mlp_dim** (`int`, *optional*, defaults to `3072`): MLP inner width in the text blocks.
- **max_seq_len** (`int`, *optional*, defaults to `64`): text input length.
- **input_tensor** (`NoneType`, *optional*, defaults to `None`): pre-existing input tensors to build on.
- **name** (`str`, *optional*, defaults to `'SigLIP2ZeroShotClassify'`): model name.

### SigLIP2ImageClassify

```python
SigLIP2ImageClassify(
    num_classes=1000,
    image_size=224,
    patch_size=16,
    vision_hidden_dim=768,
    vision_num_layers=12,
    vision_num_heads=12,
    vision_mlp_dim=3072,
    input_tensor=None,
    name="SigLIPImageClassify",
)
```

SigLIP 2 vision tower + linear image-classification head.

**Parameters** are inherited from `SigLIPImageClassify`.

**Parameters**

- **num_classes** (`int`, *optional*, defaults to `1000`): number of output classes.
- **image_size** (`int`, *optional*, defaults to `224`): input image spec. An `int` builds an `N x N x 3` input, a 2-tuple `(H, W)` assumes 3 channels, and a 3-tuple follows the active `keras.config.image_data_format()`.
- **patch_size** (`int`, *optional*, defaults to `16`): ViT patch size.
- **vision_hidden_dim** (`int`, *optional*, defaults to `768`): ViT hidden dimension.
- **vision_num_layers** (`int`, *optional*, defaults to `12`): ViT encoder depth.
- **vision_num_heads** (`int`, *optional*, defaults to `12`): ViT attention heads.
- **vision_mlp_dim** (`int`, *optional*, defaults to `3072`): MLP inner width in the vision blocks.
- **input_tensor** (`NoneType`, *optional*, defaults to `None`): pre-existing input tensors to build on.
- **name** (`str`, *optional*, defaults to `'SigLIPImageClassify'`): model name.

### SigLIP2VisionModel

```python
SigLIP2VisionModel(name="SigLIP2VisionModel")
```

SigLIP 2 vision tower as a standalone model.

**Parameters** are inherited from `SigLIP2VisionModel`.

**Parameters**

- **name** (`str`, *optional*, defaults to `'SigLIP2VisionModel'`): model name.

### SigLIP2TextModel

```python
SigLIP2TextModel(name="SigLIP2TextModel")
```

SigLIP 2 text tower as a standalone model.

**Parameters** are inherited from `SigLIP2TextModel`.

**Parameters**

- **name** (`str`, *optional*, defaults to `'SigLIP2TextModel'`): model name.

> **`SigLIP2Model` gives you embeddings, `SigLIP2ZeroShotClassify` gives you
> logits.**

## Preprocessing

### SigLIP2ImageProcessor

```python
SigLIP2ImageProcessor(
    image_resolution=224,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    do_center_crop=True,
    do_normalize=True,
    do_resize=True,
    data_format=None,
)
```

Image processor for SigLIP 2 models.

**Parameters**

- **image_resolution** (`int`, *optional*, defaults to `224`): target square resolution.
- **mean** (`tuple`, *optional*, defaults to `(0.5, 0.5, 0.5)`): per-channel normalization mean.
- **std** (`tuple`, *optional*, defaults to `(0.5, 0.5, 0.5)`): per-channel normalization std.
- **do_center_crop** (`bool`, *optional*, defaults to `True`): center-crop after the resize.
- **do_normalize** (`bool`, *optional*, defaults to `True`): apply mean/std normalization.
- **do_resize** (`bool`, *optional*, defaults to `True`): resize before cropping.
- **data_format** (`NoneType`, *optional*, defaults to `None`): `"channels_last"` or `"channels_first"`. Defaults to `keras.config.image_data_format()`.

### SigLIP2Tokenizer

```python
SigLIP2Tokenizer(
    variant=None,
    tokenizer_file=None,
    max_seq_len=64,
    pad_token="<pad>",
    bos_token="<bos>",
    eos_token="<eos>",
    unk_token="<unk>",
)
```

SigLIP2 (Gemma) SentencePiece tokenizer (``tokenizers`` Rust backend).

**Parameters**

- **variant** (`NoneType`, *optional*, defaults to `None`): variant key, used to fetch the matching tokenizer files.
- **tokenizer_file** (`NoneType`, *optional*, defaults to `None`): explicit `tokenizer.json` path, overriding `variant`.
- **max_seq_len** (`int`, *optional*, defaults to `64`): text input length.
- **pad_token** (`str`, *optional*, defaults to `'<pad>'`): padding token string.
- **bos_token** (`str`, *optional*, defaults to `'<bos>'`): begin-of-sequence token string.
- **eos_token** (`str`, *optional*, defaults to `'<eos>'`): end-of-sequence token string.
- **unk_token** (`str`, *optional*, defaults to `'<unk>'`): unknown-token string.

### SigLIP2Processor

```python
SigLIP2Processor(
    image_resolution=224,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    do_center_crop=True,
    do_normalize=True,
    do_resize=True,
    variant=None,
    tokenizer_file=None,
    max_seq_len=64,
    pad_token="<pad>",
    bos_token="<bos>",
    eos_token="<eos>",
    unk_token="<unk>",
    tokenizer=None,
    image_processor=None,
)
```

Combined processor for SigLIP 2 models: image + Gemma text.

**Parameters**

- **image_resolution** (`int`, *optional*, defaults to `224`): target square resolution.
- **mean** (`tuple`, *optional*, defaults to `(0.5, 0.5, 0.5)`): per-channel normalization mean.
- **std** (`tuple`, *optional*, defaults to `(0.5, 0.5, 0.5)`): per-channel normalization std.
- **do_center_crop** (`bool`, *optional*, defaults to `True`): center-crop after the resize.
- **do_normalize** (`bool`, *optional*, defaults to `True`): apply mean/std normalization.
- **do_resize** (`bool`, *optional*, defaults to `True`): resize before cropping.
- **variant** (`NoneType`, *optional*, defaults to `None`): variant key, used to fetch the matching tokenizer files.
- **tokenizer_file** (`NoneType`, *optional*, defaults to `None`): explicit `tokenizer.json` path, overriding `variant`.
- **max_seq_len** (`int`, *optional*, defaults to `64`): text input length.
- **pad_token** (`str`, *optional*, defaults to `'<pad>'`): padding token string.
- **bos_token** (`str`, *optional*, defaults to `'<bos>'`): begin-of-sequence token string.
- **eos_token** (`str`, *optional*, defaults to `'<eos>'`): end-of-sequence token string.
- **unk_token** (`str`, *optional*, defaults to `'<unk>'`): unknown-token string.
- **tokenizer** (`NoneType`, *optional*, defaults to `None`): a pre-built tokenizer, instead of building one.
- **image_processor** (`NoneType`, *optional*, defaults to `None`): a pre-built image processor.

## Model Variants

Load any of these with `from_weights("zeromodels/<variant id>")`.

| Variant id | Image size | Patch | Weights |
|---|---:|---:|---|
| `siglip2_base_p16_224` | 224 | 16 | hub |
| `siglip2_base_p16_256` | 256 | 16 | hub |
| `siglip2_base_p16_384` | 384 | 16 | hub |
| `siglip2_base_p16_512` | 512 | 16 | hub |
| `siglip2_base_p32_256` | 256 | 32 | hub |
| `siglip2_large_p16_256` | 256 | 16 | hub |
| `siglip2_large_p16_384` | 384 | 16 | hub |
| `siglip2_large_p16_512` | 512 | 16 | hub |
| `siglip2_so400m_p14_224` | 224 | 14 | hub |
| `siglip2_so400m_p14_384` | 384 | 14 | hub |
| `siglip2_so400m_p16_256` | 256 | 16 | hub |
| `siglip2_so400m_p16_384` | 384 | 16 | hub |
| `siglip2_so400m_p16_512` | 512 | 16 | hub |

## Basic Usage: Zero-Shot Classification

<img src="../assets/data/coco_living_room.jpg" alt="A living room interior" width="380">

```python
import keras
from zeromodels.models.siglip2 import SigLIP2Processor, SigLIP2ZeroShotClassify

processor = SigLIP2Processor.from_weights("zeromodels/siglip2_base_p16_224")
model = SigLIP2ZeroShotClassify.from_weights("zeromodels/siglip2_base_p16_224")

labels = [
    "a photo of a living room",
    "a photo of a surfer riding a wave",
    "a photo of a bear",
    "a photo of green apples",
]
inputs = processor(text=labels, image_paths="assets/data/coco_living_room.jpg")
output = model(
    {
        "images": inputs["images"],
        "token_ids": inputs["input_ids"],
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
0.999972  a photo of a living room
0.000005  a photo of a surfer riding a wave
0.000007  a photo of a bear
0.000016  a photo of green apples
```

As with SigLIP, the processor returns `input_ids` and the model takes
`token_ids`, with no `padding_mask`.

## Batch Processing Multiple Images

Pass a list of paths. `image_logits` becomes `(num_images, num_texts)`, one row per
image, and the same label set is scored against each:

<p>
  <img src="../assets/data/coco_living_room.jpg" alt="A living room interior" width="300">
  <img src="../assets/data/coco_surfer.jpg" alt="A surfer riding a wave" width="300">
</p>

```python
image_paths = ["assets/data/coco_living_room.jpg", "assets/data/coco_surfer.jpg"]
labels = [
    "a photo of a living room",
    "a photo of a surfer riding a wave",
    "a photo of a bear",
    "a photo of green apples",
]

inputs = processor(text=labels, image_paths=image_paths)
output = model(
    {
        "images": inputs["images"],
        "token_ids": inputs["input_ids"],
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
assets/data/coco_living_room.jpg
  0.999972  a photo of a living room
  0.000005  a photo of a surfer riding a wave
  0.000007  a photo of a bear
  0.000016  a photo of green apples

assets/data/coco_surfer.jpg
  0.000000  a photo of a living room
  0.999970  a photo of a surfer riding a wave
  0.000029  a photo of a bear
  0.000000  a photo of green apples
```

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
`model_type` is `"siglip"` loads directly with the `hf:` prefix, including
community fine-tunes.

```python
from zeromodels.models.siglip2 import SigLIP2ZeroShotClassify

model = SigLIP2ZeroShotClassify.from_weights("hf:google/siglip2-base-patch16-224")
model = SigLIP2ZeroShotClassify.from_weights("hf:<user>/my-finetune")
```

No shape arguments are needed. The architecture is read from the repo's `config.json`
and mapped onto the constructor.

All 5 model classes accept `hf:`, as do `SigLIP2ImageProcessor`, `SigLIP2Tokenizer`,
and `SigLIP2Processor`, so you can pull the matching preprocessing from the same
repo:

```python
processor = SigLIP2Processor.from_weights("hf:google/siglip2-base-patch16-224")
```

Loading `hf:google/siglip2-base-patch16-224` and the `zeromodels/siglip2_base_p16_224` Hub
variant produces identical outputs, since they are the same checkpoint by two routes.
