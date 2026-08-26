# SigLIP

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

SigLIP is a vision + text dual encoder trained with a **pairwise sigmoid loss**
rather than CLIP's softmax contrastive loss. Because the loss is computed per
image-text pair instead of over the whole batch, training does not need a global
normalization across negatives, which makes it scale to large batches and gives
stronger zero-shot accuracy at the same model size.

The vision side is a ViT with a learned attention-pooling head (no CLS token) and
the text side is a transformer whose pooled feature is the last token.

**Paper**: [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)

## API

### SigLIPModel

```python
SigLIPModel(
    image_size=224,
    patch_size=16,
    vision_hidden_dim=768,
    vision_num_layers=12,
    vision_num_heads=12,
    vision_mlp_dim=3072,
    vocab_size=32000,
    embed_dim=768,
    text_hidden_dim=768,
    text_num_layers=12,
    text_num_heads=12,
    text_mlp_dim=3072,
    max_seq_len=64,
    input_tensor=None,
    name="SigLIPModel",
)
```

SigLIP dual encoder (no contrastive head).

**Parameters**

- **image_size** (`int`, *optional*, defaults to `224`): input image spec. An `int` builds an `N x N x 3` input, a 2-tuple `(H, W)` assumes 3 channels, and a 3-tuple follows the active `keras.config.image_data_format()`.
- **patch_size** (`int`, *optional*, defaults to `16`): ViT patch size.
- **vision_hidden_dim** (`int`, *optional*, defaults to `768`): ViT hidden dimension.
- **vision_num_layers** (`int`, *optional*, defaults to `12`): ViT encoder depth.
- **vision_num_heads** (`int`, *optional*, defaults to `12`): ViT attention heads.
- **vision_mlp_dim** (`int`, *optional*, defaults to `3072`): MLP inner width in the vision blocks.
- **vocab_size** (`int`, *optional*, defaults to `32000`): tokenizer vocabulary size.
- **embed_dim** (`int`, *optional*, defaults to `768`): shared joint embedding dimension.
- **text_hidden_dim** (`int`, *optional*, defaults to `768`): text encoder hidden dimension.
- **text_num_layers** (`int`, *optional*, defaults to `12`): text encoder depth.
- **text_num_heads** (`int`, *optional*, defaults to `12`): text encoder attention heads.
- **text_mlp_dim** (`int`, *optional*, defaults to `3072`): MLP inner width in the text blocks.
- **max_seq_len** (`int`, *optional*, defaults to `64`): text input length.
- **input_tensor** (`NoneType`, *optional*, defaults to `None`): pre-existing input tensors to build on.
- **name** (`str`, *optional*, defaults to `'SigLIPModel'`): model name.

### SigLIPZeroShotClassify

```python
SigLIPZeroShotClassify(
    image_size=224,
    patch_size=16,
    vision_hidden_dim=768,
    vision_num_layers=12,
    vision_num_heads=12,
    vision_mlp_dim=3072,
    vocab_size=32000,
    embed_dim=768,
    text_hidden_dim=768,
    text_num_layers=12,
    text_num_heads=12,
    text_mlp_dim=3072,
    max_seq_len=64,
    input_tensor=None,
    name="SigLIPZeroShotClassify",
)
```

SigLIP + sigmoid-similarity head for zero-shot classification / retrieval.

**Parameters**

- **image_size** (`int`, *optional*, defaults to `224`): input image spec. An `int` builds an `N x N x 3` input, a 2-tuple `(H, W)` assumes 3 channels, and a 3-tuple follows the active `keras.config.image_data_format()`.
- **patch_size** (`int`, *optional*, defaults to `16`): ViT patch size.
- **vision_hidden_dim** (`int`, *optional*, defaults to `768`): ViT hidden dimension.
- **vision_num_layers** (`int`, *optional*, defaults to `12`): ViT encoder depth.
- **vision_num_heads** (`int`, *optional*, defaults to `12`): ViT attention heads.
- **vision_mlp_dim** (`int`, *optional*, defaults to `3072`): MLP inner width in the vision blocks.
- **vocab_size** (`int`, *optional*, defaults to `32000`): tokenizer vocabulary size.
- **embed_dim** (`int`, *optional*, defaults to `768`): shared joint embedding dimension.
- **text_hidden_dim** (`int`, *optional*, defaults to `768`): text encoder hidden dimension.
- **text_num_layers** (`int`, *optional*, defaults to `12`): text encoder depth.
- **text_num_heads** (`int`, *optional*, defaults to `12`): text encoder attention heads.
- **text_mlp_dim** (`int`, *optional*, defaults to `3072`): MLP inner width in the text blocks.
- **max_seq_len** (`int`, *optional*, defaults to `64`): text input length.
- **input_tensor** (`NoneType`, *optional*, defaults to `None`): pre-existing input tensors to build on.
- **name** (`str`, *optional*, defaults to `'SigLIPZeroShotClassify'`): model name.

### SigLIPImageClassify

```python
SigLIPImageClassify(
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

SigLIP vision tower + linear image-classification head.

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

### SigLIPVisionModel

```python
SigLIPVisionModel(
    image_size=224,
    patch_size=16,
    vision_hidden_dim=768,
    vision_num_layers=12,
    vision_num_heads=12,
    vision_mlp_dim=3072,
    input_tensor=None,
    name="SigLIPVisionModel",
)
```

SigLIP vision tower as a standalone model.

**Parameters**

- **image_size** (`int`, *optional*, defaults to `224`): input image spec. An `int` builds an `N x N x 3` input, a 2-tuple `(H, W)` assumes 3 channels, and a 3-tuple follows the active `keras.config.image_data_format()`.
- **patch_size** (`int`, *optional*, defaults to `16`): ViT patch size.
- **vision_hidden_dim** (`int`, *optional*, defaults to `768`): ViT hidden dimension.
- **vision_num_layers** (`int`, *optional*, defaults to `12`): ViT encoder depth.
- **vision_num_heads** (`int`, *optional*, defaults to `12`): ViT attention heads.
- **vision_mlp_dim** (`int`, *optional*, defaults to `3072`): MLP inner width in the vision blocks.
- **input_tensor** (`NoneType`, *optional*, defaults to `None`): pre-existing input tensors to build on.
- **name** (`str`, *optional*, defaults to `'SigLIPVisionModel'`): model name.

### SigLIPTextModel

```python
SigLIPTextModel(
    vocab_size=32000,
    embed_dim=768,
    text_hidden_dim=768,
    text_num_layers=12,
    text_num_heads=12,
    text_mlp_dim=3072,
    max_seq_len=64,
    input_tensor=None,
    name="SigLIPTextModel",
)
```

SigLIP text tower as a standalone model.

**Parameters**

- **vocab_size** (`int`, *optional*, defaults to `32000`): tokenizer vocabulary size.
- **embed_dim** (`int`, *optional*, defaults to `768`): shared joint embedding dimension.
- **text_hidden_dim** (`int`, *optional*, defaults to `768`): text encoder hidden dimension.
- **text_num_layers** (`int`, *optional*, defaults to `12`): text encoder depth.
- **text_num_heads** (`int`, *optional*, defaults to `12`): text encoder attention heads.
- **text_mlp_dim** (`int`, *optional*, defaults to `3072`): MLP inner width in the text blocks.
- **max_seq_len** (`int`, *optional*, defaults to `64`): text input length.
- **input_tensor** (`NoneType`, *optional*, defaults to `None`): pre-existing input tensors to build on.
- **name** (`str`, *optional*, defaults to `'SigLIPTextModel'`): model name.

> **`SigLIPModel` gives you embeddings, `SigLIPZeroShotClassify` gives you
> logits.** Reach for the latter whenever you want probabilities over class
> prompts.

## Preprocessing

### SigLIPImageProcessor

```python
SigLIPImageProcessor(
    image_resolution=224,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    do_center_crop=True,
    do_normalize=True,
    do_resize=True,
    data_format=None,
)
```

Image processor for SigLIP (Sigmoid Loss for Language Image Pre-training) models. This processor handles various preprocessing steps for images to be used with SigLIP models, including resizing, center cropping, and normalization.

**Parameters**

- **image_resolution** (`int`, *optional*, defaults to `224`): target square resolution.
- **mean** (`tuple`, *optional*, defaults to `(0.5, 0.5, 0.5)`): per-channel normalization mean.
- **std** (`tuple`, *optional*, defaults to `(0.5, 0.5, 0.5)`): per-channel normalization std.
- **do_center_crop** (`bool`, *optional*, defaults to `True`): center-crop after the resize.
- **do_normalize** (`bool`, *optional*, defaults to `True`): apply mean/std normalization.
- **do_resize** (`bool`, *optional*, defaults to `True`): resize before cropping.
- **data_format** (`NoneType`, *optional*, defaults to `None`): `"channels_last"` or `"channels_first"`. Defaults to `keras.config.image_data_format()`.

### SigLIPTokenizer

```python
SigLIPTokenizer(
    variant=None,
    tokenizer_file=None,
    max_seq_len=64,
    unk_token="<unk>",
    pad_token="</s>",
    eos_token="</s>",
)
```

SigLIP SentencePiece Unigram tokenizer (``tokenizers`` Rust backend).

Pads with the **eos token** and returns no attention mask, so the padded
region cannot be found by mask. Compare full fixed-length id arrays instead.

**Parameters**

- **variant** (`NoneType`, *optional*, defaults to `None`): variant key, used to fetch the matching tokenizer files.
- **tokenizer_file** (`NoneType`, *optional*, defaults to `None`): explicit `tokenizer.json` path, overriding `variant`.
- **max_seq_len** (`int`, *optional*, defaults to `64`): text input length.
- **unk_token** (`str`, *optional*, defaults to `'<unk>'`): unknown-token string.
- **pad_token** (`str`, *optional*, defaults to `'</s>'`): padding token string.
- **eos_token** (`str`, *optional*, defaults to `'</s>'`): end-of-sequence token string.

### SigLIPProcessor

```python
SigLIPProcessor(
    image_resolution=224,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    do_center_crop=True,
    do_normalize=True,
    do_resize=True,
    variant=None,
    tokenizer_file=None,
    max_seq_len=64,
    unk_token="<unk>",
    pad_token="</s>",
    eos_token="</s>",
    tokenizer=None,
    image_processor=None,
)
```

Combined image + text processor for SigLIP.

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
- **unk_token** (`str`, *optional*, defaults to `'<unk>'`): unknown-token string.
- **pad_token** (`str`, *optional*, defaults to `'</s>'`): padding token string.
- **eos_token** (`str`, *optional*, defaults to `'</s>'`): end-of-sequence token string.
- **tokenizer** (`NoneType`, *optional*, defaults to `None`): a pre-built tokenizer, instead of building one.
- **image_processor** (`NoneType`, *optional*, defaults to `None`): a pre-built image processor.

## Model Variants

Load any of these with `from_weights("zeromodels/<variant id>")`.

| Variant id | Image size | Patch | Weights |
|---|---:|---:|---|
| `siglip_base_p16_224` | 224 | 16 | hub |
| `siglip_base_p16_256` | 256 | 16 | hub |
| `siglip_base_p16_multilingual_256` | 256 | 16 | hub |
| `siglip_base_p16_384` | 384 | 16 | hub |
| `siglip_base_p16_512` | 512 | 16 | hub |
| `siglip_large_p16_256` | 256 | 16 | hub |
| `siglip_large_p16_384` | 384 | 16 | hub |
| `siglip_so400m_p14_224` | 224 | 14 | hub |
| `siglip_so400m_p14_384` | 384 | 14 | hub |

## Basic Usage: Zero-Shot Classification

<img src="../assets/data/coco_skier.jpg" alt="A person skiing on a snowy slope" width="380">

```python
import keras
from zeromodels.models.siglip import SigLIPProcessor, SigLIPZeroShotClassify

processor = SigLIPProcessor.from_weights("zeromodels/siglip_base_p16_224")
model = SigLIPZeroShotClassify.from_weights("zeromodels/siglip_base_p16_224")

labels = [
    "a photo of a person skiing",
    "a photo of green apples",
    "a photo of a bear",
    "a photo of a living room",
]
inputs = processor(text=labels, image_paths="assets/data/coco_skier.jpg")
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
1.000000  a photo of a person skiing
0.000000  a photo of green apples
0.000000  a photo of a bear
0.000000  a photo of a living room
```

Note the input mapping: `SigLIPProcessor` returns `input_ids`, which the model
takes as `token_ids`. There is no `padding_mask`, unlike MetaCLIP 2.

## Batch Processing Multiple Images

Pass a list of paths. `image_logits` becomes `(num_images, num_texts)`, one row per
image, and the same label set is scored against each:

<p>
  <img src="../assets/data/coco_skier.jpg" alt="A person skiing on a snowy slope" width="300">
  <img src="../assets/data/coco_apples.jpg" alt="Green apples in a bowl" width="300">
</p>

```python
image_paths = ["assets/data/coco_skier.jpg", "assets/data/coco_apples.jpg"]
labels = [
    "a photo of a person skiing",
    "a photo of green apples",
    "a photo of a bear",
    "a photo of a living room",
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
assets/data/coco_skier.jpg
  1.000000  a photo of a person skiing
  0.000000  a photo of green apples
  0.000000  a photo of a bear
  0.000000  a photo of a living room

assets/data/coco_apples.jpg
  0.000000  a photo of a person skiing
  1.000000  a photo of green apples
  0.000000  a photo of a bear
  0.000000  a photo of a living room
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
from zeromodels.models.siglip import SigLIPZeroShotClassify

model = SigLIPZeroShotClassify.from_weights("hf:google/siglip-base-patch16-224")
model = SigLIPZeroShotClassify.from_weights("hf:<user>/my-finetune")
```

No shape arguments are needed. The architecture is read from the repo's `config.json`
and mapped onto the constructor.

All 5 model classes accept `hf:`, as do `SigLIPImageProcessor`, `SigLIPTokenizer`,
and `SigLIPProcessor`, so you can pull the matching preprocessing from the same repo:

```python
processor = SigLIPProcessor.from_weights("hf:google/siglip-base-patch16-224")
```

Loading `hf:google/siglip-base-patch16-224` and the `zeromodels/siglip_base_p16_224` Hub
variant produces identical outputs, since they are the same checkpoint by two routes.
