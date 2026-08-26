# CLIP

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>kf_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

CLIP (Contrastive Language-Image Pre-training) is a vision + text dual-encoder trained on hundreds of millions of (image, caption) pairs with a contrastive loss. The vision side is a ViT; the text side is a small transformer with causal masking. Both encoders project to a shared embedding space, and a learnable temperature scales the cosine-similarity logits.

CLIP's similarity-matrix output makes it useful for many downstream tasks: zero-shot classification, image-text retrieval, image-image similarity, embedding extraction for diffusion models or VLMs, and (with a supervised head) standard image classification.

**Paper**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

## API

### CLIPModel

```python
CLIPModel(
    embed_dim=512,
    image_size=224,
    vision_num_layers=12,
    vision_hidden_dim=768,
    vision_patch_size=32,
    max_seq_len=77,
    vocab_size=49408,
    text_hidden_dim=512,
    text_num_heads=8,
    text_num_layers=12,
    vision_mlp_ratio=4.0,
    text_mlp_ratio=4.0,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
    input_tensor=None,
    name="CLIPModel",
)
```

The dual encoder itself, and the base the other two classes build on. Returns
projected but un-normalized embeddings, for retrieval, image-image similarity, or as
frozen features.

**Parameters**

- **embed_dim** (`int`, *optional*, defaults to `512`): shared embedding dimension, the `projection_dim` of the HF config.
- **image_size** (`int` or `tuple`, *optional*, defaults to `224`): input image spec. An `int` builds an `N x N x 3` input, a 2-tuple `(H, W)` assumes 3 channels, and a 3-tuple follows the active `keras.config.image_data_format()`.
- **vision_num_layers** (`int`, *optional*, defaults to `12`): ViT encoder depth.
- **vision_hidden_dim** (`int`, *optional*, defaults to `768`): ViT hidden dimension.
- **vision_patch_size** (`int`, *optional*, defaults to `32`): ViT patch size.
- **max_seq_len** (`int`, *optional*, defaults to `77`): text input length.
- **vocab_size** (`int`, *optional*, defaults to `49408`): tokenizer vocabulary size.
- **text_hidden_dim** (`int`, *optional*, defaults to `512`): text encoder hidden dimension.
- **text_num_heads** (`int`, *optional*, defaults to `8`): text encoder attention heads.
- **text_num_layers** (`int`, *optional*, defaults to `12`): text encoder depth.
- **vision_mlp_ratio** (`float`, *optional*, defaults to `4.0`): MLP expansion ratio in the vision blocks.
- **text_mlp_ratio** (`float`, *optional*, defaults to `4.0`): MLP expansion ratio in the text blocks.
- **hidden_act** (`str`, *optional*, defaults to `"quick_gelu"`): MLP activation. Use `"quick_gelu"` for canonical OpenAI CLIP, `"gelu"` or `"gelu_new"` for LAION and community variants.
- **layer_norm_eps** (`float`, *optional*, defaults to `1e-5`): epsilon for every LayerNorm.
- **input_tensor** (`dict`, *optional*): pre-existing input tensors to build on.
- **name** (`str`, *optional*, defaults to `"CLIPModel"`): model name.

**Returns** a `dict`:

- **image_embeddings** (`(B_img, embed_dim)`): projected, un-normalized image embeddings.
- **text_embeddings** (`(B_txt, embed_dim)`): projected, un-normalized text embeddings.

### CLIPZeroShotClassify

```python
CLIPZeroShotClassify(
    embed_dim=512,
    image_size=224,
    vision_num_layers=12,
    vision_hidden_dim=768,
    vision_patch_size=32,
    max_seq_len=77,
    vocab_size=49408,
    text_hidden_dim=512,
    text_num_heads=8,
    text_num_layers=12,
    vision_mlp_ratio=4.0,
    text_mlp_ratio=4.0,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
    input_tensor=None,
    name="CLIPZeroShotClassify",
)
```

`CLIPModel` plus the contrastive head: L2-normalize both sides, then scale the cosine
similarities by `logit_scale`. **This is the class for zero-shot classification.**

**Parameters** are identical to [CLIPModel](#clipmodel), except **name** defaults to
`"CLIPZeroShotClassify"`.

**Returns** a `dict`:

- **image_logits** (`(B_img, B_txt)`): scaled cosine similarity, image `i` against text `j`.
- **text_logits** (`(B_txt, B_img)`): its transpose.

### CLIPImageClassify

```python
CLIPImageClassify(
    num_classes=1000,
    image_size=224,
    vision_num_layers=12,
    vision_hidden_dim=768,
    vision_patch_size=16,
    vision_mlp_ratio=4.0,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
    input_tensor=None,
    name="CLIPImageClassify",
)
```

The vision tower only (mean-pooled patches, no text side) with a linear head. Needs a
checkpoint whose head was actually trained, which no Hub Keras variant has. See
[Supervised Image Classification](#supervised-image-classification-clipimageclassify).

**Parameters**

- **num_classes** (`int`, *optional*, defaults to `1000`): number of output classes.
- **image_size** (`int` or `tuple`, *optional*, defaults to `224`): as in `CLIPModel`.
- **vision_num_layers** (`int`, *optional*, defaults to `12`): ViT encoder depth.
- **vision_hidden_dim** (`int`, *optional*, defaults to `768`): ViT hidden dimension.
- **vision_patch_size** (`int`, *optional*, defaults to `16`): ViT patch size.
- **vision_mlp_ratio** (`float`, *optional*, defaults to `4.0`): MLP expansion ratio in the vision blocks.
- **hidden_act** (`str`, *optional*, defaults to `"quick_gelu"`): MLP activation.
- **layer_norm_eps** (`float`, *optional*, defaults to `1e-5`): epsilon for every LayerNorm.
- **input_tensor** (`dict`, *optional*): pre-existing input tensors to build on.
- **name** (`str`, *optional*, defaults to `"CLIPImageClassify"`): model name.

**Returns** a tensor of shape `(B, num_classes)`: raw class logits.

> **The distinction that catches people:** `CLIPModel` gives you **embeddings**,
> `CLIPZeroShotClassify` gives you **logits**. Asking `CLIPModel` for
> `output["image_logits"]` raises `KeyError`.

### CLIPVisionModel

```python
CLIPVisionModel(
    image_size=224,
    vision_num_layers=12,
    vision_hidden_dim=768,
    vision_patch_size=32,
    vision_mlp_ratio=4.0,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
    input_tensor=None,
    name="CLIPVisionModel",
)
```

The vision tower on its own: patch embedding plus the transformer stack, ending at the
post-encoder LayerNorm. No text tower and no `visual_projection`, so use it when you
only want image features.

**Parameters**

- **image_size** (`int` or `tuple`, *optional*, defaults to `224`): input image spec, as in `CLIPModel`.
- **vision_num_layers** (`int`, *optional*, defaults to `12`): ViT encoder depth.
- **vision_hidden_dim** (`int`, *optional*, defaults to `768`): ViT hidden dimension.
- **vision_patch_size** (`int`, *optional*, defaults to `32`): ViT patch size.
- **vision_mlp_ratio** (`float`, *optional*, defaults to `4.0`): MLP expansion ratio.
- **hidden_act** (`str`, *optional*, defaults to `"quick_gelu"`): MLP activation.
- **layer_norm_eps** (`float`, *optional*, defaults to `1e-5`): LayerNorm epsilon.
- **input_tensor** (`tensor`, *optional*): pre-existing input tensor.
- **name** (`str`, *optional*, defaults to `"CLIPVisionModel"`): model name.

**Returns** a `dict`:

- **last_hidden_state** (`(B, num_patches + 1, vision_hidden_dim)`): per-token features, CLS first.
- **pooler_output** (`(B, vision_hidden_dim)`): the post-LayerNorm CLS token.

### CLIPTextModel

```python
CLIPTextModel(
    max_seq_len=77,
    vocab_size=49408,
    text_hidden_dim=512,
    text_num_heads=8,
    text_num_layers=12,
    text_mlp_ratio=4.0,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
    input_tensor=None,
    name="CLIPTextModel",
)
```

The text tower on its own: token and positional embedding, a causal-masked transformer
stack, the post-encoder LayerNorm, and the EOT-position pluck. No vision tower and no
`text_projection`.

**Parameters**

- **max_seq_len** (`int`, *optional*, defaults to `77`): text input length.
- **vocab_size** (`int`, *optional*, defaults to `49408`): tokenizer vocabulary size.
- **text_hidden_dim** (`int`, *optional*, defaults to `512`): text encoder hidden dimension.
- **text_num_heads** (`int`, *optional*, defaults to `8`): text encoder attention heads.
- **text_num_layers** (`int`, *optional*, defaults to `12`): text encoder depth.
- **text_mlp_ratio** (`float`, *optional*, defaults to `4.0`): MLP expansion ratio.
- **hidden_act** (`str`, *optional*, defaults to `"quick_gelu"`): MLP activation.
- **layer_norm_eps** (`float`, *optional*, defaults to `1e-5`): LayerNorm epsilon.
- **input_tensor** (`dict`, *optional*): pre-existing inputs, keyed `"token_ids"` and `"padding_mask"`.
- **name** (`str`, *optional*, defaults to `"CLIPTextModel"`): model name.

**Returns** a `dict`:

- **last_hidden_state** (`(B, max_seq_len, text_hidden_dim)`): per-token features.
- **pooler_output** (`(B, text_hidden_dim)`): the feature at the EOT position.

### CLIPImageEmbed

```python
CLIPImageEmbed(
    embed_dim=512,
    image_size=224,
    vision_num_layers=12,
    vision_hidden_dim=768,
    vision_patch_size=32,
    vision_mlp_ratio=4.0,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
    input_tensor=None,
    name="CLIPImageEmbed",
)
```

`CLIPVisionModel` plus the bias-free `visual_projection`, producing the same image side
as `CLIPModel` without instantiating the text tower or `logit_scale`.

**Parameters** are those of `CLIPVisionModel`, plus:

- **embed_dim** (`int`, *optional*, defaults to `512`): shared joint embedding dimension.

**Returns** a `dict`:

- **image_embeds** (`(B, embed_dim)`): projected image embeddings.
- **last_hidden_state** (`(B, num_patches + 1, vision_hidden_dim)`): pre-projection features.

### CLIPTextEmbed

```python
CLIPTextEmbed(
    embed_dim=512,
    max_seq_len=77,
    vocab_size=49408,
    text_hidden_dim=512,
    text_num_heads=8,
    text_num_layers=12,
    text_mlp_ratio=4.0,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
    input_tensor=None,
    name="CLIPTextEmbed",
)
```

`CLIPTextModel` plus the bias-free `text_projection`, producing the same text side as
`CLIPModel`.

**Parameters** are those of `CLIPTextModel`, plus:

- **embed_dim** (`int`, *optional*, defaults to `512`): shared joint embedding dimension.

**Returns** a `dict`:

- **text_embeds** (`(B, embed_dim)`): projected text embeddings.
- **last_hidden_state** (`(B, max_seq_len, text_hidden_dim)`): pre-projection features.

> **Key names differ from `CLIPModel`.** These two return `image_embeds` /
> `text_embeds`, while `CLIPModel` returns `image_embeddings` / `text_embeddings`.

## Preprocessing

### CLIPImageProcessor

```python
CLIPImageProcessor(
    image_resolution=224,
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
    do_center_crop=True,
    do_normalize=True,
    do_resize=True,
    data_format=None,
)
```

Resizes the shortest edge to `image_resolution` with PIL bicubic, center-crops to a
square, rescales to `[0, 1]`, and normalizes.

**Parameters**

- **image_resolution** (`int`, *optional*, defaults to `224`): target square resolution.
- **mean** (`tuple`, *optional*, defaults to the OpenAI CLIP mean): per-channel normalization mean.
- **std** (`tuple`, *optional*, defaults to the OpenAI CLIP std): per-channel normalization std.
- **do_center_crop** (`bool`, *optional*, defaults to `True`): center-crop after the resize.
- **do_normalize** (`bool`, *optional*, defaults to `True`): apply mean/std normalization.
- **do_resize** (`bool`, *optional*, defaults to `True`): resize before cropping.
- **data_format** (`str`, *optional*): `"channels_last"` or `"channels_first"`. Defaults to `keras.config.image_data_format()`.

**Call** `processor(image)`, where `image` is a path, a list of paths, a PIL image, or
an array. **Returns** a `dict`:

- **pixel_values** (`(B, H, W, 3)`): preprocessed images, in the configured data format.

### CLIPTokenizer

```python
CLIPTokenizer(
    variant=None,
    tokenizer_file=None,
    max_seq_len=77,
    unk_token="<|endoftext|>",
    bos_token="<|startoftext|>",
    eos_token="<|endoftext|>",
    pad_token="<|endoftext|>",
)
```

Byte-level BPE on the `tokenizers` Rust backend. Loads the variant's `tokenizer.json`
from the `clip` Hub repo and applies CLIP's truncation and `<|endoftext|>` padding.

**Parameters**

- **variant** (`str`, *optional*): CLIP variant key. Defaults to `"clip_vit_base_16"`.
- **tokenizer_file** (`str`, *optional*): explicit `tokenizer.json` path, overriding `variant`.
- **max_seq_len** (`int`, *optional*, defaults to `77`): padded and truncated length.
- **unk_token** / **bos_token** / **eos_token** / **pad_token** (`str`, *optional*): special token strings.

**Call** `tokenizer(inputs)` with a string or list of strings. **Returns** a `dict`:

- **input_ids** (`(B, max_seq_len)`): token ids.
- **attention_mask** (`(B, max_seq_len)`): `1` for real tokens, `0` for padding.

### CLIPProcessor

```python
CLIPProcessor(
    image_resolution=224,
    mean=...,
    std=...,
    do_center_crop=True,
    do_normalize=True,
    do_resize=True,
    variant=None,
    tokenizer_file=None,
    max_seq_len=77,
    unk_token="<|endoftext|>",
    bos_token="<|startoftext|>",
    eos_token="<|endoftext|>",
    pad_token="<|endoftext|>",
    tokenizer=None,
    image_processor=None,
)
```

A `CLIPImageProcessor` and a `CLIPTokenizer` behind one callable. Takes the union of
both constructors' arguments, plus:

- **tokenizer** (`CLIPTokenizer`, *optional*): a pre-built tokenizer, instead of building one.
- **image_processor** (`CLIPImageProcessor`, *optional*): a pre-built image processor.

**Call** `processor(text=None, images=None, image_paths=None)`. **Returns** a `dict`:

- **input_ids** (`(B_txt, max_seq_len)`): token ids.
- **attention_mask** (`(B_txt, max_seq_len)`): padding mask.
- **images** (`(B_img, H, W, 3)`): preprocessed images.

The model input keys differ from these, so remap when calling: `input_ids` becomes
`token_ids` and `attention_mask` becomes `padding_mask`, as the examples below do.


## Model Variants

Variant ids for `CLIPModel.from_weights`:

| Variant id              | Params  | Patch | Resolution | Source             |
|-------------------------|--------:|------:|-----------:|--------------------|
| `clip_vit_base_16`      | ~150 M  |    16 |        224 | openai             |
| `clip_vit_base_32`      | ~151 M  |    32 |        224 | openai             |
| `clip_vit_large_14`     | ~428 M  |    14 |        224 | openai             |
| `clip_vit_large_14_336` | ~428 M  |    14 |        336 | openai             |
| `clip_vit_g_14`         | ~1.4 B  |    14 |        224 | laion2b s12B b42K  |
| `clip_vit_bigg_14`      | ~2.5 B  |    14 |        224 | laion2b 39B b160k  |


## Basic Usage: Zero-Shot Classification

<img src="../assets/data/coco_cats.jpg" alt="Two cats on a pink blanket" width="380">

```python
import keras
from zeromodels.models.clip import CLIPProcessor, CLIPZeroShotClassify

processor = CLIPProcessor.from_weights("zeromodels/clip_vit_base_16")
model = CLIPZeroShotClassify.from_weights("zeromodels/clip_vit_base_16")

labels = [
    "a photo of two cats",
    "a photo of a bear",
    "a photo of a skier",
    "a photo of green apples",
]
inputs = processor(text=labels, image_paths="assets/data/coco_cats.jpg")
output = model(
    {
        "images": inputs["images"],
        "token_ids": inputs["input_ids"],
        "padding_mask": inputs["attention_mask"],
    }
)

# (1, 4): one image, four class prompts. Softmax over the text axis.
probs = keras.ops.convert_to_numpy(
    keras.ops.softmax(output["image_logits"], axis=-1)
).squeeze()
for label, p in zip(labels, probs):
    print(f"{p:.6f}  {label}")
```

```
0.999611  a photo of two cats
0.000001  a photo of a bear
0.000003  a photo of a skier
0.000386  a photo of green apples
```

Use `CLIPZeroShotClassify`, not `CLIPModel`: the latter returns
`image_embeddings` / `text_embeddings` and has no `image_logits` key.

### Batch Processing Multiple Images

Pass a list of images. `image_logits` becomes `(num_images, num_texts)`, one row per
image, and the same label set is scored against each:

<p>
  <img src="../assets/data/coco_cats.jpg" alt="Two cats on a pink blanket" width="300">
  <img src="../assets/data/coco_bear.jpg" alt="A brown bear" width="300">
</p>

```python
import keras
from zeromodels.models.clip import CLIPProcessor, CLIPZeroShotClassify

processor = CLIPProcessor.from_weights("zeromodels/clip_vit_base_16")
model = CLIPZeroShotClassify.from_weights("zeromodels/clip_vit_base_16")

image_paths = ["assets/data/coco_cats.jpg", "assets/data/coco_bear.jpg"]
labels = [
    "a photo of two cats",
    "a photo of a bear",
    "a photo of a skier",
    "a photo of green apples",
]

inputs = processor(text=labels, image_paths=image_paths)
output = model(
    {
        "images": inputs["images"],
        "token_ids": inputs["input_ids"],
        "padding_mask": inputs["attention_mask"],
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
assets/data/coco_cats.jpg
  0.999611  a photo of two cats
  0.000001  a photo of a bear
  0.000003  a photo of a skier
  0.000386  a photo of green apples

assets/data/coco_bear.jpg
  0.000001  a photo of two cats
  0.999802  a photo of a bear
  0.000192  a photo of a skier
  0.000004  a photo of green apples
```

## Supervised Image Classification: `CLIPImageClassify`

Mirrors HF's `CLIPForImageClassification`: the CLIP vision encoder feeds a mean-pool over the patch tokens (CLS excluded) and a single linear `classifier` Dense producing `num_classes` logits. The text tower, visual projection, and `logit_scale` are **not** built.

The classifier head is a plain `Dense`, so this class is only meaningful with a
checkpoint whose head was actually trained. None of the Hub checkpoint variants in the
table above carry one: they are all base CLIP, so
`CLIPImageClassify.from_weights("zeromodels/clip_vit_base_16")` leaves the head randomly
initialized and its predictions are meaningless. Point it at a fine-tune, or train
the head yourself.

```python
import keras
from zeromodels.models.clip import CLIPImageClassify, CLIPImageProcessor

# A checkpoint whose classifier head was trained
model = CLIPImageClassify.from_weights("hf:<user>/clip-finetuned-imagenet")
image_processor = CLIPImageProcessor.from_weights("zeromodels/clip_vit_base_16")

inputs = image_processor("cat.jpg")  # {"pixel_values": (1, 224, 224, 3)}
logits = model(inputs["pixel_values"])  # (B, num_classes)
pred = keras.ops.argmax(logits, axis=-1)
```

The processor returns a **dict**, so index `["pixel_values"]` before calling the
model rather than passing the dict straight in.

Construct from scratch for fine-tuning on a new dataset:

```python
model = CLIPImageClassify(
    num_classes=10,  # your class count
    image_size=224,
    vision_num_layers=12,
    vision_hidden_dim=768,
    vision_patch_size=16,
    vision_mlp_ratio=4.0,
    hidden_act="quick_gelu",  # or "gelu" / "gelu_new"
    layer_norm_eps=1e-5,
)
model.output_shape  # (None, 10)
```

Mind the argument names. Keras forwards unrecognized keywords up to `Model`, so a
typo like `num_labels=10` or `vision_width=768` is swallowed silently and you get a
model built from the **defaults** (`num_classes=1000`) with no error raised.

You can also warm-start the vision encoder from a `CLIPModel` checkpoint (the encoder weight names match across both classes):

```python
src = CLIPModel.from_weights("zeromodels/clip_vit_base_16")
ac = CLIPImageClassify(
    num_classes=10,
    image_size=224,
    vision_num_layers=12,
    vision_hidden_dim=768,
    vision_patch_size=16,
    vision_mlp_ratio=4.0,
)
# Transfer the vision encoder weights; leave the classifier random
for src_layer, dst_layer in zip(src.layers, ac.layers):
    if "vision_model" in dst_layer.name and src_layer.name == dst_layer.name:
        dst_layer.set_weights(src_layer.get_weights())
```

## Data Format

**Both the models and the processors support `channels_last` and `channels_first`.**
Neither is hard-coded to a layout, so the whole pipeline runs either way.

They pick the format differently, which is the one thing to keep straight:

| | How it picks the format |
|---|---|
| Processors | A `data_format` kwarg, per instance. `None` (the default) resolves to `keras.config.image_data_format()`. |
| Models | Read `keras.config.image_data_format()` when they are **constructed**. There is no `data_format` argument. |

So a processor can be overridden on its own, while a model follows the global setting
in force at build time.

### Overriding the processor only

```python
CLIPImageProcessor(data_format="channels_last")("photo.jpg")
# {"pixel_values": (1, 224, 224, 3)}

CLIPImageProcessor(data_format="channels_first")("photo.jpg")
# {"pixel_values": (1, 3, 224, 224)}
```

### Switching the whole pipeline

Set the global format before constructing the model, and both sides agree:

```python
import keras

keras.config.set_image_data_format("channels_first")

processor = CLIPProcessor.from_weights("zeromodels/clip_vit_base_16")
model = CLIPZeroShotClassify.from_weights("zeromodels/clip_vit_base_16")

inputs = processor(text=labels, image_paths="assets/data/coco_cats.jpg")
# inputs["images"] is (1, 3, 224, 224)
output = model(
    {
        "images": inputs["images"],
        "token_ids": inputs["input_ids"],
        "padding_mask": inputs["attention_mask"],
    }
)
```

The probabilities are the same under either layout. Only the tensor shape changes.

Note that `keras.config.set_image_data_format` is global state. Set it once at the top
of a script rather than toggling it between calls, since already-built models keep the
layout they were constructed with.

## Loading Fine-tuned and Community Weights

You are not limited to the official variants above. Any Hugging Face repo whose
`model_type` is `"clip"` loads directly with the `hf:` prefix: the original OpenAI
checkpoints, LAION variants, and arbitrary user fine-tunes.

```python
from zeromodels.models.clip import CLIPZeroShotClassify

# The original OpenAI checkpoints
model = CLIPZeroShotClassify.from_weights("hf:openai/clip-vit-base-patch16")

# A LAION variant, which uses "gelu" rather than "quick_gelu"
model = CLIPZeroShotClassify.from_weights("hf:laion/CLIP-ViT-B-16-laion2B-s34B-b88K")

# Somebody's fine-tune
model = CLIPZeroShotClassify.from_weights("hf:<user>/clip-finetuned-on-my-data")
```

No shape arguments are needed. The architecture is read from the repo's
`config.json` and mapped onto the constructor. That includes `hidden_act`, which
matters in practice: OpenAI checkpoints set it to `"quick_gelu"` while LAION ones use
`"gelu"`, and reading it from the config is what keeps both bit-close to their
reference.

All seven model classes accept `hf:`, as do `CLIPProcessor`, `CLIPImageProcessor`, and
`CLIPTokenizer`, so you can pull the matching preprocessing from the same repo:

```python
processor = CLIPProcessor.from_weights("hf:openai/clip-vit-base-patch16")
```

Loading `hf:openai/clip-vit-base-patch16` and the `zeromodels/clip_vit_base_16` Hub variant
produces identical outputs, since they are the same checkpoint by two routes.
