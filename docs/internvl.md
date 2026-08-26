# InternVL

<div class="kf-note kf-note--convert">
<b>On-the-fly conversion:</b> these weights are <b>not</b> mirrored as preconverted
<code>.weights.h5</code> under <code>zeromodels/</code>.
<code>from_weights("&lt;variant&gt;")</code> downloads the original safetensors
from the Hub and converts them in process on every load, because checkpoints this large are
impractical to re-host.
Pass <code>cache_converted=True</code> to keep the converted result and skip the download and
conversion next time. See <a href="../loading_weights/">Loading Weights</a>.
</div>

OpenGVLab's InternVL3 vision-language models, ported to pure Keras 3. An
InternViT tower feeds a pixel-shuffle downsampler and an MLP connector into an
inlined Qwen2 text decoder. Images are tiled dynamically: the processor picks an
aspect-ratio-matched tile grid (`min_patches`..`max_patches`) plus an optional
thumbnail, and each tile becomes `image_seq_length` tokens.

Links:

- Paper: [InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks (arXiv:2312.14238)](https://arxiv.org/abs/2312.14238)
- HF docs: [transformers/model_doc/internvl](https://huggingface.co/docs/transformers/model_doc/internvl)

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `internvl3-1b` | [`OpenGVLab/InternVL3-1B-hf`](https://huggingface.co/OpenGVLab/InternVL3-1B-hf) |
| `internvl3-2b` | [`OpenGVLab/InternVL3-2B-hf`](https://huggingface.co/OpenGVLab/InternVL3-2B-hf) |
| `internvl3-8b` | [`OpenGVLab/InternVL3-8B-hf`](https://huggingface.co/OpenGVLab/InternVL3-8B-hf) |
| `internvl3-14b` | [`OpenGVLab/InternVL3-14B-hf`](https://huggingface.co/OpenGVLab/InternVL3-14B-hf) |
| `internvl3-38b` | [`OpenGVLab/InternVL3-38B-hf`](https://huggingface.co/OpenGVLab/InternVL3-38B-hf) |
| `internvl3-78b` | [`OpenGVLab/InternVL3-78B-hf`](https://huggingface.co/OpenGVLab/InternVL3-78B-hf) |
| `internvl3.5-1b` | [`OpenGVLab/InternVL3_5-1B-HF`](https://huggingface.co/OpenGVLab/InternVL3_5-1B-HF) |
| `internvl3.5-2b` | [`OpenGVLab/InternVL3_5-2B-HF`](https://huggingface.co/OpenGVLab/InternVL3_5-2B-HF) |
| `internvl3.5-4b` | [`OpenGVLab/InternVL3_5-4B-HF`](https://huggingface.co/OpenGVLab/InternVL3_5-4B-HF) |
| `internvl3.5-8b` | [`OpenGVLab/InternVL3_5-8B-HF`](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF) |
| `internvl3.5-14b` | [`OpenGVLab/InternVL3_5-14B-HF`](https://huggingface.co/OpenGVLab/InternVL3_5-14B-HF) |
| `internvl3.5-38b` | [`OpenGVLab/InternVL3_5-38B-HF`](https://huggingface.co/OpenGVLab/InternVL3_5-38B-HF) |
| `internvl3.5-30b-a3b` | [`OpenGVLab/InternVL3_5-30B-A3B-HF`](https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B-HF) |

The InternVL3 rows use a Qwen2.5 text tower; InternVL3.5 dense uses Qwen3, and
`internvl3.5-30b-a3b` uses a Qwen3-MoE tower. One class loads them all.

## API

### `InternVLModel`

InternVL3 multimodal backbone: InternViT tower + pixel-shuffle projector + Qwen2-style decoder.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `151674` | token vocabulary size |
| `embed_dim` | `896` | text model width |
| `mlp_dim` | `4864` | MLP inner width |
| `num_layers` | `24` | decoder blocks |
| `num_heads` | `14` | query heads |
| `num_kv_heads` | `2` | key/value heads (GQA) |
| `norm_eps` | `1e-06` | normalization epsilon |
| `rope_theta` | `1000000.0` | rotary base frequency |
| `tie_embeddings` | `False` | reuse embeddings as the LM head |
| `vision_embed_dim` | `1024` | vision tower width |
| `vision_mlp_dim` | `4096` | vision MLP width |
| `vision_num_layers` | `24` | vision tower depth |
| `vision_num_heads` | `16` | vision attention heads |
| `image_size` | `448` | expected image resolution |
| `patch_size` | `14` | patch size |
| `vision_attention_bias` | `True` |  |
| `vision_qk_norm` | `False` |  |
| `vision_norm_type` | `'layer_norm'` |  |
| `vision_norm_eps` | `1e-06` | vision tower norm epsilon |
| `vision_layer_scale_init` | `0.1` |  |
| `downsample_ratio` | `0.5` |  |
| `image_token_id` | `151667` | placeholder token id expanded per image |

### `InternVLConditionalGenerate`

InternVL3 with an LM head + fast ``.generate()`` (image+text -> text).

```python
generate(
    input_ids,
    attention_mask=None,
    max_new_tokens=None,
    eos_token_id=None,
    sampler=None,
    seed=None,
    **prefill_inputs,
)
```

Image and video tensors ride along as `**prefill_inputs`; the processor
produces them for you.

### `InternVLTextModel`

Qwen2-style causal decoder: ``embed -> num_layers x InternVLDecoderLayer -> RMSNorm``.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | required | token vocabulary size |
| `embed_dim` | required | text model width |
| `mlp_dim` | required | MLP inner width |
| `num_layers` | required | decoder blocks |
| `num_heads` | required | query heads |
| `num_kv_heads` | required | key/value heads (GQA) |
| `head_dim` | `None` | per-head width |
| `norm_eps` | `1e-06` | normalization epsilon |

### `InternVLVisionModel`

InternViT vision tower: conv patch embed + CLS/pos embeddings -> layer-scaled pre-norm blocks (-> optional final LayerNorm).

| Arg | Default | Meaning |
|---|---|---|
| `embed_dim` | required | text model width |
| `mlp_dim` | required | MLP inner width |
| `num_layers` | required | decoder blocks |
| `num_heads` | required | query heads |
| `image_size` | `448` | expected image resolution |
| `patch_size` | `14` | patch size |
| `attention_bias` | `True` | add bias to the QKV projections |
| `qk_norm` | `False` | normalize queries and keys before attention |
| `norm_type` | `'layer_norm'` | normalization layer to use |
| `norm_eps` | `1e-06` | normalization epsilon |
| `layer_scale_init` | `0.1` | initial LayerScale value |
| `use_mean_pooling` | `True` | when set, the tower's final norm is skipped |

### `InternVLImageProcessor`

InternVL dynamic-tiling image processor (HF GotOcr2 recipe).

| Arg | Default | Meaning |
|---|---|---|
| `size` | `448` | target resolution |
| `min_patches` | `1` | fewest tiles per image |
| `max_patches` | `12` | most tiles per image |
| `crop_to_patches` | `True` | tile the image dynamically by aspect ratio |
| `use_thumbnail` | `True` | append a whole-image thumbnail tile |
| `image_mean` | `(0.485, 0.456, 0.406)` | per-channel normalization mean |
| `image_std` | `(0.229, 0.224, 0.225)` | per-channel normalization std |

### `InternVLTokenizer`

InternVL3 Qwen2-BPE tokenizer (``tokenizers`` backend).

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `None` | Hub repo to pull tokenizer/processor files from |
| `tokenizer_file` | `None` | explicit path to a `tokenizer.json` |

### `InternVLProcessor`

Image + text -> model inputs for the InternVL3 models.

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `None` | Hub repo to pull tokenizer/processor files from |
| `image_seq_length` | `256` | tokens each image tile expands to |
| `tokenizer` | `None` | override the default tokenizer |
| `image_processor` | `None` | override the default image processor |

## End-to-end example

### Single input (image + text)

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.models.internvl import InternVLConditionalGenerate, InternVLProcessor

model = InternVLConditionalGenerate.from_weights("internvl3-1b")
processor = InternVLProcessor.from_weights("internvl3-1b")

image = Image.open("photo.jpg")
inputs = processor(
    conversation=[
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in one sentence."},
            ],
        }
    ]
)
outputs = model.generate(**inputs, max_new_tokens=64)

print(processor.decode(outputs[0]))
```

### Several images in one conversation

Add one image content item per image. The processor expands each marker to
that image's own patch count:

```python
inputs = processor(
    conversation=[
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open("a.jpg")},
                {"type": "image", "image": Image.open("b.jpg")},
                {"type": "text", "text": "What differs between these two images?"},
            ],
        }
    ]
)
outputs = model.generate(**inputs, max_new_tokens=64)
```

### Batch

Pass a list of conversations. Each one is rendered separately and takes only
the images its own markers claim, so the conversations do not need the same
number of images or images of the same size:

```python
conversations = [
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open("a.jpg")},
                {"type": "text", "text": "What is in this image?"},
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open("b.jpg")},
                {"type": "image", "image": Image.open("c.jpg")},
                {"type": "text", "text": "What differs between these?"},
            ],
        }
    ],
]
inputs = processor(conversation=conversations)
outputs = model.generate(**inputs, max_new_tokens=64)

for text in processor.batch_decode(outputs):
    print(text)
```

Text-only prompts batch the same way: pass `text=[...]` with no `images`.

### Text only

`InternVLTokenizer` encodes raw text: it has no chat template, so pass a prompt you
have rendered yourself (or go through the processor above).

```python
from zeromodels.models.internvl import InternVLTokenizer

tokenizer = InternVLTokenizer.from_weights("internvl3-1b")
inputs = tokenizer("Who wrote Dune?")
outputs = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = InternVLConditionalGenerate.from_weights(
    "internvl3-1b", quantization="int8", load_dtype="bfloat16"
)
```
