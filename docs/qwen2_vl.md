# Qwen2-VL

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code>, <code>zm_preprocessor.json</code> and
<code>tokenizer.json</code> plus the Keras weights: <code>model.weights.h5</code>, or a
sharded <code>model.weights.json</code> + shards for the 72B). Load the model and
processor with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Alibaba's Qwen2-VL vision-language models, ported to pure Keras 3. A
native-resolution ViT consumes images at their own aspect ratio: the processor
smart-resizes so both sides are multiples of `patch_size * spatial_merge_size`,
flattens into patches, and reports an `image_grid_thw` grid. The text decoder
uses M-RoPE (multimodal rotary embeddings split across time/height/width).

Links:

- HF collection: [Qwen2-VL](https://huggingface.co/collections/zeromodels/qwen2-vl-6a7cda6f1cbf2cf66e7b5d36)
- Paper: [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- HF docs: [transformers/model_doc/qwen2_vl](https://huggingface.co/docs/transformers/model_doc/qwen2_vl)

See also [qwen2_5_vl.md](qwen2_5_vl.md), [qwen3_vl.md](qwen3_vl.md).

## Variants

Preconverted, bf16 weights are hosted under `zeromodels/`. Load with
`from_weights("zeromodels/<variant>")`; the `-instruct` suffix marks
instruction-tuned checkpoints (use the chat template via `Qwen2VLProcessor`), bare
names are base models. The 2B / 7B sizes are Apache 2.0; the 72B is under the Qwen
license.

| Variant | Hub |
|---|---|
| `qwen2-vl-2b` | [`zeromodels/qwen2-vl-2b`](https://huggingface.co/zeromodels/qwen2-vl-2b) |
| `qwen2-vl-2b-instruct` | [`zeromodels/qwen2-vl-2b-instruct`](https://huggingface.co/zeromodels/qwen2-vl-2b-instruct) |
| `qwen2-vl-7b` | [`zeromodels/qwen2-vl-7b`](https://huggingface.co/zeromodels/qwen2-vl-7b) |
| `qwen2-vl-7b-instruct` | [`zeromodels/qwen2-vl-7b-instruct`](https://huggingface.co/zeromodels/qwen2-vl-7b-instruct) |
| `qwen2-vl-72b` | [`zeromodels/qwen2-vl-72b`](https://huggingface.co/zeromodels/qwen2-vl-72b) |
| `qwen2-vl-72b-instruct` | [`zeromodels/qwen2-vl-72b-instruct`](https://huggingface.co/zeromodels/qwen2-vl-72b-instruct) |

Upstream Qwen safetensors also load directly via the `hf:` prefix, e.g.
`from_weights("hf:Qwen/Qwen2-VL-7B-Instruct")`, which converts them in process (pass
`cache_converted=True` to keep the result). See [Loading Weights](loading_weights.md).

## API

### `Qwen2VLModel`

Qwen2-VL multimodal backbone: vision tower + Qwen2 decoder fused by M-RoPE.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `151936` | token vocabulary size |
| `embed_dim` | `1536` | text model width |
| `mlp_dim` | `8960` | MLP inner width |
| `num_layers` | `28` | decoder blocks |
| `num_heads` | `12` | query heads |
| `num_kv_heads` | `2` | key/value heads (GQA) |
| `norm_eps` | `1e-06` | normalization epsilon |
| `rope_theta` | `1000000.0` | rotary base frequency |
| `mrope_section` | `(16, 24, 24)` | M-RoPE split across time/height/width |
| `tie_embeddings` | `True` | reuse embeddings as the LM head |
| `vision_depth` | `32` | vision tower depth |
| `vision_embed_dim` | `1280` | vision tower width |
| `vision_num_heads` | `16` | vision attention heads |
| `vision_mlp_ratio` | `4` |  |
| `patch_size` | `14` | patch size |
| `spatial_merge_size` | `2` | patch-merge factor before the decoder |
| `temporal_patch_size` | `2` | frames per temporal patch |
| `in_channels` | `3` | input image channels |
| `image_token_id` | `151655` | placeholder token id expanded per image |
| `video_token_id` | `151656` | placeholder token id expanded per video |
| `vision_start_token_id` | `151652` | token id opening a vision span |
| `vision_end_token_id` | `151653` | token id closing a vision span |

### `Qwen2VLConditionalGenerate`

Qwen2-VL with an LM head + fast ``.generate()`` (image+text -> text).

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

### `Qwen2VLTextGenerate`

Text-only counterpart of `Qwen2VLConditionalGenerate`, built with no vision tower
(`build_vision=False`), so `.generate()` takes just token ids. It reads only the language
model out of a Qwen2-VL checkpoint: `hf:` conversion copies just the text weights (the
vision keys are never touched), and a zeromodels repo declaring
`Qwen2VLConditionalGenerate` is read through `FULL_CHECKPOINT_SOURCES`.
Set `config_class = Qwen2VLTextConfig`.

```python
from zeromodels.models.qwen2_vl import Qwen2VLTextGenerate, Qwen2VLTokenizer

model = Qwen2VLTextGenerate.from_weights("zeromodels/qwen2-vl-2b")
tokenizer = Qwen2VLTokenizer.from_weights("zeromodels/qwen2-vl-2b")
outputs = model.generate(**tokenizer("Who wrote Dune?"), max_new_tokens=32)
print(tokenizer.decode(outputs[0]))
```

### `Qwen2VLTextModel`

Qwen2 causal decoder: ``embed -> num_layers x Qwen2VLDecoderLayer -> RMSNorm``.

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

### `Qwen2VLVisionModel`

Qwen2-VL vision tower: patch-embed -> rotary blocks -> 2x2 merger.

| Arg | Default | Meaning |
|---|---|---|
| `embed_dim` | required | text model width |
| `depth` | required | vision tower depth |
| `num_heads` | required | query heads |
| `llm_hidden_size` | required |  |
| `mlp_ratio` | `4` |  |
| `spatial_merge_size` | `2` | patch-merge factor before the decoder |

### `Qwen2VLImageProcessor`

Turn PIL/array images into ``{"pixel_values", "image_grid_thw"}``.

| Arg | Default | Meaning |
|---|---|---|
| `patch_size` | `14` | patch size |
| `spatial_merge_size` | `2` | patch-merge factor before the decoder |
| `temporal_patch_size` | `2` | frames per temporal patch |
| `min_pixels` | `3136` | smallest allowed pixel budget |
| `max_pixels` | `1003520` | largest allowed pixel budget |
| `image_mean` | `(0.48145466, 0.4578275, 0.40821073)` | per-channel normalization mean |
| `image_std` | `(0.26862954, 0.26130258, 0.27577711)` | per-channel normalization std |

### `Qwen2VLTokenizer`

Qwen2 BPE tokenizer (``tokenizers`` backend).

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `None` | Hub repo to pull tokenizer/processor files from |
| `tokenizer_file` | `None` | explicit path to a `tokenizer.json` |

### `Qwen2VLProcessor`

Image + text -> model inputs for the Qwen-VL models.

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `None` | Hub repo to pull tokenizer/processor files from |
| `patch_size` | `14` | patch size |
| `spatial_merge_size` | `2` | patch-merge factor before the decoder |
| `temporal_patch_size` | `2` | frames per temporal patch |
| `tokenizer` | `None` | override the default tokenizer |
| `image_processor` | `None` | override the default image processor |

## End-to-end example

### Single input (image + text)

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.models.qwen2_vl import Qwen2VLConditionalGenerate, Qwen2VLProcessor

model = Qwen2VLConditionalGenerate.from_weights("zeromodels/qwen2-vl-2b")
processor = Qwen2VLProcessor.from_weights("zeromodels/qwen2-vl-2b")

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

For text-only generation use `Qwen2VLTextGenerate`, which drops the vision tower and
loads just the language model. `Qwen2VLTokenizer` encodes raw text (no chat template, so
pass a prompt you have rendered yourself).

```python
from zeromodels.models.qwen2_vl import Qwen2VLTextGenerate, Qwen2VLTokenizer

model = Qwen2VLTextGenerate.from_weights("zeromodels/qwen2-vl-2b")
tokenizer = Qwen2VLTokenizer.from_weights("zeromodels/qwen2-vl-2b")
outputs = model.generate(**tokenizer("Who wrote Dune?"), max_new_tokens=32)
print(tokenizer.decode(outputs[0]))
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = Qwen2VLConditionalGenerate.from_weights(
    "zeromodels/qwen2-vl-2b", quantization="int8", load_dtype="bfloat16"
)
```
