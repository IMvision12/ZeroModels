# Qwen2.5-VL

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> and <code>zm_preprocessor.json</code> plus the
Keras weights: <code>model.weights.h5</code>, or a sharded <code>model.weights.json</code> +
shards for the larger checkpoints). Load the model and processor with
<code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Alibaba's Qwen2.5-VL vision-language models, ported to pure Keras 3. It keeps
Qwen2-VL's native-resolution ViT and M-RoPE decoder, and reuses that family's
processor and image processor directly.

Parity trap from the port: `rms_norm_eps` defaults to 1e-5 while several
checkpoints specify 1e-6. The value is read from the checkpoint config, so do
not override it by hand.

Links:

- HF collection: [Qwen2.5-VL](https://huggingface.co/collections/zeromodels/qwen25-vl-6a7cc9f463d6956b6c3ba911)
- Paper: [Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- HF docs: [transformers/model_doc/qwen2_5_vl](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl)

See also [qwen2_vl.md](qwen2_vl.md), [qwen3_vl.md](qwen3_vl.md).

## Variants

Load any of these with `from_weights("zeromodels/<variant>")`.

| Variant | Hub |
|---|---|
| `qwen2.5-vl-3b-instruct` | [`zeromodels/qwen2.5-vl-3b-instruct`](https://huggingface.co/zeromodels/qwen2.5-vl-3b-instruct) |
| `qwen2.5-vl-7b-instruct` | [`zeromodels/qwen2.5-vl-7b-instruct`](https://huggingface.co/zeromodels/qwen2.5-vl-7b-instruct) |
| `qwen2.5-vl-32b-instruct` | [`zeromodels/qwen2.5-vl-32b-instruct`](https://huggingface.co/zeromodels/qwen2.5-vl-32b-instruct) |
| `qwen2.5-vl-72b-instruct` | [`zeromodels/qwen2.5-vl-72b-instruct`](https://huggingface.co/zeromodels/qwen2.5-vl-72b-instruct) |

## API

### `Qwen2_5VLModel`

Qwen2.5-VL multimodal backbone: windowed vision tower + Qwen2.5 decoder.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `151936` | token vocabulary size |
| `embed_dim` | `2048` | text model width |
| `mlp_dim` | `11008` | MLP inner width |
| `num_layers` | `36` | decoder blocks |
| `num_heads` | `16` | query heads |
| `num_kv_heads` | `2` | key/value heads (GQA) |
| `norm_eps` | `1e-06` | normalization epsilon |
| `rope_theta` | `1000000.0` | rotary base frequency |
| `mrope_section` | `(16, 24, 24)` | M-RoPE split across time/height/width |
| `tie_embeddings` | `True` | reuse embeddings as the LM head |
| `vision_depth` | `32` | vision tower depth |
| `vision_embed_dim` | `1280` | vision tower width |
| `vision_mlp_dim` | `3420` | vision MLP width |
| `vision_num_heads` | `16` | vision attention heads |
| `vision_out_dim` | `None` | projector output width (matches the decoder) |
| `window_size` | `112` |  |
| `fullatt_block_indexes` | `(7, 15, 23, 31)` |  |
| `tokens_per_second` | `2` |  |
| `patch_size` | `14` | patch size |
| `spatial_merge_size` | `2` | patch-merge factor before the decoder |
| `temporal_patch_size` | `2` | frames per temporal patch |
| `in_channels` | `3` | input image channels |
| `image_token_id` | `151655` | placeholder token id expanded per image |
| `video_token_id` | `151656` | placeholder token id expanded per video |
| `vision_start_token_id` | `151652` | token id opening a vision span |
| `vision_end_token_id` | `151653` | token id closing a vision span |

### `Qwen2_5VLConditionalGenerate`

Qwen2.5-VL with an LM head + fast ``.generate()`` (image+text -> text).

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

### `Qwen2_5VLTextGenerate`

Text-only counterpart of `Qwen2_5VLConditionalGenerate`, built with no vision tower
(`build_vision=False`), so `.generate()` takes just token ids. It reads only the language
model out of a Qwen2.5-VL checkpoint: `hf:` conversion copies just the text weights, and a
zeromodels repo declaring `Qwen2_5VLConditionalGenerate` is read through
`FULL_CHECKPOINT_SOURCES`. Qwen2.5-VL ships no dedicated tokenizer class, so drive text
through `Qwen2_5VLProcessor` with a text-only conversation. Set
`config_class = Qwen2_5VLTextConfig`.

```python
from zeromodels.models.qwen2_5_vl import Qwen2_5VLTextGenerate, Qwen2_5VLProcessor

model = Qwen2_5VLTextGenerate.from_weights("zeromodels/qwen2.5-vl-3b-instruct")
processor = Qwen2_5VLProcessor.from_weights("zeromodels/qwen2.5-vl-3b-instruct")
conversation = [
    {"role": "user", "content": [{"type": "text", "text": "Who wrote Dune?"}]},
]
inputs = processor(conversation)
outputs = model.generate(**inputs, max_new_tokens=32)
print(processor.decode(outputs[0]))
```

### `Qwen2_5VLTextModel`

Qwen2.5 causal decoder: ``embed -> N x Qwen2_5VLDecoderLayer -> RMSNorm``.

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

### `Qwen2_5VLVisionModel`

Qwen2.5-VL vision tower: patch-embed -> windowed blocks -> 2x2 merger.

| Arg | Default | Meaning |
|---|---|---|
| `embed_dim` | required | text model width |
| `depth` | required | vision tower depth |
| `num_heads` | required | query heads |
| `intermediate_size` | required | MLP inner width |
| `out_hidden_size` | required | projector output width |
| `window_size` | required |  |
| `fullatt_block_indexes` | required |  |
| `patch_size` | `14` | patch size |
| `spatial_merge_size` | `2` | patch-merge factor before the decoder |

### `Qwen2_5VLProcessor`

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
from zeromodels.models.qwen2_5_vl import (
    Qwen2_5VLConditionalGenerate,
    Qwen2_5VLProcessor,
)

model = Qwen2_5VLConditionalGenerate.from_weights("zeromodels/qwen2.5-vl-3b-instruct")
processor = Qwen2_5VLProcessor.from_weights("zeromodels/qwen2.5-vl-3b-instruct")

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

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = Qwen2_5VLConditionalGenerate.from_weights(
    "zeromodels/qwen2.5-vl-3b-instruct",
    quantization="int8",
    load_dtype="bfloat16",
)
```
