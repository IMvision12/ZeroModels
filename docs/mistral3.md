# Mistral 3

<div class="kf-note kf-note--convert">
<b>On-the-fly conversion:</b> these weights are <b>not</b> mirrored as preconverted
<code>.weights.h5</code> under <code>zeromodels/</code>.
<code>from_weights("&lt;variant&gt;")</code> downloads the original safetensors
from the Hub and converts them in process on every load, because checkpoints this large are
impractical to re-host.
Pass <code>cache_converted=True</code> to keep the converted result and skip the download and
conversion next time. See <a href="../loading_weights/">Loading Weights</a>.
</div>

Mistral AI's vision-language models, ported to pure Keras 3. A Pixtral-style
native-resolution vision tower feeds a patch-merging projector
(`spatial_merge_size`) into the Mistral text decoder. Images are resized so the
longest edge fits `longest_edge` while staying on the patch grid.

Links:

- HF docs: [transformers/model_doc/mistral3](https://huggingface.co/docs/transformers/model_doc/mistral3)

See also [mistral.md](mistral.md), [mixtral.md](mixtral.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `mistral-small-3.1-24b-instruct` | [`mistralai/Mistral-Small-3.1-24B-Instruct-2503`](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) |
| `mistral-small-3.2-24b-instruct` | [`mistralai/Mistral-Small-3.2-24B-Instruct-2506`](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506) |

## API

### `Mistral3Model`

Mistral 3 multimodal backbone (Mistral Small 3.1/3.2): Pixtral vision tower + 2x2 patch-merging projector + Mistral text decoder.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `131072` | token vocabulary size |
| `embed_dim` | `5120` | text model width |
| `mlp_dim` | `32768` | MLP inner width |
| `num_layers` | `40` | decoder blocks |
| `num_heads` | `32` | query heads |
| `num_kv_heads` | `8` | key/value heads (GQA) |
| `head_dim` | `128` | per-head width |
| `norm_eps` | `1e-05` | normalization epsilon |
| `rope_theta` | `1000000000.0` | rotary base frequency |
| `tie_embeddings` | `False` | reuse embeddings as the LM head |
| `vision_embed_dim` | `1024` | vision tower width |
| `vision_mlp_dim` | `4096` | vision MLP width |
| `vision_num_layers` | `24` | vision tower depth |
| `vision_num_heads` | `16` | vision attention heads |
| `image_size` | `1540` | expected image resolution |
| `patch_size` | `14` | patch size |
| `vision_rope_theta` | `10000.0` | rotary base in the vision tower |
| `spatial_merge_size` | `2` | patch-merge factor before the decoder |
| `projector_bias` | `False` |  |
| `image_token_id` | `10` | placeholder token id expanded per image |

### `Mistral3ConditionalGenerate`

Mistral 3 with an LM head + fast ``.generate()`` (image+text -> text).

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

### `Mistral3TextModel`

Mistral causal decoder: ``embed -> num_layers x Mistral3DecoderLayer -> RMSNorm``.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | required | token vocabulary size |
| `embed_dim` | required | text model width |
| `mlp_dim` | required | MLP inner width |
| `num_layers` | required | decoder blocks |
| `num_heads` | required | query heads |
| `num_kv_heads` | required | key/value heads (GQA) |
| `head_dim` | `None` | per-head width |
| `norm_eps` | `1e-05` | normalization epsilon |

### `Mistral3VisionModel`

Pixtral vision tower: conv patch embed -> RMS pre-norm -> 2D-rotary blocks, over a packed variable-resolution patch sequence.

| Arg | Default | Meaning |
|---|---|---|
| `embed_dim` | required | text model width |
| `mlp_dim` | required | MLP inner width |
| `num_layers` | required | decoder blocks |
| `num_heads` | required | query heads |
| `image_size` | `1540` | expected image resolution |
| `patch_size` | `14` | patch size |
| `rope_theta` | `10000.0` | rotary base frequency |

### `Mistral3ImageProcessor`

Pixtral variable-resolution image processor (Mistral 3 recipe).

| Arg | Default | Meaning |
|---|---|---|
| `longest_edge` | `1540` | longest allowed image edge |
| `patch_size` | `14` | patch size |
| `image_mean` | `(0.48145466, 0.4578275, 0.40821073)` | per-channel normalization mean |
| `image_std` | `(0.26862954, 0.26130258, 0.27577711)` | per-channel normalization std |

### `Mistral3Tokenizer`

Mistral 3 Tekken tokenizer (``tokenizers`` backend).

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `None` | Hub repo to pull tokenizer/processor files from |
| `tokenizer_file` | `None` | explicit path to a `tokenizer.json` |

### `Mistral3Processor`

Image + text -> model inputs for the Mistral 3 (Small 3.1/3.2) models.

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `None` | Hub repo to pull tokenizer/processor files from |
| `patch_size` | `14` | patch size |
| `spatial_merge_size` | `2` | patch-merge factor before the decoder |
| `tokenizer` | `None` | override the default tokenizer |
| `image_processor` | `None` | override the default image processor |

## End-to-end example

### Single input (image + text)

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.models.mistral3 import Mistral3ConditionalGenerate, Mistral3Processor

model = Mistral3ConditionalGenerate.from_weights("mistral-small-3.1-24b-instruct")
processor = Mistral3Processor.from_weights("mistral-small-3.1-24b-instruct")

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

`Mistral3Tokenizer` encodes raw text: it has no chat template, so pass a prompt you
have rendered yourself (or go through the processor above).

```python
from zeromodels.models.mistral3 import Mistral3Tokenizer

tokenizer = Mistral3Tokenizer.from_weights("mistral-small-3.1-24b-instruct")
inputs = tokenizer("Who wrote Dune?")
outputs = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = Mistral3ConditionalGenerate.from_weights(
    "mistral-small-3.1-24b-instruct",
    quantization="int8",
    load_dtype="bfloat16",
)
```
