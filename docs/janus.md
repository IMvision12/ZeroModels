# Janus-Pro

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>kf_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

DeepSeek's Janus-Pro unified multimodal models, ported to pure Keras 3. A SigLIP
tower (with exact-gelu, matching the reference) and an MLP connector feed a
DeepSeek text decoder.

Scope note: this port covers the **understanding** path (image + text -> text).
Janus's VQ-VAE image-generation branch is not ported.

Links:

- Paper: [Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- HF docs: [transformers/model_doc/janus](https://huggingface.co/docs/transformers/model_doc/janus)

See also [deepseek_vl.md](deepseek_vl.md).

## Variants

Load any of these with `from_weights("zeromodels/<variant>")`.

| Variant | Hub |
|---|---|
| `janus_pro_1b` | [`zeromodels/janus_pro_1b`](https://huggingface.co/zeromodels/janus_pro_1b) |
| `janus_pro_7b` | [`zeromodels/janus_pro_7b`](https://huggingface.co/zeromodels/janus_pro_7b) |

## API

### `JanusModel`

Janus-Pro multimodal understanding backbone: SigLIP-style tower + depth-2 GELU aligner + Llama decoder.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `102400` | token vocabulary size |
| `embed_dim` | `2048` | text model width |
| `mlp_dim` | `5632` | MLP inner width |
| `num_layers` | `24` | decoder blocks |
| `num_heads` | `16` | query heads |
| `num_kv_heads` | `16` | key/value heads (GQA) |
| `head_dim` | `128` | per-head width |
| `norm_eps` | `1e-06` | normalization epsilon |
| `rope_theta` | `10000.0` | rotary base frequency |
| `tie_embeddings` | `False` | reuse embeddings as the LM head |
| `vision_embed_dim` | `1024` | vision tower width |
| `vision_mlp_dim` | `4096` | vision MLP width |
| `vision_num_layers` | `24` | vision tower depth |
| `vision_num_heads` | `16` | vision attention heads |
| `image_size` | `384` | expected image resolution |
| `patch_size` | `16` | patch size |
| `vision_norm_eps` | `1e-06` | vision tower norm epsilon |
| `image_token_id` | `100581` | placeholder token id expanded per image |

### `JanusConditionalGenerate`

DeepSeek-VL with an LM head + fast ``.generate()`` (image+text -> text).

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

### `JanusVisionModel`

SigLIP vision tower: biased conv patch embed + learned position embeddings -> pre-LN encoder blocks (exact gelu) -> final LayerNorm.

| Arg | Default | Meaning |
|---|---|---|
| `embed_dim` | required | text model width |
| `mlp_dim` | required | MLP inner width |
| `num_layers` | required | decoder blocks |
| `num_heads` | required | query heads |
| `image_size` | `384` | expected image resolution |
| `patch_size` | `16` | patch size |
| `norm_eps` | `1e-06` | normalization epsilon |

### `JanusImageProcessor`

Preprocess images for Janus-Pro.

| Arg | Default | Meaning |
|---|---|---|
| `size` | `384` | target resolution |
| `min_size` | `14` | smallest allowed edge |
| `background_color` | `(127, 127, 127)` | pad colour for letterboxing |
| `image_mean` | `(0.5, 0.5, 0.5)` | per-channel normalization mean |
| `image_std` | `(0.5, 0.5, 0.5)` | per-channel normalization std |
| `data_format` | `None` | `channels_last` or `channels_first` |

### `JanusProcessor`

Image + text -> model inputs for Janus-Pro (understanding path).

| Arg | Default | Meaning |
|---|---|---|
| `variant` | `None` | variant whose tokenizer/processor files to fetch |
| `hf_id` | `None` | Hub repo to pull tokenizer/processor files from |
| `num_image_tokens` | `576` | tokens each image expands to |
| `use_default_system_prompt` | `True` |  |
| `tokenizer` | `None` | override the default tokenizer |
| `image_processor` | `None` | override the default image processor |

### `JanusTokenizer`

Janus-Pro BPE tokenizer (``tokenizers`` backend).

| Arg | Default | Meaning |
|---|---|---|
| `variant` | `None` | variant whose tokenizer/processor files to fetch |
| `hf_id` | `None` | Hub repo to pull tokenizer/processor files from |
| `tokenizer_file` | `None` | explicit path to a `tokenizer.json` |

## End-to-end example

### Single input (image + text)

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.models.janus import JanusConditionalGenerate, JanusProcessor

model = JanusConditionalGenerate.from_weights("zeromodels/janus_pro_1b")
processor = JanusProcessor.from_weights("zeromodels/janus_pro_1b")

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

`JanusTokenizer` encodes raw text: it has no chat template, so pass a prompt you
have rendered yourself (or go through the processor above).

```python
from zeromodels.models.janus import JanusTokenizer

tokenizer = JanusTokenizer.from_weights("zeromodels/janus_pro_1b")
inputs = tokenizer("Who wrote Dune?")
outputs = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = JanusConditionalGenerate.from_weights(
    "zeromodels/janus_pro_1b",
    quantization="int8",
    load_dtype="bfloat16",
)
```
