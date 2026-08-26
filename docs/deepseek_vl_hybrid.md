# DeepSeek-VL Hybrid

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

The dual-tower DeepSeek-VL variant, ported to pure Keras 3. It runs the SigLIP
tower alongside a **high-resolution SAM-style tower** (`high_res_*` arguments)
and fuses both feature streams before the text decoder, which is what lets it
read fine detail such as small text.

Build note: the high-res tower only builds when image inputs are passed, so a
text-only forward leaves those sublayers unbuilt.

Links:

- Paper: [DeepSeek-VL: Towards Real-World Vision-Language Understanding (arXiv:2403.05525)](https://arxiv.org/abs/2403.05525)

See also [deepseek_vl.md](deepseek_vl.md), [janus.md](janus.md).

## Variants

Load any of these with `from_weights("zeromodels/<variant>")`.

| Variant | Hub |
|---|---|
| `deepseek_vl_7b_chat` | [`zeromodels/deepseek_vl_7b_chat`](https://huggingface.co/zeromodels/deepseek_vl_7b_chat) |
| `deepseek_vl_7b_base` | [`zeromodels/deepseek_vl_7b_base`](https://huggingface.co/zeromodels/deepseek_vl_7b_base) |

## API

### `DeepseekVLHybridModel`

DeepSeek-VL Hybrid (7B) backbone: dual vision (SigLIP @384 + SAM @1024) + 3-way aligner + Llama-7B decoder.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `102400` | token vocabulary size |
| `embed_dim` | `4096` | text model width |
| `mlp_dim` | `11008` | MLP inner width |
| `num_layers` | `30` | decoder blocks |
| `num_heads` | `32` | query heads |
| `num_kv_heads` | `32` | key/value heads (GQA) |
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
| `high_res_embed_dim` | `768` |  |
| `high_res_mlp_dim` | `3072` |  |
| `high_res_num_layers` | `12` |  |
| `high_res_num_heads` | `12` |  |
| `high_res_image_size` | `1024` |  |
| `high_res_patch_size` | `16` |  |
| `high_res_output_channels` | `256` |  |
| `high_res_window_size` | `14` |  |
| `high_res_global_attn_indexes` | `(2, 5, 8, 11)` |  |
| `high_res_norm_eps` | `1e-06` |  |
| `image_token_id` | `100015` | placeholder token id expanded per image |

### `DeepseekVLHybridConditionalGenerate`

DeepSeek-VL Hybrid with an LM head + fast ``.generate()``.

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

### `DeepseekVLHybridImageProcessor`

Dual-resolution preprocessing for DeepSeek-VL Hybrid (7B).

| Arg | Default | Meaning |
|---|---|---|
| `size` | `384` | target resolution |
| `high_res_size` | `1024` |  |
| `min_size` | `14` | smallest allowed edge |
| `image_mean` | `(0.5, 0.5, 0.5)` | per-channel normalization mean |
| `image_std` | `(0.5, 0.5, 0.5)` | per-channel normalization std |
| `high_res_image_mean` | `(0.48145466, 0.4578275, 0.40821073)` |  |
| `high_res_image_std` | `(0.26862954, 0.26130258, 0.27577711)` |  |
| `background_color` | `(127, 127, 127)` | pad colour for letterboxing |
| `high_res_background_color` | `(122, 116, 104)` |  |
| `data_format` | `None` | `channels_last` or `channels_first` |

### `DeepseekVLHybridProcessor`

Image + text -> model inputs for DeepSeek-VL Hybrid (7B).

| Arg | Default | Meaning |
|---|---|---|
| `variant` | `None` | variant whose tokenizer/processor files to fetch |
| `hf_id` | `None` | Hub repo to pull tokenizer/processor files from |
| `num_image_tokens` | `576` | tokens each image expands to |
| `tokenizer` | `None` | override the default tokenizer |
| `image_processor` | `None` | override the default image processor |

### `DeepseekVLHybridTokenizer`

DeepSeek-VL Hybrid (7B) tokenizer.

| Arg | Default | Meaning |
|---|---|---|
| `variant` | `None` | variant whose tokenizer/processor files to fetch |
| `hf_id` | `None` | Hub repo to pull tokenizer/processor files from |
| `tokenizer_file` | `None` | explicit path to a `tokenizer.json` |

## Data Format

The model reads `keras.config.image_data_format()` when it is constructed and accepts both
`channels_last` and `channels_first` `pixel_values`. For `channels_first` input, the image is
transposed to `channels_last` at the SAM high-res tower's entry (the "door"), so its
convolutions and resizes run in `channels_last` internally. The soft image tokens are
identical under either layout.

## End-to-end example

### Single input (image + text)

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.models.deepseek_vl_hybrid import (
    DeepseekVLHybridConditionalGenerate,
    DeepseekVLHybridProcessor,
)

model = DeepseekVLHybridConditionalGenerate.from_weights(
    "zeromodels/deepseek_vl_7b_chat"
)
processor = DeepseekVLHybridProcessor.from_weights("zeromodels/deepseek_vl_7b_chat")

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

`DeepseekVLHybridTokenizer` encodes raw text: it has no chat template, so pass a prompt you
have rendered yourself (or go through the processor above).

```python
from zeromodels.models.deepseek_vl_hybrid import DeepseekVLHybridTokenizer

tokenizer = DeepseekVLHybridTokenizer.from_weights("zeromodels/deepseek_vl_7b_chat")
inputs = tokenizer("Who wrote Dune?")
outputs = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = DeepseekVLHybridConditionalGenerate.from_weights(
    "zeromodels/deepseek_vl_7b_chat",
    quantization="int8",
    load_dtype="bfloat16",
)
```
