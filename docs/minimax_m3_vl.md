# MiniMax-M3-VL

<div class="kf-note kf-note--convert">
<b>On-the-fly conversion:</b> these weights are <b>not</b> mirrored as preconverted
<code>.weights.h5</code> under <code>zeromodels/</code>.
<code>from_weights("&lt;variant&gt;")</code> downloads the original safetensors
from the Hub and converts them in process on every load, because checkpoints this large are
impractical to re-host.
Pass <code>cache_converted=True</code> to keep the converted result and skip the download and
conversion next time. See <a href="../loading_weights/">Loading Weights</a>.
</div>

MiniMax's M3 vision-language model, ported to pure Keras 3. A native-resolution
vision tower and patch-merging projector feed a MiniMax sparse decoder.

Memory is governed by **total** parameters, not active ones.


See also [minimax.md](minimax.md), [minimax_m2.md](minimax_m2.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `minimax-m3` | [`MiniMaxAI/MiniMax-M3`](https://huggingface.co/MiniMaxAI/MiniMax-M3) |

## API

### `MiniMaxM3VLModel`

MiniMax-M3 vision-language backbone (MiniMaxAI/MiniMax-M3).

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `200064` | token vocabulary size |
| `embed_dim` | `6144` | text model width |
| `mlp_dim` | `3072` | MLP inner width |
| `dense_mlp_dim` | `12288` |  |
| `shared_mlp_dim` | `3072` |  |
| `num_layers` | `60` | decoder blocks |
| `num_heads` | `64` | query heads |
| `num_kv_heads` | `4` | key/value heads (GQA) |
| `head_dim` | `128` | per-head width |
| `num_experts` | `128` | expert count |
| `num_experts_per_tok` | `4` | experts routed per token |
| `routed_scaling_factor` | `2.0` | scale applied to routed-expert output |
| `layer_types` | `None` |  |
| `mlp_layer_types` | `None` |  |
| `index_n_heads` | `4` |  |
| `index_head_dim` | `128` |  |
| `index_block_size` | `128` |  |
| `index_topk_blocks` | `16` |  |
| `index_local_blocks` | `1` |  |
| `swiglu_alpha` | `1.702` |  |
| `swiglu_limit` | `7.0` |  |
| `partial_rotary_factor` | `0.5` | fraction of each head that gets rotated |
| `rope_theta` | `5000000.0` | rotary base frequency |
| `norm_eps` | `1e-06` | normalization epsilon |
| `vision_embed_dim` | `1280` | vision tower width |
| `vision_mlp_dim` | `5120` | vision MLP width |
| `vision_num_layers` | `32` | vision tower depth |
| `vision_num_heads` | `16` | vision attention heads |
| `patch_size` | `14` | patch size |
| `temporal_patch_size` | `2` | frames per temporal patch |
| `spatial_merge_size` | `2` | patch-merge factor before the decoder |
| `vision_rope_theta` | `10000.0` | rotary base in the vision tower |
| `vision_norm_eps` | `1e-05` | vision tower norm epsilon |
| `projector_dim` | `6144` |  |
| `image_token_id` | `200025` | placeholder token id expanded per image |
| `video_token_id` | `200026` | placeholder token id expanded per video |
| `tie_embeddings` | `False` | reuse embeddings as the LM head |

### `MiniMaxM3VLConditionalGenerate`

MiniMax-M3 VL with an LM head + fast ``.generate()`` (image+text -> text).

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

### `MiniMaxM3VLVisionModel`

MiniMax-M3 vision tower: Conv3d-equivalent patch embed + 3D RoPE + CLIP-style pre-LN blocks over the packed patch sequence (no final norm).

| Arg | Default | Meaning |
|---|---|---|
| `embed_dim` | `1280` | text model width |
| `mlp_dim` | `5120` | MLP inner width |
| `num_layers` | `32` | decoder blocks |
| `num_heads` | `16` | query heads |
| `patch_size` | `14` | patch size |
| `temporal_patch_size` | `2` | frames per temporal patch |
| `spatial_merge_size` | `2` | patch-merge factor before the decoder |
| `rope_theta` | `10000.0` | rotary base frequency |
| `norm_eps` | `1e-05` | normalization epsilon |

### `MiniMaxM3VLImageProcessor`

Preprocess images (or video frames) for MiniMax-M3 VL.

| Arg | Default | Meaning |
|---|---|---|
| `patch_size` | `14` | patch size |
| `temporal_patch_size` | `2` | frames per temporal patch |
| `merge_size` | `2` | patch-merge factor |
| `min_pixels` | `3136` | smallest allowed pixel budget |
| `max_pixels` | `451584` | largest allowed pixel budget |
| `image_mean` | `(0.48145466, 0.4578275, 0.40821073)` | per-channel normalization mean |
| `image_std` | `(0.26862954, 0.26130258, 0.27577711)` | per-channel normalization std |

### `MiniMaxM3VLProcessor`

Image / video + text -> model inputs for MiniMax-M3 VL.

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `None` | Hub repo to pull tokenizer/processor files from |
| `tokenizer` | `None` | override the default tokenizer |
| `image_processor` | `None` | override the default image processor |

### `MiniMaxM3VLTokenizer`

MiniMax-M3 BPE tokenizer (``tokenizers`` backend, ~200k vocab).

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `None` | Hub repo to pull tokenizer/processor files from |
| `tokenizer_file` | `None` | explicit path to a `tokenizer.json` |

## End-to-end example

### Single input (image + text)

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.models.minimax_m3_vl import (
    MiniMaxM3VLConditionalGenerate,
    MiniMaxM3VLProcessor,
)

model = MiniMaxM3VLConditionalGenerate.from_weights("minimax-m3")
processor = MiniMaxM3VLProcessor.from_weights("minimax-m3")

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

`MiniMaxM3VLTokenizer` encodes raw text: it has no chat template, so pass a prompt you
have rendered yourself (or go through the processor above).

```python
from zeromodels.models.minimax_m3_vl import MiniMaxM3VLTokenizer

tokenizer = MiniMaxM3VLTokenizer.from_weights("minimax-m3")
inputs = tokenizer("Who wrote Dune?")
outputs = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = MiniMaxM3VLConditionalGenerate.from_weights(
    "minimax-m3", quantization="int8", load_dtype="bfloat16"
)
```
