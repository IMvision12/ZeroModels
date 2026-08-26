# Kimi K2.5

<div class="kf-note kf-note--convert">
<b>On-the-fly conversion:</b> these weights are <b>not</b> mirrored as preconverted
<code>.weights.h5</code> under <code>zeromodels/</code>.
<code>from_weights("&lt;variant&gt;")</code> downloads the original safetensors
from the Hub and converts them in process on every load, because checkpoints this large are
impractical to re-host.
Pass <code>cache_converted=True</code> to keep the converted result and skip the download and
conversion next time. See <a href="../loading_weights/">Loading Weights</a>.
</div>

Moonshot's Kimi K2.5 / K2.6 / K2.7 vision-language models, ported to pure Keras 3.
All three share one architecture and tokenizer: a MoonViT native-resolution
vision tower (with temporal support for video) and a DeepSeek-V3-style text
decoder, so MLA and DeepSeekMoE apply here too.

The learned position grid is bicubic-interpolated with the cubic math spelled out
in `keras.ops`, because `ops.image.resize` is backend-divergent for bicubic.
Memory is governed by **total** parameters, not active ones.


See also [deepseek_v3.md](deepseek_v3.md).

## Variants

Load any of these with `from_weights("<variant>")`.

| Variant | Hub |
|---|---|
| `kimi-k2.5` | [`moonshotai/Kimi-K2.5`](https://huggingface.co/moonshotai/Kimi-K2.5) |
| `kimi-k2.6` | [`moonshotai/Kimi-K2.6`](https://huggingface.co/moonshotai/Kimi-K2.6) |
| `kimi-k2.7-code` | [`moonshotai/Kimi-K2.7-Code`](https://huggingface.co/moonshotai/Kimi-K2.7-Code) |

## API

### `KimiK25Model`

Kimi K2.5 / K2.6 / K2.7-Code: MoonViT + DeepSeek-V3 MoE decoder.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `163840` | token vocabulary size |
| `embed_dim` | `7168` | text model width |
| `num_layers` | `61` | decoder blocks |
| `num_heads` | `64` | query heads |
| `mlp_dim` | `18432` | MLP inner width |
| `moe_mlp_dim` | `2048` | per-expert inner width |
| `num_experts` | `384` | expert count |
| `num_experts_per_tok` | `8` | experts routed per token |
| `n_shared_experts` | `1` | always-on shared experts |
| `n_group` | `1` | routing groups (node-limited routing) |
| `topk_group` | `1` | groups kept per token |
| `norm_topk_prob` | `True` | renormalize the top-k router weights |
| `routed_scaling_factor` | `2.827` | scale applied to routed-expert output |
| `first_k_dense` | `1` | leading layers left dense instead of MoE |
| `q_lora_rank` | `1536` | query bottleneck rank (MLA) |
| `kv_lora_rank` | `512` | key/value bottleneck rank (MLA) |
| `qk_nope_head_dim` | `128` | non-rotary part of each head (MLA) |
| `qk_rope_head_dim` | `64` | rotary part of each head (MLA) |
| `v_head_dim` | `128` | value head width (MLA) |
| `rope_theta` | `50000.0` | rotary base frequency |
| `rope_scaling` | `None` |  |
| `norm_eps` | `1e-05` | normalization epsilon |
| `max_position_embeddings` | `262144` | longest position index the model builds |
| `tie_embeddings` | `False` | reuse embeddings as the LM head |
| `vision_embed_dim` | `1152` | vision tower width |
| `vision_depth` | `27` | vision tower depth |
| `vision_num_heads` | `16` | vision attention heads |
| `vision_mlp_dim` | `4304` | vision MLP width |
| `vision_patch_size` | `14` | vision patch size |
| `pos_emb_height` | `64` |  |
| `pos_emb_width` | `64` |  |
| `pos_emb_time` | `4` |  |
| `merge_kernel` | `(2, 2)` | patch-merge kernel |
| `vision_rope_theta` | `10000.0` | rotary base in the vision tower |
| `projection_hidden_size` | `1152` |  |
| `projection_norm_eps` | `1e-05` |  |
| `image_token_id` | `163605` | placeholder token id expanded per image |
| `video_token_id` | `163840` | placeholder token id expanded per video |
| `vision_start_token_id` | `163602` | token id opening a vision span |
| `vision_end_token_id` | `163604` | token id closing a vision span |

### `KimiK25ConditionalGenerate`

Kimi K2.5 with an LM head + fast ``.generate()`` (image/video+text -> text).

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

### `KimiK25VisionModel`

MoonViT with a temporal axis: native-resolution packed ViT for images and video.

| Arg | Default | Meaning |
|---|---|---|
| `embed_dim` | `1152` | text model width |
| `depth` | `27` | vision tower depth |
| `num_heads` | `16` | query heads |
| `mlp_dim` | `4304` | MLP inner width |
| `patch_size` | `14` | patch size |
| `pos_emb_height` | `64` |  |
| `pos_emb_width` | `64` |  |
| `pos_emb_time` | `4` |  |
| `merge_kernel` | `(2, 2)` | patch-merge kernel |
| `in_channels` | `3` | input image channels |
| `rope_theta` | `10000.0` | rotary base frequency |

### `KimiK25MultimodalProjection`

Patch-merger projector: norm over the ViT width, then a two-layer GELU MLP.

| Arg | Default | Meaning |
|---|---|---|
| `mm_dim` | required |  |
| `embed_dim` | required | text model width |
| `norm_eps` | `1e-05` | normalization epsilon |

### `KimiK25Tokenizer`

Kimi K2.5 / K2.6 / K2.7-Code tiktoken tokenizer (163840 tokens).

| Arg | Default | Meaning |
|---|---|---|
| `vocab_file` | `None` |  |
| `hf_id` | `None` | Hub repo to pull tokenizer/processor files from |

### `KimiK25ImageProcessor`

Native-resolution (NaViT) patch preprocessor for Kimi K2.5's MoonViT.

| Arg | Default | Meaning |
|---|---|---|
| `patch_size` | `14` | patch size |
| `merge_size` | `2` | patch-merge factor |
| `max_patches` | `16384` | most tiles per image |
| `max_side` | `512` |  |
| `image_mean` | `(0.5, 0.5, 0.5)` | per-channel normalization mean |
| `image_std` | `(0.5, 0.5, 0.5)` | per-channel normalization std |

### `KimiK25Processor`

Text + image processor for Kimi K2.5 / K2.6 / K2.7-Code.

| Arg | Default | Meaning |
|---|---|---|
| `tokenizer` | `None` | override the default tokenizer |
| `image_processor` | `None` | override the default image processor |

## End-to-end example

### Single input (image + text)

`KimiK25Processor` takes an already-rendered prompt rather than a message list:
it does not apply a chat template. The prompt must carry one
`<|media_pad|>` marker per image, which the processor expands
into the right number of patch tokens.

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.models.kimi_k25 import KimiK25ConditionalGenerate, KimiK25Processor

model = KimiK25ConditionalGenerate.from_weights("kimi-k2.5")
processor = KimiK25Processor.from_weights("kimi-k2.5")

image = Image.open("photo.jpg")
prompt = (
    "<|im_user|>user<|im_middle|>"
    "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
    "Describe this image in one sentence.<|im_end|>"
    "<|im_assistant|>assistant<|im_middle|>"
)
inputs = processor(text=prompt, images=[image])
outputs = model.generate(**inputs, max_new_tokens=64)

print(processor.decode(outputs[0]))
```

The marker layout above is the checkpoint's own chat template. It appends an
open `<think>` block after `<|im_assistant|>assistant<|im_middle|>`; leave that
out (as here) for a direct answer.

### Batch

Pass a list of prompts and the matching images. Each prompt takes the images
its own markers claim, the processor pads them, and `generate` runs the batch
together:

```python
def build_prompt(question):
    return (
        "<|im_user|>user<|im_middle|>"
        "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
        f"{question}<|im_end|>"
        "<|im_assistant|>assistant<|im_middle|>"
    )


prompts = [
    build_prompt("What is in this image?"),
    build_prompt("Describe the colours."),
]
images = [Image.open("a.jpg"), Image.open("b.jpg")]
inputs = processor(text=prompts, images=images)
outputs = model.generate(**inputs, max_new_tokens=64)

for text in processor.batch_decode(outputs):
    print(text)
```

### Text only

`KimiK25Tokenizer` encodes raw text: it has no chat template, so pass a prompt you
have rendered yourself (or go through the processor above).

```python
from zeromodels.models.kimi_k25 import KimiK25Tokenizer

tokenizer = KimiK25Tokenizer.from_weights("kimi-k2.5")
inputs = tokenizer("Who wrote Dune?")
outputs = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = KimiK25ConditionalGenerate.from_weights(
    "kimi-k2.5", quantization="int8", load_dtype="bfloat16"
)
```
