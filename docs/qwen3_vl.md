# Qwen3-VL

<div class="kf-note kf-note--weights">
<b>Weights:</b> the eight dense sizes (2B / 4B / 8B / 32B, each Instruct + Thinking)
are hosted as pretrained Keras weights on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>kf_config.json</code> and <code>kf_preprocessor.json</code> plus a
sharded <code>model.weights.json</code> + shards and a <code>tokenizer.json</code>). Load the
model and processor with <code>from_weights("zeromodels/&lt;variant&gt;")</code>. The MoE
sizes (30B-A3B / 235B-A22B) are a separate architecture and are not hosted here.
</div>

Alibaba's Qwen3-VL vision-language models, ported to pure Keras 3. It follows the
Qwen-VL line (native-resolution ViT, M-RoPE decoder) with a 16px patch.

Links:

- HF collection: [Qwen3-VL](https://huggingface.co/collections/zeromodels/qwen3-vl-6a7d7677c2926ecbddb1ed0a)
- Paper: [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388)
- HF docs: [transformers/model_doc/qwen3_vl](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)

See also [qwen2_vl.md](qwen2_vl.md), [qwen2_5_vl.md](qwen2_5_vl.md).

## Variants

Preconverted, bf16 weights are hosted under `zeromodels/`. Load the model and processor
with `from_weights("zeromodels/<variant>")`; the `-instruct` sizes are the
instruction-tuned checkpoints and `-thinking` the reasoning checkpoints. Qwen3-VL is
Apache 2.0.

| Variant | Hub |
|---|---|
| `qwen3-vl-2b-instruct` | [`zeromodels/qwen3-vl-2b-instruct`](https://huggingface.co/zeromodels/qwen3-vl-2b-instruct) |
| `qwen3-vl-2b-thinking` | [`zeromodels/qwen3-vl-2b-thinking`](https://huggingface.co/zeromodels/qwen3-vl-2b-thinking) |
| `qwen3-vl-4b-instruct` | [`zeromodels/qwen3-vl-4b-instruct`](https://huggingface.co/zeromodels/qwen3-vl-4b-instruct) |
| `qwen3-vl-4b-thinking` | [`zeromodels/qwen3-vl-4b-thinking`](https://huggingface.co/zeromodels/qwen3-vl-4b-thinking) |
| `qwen3-vl-8b-instruct` | [`zeromodels/qwen3-vl-8b-instruct`](https://huggingface.co/zeromodels/qwen3-vl-8b-instruct) |
| `qwen3-vl-8b-thinking` | [`zeromodels/qwen3-vl-8b-thinking`](https://huggingface.co/zeromodels/qwen3-vl-8b-thinking) |
| `qwen3-vl-32b-instruct` | [`zeromodels/qwen3-vl-32b-instruct`](https://huggingface.co/zeromodels/qwen3-vl-32b-instruct) |
| `qwen3-vl-32b-thinking` | [`zeromodels/qwen3-vl-32b-thinking`](https://huggingface.co/zeromodels/qwen3-vl-32b-thinking) |

Upstream Qwen safetensors also load directly via the `hf:` prefix, e.g.
`from_weights("hf:Qwen/Qwen3-VL-4B-Instruct")`, which converts them in process (pass
`cache_converted=True` to keep the result). See [Loading Weights](loading_weights.md).

## API

### `Qwen3VLModel`

Qwen3-VL multimodal backbone: vision tower + Qwen3 decoder + DeepStack.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `151936` | token vocabulary size |
| `embed_dim` | `2048` | text model width |
| `mlp_dim` | `6144` | MLP inner width |
| `num_layers` | `28` | decoder blocks |
| `num_heads` | `16` | query heads |
| `num_kv_heads` | `8` | key/value heads (GQA) |
| `head_dim` | `128` | per-head width |
| `norm_eps` | `1e-06` | normalization epsilon |
| `rope_theta` | `5000000.0` | rotary base frequency |
| `mrope_section` | `(24, 20, 20)` | M-RoPE split across time/height/width |
| `tie_embeddings` | `True` | reuse embeddings as the LM head |
| `vision_depth` | `24` | vision tower depth |
| `vision_embed_dim` | `1024` | vision tower width |
| `vision_mlp_dim` | `4096` | vision MLP width |
| `vision_num_heads` | `16` | vision attention heads |
| `vision_out_dim` | `None` | projector output width (matches the decoder) |
| `vision_act` | `'gelu_pytorch_tanh'` |  |
| `num_position_embeddings` | `2304` | learned position grid size |
| `deepstack_visual_indexes` | `(5, 11, 17)` | vision blocks that feed a DeepStack merger |
| `patch_size` | `16` | patch size |
| `spatial_merge_size` | `2` | patch-merge factor before the decoder |
| `temporal_patch_size` | `2` | frames per temporal patch |
| `in_channels` | `3` | input image channels |
| `image_token_id` | `151655` | placeholder token id expanded per image |
| `video_token_id` | `151656` | placeholder token id expanded per video |
| `vision_start_token_id` | `151652` | token id opening a vision span |
| `vision_end_token_id` | `151653` | token id closing a vision span |

### `Qwen3VLConditionalGenerate`

Qwen3-VL with an LM head + fast ``.generate()`` (image+text -> text).

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `151936` | token vocabulary size |
| `embed_dim` | `2048` | text model width |
| `mlp_dim` | `6144` | MLP inner width |
| `num_layers` | `28` | decoder blocks |
| `num_heads` | `16` | query heads |
| `num_kv_heads` | `8` | key/value heads (GQA) |
| `head_dim` | `128` | per-head width |
| `norm_eps` | `1e-06` | normalization epsilon |
| `rope_theta` | `5000000.0` | rotary base frequency |
| `mrope_section` | `(24, 20, 20)` | M-RoPE split across time/height/width |
| `tie_embeddings` | `True` | reuse embeddings as the LM head |
| `vision_depth` | `24` | vision tower depth |
| `vision_embed_dim` | `1024` | vision tower width |
| `vision_mlp_dim` | `4096` | vision MLP width |
| `vision_num_heads` | `16` | vision attention heads |
| `vision_out_dim` | `None` | projector output width (matches the decoder) |
| `vision_act` | `'gelu_pytorch_tanh'` |  |
| `num_position_embeddings` | `2304` | learned position grid size |
| `deepstack_visual_indexes` | `(5, 11, 17)` | vision blocks that feed a DeepStack merger |
| `patch_size` | `16` | patch size |
| `spatial_merge_size` | `2` | patch-merge factor before the decoder |
| `temporal_patch_size` | `2` | frames per temporal patch |
| `in_channels` | `3` | input image channels |
| `image_token_id` | `151655` | placeholder token id expanded per image |
| `video_token_id` | `151656` | placeholder token id expanded per video |
| `vision_start_token_id` | `151652` | token id opening a vision span |
| `vision_end_token_id` | `151653` | token id closing a vision span |

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

### `Qwen3VLTextGenerate`

Text-only counterpart of `Qwen3VLConditionalGenerate`, built with no vision tower
(`build_vision=False`), so `.generate()` takes just token ids. It reads only the language
model out of a Qwen3-VL checkpoint: `hf:` conversion copies just the text weights, and a
zeromodels repo declaring `Qwen3VLConditionalGenerate` is read through
`FULL_CHECKPOINT_SOURCES`. Qwen3-VL ships no dedicated tokenizer class, so drive text
through `Qwen3VLProcessor` with a text-only conversation. Set
`config_class = Qwen3VLTextConfig`.

```python
from zeromodels.models.qwen3_vl import Qwen3VLTextGenerate, Qwen3VLProcessor

model = Qwen3VLTextGenerate.from_weights("zeromodels/qwen3-vl-2b-instruct")
processor = Qwen3VLProcessor.from_weights("zeromodels/qwen3-vl-2b-instruct")
conversation = [
    {"role": "user", "content": [{"type": "text", "text": "Who wrote Dune?"}]},
]
inputs = processor(conversation)
outputs = model.generate(**inputs, max_new_tokens=32)
print(processor.decode(outputs[0]))
```

### `Qwen3VLTextModel`

Qwen3 causal decoder with DeepStack visual-feature injection.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | required | token vocabulary size |
| `embed_dim` | required | text model width |
| `mlp_dim` | required | MLP inner width |
| `num_layers` | required | decoder blocks |
| `num_heads` | required | query heads |
| `num_kv_heads` | required | key/value heads (GQA) |
| `head_dim` | required | per-head width |
| `norm_eps` | `1e-06` | normalization epsilon |

### `Qwen3VLVisionModel`

Qwen3-VL vision tower: learned pos-embeds -> GELU blocks -> merger + DeepStack.

| Arg | Default | Meaning |
|---|---|---|
| `embed_dim` | required | text model width |
| `depth` | required | vision tower depth |
| `num_heads` | required | query heads |
| `intermediate_size` | required | MLP inner width |
| `out_hidden_size` | required | projector output width |
| `num_position_embeddings` | required | learned position grid size |
| `deepstack_visual_indexes` | required | vision blocks that feed a DeepStack merger |
| `hidden_act` | `'gelu_pytorch_tanh'` |  |
| `patch_size` | `16` | patch size |
| `spatial_merge_size` | `2` | patch-merge factor before the decoder |

### `Qwen3VLProcessor`

Qwen3-VL image+text processor: like :class:`Qwen2VLProcessor` but with a 16px patch and the Qwen3-VL image normalization (``[0.5]*3``).

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `'Qwen/Qwen3-VL-2B-Instruct'` | Hub repo to pull tokenizer/processor files from |
| `patch_size` | `16` | patch size |
| `spatial_merge_size` | `2` | patch-merge factor before the decoder |
| `temporal_patch_size` | `2` | frames per temporal patch |

## End-to-end example

### Single input (image + text)

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.models.qwen3_vl import Qwen3VLConditionalGenerate, Qwen3VLProcessor

model = Qwen3VLConditionalGenerate.from_weights("zeromodels/qwen3-vl-2b-instruct")
processor = Qwen3VLProcessor.from_weights("zeromodels/qwen3-vl-2b-instruct")

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
model = Qwen3VLConditionalGenerate.from_weights(
    "zeromodels/qwen3-vl-2b-instruct", quantization="int8", load_dtype="bfloat16"
)
```
