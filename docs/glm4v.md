# GLM-4V

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + a sharded <code>model.weights.json</code>,
stored in <b>bfloat16</b>). Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>,
or convert an original checkpoint on the fly with
<code>from_weights("hf:zai-org/GLM-4.1V-9B-Thinking")</code>. See
<a href="../loading_weights/">Loading Weights</a>.
</div>

Zhipu's GLM-4V vision-language models, ported to pure Keras 3. A ViT tower with
learned position embeddings (bicubic-interpolated to the native grid) feeds the
GLM text decoder, which uses M-RoPE across time/height/width.

The bicubic interpolation of the learned position grid is spelled out in
`keras.ops` rather than using `ops.image.resize`, because that op is
backend-divergent for bicubic and would break parity on jax/tf.

Links:

- Paper: [GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning (arXiv:2507.01006)](https://arxiv.org/abs/2507.01006)
- HF docs: [transformers/model_doc/glm4v](https://huggingface.co/docs/transformers/model_doc/glm4v)

See also [glm4v_moe.md](glm4v_moe.md), [glm4.md](glm4.md).

## Variants

Load any of these with `from_weights("zeromodels/<variant>")` (or convert the
upstream checkpoint on the fly with `from_weights("hf:<upstream>")`).

| Variant | Hosted | Upstream |
|---|---|---|
| `glm-4.1v-9b-thinking` | `zeromodels/glm-4.1v-9b-thinking` | [`zai-org/GLM-4.1V-9B-Thinking`](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking) |
| `glm-4.1v-9b-base` | `zeromodels/glm-4.1v-9b-base` | [`zai-org/GLM-4.1V-9B-Base`](https://huggingface.co/zai-org/GLM-4.1V-9B-Base) |
| `glm-4.6v-flash` | `zeromodels/glm-4.6v-flash` | [`zai-org/GLM-4.6V-Flash`](https://huggingface.co/zai-org/GLM-4.6V-Flash) |

All three are dense (`glm4v`) VLMs sharing the GLM-4V vision tower and a GLM-4
decoder; GLM-4.6V-Flash is the newest ~10B checkpoint.

## API

### `Glm4vModel`

GLM-4V multimodal backbone: vision tower + GLM-4 decoder fused by M-RoPE.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `151552` | token vocabulary size |
| `embed_dim` | `4096` | text model width |
| `mlp_dim` | `13696` | MLP inner width |
| `num_layers` | `40` | decoder blocks |
| `num_heads` | `32` | query heads |
| `num_kv_heads` | `2` | key/value heads (GQA) |
| `partial_rotary_factor` | `0.5` | fraction of each head that gets rotated |
| `norm_eps` | `1e-05` | normalization epsilon |
| `rope_theta` | `10000.0` | rotary base frequency |
| `mrope_section` | `(8, 12, 12)` | M-RoPE split across time/height/width |
| `tie_embeddings` | `False` | reuse embeddings as the LM head |
| `vision_depth` | `24` | vision tower depth |
| `vision_embed_dim` | `1536` | vision tower width |
| `vision_num_heads` | `12` | vision attention heads |
| `vision_mlp_dim` | `13696` | vision MLP width |
| `vision_out_dim` | `4096` | projector output width (matches the decoder) |
| `image_size` | `336` | expected image resolution |
| `patch_size` | `14` | patch size |
| `spatial_merge_size` | `2` | patch-merge factor before the decoder |
| `temporal_patch_size` | `2` | frames per temporal patch |
| `in_channels` | `3` | input image channels |
| `vision_norm_eps` | `1e-05` | vision tower norm epsilon |
| `image_token_id` | `151343` | placeholder token id expanded per image |
| `video_token_id` | `151344` | placeholder token id expanded per video |
| `image_start_token_id` | `151339` | token id opening an image span |
| `image_end_token_id` | `151340` | token id closing an image span |
| `video_start_token_id` | `151341` |  |
| `video_end_token_id` | `151342` |  |

### `Glm4vConditionalGenerate`

GLM-4V with an LM head + fast ``.generate()`` (image+text -> text).

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

### `Glm4vTextModel`

GLM-4V text decoder: ``embed -> num_layers x Glm4DecoderLayer -> RMSNorm``.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | required | token vocabulary size |
| `embed_dim` | required | text model width |
| `mlp_dim` | required | MLP inner width |
| `num_layers` | required | decoder blocks |
| `num_heads` | required | query heads |
| `num_kv_heads` | required | key/value heads (GQA) |
| `head_dim` | required | per-head width |
| `rotary_dim` | required |  |
| `norm_eps` | `1e-05` | normalization epsilon |

### `Glm4vVisionModel`

GLM-4V vision tower.

| Arg | Default | Meaning |
|---|---|---|
| `embed_dim` | `1536` | text model width |
| `depth` | `24` | vision tower depth |
| `num_heads` | `12` | query heads |
| `out_hidden_size` | `4096` | projector output width |
| `intermediate_size` | `13696` | MLP inner width |
| `image_size` | `336` | expected image resolution |
| `patch_size` | `14` | patch size |
| `spatial_merge_size` | `2` | patch-merge factor before the decoder |
| `norm_eps` | `1e-05` | normalization epsilon |
| `rope_theta` | `10000.0` | rotary base frequency |

### `Glm4vTokenizer`

GLM-4V BPE tokenizer (``tokenizers`` backend) with vision specials.

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `None` | Hub repo to pull tokenizer/processor files from |
| `tokenizer_file` | `None` | explicit path to a `tokenizer.json` |

### `Glm4vImageProcessor`

Turn PIL/array images into ``{"pixel_values", "image_grid_thw"}``.

| Arg | Default | Meaning |
|---|---|---|
| `patch_size` | `14` | patch size |
| `spatial_merge_size` | `2` | patch-merge factor before the decoder |
| `temporal_patch_size` | `2` | frames per temporal patch |
| `min_pixels` | `12544` | smallest allowed pixel budget |
| `max_pixels` | `9633792` | largest allowed pixel budget |
| `image_mean` | `(0.48145466, 0.4578275, 0.40821073)` | per-channel normalization mean |
| `image_std` | `(0.26862954, 0.26130258, 0.27577711)` | per-channel normalization std |

### `Glm4vProcessor`

Image + text -> model inputs for GLM-4V.

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
from zeromodels.models.glm4v import Glm4vConditionalGenerate, Glm4vProcessor

model = Glm4vConditionalGenerate.from_weights("zeromodels/glm-4.1v-9b-thinking")
processor = Glm4vProcessor.from_weights("zeromodels/glm-4.1v-9b-thinking")

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

`Glm4vTokenizer` encodes raw text: it has no chat template, so pass a prompt you
have rendered yourself (or go through the processor above).

```python
from zeromodels.models.glm4v import Glm4vTokenizer

tokenizer = Glm4vTokenizer.from_weights("zeromodels/glm-4.1v-9b-thinking")
inputs = tokenizer("Who wrote Dune?")
outputs = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = Glm4vConditionalGenerate.from_weights(
    "zeromodels/glm-4.1v-9b-thinking", quantization="int8", load_dtype="bfloat16"
)
```
