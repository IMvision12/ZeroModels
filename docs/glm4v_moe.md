# GLM-4.5V (GLM-4V MoE)

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + a sharded <code>model.weights.json</code>).
Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>, or convert an original
checkpoint on the fly with <code>from_weights("hf:zai-org/GLM-4.5V")</code>. See
<a href="../loading_weights/">Loading Weights</a>.
</div>

The Mixture-of-Experts GLM-4V, ported to pure Keras 3. It pairs the GLM-4V vision
tower with a GLM-4.5-style sparse decoder: routed experts plus shared experts and
node-limited routing.

Memory is governed by **total** parameters, not active ones: every expert stays
resident.

Like the upstream checkpoints, the weights are stored **bfloat16** except the MoE
router correction bias (`e_score_correction_bias`, one per MoE layer), kept in
**float32** so bf16 rounding cannot flip the group / expert top-k selection. This
mirrors transformers' `_keep_in_fp32_modules_strict`; the hosted `zm_config.json`
records it as `weight_dtype: "bfloat16"` + `weight_dtype_overrides:
{"e_score_correction_bias": "float32"}`.

Links:

- Paper: [GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning (arXiv:2507.01006)](https://arxiv.org/abs/2507.01006)
- HF docs: [transformers/model_doc/glm4v_moe](https://huggingface.co/docs/transformers/model_doc/glm4v_moe)

See also [glm4v.md](glm4v.md), [glm4_moe.md](glm4_moe.md).

## Variants

Load any of these with `from_weights("zeromodels/<variant>")` (or convert the
upstream checkpoint on the fly with `from_weights("hf:<upstream>")`).

| Variant | Hosted | Upstream |
|---|---|---|
| `glm-4.5v` | `zeromodels/glm-4.5v` | [`zai-org/GLM-4.5V`](https://huggingface.co/zai-org/GLM-4.5V) |
| `glm-4.6v` | `zeromodels/glm-4.6v` | [`zai-org/GLM-4.6V`](https://huggingface.co/zai-org/GLM-4.6V) |

Both are ~108B MoE (`glm4v_moe`) VLMs: the GLM-4V vision tower on a GLM-4.5 MoE
decoder. GLM-4.6V is the newer checkpoint.

## API

### `Glm4vMoeModel`

GLM-4.5V multimodal backbone: GLM-4V vision tower + GLM-4.5 MoE decoder.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `151424` | token vocabulary size |
| `embed_dim` | `4096` | text model width |
| `mlp_dim` | `10944` | MLP inner width |
| `moe_mlp_dim` | `1408` | per-expert inner width |
| `num_layers` | `46` | decoder blocks |
| `num_heads` | `96` | query heads |
| `num_kv_heads` | `8` | key/value heads (GQA) |
| `head_dim` | `128` | per-head width |
| `num_experts` | `128` | expert count |
| `num_experts_per_tok` | `8` | experts routed per token |
| `n_shared_experts` | `1` | always-on shared experts |
| `n_group` | `1` | routing groups (node-limited routing) |
| `topk_group` | `1` | groups kept per token |
| `norm_topk_prob` | `True` | renormalize the top-k router weights |
| `routed_scaling_factor` | `1.0` | scale applied to routed-expert output |
| `first_k_dense` | `1` | leading layers left dense instead of MoE |
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
| `image_token_id` | `151363` | placeholder token id expanded per image |
| `video_token_id` | `151364` | placeholder token id expanded per video |
| `image_start_token_id` | `151339` | token id opening an image span |
| `image_end_token_id` | `151340` | token id closing an image span |
| `video_start_token_id` | `151341` |  |
| `video_end_token_id` | `151342` |  |

### `Glm4vMoeConditionalGenerate`

GLM-4.5V with an LM head + fast ``.generate()`` (image+text -> text).

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

### `Glm4vMoeTextModel`

GLM-4.5V text decoder: ``embed -> num_layers x Glm4MoeDecoderLayer -> RMSNorm``.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | required | token vocabulary size |
| `embed_dim` | required | text model width |
| `mlp_dim` | required | MLP inner width |
| `moe_mlp_dim` | required | per-expert inner width |
| `num_layers` | required | decoder blocks |
| `num_heads` | required | query heads |
| `num_kv_heads` | required | key/value heads (GQA) |
| `head_dim` | required | per-head width |
| `rotary_dim` | required |  |
| `num_experts` | required | expert count |
| `num_experts_per_tok` | required | experts routed per token |
| `n_shared_experts` | required | always-on shared experts |
| `n_group` | required | routing groups (node-limited routing) |
| `topk_group` | required | groups kept per token |
| `norm_topk_prob` | required | renormalize the top-k router weights |
| `routed_scaling_factor` | required | scale applied to routed-expert output |
| `first_k_dense` | required | leading layers left dense instead of MoE |
| `norm_eps` | `1e-05` | normalization epsilon |

### `Glm4vMoeTokenizer`

GLM-4.5V BPE tokenizer (``tokenizers`` backend) with vision specials.

| Arg | Default | Meaning |
|---|---|---|
| `hf_id` | `None` | Hub repo to pull tokenizer/processor files from |
| `tokenizer_file` | `None` | explicit path to a `tokenizer.json` |

### `Glm4vMoeProcessor`

Image + text -> model inputs for GLM-4.5V.

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
from zeromodels.models.glm4v_moe import Glm4vMoeConditionalGenerate, Glm4vMoeProcessor

model = Glm4vMoeConditionalGenerate.from_weights("zeromodels/glm-4.5v")
processor = Glm4vMoeProcessor.from_weights("zeromodels/glm-4.5v")

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

`Glm4vMoeTokenizer` encodes raw text: it has no chat template, so pass a prompt you
have rendered yourself (or go through the processor above).

```python
from zeromodels.models.glm4v_moe import Glm4vMoeTokenizer

tokenizer = Glm4vMoeTokenizer.from_weights("zeromodels/glm-4.5v")
inputs = tokenizer("Who wrote Dune?")
outputs = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = Glm4vMoeConditionalGenerate.from_weights(
    "zeromodels/glm-4.5v", quantization="int8", load_dtype="bfloat16"
)
```
