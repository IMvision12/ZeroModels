# Qwen3.5-MoE

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>kf_config.json</code> and <code>kf_preprocessor.json</code> plus a
sharded <code>model.weights.json</code> + shards and a <code>tokenizer.json</code>). Load the
model and processor with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Alibaba's Qwen3.5-MoE vision-language models, ported to pure Keras 3. Qwen3.5-MoE is a
**multimodal Mixture-of-Experts** model (`Qwen3_5MoeForConditionalGeneration`): a
Qwen3-VL-style vision tower feeds a **Qwen3-Next hybrid MoE** text decoder, so it pairs
image + video understanding with the sparse hybrid backbone.

- **Text backbone**: the Qwen3-Next hybrid, most blocks are Gated-DeltaNet linear
  attention with a gated full-attention block every fourth layer (GQA, per-head QK-norm,
  partial-rotary **interleaved M-RoPE**), and every block's MLP is a softmax-routed
  expert bank plus a sigmoid-gated shared expert. Zero-centered RMSNorm.
- **Vision tower**: the Qwen3-VL ViT (learned, interpolated position embeddings, GELU
  blocks, 2x2 patch merger) **without DeepStack**.
- **Fusion**: image/video placeholder tokens are replaced by the merged patch
  embeddings, and 3D M-RoPE positions are derived from each image's `(t, h, w)` grid.

Links:

- HF collection: [Qwen3.5-MoE](https://huggingface.co/collections/zeromodels/qwen35-moe-6a7eb77a1a41110f3195af09)
- Paper: [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388)
- HF docs: [transformers/model_doc/qwen3_5_moe](https://huggingface.co/docs/transformers/model_doc/qwen3_5_moe)

See also [qwen3_next.md](qwen3_next.md) (the text-only Qwen3-Next MoE), [qwen3_5.md](qwen3_5.md)
(the dense Qwen3.5 text backbone), [qwen3_vl.md](qwen3_vl.md) (the dense Qwen3-VL).

## Variants

Preconverted, bf16 weights are hosted under `zeromodels/`. Load the model and processor
with `from_weights("zeromodels/<variant>")`; the `-base` suffix marks the base
(non-instruction-tuned) checkpoints. Qwen3.5-MoE is Apache 2.0.

| Variant | Hub |
|---|---|
| `qwen3.5-35b-a3b` | [`zeromodels/qwen3.5-35b-a3b`](https://huggingface.co/zeromodels/qwen3.5-35b-a3b) |
| `qwen3.5-35b-a3b-base` | [`zeromodels/qwen3.5-35b-a3b-base`](https://huggingface.co/zeromodels/qwen3.5-35b-a3b-base) |
| `qwen3.5-122b-a10b` | [`zeromodels/qwen3.5-122b-a10b`](https://huggingface.co/zeromodels/qwen3.5-122b-a10b) |

Upstream Qwen safetensors also load directly via the `hf:` prefix, e.g.
`from_weights("hf:Qwen/Qwen3.5-35B-A3B")`, which converts them in process (pass
`cache_converted=True` to keep the result). See [Loading Weights](loading_weights.md).

## API

### `Qwen3_5MoeModel`

The multimodal backbone (vision tower + Qwen3-Next MoE decoder), no LM head. Returns
`{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `248320` | token vocabulary size |
| `embed_dim` | `2048` | text model width |
| `num_layers` | `40` | decoder blocks |
| `num_heads` | `16` | query heads (full-attention layers) |
| `num_kv_heads` | `2` | key/value heads (GQA) |
| `head_dim` | `256` | per-head width |
| `partial_rotary_factor` | `0.25` | fraction of `head_dim` that gets rotary |
| `mrope_section` | `(11, 11, 10)` | interleaved M-RoPE split (time/height/width) |
| `full_attention_interval` | `4` | full-attention block every Nth layer |
| `num_experts` | `256` | routed experts |
| `num_experts_per_tok` | `8` | experts routed per token |
| `moe_mlp_dim` | `512` | per-expert inner width |
| `shared_mlp_dim` | `512` | shared-expert inner width |
| `linear_*` | (Qwen3-Next) | Gated-DeltaNet head dims / counts / conv kernel |
| `vision_depth` | `27` | vision blocks |
| `vision_embed_dim` | `1152` | vision tower width |
| `vision_out_dim` | `embed_dim` | merger output width |
| `num_position_embeddings` | `2304` | learned vision position grid |
| `patch_size` | `16` | vision patch size |
| `image_token_id` | `248056` | placeholder token expanded per image |
| `video_token_id` | `248057` | placeholder token expanded per video |

### `Qwen3_5MoeConditionalGenerate`

`Qwen3_5MoeModel` plus a (tied) LM head and fast `.generate()` (image+text -> text).
Returns `{"logits": (batch, seq, vocab_size)}`. Prefill runs the vision encoder +
M-RoPE into a **hybrid** per-layer cache (fixed-slot KV for the full-attention layers,
`(conv_state, recurrent_state)` for the Gated-DeltaNet layers); decode advances one text
token at a time.

### `Qwen3_5MoeProcessor`

Image/video + text processor (ChatML + image-pad expansion), a 16px-patch Qwen-VL
processor with the Qwen3.5-MoE tokenizer and video processor.

## End-to-end example

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.models.qwen3_5_moe import (
    Qwen3_5MoeConditionalGenerate,
    Qwen3_5MoeProcessor,
)

model = Qwen3_5MoeConditionalGenerate.from_weights("zeromodels/qwen3.5-35b-a3b")
processor = Qwen3_5MoeProcessor.from_weights("zeromodels/qwen3.5-35b-a3b")

inputs = processor(
    conversation=[
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open("photo.jpg")},
                {"type": "text", "text": "Describe this image in one sentence."},
            ],
        }
    ]
)
outputs = model.generate(**inputs, max_new_tokens=64)

print(processor.decode(outputs[0]))
```

### Lower memory

Larger checkpoints load in bf16 or weight-only quantized. See
[quantization.md](quantization.md):

```python
model = Qwen3_5MoeConditionalGenerate.from_weights(
    "zeromodels/qwen3.5-35b-a3b", quantization="int8", load_dtype="bfloat16"
)
```
