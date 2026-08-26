# Qwen3-VL-MoE

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>kf_config.json</code> and <code>kf_preprocessor.json</code> plus a
sharded <code>model.weights.json</code> + shards and a <code>tokenizer.json</code>). Load the
model and processor with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

The Mixture-of-Experts variant of Qwen3-VL, ported to pure Keras 3. It is **exactly**
Qwen3-VL, the same DeepStack vision tower, interleaved M-RoPE, and per-head QK-norm GQA
attention, except each text decoder block's MLP is a **sparse Mixture-of-Experts** block:
a float32-softmax top-k router over fused SwiGLU experts, with **no shared expert** (the
Qwen3-MoE recipe). Layers off the `decoder_sparse_step` cadence (or in `mlp_only_layers`)
stay dense.

Memory is governed by **total** parameters, not active ones.

Links:

- HF collection: [Qwen3-VL-MoE](https://huggingface.co/collections/zeromodels/qwen3-vl-moe-6a7eb7d3e6d95b296dae7d0d)
- Paper: [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388)
- HF docs: [transformers/model_doc/qwen3_vl_moe](https://huggingface.co/docs/transformers/model_doc/qwen3_vl_moe)

See also [qwen3_vl.md](qwen3_vl.md) (the dense Qwen3-VL), [qwen3_moe.md](qwen3_moe.md)
(the text-only Qwen3-MoE recipe reused here), [qwen3_5_moe.md](qwen3_5_moe.md) (the
Qwen3.5 MoE VLM, a different hybrid backbone).

## Variants

Preconverted, bf16 weights are hosted under `zeromodels/`. Load the model and processor
with `from_weights("zeromodels/<variant>")`; the `-instruct` sizes are the
instruction-tuned checkpoints and `-thinking` the reasoning checkpoints. Qwen3-VL-MoE is
Apache 2.0.

| Variant | Hub |
|---|---|
| `qwen3-vl-30b-a3b-instruct` | [`zeromodels/qwen3-vl-30b-a3b-instruct`](https://huggingface.co/zeromodels/qwen3-vl-30b-a3b-instruct) |
| `qwen3-vl-30b-a3b-thinking` | [`zeromodels/qwen3-vl-30b-a3b-thinking`](https://huggingface.co/zeromodels/qwen3-vl-30b-a3b-thinking) |
| `qwen3-vl-235b-a22b-instruct` | [`zeromodels/qwen3-vl-235b-a22b-instruct`](https://huggingface.co/zeromodels/qwen3-vl-235b-a22b-instruct) |
| `qwen3-vl-235b-a22b-thinking` | [`zeromodels/qwen3-vl-235b-a22b-thinking`](https://huggingface.co/zeromodels/qwen3-vl-235b-a22b-thinking) |

Upstream Qwen safetensors also load directly via the `hf:` prefix, e.g.
`from_weights("hf:Qwen/Qwen3-VL-30B-A3B-Instruct")`, which converts them in process (pass
`cache_converted=True` to keep the result). See [Loading Weights](loading_weights.md).

## API

### `Qwen3VLMoeModel`

The multimodal backbone (vision tower + DeepStack + Qwen3-VL MoE decoder), no LM head.
Returns `{"last_hidden_state": (batch, seq, embed_dim)}`.

| Arg | Default | Meaning |
|---|---|---|
| `vocab_size` | `151936` | token vocabulary size |
| `embed_dim` | `2048` | text model width |
| `mlp_dim` | `5632` | dense-MLP width (non-MoE layers) |
| `num_layers` | `24` | decoder blocks |
| `num_heads` | `16` | query heads |
| `num_kv_heads` | `16` | key/value heads (GQA) |
| `head_dim` | `128` | per-head width |
| `mrope_section` | `(24, 20, 20)` | interleaved M-RoPE split (time/height/width) |
| `num_experts` | `60` | routed experts |
| `num_experts_per_tok` | `4` | experts routed per token |
| `moe_mlp_dim` | `1408` | per-expert inner width |
| `norm_topk_prob` | `True` | renormalize the top-k router weights |
| `decoder_sparse_step` | `1` | MoE every Nth layer |
| `mlp_only_layers` | `()` | layer indices forced dense |
| `vision_depth` | `27` | vision blocks |
| `vision_out_dim` | `embed_dim` | merger output width |
| `deepstack_visual_indexes` | `(8, 16, 24)` | vision blocks feeding DeepStack |
| `patch_size` | `16` | vision patch size |
| `image_token_id` | `151655` | placeholder token expanded per image |
| `video_token_id` | `151656` | placeholder token expanded per video |

### `Qwen3VLMoeConditionalGenerate`

`Qwen3VLMoeModel` plus a (tied) LM head and fast `.generate()` (image+text -> text).
Returns `{"logits": (batch, seq, vocab_size)}`. Same fast multimodal generation as
Qwen3-VL: vision encoder + M-RoPE prefill into a fixed KV cache (DeepStack threaded
through prefill), then text-only decode. The MoE MLP does not change the cache structure.

### `Qwen3VLMoeTextGenerate`

Text-only counterpart of `Qwen3VLMoeConditionalGenerate`, built with no vision tower
(`build_vision=False`), so `.generate()` takes just token ids. It reads only the MoE
language model out of a Qwen3-VL-MoE checkpoint: `hf:` conversion copies just the text
weights, and a zeromodels repo declaring `Qwen3VLMoeConditionalGenerate` is read through
`FULL_CHECKPOINT_SOURCES`. Set `config_class = Qwen3VLMoeTextConfig`.

```python
from zeromodels.models.qwen3_vl_moe import Qwen3VLMoeTextGenerate, Qwen3VLMoeTokenizer

model = Qwen3VLMoeTextGenerate.from_weights("zeromodels/qwen3-vl-30b-a3b-instruct")
tokenizer = Qwen3VLMoeTokenizer.from_weights("zeromodels/qwen3-vl-30b-a3b-instruct")
outputs = model.generate(**tokenizer("Who wrote Dune?"), max_new_tokens=32)
print(tokenizer.decode(outputs[0]))
```

### `Qwen3VLMoeProcessor`

Image/video + text processor (ChatML + image-pad expansion), a 16px-patch Qwen-VL
processor with the Qwen3-VL-MoE tokenizer and video processor.

## End-to-end example

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.models.qwen3_vl_moe import (
    Qwen3VLMoeConditionalGenerate,
    Qwen3VLMoeProcessor,
)

model = Qwen3VLMoeConditionalGenerate.from_weights(
    "zeromodels/qwen3-vl-30b-a3b-instruct"
)
processor = Qwen3VLMoeProcessor.from_weights("zeromodels/qwen3-vl-30b-a3b-instruct")

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
model = Qwen3VLMoeConditionalGenerate.from_weights(
    "zeromodels/qwen3-vl-30b-a3b-instruct", quantization="int8", load_dtype="bfloat16"
)
```
