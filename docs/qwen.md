# Qwen (text & vision-language)

<div class="kf-note kf-note--weights">
<b>Weights:</b> the Qwen families are hosted as preconverted Keras weights on Hugging Face
under <a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>; load
with <code>from_weights("zeromodels/&lt;variant&gt;")</code>. Upstream Qwen safetensors
also convert on the fly via the <code>hf:</code> prefix. See the per-family pages for the
exact variant lists.
</div>

Alibaba's Qwen family in **pure Keras 3**: both the text LLMs and the
image+text multimodal LLMs: with bit-close parity to HuggingFace
(verified on real checkpoints, see below). One implementation per family runs
unmodified on **TensorFlow / Torch / JAX**.

**Papers**:
[Qwen2](https://arxiv.org/abs/2407.10671) ·
[Qwen3](https://arxiv.org/abs/2505.09388) ·
[Qwen2-VL](https://arxiv.org/abs/2409.12191) ·
[Qwen2.5-VL](https://arxiv.org/abs/2502.13923)

| Family | Module | Kind | Text decoder |
|---|---|---|---|
| Qwen2 | `zeromodels.models.qwen2` | text | Qwen2 (GQA, **qkv bias**, 1-D RoPE) |
| Qwen3 | `zeromodels.models.qwen3` | text | Qwen3 (**QK-norm**, no qkv bias) |
| Qwen3.5 | `zeromodels.models.qwen3_5` | text | **Qwen3-Next hybrid** (Gated-DeltaNet + gated full attention) |
| Qwen2-VL | `zeromodels.models.qwen2_vl` | image+text | Qwen2 |
| Qwen2.5-VL | `zeromodels.models.qwen2_5_vl` | image+text | Qwen2.5 (windowed vision) |
| Qwen3-VL | `zeromodels.models.qwen3_vl` | image+text | Qwen3 (interleaved M-RoPE, DeepStack) |

## Loading

Each family exposes two classes:

- **`*Model`**: base model; its `call` returns features (`last_hidden_state`).
- **`*Generate`**: adds the LM head + greedy `.generate()`; `call` returns `logits`.

The canonical path is the **hosted repo id** `zeromodels/<variant>` (preconverted
bf16 Keras weights + `zm_config.json`); a raw `hf:` id also converts any matching
`model_type` on the fly:

```python
from zeromodels.models.qwen3 import Qwen3TextGenerate
from zeromodels.models.qwen2_vl import Qwen2VLConditionalGenerate

gen = Qwen3TextGenerate.from_weights("zeromodels/qwen3-4b")  # text
gen = Qwen2VLConditionalGenerate.from_weights(
    "zeromodels/qwen2-vl-7b-instruct"
)  # multimodal
# raw hf: ids convert from the upstream safetensors:
gen = Qwen3TextGenerate.from_weights("hf:Qwen/Qwen3-4B")
```

### Available variants

Text:

| Family | Variants (`from_weights("…")`) |
|---|---|
| Qwen2 | `qwen2-{0.5b,1.5b,7b,72b}` and each `-instruct` |
| Qwen3 | `qwen3-{0.6b,1.7b,4b,8b,14b}` and each `-base`; `qwen3-32b` |
| Qwen3.5 | `qwen3.5-{0.8b,2b,4b,9b}` and each `-base`; `qwen3.5-27b` |

Vision-language:

| Family | Variants |
|---|---|
| Qwen2-VL | `qwen2-vl-{2b,7b,72b}` and each `-instruct` |
| Qwen2.5-VL | `qwen2.5-vl-{3b,7b,32b,72b}-instruct` (instruct-only series) |
| Qwen3-VL | `qwen3-vl-{2b,4b,8b,32b}-instruct` and each `-thinking` |

Mixture-of-Experts (own folders; sparse expert blocks, see each page):

| Family | Module | Kind | Variants |
|---|---|---|---|
| Qwen2-MoE | `qwen2_moe` | text | `qwen1.5-moe-a2.7b`(`-chat`); `qwen2-57b-a14b`(`-instruct`) |
| Qwen3-MoE | `qwen3_moe` | text | `qwen3-30b-a3b`(`-base`, `-instruct-2507`, `-thinking-2507`) |
| Qwen3-Next | `qwen3_next` | text | `qwen3-next-80b-a3b-{instruct,thinking}` |
| Qwen3-VL-MoE | `qwen3_vl_moe` | image+text | `qwen3-vl-30b-a3b-{instruct,thinking}` |
| Qwen3.5-MoE | `qwen3_5_moe` | image+text | `qwen3.5-{35b-a3b,35b-a3b-base,122b-a10b}` |

Quantized repos (`-AWQ`, `-GPTQ-*`, GGUF) are out of scope (the converter reads bf16/fp
safetensors).

> **Qwen3.5 comes in two forms here.** The dense text backbone is `qwen3_5`
> (`model_type` `qwen3_5` / `qwen3_5_text`, this page's Qwen3.5 row); the released
> Mixture-of-Experts checkpoints (`Qwen3_5ForConditionalGeneration`) are the multimodal
> `qwen3_5_moe` port (see [qwen3_5_moe.md](qwen3_5_moe.md)).

## Verified parity

Validated against the HF reference (eager attention) on a real forward pass:
**argmax agreement 1.0000** at every position; text generation is **token-exact**
greedy:

| Model | Checkpoint | max \|Δ logits\| | argmax |
|---|---|---|---|
| Qwen2 | `Qwen/Qwen2-0.5B` | 3.1e-5 | 1.0000 |
| Qwen3 | `Qwen/Qwen3-0.6B` | 2.2e-5 | 1.0000 |
| Qwen3.5 | `Qwen/Qwen3.5-0.8B` | 1.5e-5 | 1.0000 |
| Qwen2-VL | `Qwen/Qwen2-VL-2B-Instruct` | 7.3e-4 | 1.0000 |
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-3B-Instruct` | 4.1e-4 | 1.0000 |
| Qwen3-VL | `Qwen/Qwen3-VL-2B-Instruct` | 3.9e-3 | 1.0000 |

Individual primitives (RMSNorm, GQA + M-RoPE attention, SwiGLU, vision rotary,
KV cache) match HF to ~1e-7 in isolation. The Qwen3.5 residual is
chunked-vs-recurrent Gated-DeltaNet fp accumulation: the kernels are
algebraically identical.

## Forward pass

`*Model` returns features; `*Generate` adds logits. Text takes just token ids;
VL adds pre-patchified pixels:

```python
# text
gen({"input_ids": input_ids})["logits"]  # (B, L, vocab_size)

# vision-language: images; placeholders sit inside input_ids
inputs = {
    "input_ids": input_ids,  # (B, L) int, image placeholders
    "pixel_values": pixel_values,  # (num_patches, patch_dim) image patches
    "image_grid_thw": image_grid_thw,  # (num_images, 3) per-image (t, h, w)
}
gen(inputs)["logits"]  # (B, L, vocab_size)
```

The image block is optional; its embeddings scatter into the `<|image_pad|>` slots.

These are token-id (and pre-flattened-patch) models: **no spatial H/W axes**, so
`channels_first/last` does not apply (handled like the audio models). The VL
patch-embed Conv3d (kernel == stride) is implemented as a `Dense`.

## Generation

`.generate()` is greedy decoding with a KV cache. Qwen3.5 additionally carries
the per-layer conv state + delta-rule recurrent state for its linear layers; the
VL families carry incremental M-RoPE positions (each new token's position is
`cache_len + rope_delta` on all three axes), and Qwen3-VL injects its
**DeepStack** vision features into the first decoder layers during prefill.

The API is the same shape for both: build inputs from a chat list, then
`model.generate(**inputs, max_new_tokens=...)`. **LLMs use the tokenizer**
(text only); **VLMs use the processor** (tokenizer + image processor), with images
inline in the conversation via `path` / `url` / a PIL image.

Load the tokenizer / processor with `.from_weights(...)`, passing the **same**
identifier you give the model, so its files match the checkpoint, e.g.
`Qwen3Tokenizer.from_weights("zeromodels/qwen3-0.6b")` or
`Qwen2VLProcessor.from_weights("zeromodels/qwen2-vl-2b-instruct")`. The `hf:` prefix
works too (`Qwen2Tokenizer.from_weights("hf:Qwen/Qwen2-7B-Instruct")`), and the bare
`Qwen2Tokenizer()` / `Qwen2VLProcessor()` constructors fall back to a default Qwen repo.

```python
# text LLM: tokenizer takes the chat messages
from zeromodels.models.qwen3 import Qwen3TextGenerate, Qwen3Tokenizer

model = Qwen3TextGenerate.from_weights("zeromodels/qwen3-0.6b")
tokenizer = Qwen3Tokenizer.from_weights("zeromodels/qwen3-0.6b")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Name three prime numbers."},
]
inputs = tokenizer(messages)
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))

# vision-language: processor takes the conversation (images inline)
from zeromodels.models.qwen2_vl import Qwen2VLConditionalGenerate, Qwen2VLProcessor

model = Qwen2VLConditionalGenerate.from_weights("zeromodels/qwen2-vl-2b-instruct")
processor = Qwen2VLProcessor.from_weights("zeromodels/qwen2-vl-2b-instruct")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image.jpg"},
            {"type": "text", "text": "What happened in the image?"},
        ],
    },
]
inputs = processor(conversation)
outputs = model.generate(**inputs, max_new_tokens=128)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

## Image processor (VL)

`Qwen2VLImageProcessor` is a pure-Python port of HF's: smart-resize each image so
both sides are multiples of `patch_size · spatial_merge_size`, CLIP-normalize,
repeat the frame to fill `temporal_patch_size`, and reshape into the
`(num_patches, patch_dim)` layout with a matching `image_grid_thw`. Grids match
HF exactly; pixels match to a small bicubic tolerance.

```python
from zeromodels.models.qwen2_vl.qwen2_vl_image_processor import Qwen2VLImageProcessor

feat = Qwen2VLImageProcessor()(pil_image)  # {"pixel_values", "image_grid_thw"}
```

## Architecture notes

### Text families

| | Qwen2 | Qwen3 | Qwen3.5 |
|---|---|---|---|
| Token mixer | GQA attention | GQA attention | **hybrid** linear / full |
| Linear attention |: |, | **Gated-DeltaNet** (conv1d + delta rule) |
| QK-norm | no | **yes** | yes (full layers) |
| QKV bias | **yes** | no | no |
| RoPE | 1-D full | 1-D full | **partial** (factor 0.25) |
| Norm | RMSNorm | RMSNorm | **zero-centered** `(1+w)` + gated |
| Output gate |, |, | **sigmoid gate** (full attention) |

For pure text, Qwen3.5's three M-RoPE position axes coincide, so rotary reduces
to standard 1-D partial rope.

### Vision-language families

| | Qwen2-VL | Qwen2.5-VL | Qwen3-VL |
|---|---|---|---|
| Vision norm / MLP | LayerNorm / GELU | RMSNorm / SwiGLU | LayerNorm / GELU |
| Vision attention | full | **windowed** (+ full at some layers) | full |
| Vision positions | 2-D rotary | 2-D rotary | 2-D rotary + **learned** (interpolated) |
| Extra vision |: | `tokens_per_second` (video) | **DeepStack** fusion |
| Patch size | 14 | 14 | 16 |
| Text decoder | Qwen2 (qkv bias) | Qwen2.5 (qkv bias) | **Qwen3** (QK-norm, no qkv bias) |
| M-RoPE | sectioned | sectioned | **interleaved** |

The shared VL primitives live in `qwen2_vl/qwen2_vl_layers.py`; Qwen2.5-VL and
Qwen3-VL subclass from Qwen2-VL. The text families are each self-contained (no
cross-family imports); every family's `convert_*_hf_to_keras.py` maps the HF
safetensors to Keras.

## Citation

```bibtex
@article{Qwen2,        title={Qwen2 Technical Report},   author={Yang, An and others}, journal={arXiv:2407.10671}, year={2024}}
@article{Qwen3,        title={Qwen3 Technical Report},   author={Yang, An and others}, journal={arXiv:2505.09388}, year={2025}}
@article{Qwen2VL,      title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution}, author={Wang, Peng and others}, journal={arXiv:2409.12191}, year={2024}}
@article{Qwen2.5-VL,   title={Qwen2.5-VL Technical Report}, author={Bai, Shuai and others}, journal={arXiv:2502.13923}, year={2025}}
```
