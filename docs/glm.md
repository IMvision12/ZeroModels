# GLM (text & vision-language)

<div class="kf-note kf-note--weights">
<b>Weights:</b> most GLM families are hosted as preconverted Keras weights under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a> (each repo
carries <code>kf_config.json</code> + the <code>.weights.h5</code>) — load with
<code>from_weights("zeromodels/&lt;variant&gt;")</code>. The largest checkpoints
(full-size GLM-4.5 / GLM-4.6 and the GLM-5 MoE series) are not re-hosted; convert them on
the fly with a raw <code>hf:</code> id. See <a href="../loading_weights/">Loading Weights</a>.
</div>

Zhipu / Z.ai's **GLM** family in pure Keras 3: the text LLMs (GLM-4 through the
GLM-5 MoE series) and the GLM-4V vision-language models: one implementation per
family, runnable unchanged on **TensorFlow / Torch / JAX** and bit-close to the
HuggingFace reference.

**Papers / refs**:
[ChatGLM (GLM-4)](https://arxiv.org/abs/2406.12793) ·
[GLM-4.5](https://huggingface.co/zai-org/GLM-4.5) ·
[GLM-4.1V](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking) ·
[GLM-5.2](https://huggingface.co/zai-org/GLM-5.2)

| Family | Module | Kind | Decoder |
|---|---|---|---|
| GLM-4-9B | `zeromodels.models.glm` | text | GLM (interleaved partial RoPE) |
| GLM-4-0414 / GLM-Z1 | `zeromodels.models.glm4` | text | GLM + sandwich norms |
| GLM-4.5 / GLM-4.6 | `zeromodels.models.glm4_moe` | text (MoE) | grouped-top-k router + shared expert, NeoX partial RoPE |
| GLM-5 / GLM-5.1 / GLM-5.2 | `zeromodels.models.glm5_moe` | text (MoE) | **MLA + DSA** (DeepSeek Sparse Attention) + DeepSeekMoE |
| GLM-4.1V | `zeromodels.models.glm4v` | image+video+text | GLM + Qwen2-VL-class M-RoPE vision |
| GLM-4.5V | `zeromodels.models.glm4v_moe` | image+video+text (MoE) | GLM-4.5 MoE + GLM-4V vision |

Each family exposes a `*Model` (features, `call` → `last_hidden_state`) and a
`*Generate` (adds the LM head + `.generate()`, `call` → `logits`), plus a
`*Tokenizer` (and a `*Processor` for the VL families).

## Loading

Hosted checkpoints load by their `zeromodels/<variant>` repo id (preconverted
bf16 `.weights.h5` + `kf_config.json`). A raw `hf:` id converts any matching
upstream checkpoint on the fly (safetensors mapped at load time; FP8 MoE
dequantized) — used for the checkpoints too large to re-host.

```python
from zeromodels.models.glm import GlmTextGenerate
from zeromodels.models.glm4_moe import Glm4MoeTextGenerate

gen = GlmTextGenerate.from_weights("zeromodels/glm-4-9b-chat")  # text
gen = Glm4MoeTextGenerate.from_weights("zeromodels/glm-4.5-air")  # text MoE
# a raw hf: id converts on the fly (e.g. the not-hosted full-size GLM-4.5)
gen = Glm4MoeTextGenerate.from_weights("hf:zai-org/GLM-4.5")
```

### Available variants

Text (load hosted ones as `zeromodels/<variant>`):

| Family | Variants | Hosted? | Upstream |
|---|---|---|---|
| GLM-4-9B | `glm-4-9b`, `glm-4-9b-chat`, `glm-4-9b-chat-1m` | yes | `zai-org/glm-4-9b{,-chat,-chat-1m}-hf` |
| GLM-4-0414 / GLM-Z1 | `glm-4-9b-0414`, `glm-4-32b-0414`, `glm-4-32b-base-0414`, `glm-z1-9b-0414`, `glm-z1-32b-0414` | yes | `zai-org/GLM-4-*-0414`, `zai-org/GLM-Z1-*-0414` |
| GLM-4.5 (MoE) | `glm-4.5-air`, `glm-4.5-air-base` | yes | `zai-org/GLM-4.5-Air{,-Base}` |
| GLM-4.5 / GLM-4.6 (MoE, full) | `glm-4.5`, `glm-4.6` | `hf:` only | `zai-org/GLM-4.5`, `zai-org/GLM-4.6` |
| GLM-5 / GLM-5.1 / GLM-5.2 (MoE) | `glm5`, `glm5_1`, `glm5_2` | `hf:` only | `zai-org/GLM-5{,.1,.2}` |

Vision-language:

| Family | Variants | Hosted? | Upstream |
|---|---|---|---|
| GLM-4.1V / 4.6V-Flash | `glm-4.1v-9b-thinking`, `glm-4.1v-9b-base`, `glm-4.6v-flash` | yes | `zai-org/GLM-4.1V-9B-{Thinking,Base}`, `zai-org/GLM-4.6V-Flash` |
| GLM-4.5V / 4.6V (MoE) | `glm-4.5v`, `glm-4.6v` | yes | `zai-org/GLM-4.5V`, `zai-org/GLM-4.6V` |

## Generation

`.generate()` is greedy decoding with a KV cache. **LLMs use the tokenizer**
(text only); **VLMs use the processor** (tokenizer + image/video processor), with
images inline in the conversation. Load the tokenizer / processor with the **same**
identifier you give the model.

```python
# text LLM
from zeromodels.models.glm import GlmTextGenerate, GlmTokenizer

model = GlmTextGenerate.from_weights("zeromodels/glm-4-9b-chat")
tokenizer = GlmTokenizer.from_weights("zeromodels/glm-4-9b-chat")

messages = [{"role": "user", "content": "Name three prime numbers."}]
inputs = tokenizer(messages)
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))

# vision-language (GLM-4.1V)
from zeromodels.models.glm4v import Glm4vConditionalGenerate, Glm4vProcessor

model = Glm4vConditionalGenerate.from_weights("zeromodels/glm-4.1v-9b-thinking")
processor = Glm4vProcessor.from_weights("zeromodels/glm-4.1v-9b-thinking")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image.jpg"},
            {"type": "text", "text": "What is in the image?"},
        ],
    },
]
inputs = processor(conversation)
outputs = model.generate(**inputs, max_new_tokens=128)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

## Architecture notes

- **GLM-4 / GLM-4-0414** (`glm`, `glm4`): post-norm GLM block with **interleaved
  partial RoPE**; GLM-4-0414 adds sandwich (input/post) norms around attention and
  MLP. The `glm-z1-*` checkpoints are reasoning fine-tunes on the same arch.
- **GLM-4.5 / GLM-4.6** (`glm4_moe`): DeepSeek-V3-style **grouped-top-k** MoE router
  with a shared expert and fused expert einsums, **NeoX** partial RoPE (not
  interleaved); the auxiliary MTP head is dropped, and FP8-released weights are
  dequantized on load.
- **GLM-5 / 5.1 / 5.2** (`glm5_moe`): **Multi-head Latent Attention (MLA)** +
  **DeepSeek Sparse Attention (DSA)** with a Lightning-Indexer, on top of
  DeepSeekMoE. 5 ≡ 5.1 in the forward; 5.2 only changes `max_position_embeddings`
  (1M) and `rope_theta`. Cached decode skips the indexer for exact short-context
  output.
- **GLM-4V / GLM-4.5V** (`glm4v`, `glm4v_moe`): Qwen2-VL-class **M-RoPE** vision
  tower with a **learned** position embedding (bicubic-interpolated), a Conv3d patch
  embed + downsample conv, and a SwiGLU patch merger; the text side is GLM-4 (4V) or
  GLM-4.5 MoE (4.5V).

## Parity vs HuggingFace Reference

Validated against `transformers` (cloned main, eager attention) on real forward
passes: **greedy generation is token-identical**; max|Δ logits| ≈ `1.8e-7` (GLM-4),
`2.8e-7` (GLM-4-0414), `2e-7`–`3e-7` (GLM-4.5 / 4.1V / 4.5V), `~5e-8` (GLM-5 series,
fp32 tiny-config). Verified across the `torch`, `jax`, and `tensorflow` backends.

## Citation

```bibtex
@article{glm2024chatglm, title={ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools}, author={GLM Team}, journal={arXiv:2406.12793}, year={2024}}
```
