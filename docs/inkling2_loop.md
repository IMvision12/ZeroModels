# Inkling-2-Mini (`inkling2_loop`)

Thinking Machines Lab's **Inkling-2-Mini** (`thinkingmachines/Inkling-2-Mini`), ported to pure Keras 3 (JAX / Torch / TensorFlow).

Inkling-2-Mini is a **multimodal looped Mixture-of-Experts** transformer (`Inkling2LoopForConditionalGeneration`, ~26.4B total / ~3.4B–5.4B active parameters, Apache 2.0):

- **Looped Text Backbone**: 24 physical decoder layers where the middle 12 layers (`6..17`) execute 3 times sequentially with shared weights (48 logical layers), passing cross-pass messages through `Inkling2LoopMessageAttentionResidual` and rank-64 `Inkling2LoopMessageGate` controllers.
- **Hybrid Attention + Short Convolutions**: 5 sliding-window layers (`sliding_window_size=1024`) + 1 full-attention layer per 6-layer attention-residual block, with per-token attention sinks, hidden-state-conditioned relative position bias, L2-clipped Q/K/Rel states, logarithmic scaling on full-attention layers, and causal short convolutions (static + dynamic).
- **Sparse MoE**: 128 routed experts, top-8 sigmoid routing with `e_score_correction_bias`, softcapped SwiGLU, and final logit softcapping (`30.0`).
- **Vision Tower**: 23-layer ViT (`20x20` native patches, `2x2` spatial merge → `40x40` pixels per image token, bilinearly interpolated `32x32` learned position embeddings, CLS token, 2D axial RoPE, LayerScale, 2-layer GELU merger).

## Loading Weights

Upstream Hugging Face safetensors (mixed NVFP4 + MXFP8 experts) download and convert on the fly via the `hf:` prefix:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

from zeromodels.models.inkling2_loop import (
    Inkling2LoopConditionalGenerate,
    Inkling2LoopProcessor,
)

processor = Inkling2LoopProcessor.from_hf("thinkingmachines/Inkling-2-Mini")
model = Inkling2LoopConditionalGenerate.from_weights(
    "hf:thinkingmachines/Inkling-2-Mini",
    load_dtype="bfloat16",
)

messages = [{"role": "user", "content": "Explain in one sentence why the sky is blue."}]
inputs = processor(messages=messages)
generated = model.generate(inputs["input_ids"], max_new_tokens=64)
print(processor.decode(generated[0]))
```

For 40 GB GPUs, pass `quantization="fp8"` or `quantization="int4"` to `from_weights(...)`.

## API

- `Inkling2LoopConfig`, `Inkling2LoopTextConfig`, `Inkling2VisionConfig`
- `Inkling2LoopModel`: Multimodal backbone returning `{"last_hidden_state": ...}`.
- `Inkling2LoopConditionalGenerate`: Adds unpadded LM head (`200058` tokens), logit softcapping, and JAX-JIT `.generate()`.
- `Inkling2ImageProcessor`, `Inkling2LoopTokenizer`, `Inkling2LoopProcessor`
