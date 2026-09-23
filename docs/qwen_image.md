# Qwen-Image

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights will be hosted on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/qwen-image</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code> +
<code>tokenizer.json</code>). Load with <code>from_weights("zeromodels/qwen-image")</code>.
Conversion from Diffusers is offline only — on-the-fly <code>hf:</code> is not supported.
</div>

Qwen-Image text-to-image, ported to pure Keras 3 from
[Diffusers `QwenImagePipeline`](https://huggingface.co/docs/diffusers/api/pipelines/qwenimage)
(`/Qwen/Qwen-Image`). Latent flow-matching with:

- **Denoiser**: 60-layer double-stream MMDiT (`QwenImageTransformer2DModel`) over
  packed 2×2 latent patches (64-d tokens) with MS-RoPE
- **VAE**: Wan-derived `AutoencoderKLQwenImage` (16 latent channels, 8× spatial)
- **Text encoder**: Qwen2.5-VL-7B Instruct text tower (ChatML prompt template)
- **Scheduler**: `FlowMatchEulerDiscreteScheduler` with dynamic resolution shifting

`QwenImageModel` is the hosted container (transformer + VAE + text tower),
stored as one set of sharded weights (`model.weights.json`, ~53 GiB at 16-bit).
`QwenImageTextToImage` adds `generate` via `BaseDiffusion`, with **true CFG**
(separate cond/uncond forwards and prediction-norm renormalization, Diffusers
`true_cfg_scale`).

## Status

| Piece | Status |
|---|---|
| Configs / tokenizer / auto registry | Done |
| Transformer layers + model | Done (graph) |
| VAE | Done (T=1 image path) |
| Text-to-image `generate` + packed latents | Done |
| Flow-match dynamic shifting | Done |
| Diffusers → Keras weight transfer | In progress (`convert_qwen_image_diffusers_to_keras.py`) |
| Hosted `zeromodels/qwen-image` weights | Pending conversion |

## API sketch

```python
from zeromodels.models.qwen_image import QwenImageTextToImage, QwenImageTokenizer

model = QwenImageTextToImage.from_weights("zeromodels/qwen-image")
tok = QwenImageTokenizer.from_weights("zeromodels/qwen-image")
image = model.generate(
    **tok("a coffee shop entrance with a chalkboard sign"),
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=4.0,
)
```

## Variants

| Variant | Hub (planned) | Source |
|---|---|---|
| `qwen-image` | `zeromodels/qwen-image` | [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image) |
Paper / model card: [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image).
License: Apache-2.0.
