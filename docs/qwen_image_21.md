# Qwen-Image-2.1

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + sharded
<code>*.weights.json</code> / <code>*.weights.h5</code> +
<code>tokenizer.json</code>). Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Qwen-Image-2.1, ported to pure Keras 3: latent text-to-image flow-matching with a
32-layer **single-stream** block-causal DiT, a residual 64-channel KL autoencoder
(16× spatial), and the Qwen3-VL text tower. The whole model is **one container**,
`QwenImage21Model`. `QwenImage21TextToImage` adds `generate`.

Key facts of the port:

- **Unpatched latents**: the denoiser sees `(B, H · W, 64)` tokens (VAE scale 16;
  no 2×2 packing).
- **Block-causal attention**: text is causal; the target image block is
  bidirectional and can attend to all preceding text.
- **`causal_condition`**: text tokens modulate from `t = 0` (timestep-independent),
  matching Diffusers' KV-cache-ready conditioning.
- **True CFG optional**: Diffusers defaults to `true_cfg_scale=1.0` (no guidance).
  Pass `guidance_scale > 1` with a negative prompt to enable dual forwards.
- **Pre-norm text features**: the text tower returns decoder outputs *before* the
  final RMSNorm, matching Diffusers' forward hook.

Links:

- Reference: [diffusers `QwenImage21Pipeline`](https://huggingface.co/docs/diffusers/api/pipelines/qwenimage)
- Source: [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1)
- See also [qwen_image.md](qwen_image.md), [qwen3_vl.md](qwen3_vl.md)

## Variants

| Variant | Hub | Source |
|---|---|---|
| `qwen-image-2.1` | [`zeromodels/qwen-image-2.1`](https://huggingface.co/zeromodels/qwen-image-2.1) | [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) |

Default `generate_args`: 40 flow-match steps, `guidance_scale=1.0`, 1024×1024.

## API

### `QwenImage21TextToImage`

```python
from zeromodels.models.qwen_image_21 import (
    QwenImage21TextToImage,
    QwenImage21Tokenizer,
)

model = QwenImage21TextToImage.from_weights("zeromodels/qwen-image-2.1")
tokenizer = QwenImage21Tokenizer.from_weights("zeromodels/qwen-image-2.1")
image = model.generate(**tokenizer("a capybara in a wizard hat"), height=1024, width=1024)
```

Offline conversion (no on-the-fly `hf:` for diffusion)::

```bash
python -m zeromodels.models.qwen_image_21.convert_qwen_image_21_diffusers_to_keras
```
