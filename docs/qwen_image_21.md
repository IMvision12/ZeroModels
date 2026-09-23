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
(16× spatial, RGBA), and the Qwen3-VL text tower. The whole model is **one
container**, `QwenImage21Model`. `QwenImage21TextToImage` adds `generate`.

The weights are converted once, offline, and hosted: on-the-fly `hf:` conversion
is deliberately **not supported** for diffusion models.

Key facts of the port:

- **Unpatched latents**: the denoiser sees `(B, H · W, 64)` tokens (VAE scale 16;
  no 2×2 packing). At 1024px that is a `(B, 4096, 64)` sequence.
- **Block-causal attention**: text is causal; the target image block is
  bidirectional and can attend to all preceding text.
- **`causal_condition`**: text tokens modulate from `t = 0` (timestep-independent),
  matching Diffusers' KV-cache-ready conditioning.
- **True CFG optional**: Diffusers defaults to `true_cfg_scale=1.0` (no guidance).
  Pass `guidance_scale > 1` with a negative prompt to enable dual forwards.
- **Pre-norm text features**: the text tower returns decoder outputs *before* the
  final RMSNorm, matching Diffusers' forward hook on the language-model norm.
- **Schedulers match Diffusers**: `FlowMatchEulerDiscreteScheduler` with dynamic
  resolution shifting (`mu` from image sequence length) and `shift_terminal`.

Links:

- Source: [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1)
- Reference: [diffusers `QwenImage21Pipeline`](https://huggingface.co/docs/diffusers/api/pipelines/qwenimage)
- License: [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1)
- See also [qwen_image.md](qwen_image.md) (1.0 double-stream / packed),
  [qwen3_vl.md](qwen3_vl.md) (text tower)

## Variants

Preconverted, bfloat16 weights are hosted under `zeromodels/`. Load with
`from_weights("zeromodels/<variant>")`. Each repo is one container: DiT + VAE +
Qwen3-VL text tower, ~28 GiB at 16-bit.

| Variant | Hub | Source |
|---|---|---|
| `qwen-image-2.1` | [`zeromodels/qwen-image-2.1`](https://huggingface.co/zeromodels/qwen-image-2.1) | [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) |

Default `generate_args`: 40 flow-match steps, `guidance_scale=1.0`, 1024×1024.

## API

Configs are typed: `QwenImage21Config` (composite, `model_type` `"qwen_image_21"`)
over `QwenImage21TransformerConfig`, `QwenImage21VAEConfig` and
`QwenImage21TextConfig`, plus the checkpoint's `scheduler_config` and Qwen special
token ids. Each repo's `zm_config.json` parses through it; the constructor stays
flat, with the sub-config fields prefixed `transformer_` / `vae_` / `text_`.

### `QwenImage21TextToImage`

The text-to-image task: the `QwenImage21Model` container plus `BaseDiffusion`'s
`generate`. It supplies `encode_prompt` (ChatML template drop after the text
encoder), `predict_noise` on the transformer, `decode_latents` on the VAE with
mean/std un-normalization, and a `FlowMatchEulerDiscreteScheduler` from
`scheduler_config`.

```python
generate(
    input_ids,
    attention_mask=None,
    negative_input_ids=None,
    negative_attention_mask=None,
    num_inference_steps=None,
    guidance_scale=None,
    seed=None,
    latents=None,
    height=None,
    width=None,
    output_type="image",
)
```

| Arg | Default | Meaning |
|---|---|---|
| `input_ids` | required | ChatML-templated token ids, `**tokenizer(prompts)` |
| `attention_mask` | `None` | padding mask from the tokenizer |
| `negative_input_ids` | `None` | tokenized negative prompt; CFG only when set **and** `guidance_scale > 1` |
| `negative_attention_mask` | `None` | mask for the negative ids |
| `num_inference_steps` | `None` | scheduler steps; `generate_args` (40) when unset |
| `guidance_scale` | `None` | true CFG strength; `generate_args` (1.0) when unset |
| `seed` | `None` | seed for the initial latent |
| `latents` | `None` | explicit unpatched initial latent `(B, H·W, 64)` |
| `height` / `width` | `None` | output pixel size; `default_sample_size * 16` (1024) when unset |
| `output_type` | `"image"` | `"image"` for uint8 RGB, `"latent"` for sequence latents |

Returns `(batch, height, width, 3)` uint8 numpy images (VAE RGBA is cropped to RGB).

| Constructor arg | Default | Meaning |
|---|---|---|
| `scheduler` | `None` | a `BaseScheduler`; built from `scheduler_config` when unset |
| `transformer_sample_size` | `64` | latent side the DiT graph is built for (image / 16) |
| `vae_sample_size` | `1024` | image size the VAE graphs are built for |
| `transformer_*` / `vae_*` / `text_*` | Qwen-Image-2.1 | flat sub-config fields (see [Configuration](configuration.md)) |
| `bos_token_id` / `eos_token_id` / `pad_token_id` | `151643` / `151645` / `151643` | Qwen special tokens |
| `scheduler_config` | `None` | Diffusers scheduler dict (dynamic shifting, `max_shift=0.9`, ...) |
| `prompt_template_encode_start_idx` | `14` | ChatML system-prefix tokens dropped after the text encoder |
| `max_sequence_length` | `512` | prompt embed length after the template drop |

### `QwenImage21Model`

The container: one functional model whose graph is three disconnected paths.
Inputs cover the transformer (`sample`, `timestep`, `encoder_hidden_states`,
`encoder_hidden_states_mask`), the VAE (`image`, `latent`) and the text tower
(`token_ids`, `padding_mask`). Components are exposed as `.transformer`, `.vae`
and `.text_encoder`.

### `QwenImage21Transformer2DModel`

The denoiser (Diffusers' `QwenImage21Transformer2DModel`): a 32-layer
single-stream DiT over **unpatched** latents and Qwen3-VL text features, with
3-axis RoPE (`axes_dims_rope=(16, 56, 56)`). Inputs
`{"sample": (B, seq, 64), "timestep": (B,), "encoder_hidden_states": (B, text_seq, 4096)}`
(plus mask), output `{"sample": (B, seq, 64)}` target velocity.

| Arg | Default | Meaning |
|---|---|---|
| `patch_size` | `1` | unpatched (asserted) |
| `in_channels` / `out_channels` | `64` / `64` | latent token width |
| `num_layers` | `32` | single-stream blocks |
| `attention_head_dim` / `num_attention_heads` | `128` / `32` | head geometry (inner dim 4096) |
| `context_in_dim` | `4096` | text feature width from Qwen3-VL |
| `mlp_ratio` | `3` | SwiGLU expansion |
| `causal_condition` | `True` | text modulates from `t=0` |
| `sample_size` | `64` | latent spatial side the graph is built for |
| `text_seq_len` | `512` | static text length after the template drop |

### `AutoencoderKLQwenImage21`

The VAE (Diffusers' `AutoencoderKLQwenImage21`): residual encoder/decoder,
`encode(image)` → `(B, H/16, W/16, 64)` latents, `decode(latent)` →
`(B, H, W, 4)` RGBA in `[-1, 1]`. Latent normalisation uses `latents_mean` /
`latents_std` around the denoiser.

| Arg | Default | Meaning |
|---|---|---|
| `base_dim` / `decoder_base_dim` | `96` / `144` | channel widths |
| `z_dim` | `64` | latent channels |
| `dim_mult` | `(1, 2, 4, 8, 8)` | width multipliers per level |
| `scale_factor_spatial` | `16` | pixel → latent downscale |
| `input_channels` / `out_channels` | `4` / `4` | RGBA |
| `is_residual` | `True` | residual blocks |
| `sample_size` | `1024` | image size the graphs are built for |
| `latents_mean` / `latents_std` | Qwen-Image-2.1 | per-channel latent normalisation |

### `QwenImage21TextEncoderModel`

Qwen3-VL text tower only (no vision / LM head). Returns
`{"last_hidden_state"}` as **pre-final-norm** hidden states.

### Schedulers

| Scheduler | Notes |
|---|---|
| `FlowMatchEulerDiscreteScheduler` | `use_dynamic_shifting=True`, exponential time shift, `shift_terminal=0.02`, `max_shift=0.9` |

At generate time the task sets timesteps with resolution-dependent `mu` and a
linspace sigma schedule, matching Diffusers.

## Preprocessing

### `QwenImage21Tokenizer`

Qwen3 BPE with Diffusers' T2I ChatML template (`Comprehend and analyze the
provided prompt.`). Returns `{"input_ids", "attention_mask"}`. `encode_prompt`
drops `prompt_template_encode_start_idx` (14) tokens and keeps at most
`max_sequence_length` (512).

```python
QwenImage21Tokenizer(hf_id=None, tokenizer_file=None, max_seq_len=1024)
```

## End-to-end example

### Single prompt

```python
import os

os.environ["KERAS_BACKEND"] = "torch"

from PIL import Image
from zeromodels.models.qwen_image_21 import (
    QwenImage21TextToImage,
    QwenImage21Tokenizer,
)

model = QwenImage21TextToImage.from_weights("zeromodels/qwen-image-2.1")
tokenizer = QwenImage21Tokenizer.from_weights("zeromodels/qwen-image-2.1")

inputs = tokenizer("a photo of a capybara wearing a wizard hat, soft window light")
images = model.generate(
    **inputs,
    height=1024,
    width=1024,
    num_inference_steps=40,
    guidance_scale=1.0,
    seed=0,
)

Image.fromarray(images[0]).save("capybara.png")  # (1024, 1024, 3) uint8
```

### Optional true CFG

```python
inputs = tokenizer("a watercolor lighthouse at sunset")
negative = tokenizer("blurry, low quality")["input_ids"]

images = model.generate(
    **inputs,
    negative_input_ids=negative,
    guidance_scale=4.0,
    height=1024,
    width=1024,
)
```

### Reproducible latents

At 1024px the unpatched noise is `(batch, 64, 64, 64)` → `(batch, 4096, 64)`:

```python
import numpy as np
from zeromodels.models.qwen_image_21.qwen_image_21_model import pack_latents

h, w, channels = 64, 64, 64
noise = np.random.default_rng(0).standard_normal((1, h, w, channels)).astype("float32")
latents = pack_latents(noise, h, w)
images = model.generate(**tokenizer("a bowl of ramen"), latents=latents)
```

### Other resolutions

Graphs are built for a fixed size; weights are not. Rebuild with constructor
overrides (multiples of 32px; checkpoint targets 1024):

```python
model = QwenImage21TextToImage.from_weights(
    "zeromodels/qwen-image-2.1",
    transformer_sample_size=32,
    vae_sample_size=512,
)
images = model.generate(**tokenizer("a mountain lake at dawn"), height=512, width=512)
```

### Offline conversion

```bash
pip install zeromodels[conversion]
# KERAS_BACKEND=torch; prefer CPU for the full ~28 GiB bf16 build
python -m zeromodels.models.qwen_image_21.convert_qwen_image_21_diffusers_to_keras
```

`transfer_qwen_image_21(repo)` streams Diffusers shards into a Keras container and
writes sharded `*.weights.json`. Building the full text tower on GPU can OOM
during Functional shape tracing; convert on CPU, then load for CUDA inference.

## Data Format

**Channels-last for the VAE.** Transformer works on sequence tokens.

| | Shape |
|---|---|
| `latents` passed to `generate` (unpatched) | `(batch, (H/16)·(W/16), 64)` |
| VAE encode input | `(batch, H, W, 4)` RGBA in `[-1, 1]` |
| VAE decode output | `(batch, H, W, 4)`; `generate` returns RGB uint8 |
| VAE latent | `(batch, H/16, W/16, 64)` |
| Transformer `sample` | `(batch, seq, 64)` |

## Memory and speed

The bf16 container is about 28 GiB (DiT ~14 GiB + text ~14 GiB + VAE ~1.4 GiB).
Building the full Functional graph at 1024px on GPU can OOM during attention shape
tracing (layers use `compute_output_spec` to avoid materializing seq² temps). Prefer
CPU convert / build, then CUDA generate. A 1024px run needs substantial activation
headroom on an 80 GB card; prefer fused attention
(`keras.ops.dot_product_attention` / torch SDPA).

## Loading Fine-tuned Weights

The hosted checkpoint is the supported weight; any repo laid out like it
(`zm_config.json` declaring `QwenImage21Model`, sharded `*.weights.json` /
`*.weights.h5`, `tokenizer.json`) loads with `from_weights("<org>/<repo>")`. The
`hf:` prefix raises for diffusion models: convert Diffusers format once with
`convert_qwen_image_21_diffusers_to_keras.py` and host the result.
