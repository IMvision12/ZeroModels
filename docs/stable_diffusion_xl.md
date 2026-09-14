# Stable Diffusion XL

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + the float16 weights +
<code>tokenizer.json</code>). Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Stable Diffusion XL, ported to pure Keras 3: the [Stable Diffusion](stable_diffusion.md)
latent-diffusion recipe scaled up to 1024px with a 2.6B-parameter UNet and two text
encoders, plus the **refiner**, an image-to-image expert that finishes the base model's
latents, and **SDXL-Turbo**, the base distilled to a few steps. It subclasses the SD 1.x
family (`StableDiffusionXLModel` is a `StableDiffusionModel` with a fourth component,
`StableDiffusionXLTextToImage` is `StableDiffusionTextToImage` with the SDXL conditioning
hooks; the refiner classes subclass those), so everything on that page applies: one
container per repo, `generate` from `BaseDiffusion`, the schedulers, both data formats.
What changes:

- **Two text encoders**: CLIP ViT-L/14's text tower (768-d, the SD 1.x one) and OpenCLIP
  ViT-bigG/14's (1280-d, 32 layers, `gelu`, with a 1280-d `text_projection`). The prompt
  goes through both; their **penultimate** hidden states (no final LayerNorm) are
  concatenated into the UNet's 2048-d cross-attention context, and the second tower's
  projected pooled (EOT) state is the `text_embeds` micro-conditioning.
- **One tokenizer call**: both towers use the same CLIP BPE and differ only in the pad
  token (`<|endoftext|>` vs `!`), so `StableDiffusionXLTokenizer` is the first one and the
  model derives the second tower's ids from the returned `attention_mask`.
- **Micro-conditioning**: the UNet's timestep embedding also receives the pooled text
  embedding and six size / crop values (`original_size`, `crops_coords_top_left`,
  `target_size`, each `(height, width)`), sinusoidally embedded and projected by the
  `add_embedding` MLP (`addition_embed_type="text_time"`).
- **UNet**: three levels (320, 640, 1280) with no attention at the first, (1, 2, 10)
  transformer blocks per Transformer2D, (5, 10, 20) heads (64 wide), linear token
  projection, built at a 128x128 latent (1024px).
- **VAE**: the SDXL autoencoder (`scaling_factor` 0.13025), built in **float32 whatever
  the load dtype** (`force_upcast`: it overflows in float16, as in diffusers).
- **Classifier-free guidance** with no negative prompt uses **zero** embeddings
  (`force_zeros_for_empty_prompt`), not the encoded empty prompt. Default guidance is 5.0.
- **Scheduler**: Euler with `leading` timestep spacing (`steps_offset` 1) for the base
  models and the refiners, ancestral Euler with `trailing` spacing for SDXL-Turbo; all
  come from the repo's `scheduler_config`.
- **Refiner**: a second model (`StableDiffusionXLRefinerModel` /
  `StableDiffusionXLRefinerImageToImage`) with the OpenCLIP tower alone (1280-d
  context), a four-level UNet ((384, 768, 1536, 1536), attention at the middle levels, 4
  blocks per Transformer2D, 2.3B parameters) and an **aesthetic score** as the fifth
  micro-conditioning id. It refines a latent the base left partially denoised
  (`denoising_start`) or any image (`strength`); its empty negative prompt is encoded, not
  zeroed.
- **float16 weights**: SDXL was released in float16, so the repos store that (the VAE in
  float32) and `from_weights` builds the model in float16 by default (6.6 GB of weights).
  Pass `load_dtype="float32"` for a float32 model (13.9 GB).

The weights are converted once, offline, and hosted: on-the-fly `hf:` conversion is
deliberately **not supported** for diffusion models.

Links:

- Paper: [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis (arXiv:2307.01952)](https://arxiv.org/abs/2307.01952)
- Reference implementation: [diffusers `StableDiffusionXLPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl)
- Licenses: [CreativeML Open RAIL++-M](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
  (1.0 base and refiner), [SDXL 0.9 Research License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9/blob/main/LICENSE.md)
  (0.9, gated), [Stability AI Non-Commercial Research Community License](https://huggingface.co/stabilityai/sdxl-turbo/blob/main/LICENSE.md)
  (Turbo)

See also [stable_diffusion.md](stable_diffusion.md), [stable_diffusion_2.md](stable_diffusion_2.md), [clip.md](clip.md).

## Variants

Preconverted weights are hosted under `zeromodels/`. Load with
`from_weights("zeromodels/<variant>")`. A base repo is one container: UNet (2.57B) + VAE
(84M) + CLIP ViT-L/14 text encoder (123M) + OpenCLIP ViT-bigG/14 text encoder (695M),
3.47B parameters, 6.6 GB in float16 (two `model.weights.json` shards); a refiner repo is
UNet (2.26B) + VAE + OpenCLIP text encoder, 3.04B parameters, 5.8 GB.

| Variant | Hub | Class | Resolution | Sampler | License |
|---|---|---|---|---|---|
| `stable-diffusion-xl-base-1.0` | [`zeromodels/stable-diffusion-xl-base-1.0`](https://huggingface.co/zeromodels/stable-diffusion-xl-base-1.0) | `StableDiffusionXLTextToImage` | 1024 | Euler, 50 steps, guidance 5.0 | CreativeML Open RAIL++-M |
| `stable-diffusion-xl-refiner-1.0` | [`zeromodels/stable-diffusion-xl-refiner-1.0`](https://huggingface.co/zeromodels/stable-diffusion-xl-refiner-1.0) | `StableDiffusionXLRefinerImageToImage` | 1024 | Euler, 50 steps, guidance 5.0, strength 0.3 | CreativeML Open RAIL++-M |
| `stable-diffusion-xl-base-0.9` | [`zeromodels/stable-diffusion-xl-base-0.9`](https://huggingface.co/zeromodels/stable-diffusion-xl-base-0.9) | `StableDiffusionXLTextToImage` | 1024 | Euler, 50 steps, guidance 5.0 | SDXL 0.9 Research License (gated) |
| `stable-diffusion-xl-refiner-0.9` | [`zeromodels/stable-diffusion-xl-refiner-0.9`](https://huggingface.co/zeromodels/stable-diffusion-xl-refiner-0.9) | `StableDiffusionXLRefinerImageToImage` | 1024 | Euler, 50 steps, guidance 5.0, strength 0.3 | SDXL 0.9 Research License (gated) |
| `sdxl-turbo` | [`zeromodels/sdxl-turbo`](https://huggingface.co/zeromodels/sdxl-turbo) | `StableDiffusionXLTextToImage` | 512 | Euler ancestral, 1 step, no guidance | Stability AI Non-Commercial Research Community |

The 0.9 checkpoints are the research preview that preceded 1.0 (same architecture,
different weights); their upstream repos are gated behind the research license, so they
are converted with an authorized `HF_TOKEN`. SDXL-Turbo is a research-only,
non-commercial release; its repo defaults (`generate_args`) are 1 step and
`guidance_scale=0.0`, which is how it was trained (do not add guidance).

## API

Configs are typed: `StableDiffusionXLConfig` (composite, `model_type`
`"stable_diffusion_xl"`) over `StableDiffusionXLUNetConfig`, `StableDiffusionXLVAEConfig`,
`StableDiffusionTextConfig` (the CLIP ViT-L/14 tower, as SD 1.x) and
`StableDiffusionXLTextConfig2` (the OpenCLIP ViT-bigG/14 tower, with `projection_dim` and
its own `hidden_act`), plus the scheduler config, the token ids (`pad_token_id_2` for the
second tokenizer's `!`), `force_zeros_for_empty_prompt` and `requires_aesthetics_score`.
Flat constructor with `unet_` / `vae_` / `text_` / `text_2_` prefixes.
`StableDiffusionXLRefinerConfig` (`model_type` `"stable_diffusion_xl_refiner"`) is the
same with `text_config=None`, the `StableDiffusionXLRefinerUNetConfig` denoiser,
`requires_aesthetics_score=True` and `force_zeros_for_empty_prompt=False`.

### `StableDiffusionXLTextToImage`

`StableDiffusionTextToImage` over the SDXL graph; `generate` gains the micro-conditioning
arguments:

```python
generate(
    input_ids,
    attention_mask=None,
    negative_input_ids=None,
    num_inference_steps=None,
    guidance_scale=None,
    seed=None,
    latents=None,
    image=None,
    strength=None,
    denoising_start=None,
    denoising_end=None,
    output_type="image",
    original_size=None,
    crops_coords_top_left=(0, 0),
    target_size=None,
    aesthetic_score=6.0,
    negative_original_size=None,
    negative_crops_coords_top_left=None,
    negative_target_size=None,
    negative_aesthetic_score=2.5,
)
```

| Argument | Description |
|---|---|
| `input_ids`, `attention_mask` | `**tokenizer(prompts)`; the mask tells the model which positions are padding, so the second tower's `!`-padded ids can be derived (without it, everything after the first `<|endoftext|>` counts as padding). |
| `negative_input_ids` | Tokenized negative prompt, one row per prompt or a single row shared by the batch. Without it the unconditional branch is zero embeddings (`force_zeros_for_empty_prompt`), as in diffusers. |
| `num_inference_steps`, `guidance_scale`, `seed`, `latents` | As for [Stable Diffusion](stable_diffusion.md#stablediffusiontexttoimage); defaults from the repo's `generate_args` (50 steps, guidance 5.0 for the base model). |
| `image`, `strength`, `denoising_start`, `denoising_end`, `output_type` | The image-to-image and ensemble controls of [`BaseDiffusion.generate`](main_classes.md#basediffusion): refine an `image` (noised to `strength`), stop early (`denoising_end`) and hand the latent over (`output_type="latent"`), or resume one (`denoising_start`). |
| `original_size`, `crops_coords_top_left`, `target_size` | SDXL's size / crop conditioning, `(height, width)` in pixels; the sizes default to the generated image's size, the crop to `(0, 0)`. |
| `aesthetic_score` | The refiner's fifth conditioning value (6.0), in place of `target_size`; ignored by the base models. |
| `negative_*` | The same for the negative branch; each defaults to its positive value, except `negative_aesthetic_score` (2.5). |

Returns `(batch, H, W, 3)` uint8 images (or the latent); `latents` is
`(batch, 128, 128, 4)` at 1024px (`channels_first`: channels second).

The hooks, for anyone composing their own loop: `encode_prompt(input_ids, attention_mask,
original_size, crops_coords_top_left, target_size)` returns the dict the UNet takes
(`encoder_hidden_states` (batch, 77, 2048), `text_embeds` (batch, 1280), `time_ids`
(batch, 6)); `encode_negative_prompt` zeroes the first two when no negative ids are given;
`predict_noise(latents, timesteps, embeddings)` runs the UNet.

### `StableDiffusionXLRefinerImageToImage`

The refiner task: the same class over the refiner container (`text_encoder_2` alone,
`aesthetic_score` in the conditioning, `generate_args` with `strength` 0.3). Its
`generate` is the one above; with no `image` / `latents` it runs text-to-image, which the
refiner was not trained for. The two intended uses:

```python
base = StableDiffusionXLTextToImage.from_weights(
    "zeromodels/stable-diffusion-xl-base-1.0"
)
refiner = StableDiffusionXLRefinerImageToImage.from_weights(
    "zeromodels/stable-diffusion-xl-refiner-1.0"
)
tokenizer = StableDiffusionXLTokenizer.from_weights(
    "zeromodels/stable-diffusion-xl-base-1.0"
)
inputs = tokenizer("a lighthouse on a rocky coast at dawn, dramatic clouds, cinematic")

# ensemble of experts: the base denoises 80% of the schedule, the refiner the rest
latent = base.generate(
    **inputs, num_inference_steps=50, denoising_end=0.8, output_type="latent"
)
images = refiner.generate(
    **inputs, latents=latent, num_inference_steps=50, denoising_start=0.8
)

# image-to-image: refine any image (uint8 or [0, 1] float, (batch, H, W, 3))
images = refiner.generate(**inputs, image=images, strength=0.3, seed=0)
```

### `StableDiffusionXLModel`

The container: four disconnected paths in one functional model, `.unet` / `.vae` /
`.text_encoder` / `.text_encoder_2`.

| Inputs | Outputs |
|---|---|
| `sample` (B, 128, 128, 4), `timestep` (B,), `encoder_hidden_states` (B, 77, 2048), `text_embeds` (B, 1280), `time_ids` (B, 6) | `noise_pred` (B, 128, 128, 4) |
| `image` (B, 1024, 1024, 3), `latent` (B, 128, 128, 4) | `moments` (B, 128, 128, 8), `image` (B, 1024, 1024, 3) |
| `token_ids` (B, 77), `token_ids_2` (B, 77), `padding_mask` (B, 77) | `prompt_embeds` (B, 77, 2048), `pooled_prompt_embeds` (B, 1280) |

Each text tower is a functional CLIP text model (the `clip` family's layers) whose outputs
are `penultimate_hidden_state`, `last_hidden_state`, `pooler_output` and, for the second,
`text_embeds`. The UNet is the SD 1.x `UNet2DConditionModel` with the SDXL config
(`transformer_layers_per_block=(1, 2, 10)`, `addition_embed_type="text_time"`); the VAE
is `AutoencoderKL` with `force_upcast=True`. `StableDiffusionXLRefinerModel` is the same
container without the first tower (`token_ids_2` + `padding_mask` in, `prompt_embeds`
(B, 77, 1280) out) and with `time_ids` (B, 5).

## Preprocessing

### `StableDiffusionXLTokenizer`

`StableDiffusionTokenizer` (CLIP BPE, `<|startoftext|>` / `<|endoftext|>` framing,
`<|endoftext|>`-padded to 77 tokens). Returns `{"input_ids", "attention_mask"}`; pass both
to `generate`.

```python
StableDiffusionXLTokenizer(
    hf_id=None, tokenizer_file=None, max_seq_len=77, pad_token="<|endoftext|>"
)
```

A literal `!` in a prompt is encoded as OpenCLIP encodes it (the BPE token `!</w>`, id
256) in the second tower's derived ids; diffusers' `tokenizer_2`, whose pad token is `!`,
turns it into the pad id 0 instead.

## End-to-end example

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.models.stable_diffusion_xl import (
    StableDiffusionXLTextToImage,
    StableDiffusionXLTokenizer,
)

model = StableDiffusionXLTextToImage.from_weights(
    "zeromodels/stable-diffusion-xl-base-1.0"
)
tokenizer = StableDiffusionXLTokenizer.from_weights(
    "zeromodels/stable-diffusion-xl-base-1.0"
)

inputs = tokenizer(
    "a lighthouse on a rocky coast at dawn, dramatic clouds, cinematic, highly detailed"
)
images = model.generate(**inputs, num_inference_steps=30, guidance_scale=5.0, seed=3)

Image.fromarray(images[0]).save("lighthouse.png")  # (1024, 1024, 3) uint8
```

<img src="../assets/stable_diffusion_xl_lighthouse.jpg" alt="Stable Diffusion XL base 1.0: a lighthouse on a rocky coast at dawn, 1024px" width="380">

### Base + refiner

The SDXL "ensemble of experts": the base runs the first 80% of the schedule and hands its
latent to the refiner, which finishes it. Same prompt and initial latent as above:

```python
refiner = StableDiffusionXLRefinerImageToImage.from_weights(
    "zeromodels/stable-diffusion-xl-refiner-1.0"
)
latent = model.generate(
    **inputs, num_inference_steps=30, denoising_end=0.8, output_type="latent", seed=3
)
images = refiner.generate(
    **inputs, latents=latent, num_inference_steps=30, denoising_start=0.8
)
```

<img src="../assets/stable_diffusion_xl_refiner_lighthouse.jpg" alt="Stable Diffusion XL base + refiner: the lighthouse, refined" width="380">

### SDXL-Turbo

One step, no guidance, 512px (the repo's defaults):

```python
model = StableDiffusionXLTextToImage.from_weights("zeromodels/sdxl-turbo")
tokenizer = StableDiffusionXLTokenizer.from_weights("zeromodels/sdxl-turbo")
images = model.generate(
    **tokenizer(
        "a cinematic shot of a baby raccoon wearing an intricate italian priest robe"
    )
)
```

<img src="../assets/stable_diffusion_xl_turbo_raccoon.jpg" alt="SDXL-Turbo, one step: a baby raccoon in an italian priest robe, 512px" width="380">

### Negative prompts, micro-conditioning

```python
inputs = tokenizer(["a red bicycle", "a blue bicycle"])
negative = tokenizer("blurry, low quality")  # one row, shared by the batch
images = model.generate(
    **inputs,
    negative_input_ids=negative["input_ids"],
    original_size=(2048, 2048),  # "a crop of a larger image": sharper detail
    crops_coords_top_left=(0, 0),
    target_size=(1024, 1024),
    seed=0,
)
```

### Other resolutions

The graphs are built for the repo's resolution; pass overrides to build for another
(the weights are resolution-independent; multiples of 64px):

```python
model = StableDiffusionXLTextToImage.from_weights(
    "zeromodels/stable-diffusion-xl-base-1.0",
    unet_sample_size=(96, 128),
    vae_sample_size=(768, 1024),
)
images = model.generate(**tokenizer("a wide mountain valley"))  # (1, 768, 1024, 3)
```

### float32

```python
model = StableDiffusionXLTextToImage.from_weights(
    "zeromodels/stable-diffusion-xl-base-1.0", load_dtype="float32"
)
```

### Verified against diffusers

The components of `stable-diffusion-xl-base-1.0` and `stable-diffusion-xl-refiner-1.0`
were checked one by one against diffusers 0.39 / transformers in float32 on the same
inputs (the converted float16 weights computed in float32):

```
base     text_encoder penultimate      max|d|=6.1e-05   (values up to 854)
base     text_encoder_2 penultimate    max|d|=3.1e-05
base     text_encoder_2 pooled         max|d|=3.6e-06
base     unet noise_pred (t=981)       max|d|=9.3e-06
base     vae decode (512px)            max|d|=3.5e-05
refiner  text_encoder_2 penultimate    max|d|=3.1e-05
refiner  unet noise_pred (t=181)       max|d|=1.2e-05
```

The pipelines (Euler `leading` spacing, classifier-free guidance with the zeroed or
encoded negative branch, the micro-conditioning values, negative prompts, batches,
`strength`, `denoising_end` / `denoising_start`) were checked against
`StableDiffusionXLPipeline` and `StableDiffusionXLImg2ImgPipeline` on small
random-weight models, where the generated images are pixel-identical on torch and jax.
At full size in float16 (the default load; diffusers in float16 with its float32 VAE),
the lighthouse above matches diffusers to PSNR 49.7 dB (98.5% of pixels within 2 uint8
levels), the base + refiner ensemble to PSNR 51.9 dB (98.8% within 2) and SDXL-Turbo
(1 step, 512px) to PSNR 44.9 dB (92% within 2; a single step amplifies float16 rounding).
The lighthouse, zeromodels (left) and diffusers (right):

<img src="../assets/stable_diffusion_xl_text_to_image_output.jpg" alt="zeromodels (left) and diffusers (right) generations of the lighthouse from stable-diffusion-xl-base-1.0 with the same latent" width="760">

## Data Format

Both `channels_last` and `channels_first` are supported, as for
[Stable Diffusion](stable_diffusion.md#data-format); `generate` always returns
`(batch, H, W, 3)` uint8.

## Memory and speed

Measured on the torch backend with the default float16 load (weights resident: 6.8 GB for
a base model, 6.3 GB for the refiner, the VAE in float32):

| Resolution | Denoising step (guided, batch 1) | VAE decode (float32) |
|---|---|---|
| 512px | peak 7.1 GB, 1.2 s/step on an RTX 4060 Laptop | peak 8.5 GB |
| 768px | peak 8.0 GB | peak 10.5 GB |
| 1024px | peak 10.4 GB | about 14 GB |

Keras' functional executor keeps every layer output of a graph alive until the graph
returns, so the decoder's feature maps at 1024px add several GB on top of the weights: a
16 GB GPU runs 1024px comfortably, a 12 GB GPU 768px; on 8 GB, generate at 512px, or
denoise on the GPU (`output_type="latent"`) and decode the latent on the CPU (about 100 s
at 1024px) with a second, CPU-only process. In float32 the weights alone take 13.9 GB.
The UNet's attention runs the portable `"sdpa"` math by default, which materializes the
4096 x 4096 float32 logits of the 64 x 64 level at 1024px; loading with
`attn_implementation="fused"` (the backend's fused kernel, see
[`fused_attention`](main_classes.md#fused_attention)) cuts the 1024px step from a 10.3 GB
peak to 7.6 GB on the same GPU. On JAX and TensorFlow the denoiser step is compiled once
per run (see [`BaseDiffusion`](main_classes.md#basediffusion)).

## Loading Fine-tuned Weights

Any repo laid out like the hosted ones (`zm_config.json` declaring
`StableDiffusionXLModel`, the weights, `tokenizer.json`) loads with
`from_weights("<org>/<repo>")`. The `hf:` prefix raises for diffusion models: convert a
diffusers-format SDXL checkpoint once with
`zeromodels/models/stable_diffusion_xl/convert_stable_diffusion_xl_diffusers_to_keras.py`
(`transfer_stable_diffusion_xl(repo)`, `pip install zeromodels[conversion]`) and host the result.
