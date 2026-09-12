# Stable Diffusion 2

<div class="kf-note kf-note--weights">
<b>Weights:</b> pretrained Keras weights live on Hugging Face under
<a href="https://huggingface.co/zeromodels">zeromodels/&lt;variant&gt;</a>
(each repo carries <code>zm_config.json</code> + <code>model.weights.h5</code> +
<code>tokenizer.json</code>). Load with <code>from_weights("zeromodels/&lt;variant&gt;")</code>.
</div>

Stable Diffusion 2.x, ported to pure Keras 3: the [Stable Diffusion](stable_diffusion.md)
latent-diffusion architecture with the second-generation conditioning. It reuses the
SD 1.x family end to end (`StableDiffusion2Model` is `StableDiffusionModel`,
`StableDiffusion2TextToImage` is `StableDiffusionTextToImage`, each with the SD 2
configuration), so everything on that page applies: one container per repo, `generate`
from `BaseDiffusion`, the schedulers, both data formats. What changes is the config:

- **Text encoder**: OpenCLIP ViT-H/14's text tower, 1024 wide, 16 heads, `gelu`, shipped
  truncated to its 23rd (penultimate) layer, which SD 2 conditions on.
- **UNet**: 1024-d cross-attention, one head count per level (`(5, 10, 20, 20)`, a 64-wide
  head everywhere, where SD 1.x used 8 heads at every level) and a linear token
  projection in the Transformer2D blocks instead of the 1x1 conv.
- **Tokenizer**: the same CLIP BPE, padded with `!` (id 0) the OpenCLIP way rather than
  `<|endoftext|>`, so the empty prompt used for classifier-free guidance is
  `[<|startoftext|>, <|endoftext|>, !, !, ...]`.
- **768px checkpoints** (`stable-diffusion-2`, `stable-diffusion-2-1`): built at a 96x96
  latent and trained with the **v-prediction** objective; their repos carry a
  v-prediction DDIM scheduler, which `generate` picks up from `scheduler_config`.

The weights are converted once, offline, and hosted: on-the-fly `hf:` conversion is
deliberately **not supported** for diffusion models. The original `stabilityai/*` repos are
no longer on the Hub; the conversion sources are the `sd2-community` mirrors.

Links:

- Paper: [High-Resolution Image Synthesis with Latent Diffusion Models (arXiv:2112.10752)](https://arxiv.org/abs/2112.10752)
- Reference implementation: [diffusers `StableDiffusionPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img)
- License: [CreativeML Open RAIL++-M](https://huggingface.co/sd2-community/stable-diffusion-2-1/blob/main/LICENSE-MODEL)

See also [stable_diffusion.md](stable_diffusion.md), [clip.md](clip.md).

## Variants

Preconverted, float32 weights are hosted under `zeromodels/`. Load with
`from_weights("zeromodels/<variant>")`. Each repo is one container: UNet (866M) + VAE
(84M) + OpenCLIP text encoder (340M), 1.29B parameters, 5.16 GB (4.81 GiB, one
`model.weights.h5`). All four are released under the CreativeML Open RAIL++-M license.

| Variant | Hub | Resolution | Objective | Training |
|---|---|---|---|---|
| `stable-diffusion-2-base` | [`zeromodels/stable-diffusion-2-base`](https://huggingface.co/zeromodels/stable-diffusion-2-base) | 512 | epsilon | from scratch: 550k steps at 256px on LAION-5B (aesthetics >= 4.5), then 850k steps at 512px |
| `stable-diffusion-2` | [`zeromodels/stable-diffusion-2`](https://huggingface.co/zeromodels/stable-diffusion-2) | 768 | v-prediction | 2-base + 150k steps at 768px |
| `stable-diffusion-2-1-base` | [`zeromodels/stable-diffusion-2-1-base`](https://huggingface.co/zeromodels/stable-diffusion-2-1-base) | 512 | epsilon | 2-base + 220k steps at 512px (punsafe 0.98) |
| `stable-diffusion-2-1` | [`zeromodels/stable-diffusion-2-1`](https://huggingface.co/zeromodels/stable-diffusion-2-1) | 768 | v-prediction | 2 + 55k steps (punsafe 0.1) + 155k steps (punsafe 0.98) at 768px |

Use the `-base` checkpoints for 512px images and the others for 768px; each repo's
`zm_config.json` builds the graph at its native size.

## API

Configs are typed: `StableDiffusion2Config` (composite, `model_type`
`"stable_diffusion_2"`) over `StableDiffusion2UNetConfig` (a `UNet2DConditionConfig`
with the SD 2 widths), `AutoencoderKLConfig` and `StableDiffusion2TextConfig` (a
`CLIPTextConfig` with the ViT-H/14 sizes), plus the scheduler config and the token ids
(`pad_token_id` 0). Flat constructor, `unet_` / `vae_` / `text_` prefixes, like SD 1.x.

### `StableDiffusion2TextToImage`

`StableDiffusionTextToImage` with the SD 2 configuration; `generate` is unchanged:

```python
generate(
    input_ids,
    attention_mask=None,
    negative_input_ids=None,
    num_inference_steps=None,
    guidance_scale=None,
    seed=None,
    latents=None,
)
```

Returns `(batch, H, W, 3)` uint8 images at the checkpoint's resolution; `latents` is
`(batch, 64, 64, 4)` for the 512px checkpoints and `(batch, 96, 96, 4)` for the 768px
ones (`channels_first`: channels second). Defaults come from the repo's `generate_args`
(50 steps, guidance 7.5). See [Stable Diffusion](stable_diffusion.md#stablediffusiontexttoimage)
for the argument table.

### `StableDiffusion2Model`

The container, `StableDiffusionModel` with the SD 2 configuration: the same three
disconnected paths (`.unet`, `.vae`, `.text_encoder`) and the same inputs / outputs, with
`encoder_hidden_states` 1024 wide. It loads the same repo as the task class.

The components are the SD 1.x classes, `UNet2DConditionModel` (built with
`num_attention_heads=(5, 10, 20, 20)`, `use_linear_projection=True`,
`cross_attention_dim=1024`) and `AutoencoderKL`; see their tables on the
[Stable Diffusion](stable_diffusion.md#api) page.

## Preprocessing

### `StableDiffusion2Tokenizer`

`StableDiffusionTokenizer` with `!` as the pad token: CLIP BPE, `<|startoftext|>` /
`<|endoftext|>` framing, truncated and `!`-padded to 77 tokens. Returns
`{"input_ids", "attention_mask"}`.

```python
StableDiffusion2Tokenizer(hf_id=None, tokenizer_file=None, max_seq_len=77, pad_token="!")
```

## End-to-end example

```python
import os

os.environ["KERAS_BACKEND"] = "torch"  # or "jax" / "tensorflow"

from PIL import Image
from zeromodels.models.stable_diffusion_2 import (
    StableDiffusion2TextToImage,
    StableDiffusion2Tokenizer,
)

model = StableDiffusion2TextToImage.from_weights("zeromodels/stable-diffusion-2-1-base")
tokenizer = StableDiffusion2Tokenizer.from_weights("zeromodels/stable-diffusion-2-1-base")

inputs = tokenizer(
    "a steaming bowl of ramen on a wooden table, food photography, shallow depth of field"
)
images = model.generate(**inputs, num_inference_steps=50, guidance_scale=7.5, seed=2)

Image.fromarray(images[0]).save("ramen.png")  # (512, 512, 3) uint8
```

<img src="../assets/stable_diffusion_2_ramen.jpg" alt="Stable Diffusion 2.1-base: a steaming bowl of ramen on a wooden table, 512px" width="380">

### 768px, v-prediction

The 768px checkpoints need no extra arguments; the v-prediction DDIM scheduler and the
96x96 latent come from the repo:

```python
model = StableDiffusion2TextToImage.from_weights("zeromodels/stable-diffusion-2-1")
tokenizer = StableDiffusion2Tokenizer.from_weights("zeromodels/stable-diffusion-2-1")
images = model.generate(**tokenizer("a lighthouse on a cliff at dusk, oil painting"))  # (1, 768, 768, 3)
```

Negative prompts, batching, explicit `latents` for cross-backend reproducibility, other
resolutions and the container-only use work exactly as on the
[Stable Diffusion](stable_diffusion.md#end-to-end-example) page.

### Verified against diffusers

The ramen prompt above with the same initial latent, PNDM 50 steps and guidance 7.5
through `StableDiffusion2TextToImage` (left) and diffusers' `StableDiffusionPipeline`
(right) on `stable-diffusion-2-1-base`, both fp32:

<img src="../assets/stable_diffusion_2_text_to_image_output.jpg" alt="zeromodels (left) and diffusers (right) generations of a bowl of ramen from stable-diffusion-2-1-base with the same latent" width="760">

```
tokenizer input_ids           identical
text_encoder                  max|d|=1.5e-05
unet noise_pred (t=981)       max|d|=7.2e-07
vae decode                    max|d|=2.6e-06
final image (uint8)           max|d|=1  mean|d|=0.0095  PSNR=68.3 dB  identical_pixels=97.22%
```

The 768px v-prediction checkpoint (`stable-diffusion-2-1`, DDIM 20 steps, same latent)
matches the same way: UNet 8.3e-6, VAE 7.9e-5, final image max 1 uint8 level, 99.3% of
pixels identical, PSNR 74.6 dB.

## Data Format

Both `channels_last` and `channels_first` are supported, as for
[Stable Diffusion](stable_diffusion.md#data-format); `generate` always returns
`(batch, H, W, 3)` uint8.

## Memory and speed

The fp32 container is 5.16 GB. At 512px a guided batch-1 run needs a little over 7 GB of
GPU memory in eager Torch (about 1.5x diffusers' eager time); at 768px the self-attention
runs over 9216 tokens, so the guided run needs roughly 12 GB, or the CPU.

## Loading Fine-tuned Weights

Any repo laid out like the hosted ones (`zm_config.json` declaring
`StableDiffusion2Model`, the weights, `tokenizer.json`) loads with
`from_weights("<org>/<repo>")`. The `hf:` prefix raises for diffusion models: convert a
diffusers-format SD 2 checkpoint once with
`zeromodels/models/stable_diffusion_2/convert_stable_diffusion_2_diffusers_to_keras.py`
(`build_from_diffusers(repo)`, `pip install zeromodels[conversion]`) and host the result.
