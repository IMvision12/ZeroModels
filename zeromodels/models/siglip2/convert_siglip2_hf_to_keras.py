import gc
from typing import Dict

import keras
import numpy as np

from zeromodels.models.siglip.convert_siglip_hf_to_keras import (
    transfer_siglip_weights,
)
from zeromodels.models.siglip2 import SigLIP2ZeroShotClassify


def transfer_siglip2_weights(keras_model, hf_state_dict: Dict[str, np.ndarray]) -> None:
    transfer_siglip_weights(keras_model, hf_state_dict)


# Per-variant recipes (relocated from siglip2_config.py). The zeromodels repos
# hold a SigLIP2ZeroShotClassify checkpoint; its kf_config.json declares
# SigLIP2ZeroShotClassify and every SigLIP 2 head loads from it.
def _s2(
    image_size, patch_size, vh, vl, vhd, vm, embed, tl, th, tm, vocab=256000, seq=64
):
    return {
        "image_size": image_size,
        "patch_size": patch_size,
        "vision_hidden_dim": vh,
        "vision_num_layers": vl,
        "vision_num_heads": vhd,
        "vision_mlp_dim": vm,
        "vocab_size": vocab,
        "embed_dim": embed,
        "text_hidden_dim": embed,
        "text_num_layers": tl,
        "text_num_heads": th,
        "text_mlp_dim": tm,
        "max_seq_len": seq,
    }


SIGLIP2_RECIPES = {
    "siglip2_base_p16_224": _s2(224, 16, 768, 12, 12, 3072, 768, 12, 12, 3072),
    "siglip2_base_p16_256": _s2(256, 16, 768, 12, 12, 3072, 768, 12, 12, 3072),
    "siglip2_base_p16_384": _s2(384, 16, 768, 12, 12, 3072, 768, 12, 12, 3072),
    "siglip2_base_p16_512": _s2(512, 16, 768, 12, 12, 3072, 768, 12, 12, 3072),
    "siglip2_base_p32_256": _s2(256, 32, 768, 12, 12, 3072, 768, 12, 12, 3072),
    "siglip2_large_p16_256": _s2(256, 16, 1024, 24, 16, 4096, 1024, 24, 16, 4096),
    "siglip2_large_p16_384": _s2(384, 16, 1024, 24, 16, 4096, 1024, 24, 16, 4096),
    "siglip2_large_p16_512": _s2(512, 16, 1024, 24, 16, 4096, 1024, 24, 16, 4096),
    "siglip2_so400m_p14_224": _s2(224, 14, 1152, 27, 16, 4304, 1152, 27, 16, 4304),
    "siglip2_so400m_p14_384": _s2(384, 14, 1152, 27, 16, 4304, 1152, 27, 16, 4304),
    "siglip2_so400m_p16_256": _s2(256, 16, 1152, 27, 16, 4304, 1152, 27, 16, 4304),
    "siglip2_so400m_p16_384": _s2(384, 16, 1152, 27, 16, 4304, 1152, 27, 16, 4304),
    "siglip2_so400m_p16_512": _s2(512, 16, 1152, 27, 16, 4304, 1152, 27, 16, 4304),
}


if __name__ == "__main__":
    from transformers import SiglipModel

    SIGLIP2_CONVERSION_CONFIG = [
        ("siglip2_base_p16_224", "google/siglip2-base-patch16-224"),
        ("siglip2_base_p16_256", "google/siglip2-base-patch16-256"),
        ("siglip2_base_p16_384", "google/siglip2-base-patch16-384"),
        ("siglip2_base_p16_512", "google/siglip2-base-patch16-512"),
        ("siglip2_base_p32_256", "google/siglip2-base-patch32-256"),
        ("siglip2_large_p16_256", "google/siglip2-large-patch16-256"),
        ("siglip2_large_p16_384", "google/siglip2-large-patch16-384"),
        ("siglip2_large_p16_512", "google/siglip2-large-patch16-512"),
        ("siglip2_so400m_p14_224", "google/siglip2-so400m-patch14-224"),
        ("siglip2_so400m_p14_384", "google/siglip2-so400m-patch14-384"),
        ("siglip2_so400m_p16_256", "google/siglip2-so400m-patch16-256"),
        ("siglip2_so400m_p16_384", "google/siglip2-so400m-patch16-384"),
        ("siglip2_so400m_p16_512", "google/siglip2-so400m-patch16-512"),
    ]

    for variant, hf_id in SIGLIP2_CONVERSION_CONFIG:
        print(f"\n{'=' * 60}")
        print(f"Converting: {variant}  <-  {hf_id}")
        print(f"{'=' * 60}")

        hf_model = SiglipModel.from_pretrained(hf_id).eval()
        state = {k: v.detach().cpu().numpy() for k, v in hf_model.state_dict().items()}

        keras_model = SigLIP2ZeroShotClassify(**SIGLIP2_RECIPES[variant])
        transfer_siglip2_weights(keras_model, state)

        total_params = sum(int(np.prod(w.shape)) for w in keras_model.weights)
        total_gb = (total_params * 4) / (1024**3)

        del state
        gc.collect()

        if total_gb <= 5.0:
            import torch

            ctx = keras_model.max_seq_len
            vocab = keras_model.vocab_size
            ishape = keras_model.image_size
            if keras.config.image_data_format() == "channels_first":
                img_h, img_w = ishape[1], ishape[2]
            else:
                img_h, img_w = ishape[0], ishape[1]

            rng = np.random.default_rng(0)
            pixel = rng.standard_normal((2, img_h, img_w, 3)).astype(np.float32)
            token_ids = rng.integers(0, vocab - 1, size=(2, ctx)).astype(np.int32)

            with torch.no_grad():
                hf_out = hf_model(
                    pixel_values=torch.from_numpy(pixel.transpose(0, 3, 1, 2)),
                    input_ids=torch.from_numpy(token_ids.astype(np.int64)),
                )
                hf_logits = hf_out.logits_per_image.cpu().numpy()
                scale = hf_model.logit_scale.exp().item()

            k_out = keras_model(
                {"images": pixel, "token_ids": token_ids}, training=False
            )
            k_logits = keras.ops.convert_to_numpy(k_out["image_logits"])

            logits_diff = float(np.abs(hf_logits - k_logits).max())
            cosine_diff = logits_diff / (scale + 1e-8)
            print(
                f"  Max logits diff: {logits_diff:.6f}  "
                f"(cosine-level: {cosine_diff:.2e})"
            )
            if cosine_diff > 1e-2:
                raise ValueError(
                    f"{variant}: equivalence check failed "
                    f"(logits diff {logits_diff:.4f}, cosine {cosine_diff:.2e})"
                )
        else:
            print(
                f"  Equivalence check skipped (~{total_gb:.1f} GB model exceeds "
                f"RAM budget; weights validated by name-based mapping)"
            )

        if total_gb > 1.7:
            out_path = f"{variant}.weights.json"
            keras_model.save_weights(out_path, max_shard_size=1.7)
            print(f"  Saved -> {out_path} (sharded, ~{total_gb:.2f} GB)")
        else:
            out_path = f"{variant}.weights.h5"
            keras_model.save_weights(out_path)
            print(f"  Saved -> {out_path} (~{total_gb:.2f} GB)")

        del keras_model, hf_model
        keras.backend.clear_session()
        gc.collect()
