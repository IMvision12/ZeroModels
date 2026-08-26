import numpy as np
from tqdm import tqdm

from kerasformers.conversion.exceptions import WeightMappingError
from kerasformers.conversion.weight_transfer_util import transfer_weights

from .qwen2_vl_model import Qwen2VLConditionalGenerate

WEIGHT_NAME_MAPPING = {
    "token_embedding.embeddings": "model.embed_tokens.weight",
    "language_model.final_norm.weight": "model.norm.weight",
    "language_model.": "model.",
    "decoder_layer_": "layers.",
    "attention.query": "self_attn.q_proj",
    "attention.key": "self_attn.k_proj",
    "attention.value": "self_attn.v_proj",
    "attention.output_proj": "self_attn.o_proj",
    "attention_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    "mlp.gate": "mlp.gate_proj",
    "mlp.up": "mlp.up_proj",
    "mlp.down": "mlp.down_proj",
    "blocks_": "blocks.",
    "merger.mlp_fc1": "merger.mlp.0",
    "merger.mlp_fc2": "merger.mlp.2",
    "gamma": "weight",
    "beta": "bias",
    "kernel": "weight",
}


def transfer_qwen2_vl_weights(keras_model, hf_state_dict):
    """Load an HF Qwen2-VL state dict into a (freshly built) Keras model in place.

    Builds the model on a multimodal dummy input, accepts either HF key layout
    (``visual.* / model.*`` or ``model.visual.* / model.language_model.*``), maps
    each Keras weight path to its HF name, and copies it, reshaping the vision
    Conv3d patch embed ``(embed_dim, in, t, p, p) -> (in*t*p*p, embed_dim)``.
    """
    if not keras_model.built or not keras_model.weights:
        m = keras_model.spatial_merge_size
        h = w = 2 * m
        n_merged = (h * w) // (m * m)
        keras_model(
            {
                "input_ids": np.array(
                    [[0] + [keras_model.image_token_id] * n_merged + [1]], dtype="int64"
                ),
                "pixel_values": np.zeros(
                    (h * w, keras_model.patch_dim), dtype="float32"
                ),
                "image_grid_thw": np.array([[1, h, w]], dtype=np.int64),
            }
        )

    state = {}
    for k, v in hf_state_dict.items():
        if k.startswith("model.visual."):
            k = k[len("model.") :]
        elif k.startswith("model.language_model."):
            k = "model." + k[len("model.language_model.") :]
        state[k] = v

    for weight in tqdm(keras_model.weights, desc="Transferring weights to Keras"):
        # Functional model weight paths are flat (no model-name root to strip); this
        # yields the same names the old subclassed split() did.
        name = weight.path.replace("/", ".")
        for old, new in WEIGHT_NAME_MAPPING.items():
            name = name.replace(old, new)
        if name not in state:
            raise WeightMappingError(weight.path, name)
        torch_weight = state[name]
        if "patch_embed" in weight.path:
            tw = np.asarray(torch_weight)
            torch_weight = tw.reshape(tw.shape[0], -1)
        transfer_weights(weight.path, weight, torch_weight)


if __name__ == "__main__":
    import gc

    import torch
    from keras import ops
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    HF_ID = "Qwen/Qwen2-VL-2B-Instruct"
    print(f"[1/4] Loading HF {HF_ID} (float32, cpu)")
    hf = Qwen2VLForConditionalGeneration.from_pretrained(
        HF_ID, torch_dtype=torch.float32
    ).eval()
    processor = AutoProcessor.from_pretrained(HF_ID)

    print("[2/4] Building Keras model + transferring weights")
    state = {k: v.detach().cpu().numpy() for k, v in hf.state_dict().items()}
    model = Qwen2VLConditionalGenerate.from_weights(
        HF_ID.replace("Qwen/", "").lower(), load_weights=False
    )
    transfer_qwen2_vl_weights(model, state)

    print("[3/4] Building a real image+text input")
    from PIL import Image

    img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the image."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[img], return_tensors="pt")

    print("[4/4] Comparing logits")
    with torch.no_grad():
        hf_logits = hf(**inputs).logits.float().cpu().numpy()
    k_logits = ops.convert_to_numpy(
        model(
            {
                "input_ids": inputs["input_ids"].cpu().numpy(),
                "pixel_values": inputs["pixel_values"].float().cpu().numpy(),
                "image_grid_thw": inputs["image_grid_thw"].cpu().numpy(),
            }
        )["logits"]
    )
    diff = float(np.max(np.abs(hf_logits - k_logits)))
    print(f"  max abs logit diff: {diff:.6e}")
    assert diff < 1e-2, f"parity too high: {diff:.6e}"
    del hf, state
    gc.collect()
