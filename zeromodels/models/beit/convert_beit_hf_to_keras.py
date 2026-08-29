import numpy as np


def transfer_beit_weights(keras_model, state_dict):
    def get(key):
        return np.asarray(state_dict[key])

    def has(name):
        try:
            keras_model.get_layer(name)
            return True
        except ValueError:
            return False

    def assign_dense(layer, prefix, bias=True):
        layer.kernel.assign(np.transpose(get(f"{prefix}.weight")))
        if bias:
            layer.bias.assign(get(f"{prefix}.bias"))

    def assign_ln(layer, prefix):
        layer.gamma.assign(get(f"{prefix}.weight"))
        layer.beta.assign(get(f"{prefix}.bias"))

    # Patch embedding + CLS token.
    pe = keras_model.get_layer("patch_embed")
    pe.kernel.assign(
        np.transpose(
            get("beit.embeddings.patch_embeddings.projection.weight"), (2, 3, 1, 0)
        )
    )
    pe.bias.assign(get("beit.embeddings.patch_embeddings.projection.bias"))
    keras_model.get_layer("cls_token").cls_token.assign(
        get("beit.embeddings.cls_token")
    )

    depth = keras_model.num_hidden_layers
    for i in range(depth):
        # The segmentation model prunes the encoder layers past its deepest
        # out_index (their outputs are unused), so only transfer layers present.
        if not has(f"beit_layer_{i}_layernorm_before"):
            continue
        p = f"beit.encoder.layer.{i}"
        assign_ln(
            keras_model.get_layer(f"beit_layer_{i}_layernorm_before"),
            f"{p}.layernorm_before",
        )
        attn = keras_model.get_layer(f"beit_layer_{i}_attn")
        assign_dense(attn.q_proj, f"{p}.attention.attention.query")
        assign_dense(attn.k_proj, f"{p}.attention.attention.key", bias=False)
        assign_dense(attn.v_proj, f"{p}.attention.attention.value")
        assign_dense(attn.o_proj, f"{p}.attention.output.dense")
        attn.relative_position_bias_table.assign(
            get(
                f"{p}.attention.attention.relative_position_bias.relative_position_bias_table"
            )
        )
        keras_model.get_layer(f"beit_layer_{i}_layerscale_1").gamma.assign(
            get(f"{p}.lambda_1")
        )
        assign_ln(
            keras_model.get_layer(f"beit_layer_{i}_layernorm_after"),
            f"{p}.layernorm_after",
        )
        assign_dense(
            keras_model.get_layer(f"beit_layer_{i}_fc1"), f"{p}.intermediate.dense"
        )
        assign_dense(keras_model.get_layer(f"beit_layer_{i}_fc2"), f"{p}.output.dense")
        keras_model.get_layer(f"beit_layer_{i}_layerscale_2").gamma.assign(
            get(f"{p}.lambda_2")
        )

    # Classification head.
    if has("pooler_layernorm"):
        assign_ln(keras_model.get_layer("pooler_layernorm"), "beit.pooler.layernorm")
    if has("predictions"):
        assign_dense(keras_model.get_layer("predictions"), "classifier")


def transfer_beit_seg_head(keras_model, state_dict):
    def get(key):
        return np.asarray(state_dict[key])

    def assign_conv(layer, prefix, bias=False):
        layer.kernel.assign(np.transpose(get(f"{prefix}.weight"), (2, 3, 1, 0)))
        if bias:
            layer.bias.assign(get(f"{prefix}.bias"))

    def assign_bn(layer, prefix):
        layer.gamma.assign(get(f"{prefix}.weight"))
        layer.beta.assign(get(f"{prefix}.bias"))
        layer.moving_mean.assign(get(f"{prefix}.running_mean"))
        layer.moving_variance.assign(get(f"{prefix}.running_var"))

    gl = keras_model.get_layer

    # FPN neck (ConvTranspose uses the same (2, 3, 1, 0) kernel layout as Conv2d).
    assign_conv(gl("fpn1_convtranspose1"), "fpn1.0", bias=True)
    assign_bn(gl("fpn1_bn"), "fpn1.1")
    assign_conv(gl("fpn1_convtranspose2"), "fpn1.3", bias=True)
    assign_conv(gl("fpn2_convtranspose"), "fpn2.0", bias=True)

    # UPerNet decode head.
    for s in range(len(keras_model.pool_scales)):
        assign_conv(gl(f"psp_{s}_conv"), f"decode_head.psp_modules.{s}.1.conv")
        assign_bn(gl(f"psp_{s}_bn"), f"decode_head.psp_modules.{s}.1.bn")
    assign_conv(gl("psp_bottleneck_conv"), "decode_head.bottleneck.conv")
    assign_bn(gl("psp_bottleneck_bn"), "decode_head.bottleneck.bn")
    for j in range(3):
        assign_conv(gl(f"lateral_{j}_conv"), f"decode_head.lateral_convs.{j}.conv")
        assign_bn(gl(f"lateral_{j}_bn"), f"decode_head.lateral_convs.{j}.bn")
        assign_conv(gl(f"fpn_conv_{j}_conv"), f"decode_head.fpn_convs.{j}.conv")
        assign_bn(gl(f"fpn_conv_{j}_bn"), f"decode_head.fpn_convs.{j}.bn")
    assign_conv(gl("fpn_bottleneck_conv"), "decode_head.fpn_bottleneck.conv")
    assign_bn(gl("fpn_bottleneck_bn"), "decode_head.fpn_bottleneck.bn")
    assign_conv(gl("seg_classifier"), "decode_head.classifier", bias=True)


BEIT_CLASSIFY_VARIANTS = {
    "beit-base-patch16-224": "microsoft/beit-base-patch16-224",
    "beit-base-patch16-224-pt22k-ft22k": "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "beit-large-patch16-224": "microsoft/beit-large-patch16-224",
    "beit-large-patch16-512": "microsoft/beit-large-patch16-512",
}


if __name__ == "__main__":
    import importlib.metadata as _meta

    _orig = _meta.version
    _meta.version = lambda name: "0.23.0" if name == "tokenizers" else _orig(name)

    import json

    import keras
    import torch
    import transformers
    from huggingface_hub import hf_hub_download

    from zeromodels.conversion.hf_download_utils import download_hf_state_dict
    from zeromodels.models.beit import BeitImageClassify

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    for variant, hf_id in BEIT_CLASSIFY_VARIANTS.items():
        print(f"\n{'=' * 60}\n{variant}  <-  {hf_id}\n{'=' * 60}")
        hf_config = json.load(
            open(hf_hub_download(hf_id, "config.json"), encoding="utf-8")
        )
        km = BeitImageClassify(
            **BeitImageClassify.config_from_hf(hf_config), include_normalization=False
        )
        transfer_beit_weights(km, download_hf_state_dict(hf_id))

        hm = transformers.BeitForImageClassification.from_pretrained(hf_id).eval()
        size = km.image_size
        np.random.seed(0)
        pixels = np.random.rand(1, size, size, 3).astype("float32")
        norm = ((pixels - 0.5) / 0.5).astype("float32")
        with torch.no_grad():
            hf_logits = hm(
                pixel_values=torch.tensor(norm).permute(0, 3, 1, 2)
            ).logits.numpy()
        kl = keras.ops.convert_to_numpy(km(norm, training=False))
        diff = float(np.max(np.abs(hf_logits - kl)))
        cos = float(
            np.dot(hf_logits.ravel(), kl.ravel())
            / (np.linalg.norm(hf_logits.ravel()) * np.linalg.norm(kl.ravel()))
        )
        print(f"  logits max|diff|={diff:.3e}  cosine={cos:.8f}")
        del km, hm
        keras.backend.clear_session()
