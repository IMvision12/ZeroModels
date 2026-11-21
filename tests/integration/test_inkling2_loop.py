import numpy as np
import pytest
from keras import ops
from PIL import Image

from zeromodels.models.inkling2_loop import (
    Inkling2ImageProcessor,
    Inkling2LoopConditionalGenerate,
    Inkling2LoopConfig,
    Inkling2LoopModel,
)
from zeromodels.models.inkling2_loop.convert_inkling2_loop_hf_to_keras import (
    BYTE_TO_E2M1_PAIR,
    deinterleave_dim1,
    dequantize_inkling2_experts,
    merge_message_queries,
    transfer_inkling2_loop_weights,
)


def _tiny_kwargs():
    return {
        "vocab_size": 128,
        "unpadded_vocab_size": 120,
        "embed_dim": 64,
        "num_layers": 6,
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 16,
        "swa_num_heads": 4,
        "swa_num_kv_heads": 2,
        "swa_head_dim": 16,
        "sliding_window_size": 4,
        "d_rel": 4,
        "rel_extent": 8,
        "log_scaling_n_floor": 4,
        "log_scaling_alpha": 0.1,
        "norm_eps": 1e-6,
        "conv_kernel_size": 4,
        "moe_mlp_dim": 32,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "route_scale": 2.0,
        "n_layers_per_attn_res_block": 2,
        "loop_layer_index_start": 2,
        "loop_layer_index_end": 4,
        "loop_num_passes": 2,
        "loop_gate_rank": 8,
        "vision_embed_dim": 32,
        "vision_depth": 2,
        "vision_num_heads": 4,
        "vision_mlp_dim": 64,
        "in_channels": 3,
        "patch_size": 4,
        "spatial_merge_size": 2,
        "posemb_grid_size": 4,
        "projection_hidden_dim": 48,
        "projector_bias": False,
        "vision_out_dim": 64,
        "image_token_id": 10,
    }


def _randomize_weights(model, seed=0):
    rng = np.random.default_rng(seed)
    for w in model.weights:
        arr = rng.normal(0.0, 0.05, size=w.shape).astype(np.float32)
        w.assign(arr)


def test_nvfp4_and_mxfp8_dequantize_and_deinterleave():
    # NVFP4: 2 experts, 4 rows, 32 cols -> packed (2, 4, 16) uint8, scale (2, 4, 2)
    rng = np.random.default_rng(42)
    packed = rng.integers(0, 256, size=(2, 4, 16), dtype=np.uint8)
    scale = rng.uniform(0.25, 2.0, size=(2, 4, 2)).astype(np.float32)
    amax = np.array([3.0, 6.0], dtype=np.float32)
    denom = np.array([1.5, 2.0], dtype=np.float32)

    out = dequantize_inkling2_experts(
        packed, scale=scale, amax=amax, outer_scale_denominator=denom, out_dtype=np.float32
    )
    expected = (
        BYTE_TO_E2M1_PAIR[packed].reshape(2, 4, 2, 16)
        * scale[..., None]
        * (amax / denom)[:, None, None, None]
    ).reshape(2, 4, 32)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

    # MXFP8: 2 experts, 4 rows, 64 cols -> scale (2, 4, 2) uint8 ue8m0
    fp8_vals = rng.normal(0.0, 1.0, size=(2, 4, 64)).astype(np.float32)
    ue8m0 = np.array([[[127, 128]] * 4, [[126, 129]] * 4], dtype=np.uint8)
    out_mx = dequantize_inkling2_experts(fp8_vals, scale=ue8m0, out_dtype=np.float32)
    exp_mx = (
        fp8_vals.reshape(2, 4, 2, 32) * np.exp2(ue8m0.astype(np.float32) - 127.0)[..., None]
    ).reshape(2, 4, 64)
    np.testing.assert_allclose(out_mx, exp_mx, rtol=1e-5, atol=1e-5)

    # Deinterleave dim 1
    x = np.arange(2 * 6 * 4, dtype=np.float32).reshape(2, 6, 4)
    deint = deinterleave_dim1(x)
    assert deint.shape == x.shape
    np.testing.assert_array_equal(deint[:, :3, :], x[:, 0::2, :])
    np.testing.assert_array_equal(deint[:, 3:, :], x[:, 1::2, :])


def test_merge_message_queries():
    # 3 loop layers, H=4
    q_mlp = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    kv = (100 + np.arange(6 * 4, dtype=np.float32)).reshape(6, 4)
    merged = merge_message_queries(q_mlp, kv)
    assert merged.shape == (3, 4, 4)
    for i in range(3):
        np.testing.assert_array_equal(merged[i, 0], q_mlp[2 * i])
        np.testing.assert_array_equal(merged[i, 1], kv[2 * i])
        np.testing.assert_array_equal(merged[i, 2], kv[2 * i + 1])
        np.testing.assert_array_equal(merged[i, 3], q_mlp[2 * i + 1])


def test_image_processor_rgb_and_rgba():
    proc = Inkling2ImageProcessor()
    rgb = Image.fromarray(np.full((45, 70, 3), 128, dtype=np.uint8), mode="RGB")
    out = proc(rgb)
    assert out["image_grid_thw"].tolist() == [[1, 2, 2]]
    assert out["pixel_values"].shape == (4, 40, 40, 3)

    # RGBA transparency composites onto checkerboard (204 / 153)
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)  # alpha=0 everywhere
    out_rgba = proc(Image.fromarray(rgba, mode="RGBA"))
    assert out_rgba["image_grid_thw"].tolist() == [[1, 1, 1]]
    pv = out_rgba["pixel_values"][0]
    # Top-left (0,0) -> checkerboard 204; (0,8) -> checkerboard 153
    np.testing.assert_allclose(pv[0, 0, 0], (204.0 / 255.0 - 0.5) / 0.5, atol=1e-5)
    np.testing.assert_allclose(pv[0, 8, 0], (153.0 / 255.0 - 0.5) / 0.5, atol=1e-5)
    # Padded pixel (16, 0) was -1.0 before rescale
    np.testing.assert_allclose(pv[16, 0, 0], (-1.0 / 255.0 - 0.5) / 0.5, atol=1e-5)


def test_forward_multimodal_and_text_only():
    kw = _tiny_kwargs()
    model = Inkling2LoopConditionalGenerate(**kw)
    _randomize_weights(model, seed=1)

    # Text-only
    ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    out = model({"input_ids": ids})
    assert out["last_hidden_state"].shape == (1, 5, kw["embed_dim"])
    assert out["logits"].shape == (1, 5, kw["unpadded_vocab_size"])

    # Multimodal: 1 image with grid (1, 2, 2) -> 4 image tokens
    mm_ids = np.array([[1, 10, 10, 10, 10, 2]], dtype=np.int32)
    pv = np.random.default_rng(2).normal(size=(4, 8, 8, 3)).astype(np.float32)
    grid = np.array([[1, 2, 2]], dtype=np.int32)
    mm_out = model({"input_ids": mm_ids, "pixel_values": pv, "image_grid_thw": grid})
    assert mm_out["logits"].shape == (1, 6, kw["unpadded_vocab_size"])
    assert np.all(np.isfinite(ops.convert_to_numpy(mm_out["logits"])))


def test_prefill_vs_incremental_decode_equivalence():
    """Verify full-sequence forward matches build_cache + token-by-token call_with_cache
    across pre-loop, looped (multi-pass with cross-pass messages), and post-loop layers."""
    kw = _tiny_kwargs()
    model = Inkling2LoopConditionalGenerate(**kw)
    _randomize_weights(model, seed=7)

    seq = np.array([[2, 5, 8, 11, 14, 17, 20, 23]], dtype=np.int32)
    full_logits = ops.convert_to_numpy(model({"input_ids": seq})["logits"])

    # Prefill first 4 tokens, then decode tokens 4, 5, 6, 7 one by one
    prefill_len = 4
    max_len = seq.shape[1]
    cache, logits_p = model.build_cache(
        ops.convert_to_tensor(seq[:, :prefill_len]), None, max_len
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(logits_p), full_logits[:, prefill_len - 1, :], rtol=1e-4, atol=1e-4
    )

    for step in range(prefill_len, max_len):
        tok = ops.convert_to_tensor(seq[:, step : step + 1])
        logits_s, cache = model.call_with_cache(tok, cache, step)
        np.testing.assert_allclose(
            ops.convert_to_numpy(logits_s),
            full_logits[:, step, :],
            rtol=1e-4,
            atol=1e-4,
        )


def test_generate_and_serialization():
    kw = _tiny_kwargs()
    model = Inkling2LoopConditionalGenerate(**kw)
    _randomize_weights(model, seed=3)

    prompt = np.array([[1, 2, 3, 4]], dtype=np.int32)
    gen = model.generate(prompt, max_new_tokens=4)
    assert gen.shape == (1, 4)

    # Config roundtrip
    cfg = model.get_config()
    rebuilt = Inkling2LoopConditionalGenerate.from_config(cfg)
    assert len(rebuilt.weights) == len(model.weights)

    # BaseConfig roundtrip
    base_cfg = Inkling2LoopConfig()
    assert base_cfg.model_type == "inkling2_loop"
    assert isinstance(Inkling2LoopModel(**kw), Inkling2LoopModel)


def test_synthetic_hf_weight_transfer():
    """Verify transfer_inkling2_loop_weights maps every single HF key cleanly."""
    kw = _tiny_kwargs()
    model = Inkling2LoopConditionalGenerate(**kw)
    rng = np.random.default_rng(99)

    H = kw["embed_dim"]
    V = kw["vocab_size"]
    E = kw["num_experts"]
    I = kw["moe_mlp_dim"]
    K = kw["conv_kernel_size"]
    NH, NKV, HD = kw["num_heads"], kw["num_kv_heads"], kw["head_dim"]
    DREL, REXT = kw["d_rel"], kw["rel_extent"]
    R = kw["loop_gate_rank"]
    VH, VMLP = kw["vision_embed_dim"], kw["vision_mlp_dim"]
    VP = kw["patch_size"]
    VM = kw["spatial_merge_size"]
    VG = kw["posemb_grid_size"]
    VPROJ = kw["projection_hidden_dim"]

    sd = {
        "model.llm.embed.weight": rng.normal(0, 0.02, (V, H)).astype(np.float32),
        "model.llm.embed_norm.weight": np.ones((H,), dtype=np.float32),
        "model.llm.norm.weight": np.ones((H,), dtype=np.float32),
        "model.llm.unembed.weight": rng.normal(0, 0.02, (V, H)).astype(np.float32),
        "model.llm.attn_res.output_query": rng.normal(0, 0.02, (H,)).astype(np.float32),
        "model.llm.middle_cycle_cross_pass_queries": rng.normal(0, 0.02, (4, H)).astype(np.float32),
        "model.llm.middle_cycle_cross_pass_kv_queries": rng.normal(0, 0.02, (4, H)).astype(np.float32),
        "model.visual.backbone.embeddings.patch_embedding.weight": rng.normal(
            0, 0.02, (VH, 3 * VP * VP)
        ).astype(np.float32),
        "model.visual.backbone.embeddings.position_embedding": rng.normal(
            0, 0.02, (VG * VG, VH)
        ).astype(np.float32),
        "model.visual.backbone.embeddings.class_embedding": rng.normal(0, 0.02, (VH,)).astype(np.float32),
        "model.visual.backbone.embeddings.cls_position_embedding": rng.normal(0, 0.02, (VH,)).astype(np.float32),
        "model.visual.backbone.ln_pre.weight": np.ones((VH,), dtype=np.float32),
        "model.visual.backbone.ln_pre.bias": np.zeros((VH,), dtype=np.float32),
        "model.visual.proj_in.weight": rng.normal(0, 0.02, (VPROJ, VM * VM * VH)).astype(np.float32),
        "model.visual.proj_out.weight": rng.normal(0, 0.02, (H, VPROJ)).astype(np.float32),
    }
    for vi in range(kw["vision_depth"]):
        vb = f"model.visual.backbone.layers.{vi}"
        sd[f"{vb}.ln_1.weight"] = np.ones((VH,), dtype=np.float32)
        sd[f"{vb}.ln_1.bias"] = np.zeros((VH,), dtype=np.float32)
        sd[f"{vb}.ln_2.weight"] = np.ones((VH,), dtype=np.float32)
        sd[f"{vb}.ln_2.bias"] = np.zeros((VH,), dtype=np.float32)
        sd[f"{vb}.ls_1.gamma"] = np.full((VH,), 0.1, dtype=np.float32)
        sd[f"{vb}.ls_2.gamma"] = np.full((VH,), 0.1, dtype=np.float32)
        for p in ("q_proj", "k_proj", "v_proj", "out_proj"):
            sd[f"{vb}.self_attn.{p}.weight"] = rng.normal(0, 0.02, (VH, VH)).astype(np.float32)
            sd[f"{vb}.self_attn.{p}.bias"] = np.zeros((VH,), dtype=np.float32)
        sd[f"{vb}.mlp.c_fc.weight"] = rng.normal(0, 0.02, (VMLP, VH)).astype(np.float32)
        sd[f"{vb}.mlp.c_fc.bias"] = np.zeros((VMLP,), dtype=np.float32)
        sd[f"{vb}.mlp.c_proj.weight"] = rng.normal(0, 0.02, (VH, VMLP)).astype(np.float32)
        sd[f"{vb}.mlp.c_proj.bias"] = np.zeros((VH,), dtype=np.float32)

    for i in range(kw["num_layers"]):
        is_sliding = model.layer_types[i] == "hybrid_sliding"
        is_loop = kw["loop_layer_index_start"] <= i < kw["loop_layer_index_end"]
        b = f"model.llm.layers.{i}"
        sd[f"model.llm.attn_res.mlp_queries.{i}"] = rng.normal(0, 0.02, (H,)).astype(np.float32)
        if i > 0:
            q_shape = (H,) if is_sliding else (3, H)
            sd[f"model.llm.attn_res.attn_queries.{i - 1}"] = rng.normal(0, 0.02, q_shape).astype(np.float32)
        sd[f"{b}.mlp_norm.weight"] = np.ones((H,), dtype=np.float32)
        sd[f"{b}.attn_sconv.weight"] = rng.normal(0, 0.02, (H, 1, K)).astype(np.float32)
        sd[f"{b}.mlp_sconv.weight"] = rng.normal(0, 0.02, (H, 1, K)).astype(np.float32)
        sd[f"{b}.attn.wq_du.weight"] = rng.normal(0, 0.02, (NH * HD, H)).astype(np.float32)
        sd[f"{b}.attn.q_l2_clip.scale"] = np.ones((NH, HD), dtype=np.float32)
        sd[f"{b}.attn.wk_dv.weight"] = rng.normal(0, 0.02, (NKV * HD, H)).astype(np.float32)
        sd[f"{b}.attn.wv_dv.weight"] = rng.normal(0, 0.02, (NKV * HD, H)).astype(np.float32)
        sd[f"{b}.attn.gate_attn.weight"] = rng.normal(0, 0.02, (NH * HD, H)).astype(np.float32)
        sd[f"{b}.attn.wo_ud.weight"] = rng.normal(0, 0.02, (H, NH * HD)).astype(np.float32)
        sd[f"{b}.attn.attn_sink_proj.weight"] = rng.normal(0, 0.02, (NH, H)).astype(np.float32)
        sd[f"{b}.attn.wr_du.weight"] = rng.normal(0, 0.02, (NH * DREL, H)).astype(np.float32)
        sd[f"{b}.attn.rel_l2_clip.scale"] = np.ones((NH, DREL), dtype=np.float32)
        ext = kw["sliding_window_size"] if is_sliding else REXT
        sd[f"{b}.attn.rel_logits_proj.proj"] = rng.normal(0, 0.02, (DREL, ext)).astype(np.float32)
        if is_sliding:
            sd[f"{b}.attn_norm.weight"] = np.ones((H,), dtype=np.float32)
            sd[f"{b}.attn.k_sconv.weight"] = rng.normal(0, 0.02, (NKV * HD, 1, K)).astype(np.float32)
            sd[f"{b}.attn.v_sconv.weight"] = rng.normal(0, 0.02, (NKV * HD, 1, K)).astype(np.float32)
        else:
            sd[f"{b}.q_input_norm.weight"] = np.ones((H,), dtype=np.float32)
            sd[f"{b}.k_input_norm.weight"] = np.ones((H,), dtype=np.float32)
            sd[f"{b}.v_input_norm.weight"] = np.ones((H,), dtype=np.float32)
            sd[f"{b}.attn.k_sconv.static_w"] = rng.normal(0, 0.02, (K, NKV * HD)).astype(np.float32)
            sd[f"{b}.attn.k_sconv.weight_proj"] = rng.normal(0, 0.02, (NKV * K, H)).astype(np.float32)
            sd[f"{b}.attn.v_sconv.static_w"] = rng.normal(0, 0.02, (K, NKV * HD)).astype(np.float32)
            sd[f"{b}.attn.v_sconv.weight_proj"] = rng.normal(0, 0.02, (NKV * K, H)).astype(np.float32)

        sd[f"{b}.mlp.gate.weight"] = rng.normal(0, 0.02, (E, H)).astype(np.float32)
        sd[f"{b}.mlp.gate.bias"] = np.zeros((E,), dtype=np.float32)
        # Test NVFP4 on layers 0..3 and MXFP8 on layers 4..5
        if i < 4:
            sd[f"{b}.mlp.experts.w13_weight"] = rng.integers(0, 256, (E, 2 * I, H // 2), dtype=np.uint8)
            sd[f"{b}.mlp.experts.w13_weight_scale"] = np.ones((E, 2 * I, H // 16), dtype=np.float32)
            sd[f"{b}.mlp.experts.w13_weight_amax"] = np.ones((E,), dtype=np.float32)
            sd[f"{b}.mlp.experts.w13_weight_outer_scale_denominator"] = np.ones((E,), dtype=np.float32)
            sd[f"{b}.mlp.experts.w2_weight"] = rng.integers(0, 256, (E, H, I // 2), dtype=np.uint8)
            sd[f"{b}.mlp.experts.w2_weight_scale"] = np.ones((E, H, I // 16), dtype=np.float32)
            sd[f"{b}.mlp.experts.w2_weight_amax"] = np.ones((E,), dtype=np.float32)
            sd[f"{b}.mlp.experts.w2_weight_outer_scale_denominator"] = np.ones((E,), dtype=np.float32)
        else:
            sd[f"{b}.mlp.experts.w13_weight"] = rng.normal(0, 0.02, (E, 2 * I, H)).astype(np.float32)
            sd[f"{b}.mlp.experts.w13_weight_scale"] = np.full((E, 2 * I, H // 32), 127, dtype=np.uint8)
            sd[f"{b}.mlp.experts.w2_weight"] = rng.normal(0, 0.02, (E, H, I)).astype(np.float32)
            sd[f"{b}.mlp.experts.w2_weight_scale"] = np.full((E, H, I // 32), 127, dtype=np.uint8)

        if is_loop:
            for c in ("attn", "k", "v", "mlp"):
                sd[f"{b}.controllers.{c}.down_weight"] = rng.normal(0, 0.02, (R, 2 * H)).astype(np.float32)
                sd[f"{b}.controllers.{c}.forget_weight"] = rng.normal(0, 0.02, (H, R)).astype(np.float32)
                sd[f"{b}.controllers.{c}.gain_weight"] = rng.normal(0, 0.02, (H, R)).astype(np.float32)

    transfer_inkling2_loop_weights(model, sd)
    out = model({"input_ids": np.array([[1, 2, 3]], dtype=np.int32)})
    assert np.all(np.isfinite(ops.convert_to_numpy(out["logits"])))


@pytest.mark.gpu
def test_hf_parity_inkling2_mini():
    """End-to-end download, conversion, and logit parity against HF transformers."""
    import torch
    from transformers import AutoProcessor, Inkling2LoopForConditionalGeneration

    repo_id = "thinkingmachines/Inkling-2-Mini"
    hf_proc = AutoProcessor.from_pretrained(repo_id)
    messages = [{"role": "user", "content": "Say hello in one word."}]
    inputs = hf_proc.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    zm_model = Inkling2LoopConditionalGenerate.from_weights(
        f"hf:{repo_id}", load_dtype="bfloat16"
    )
    zm_logits = ops.convert_to_numpy(
        zm_model({"input_ids": inputs["input_ids"].numpy()})["logits"]
    )

    hf_model = Inkling2LoopForConditionalGeneration.from_pretrained(
        repo_id, dtype=torch.bfloat16, device_map="cpu"
    )
    with torch.no_grad():
        hf_logits = hf_model(input_ids=inputs["input_ids"]).logits.float().numpy()

    assert zm_logits.shape == hf_logits.shape
    zm_top = int(np.argmax(zm_logits[0, -1]))
    hf_top = int(np.argmax(hf_logits[0, -1]))
    assert zm_top == hf_top, f"Top-1 mismatch: zm={zm_top} vs hf={hf_top}"
    cos = float(
        np.dot(zm_logits[0, -1], hf_logits[0, -1])
        / (np.linalg.norm(zm_logits[0, -1]) * np.linalg.norm(hf_logits[0, -1]))
    )
    assert cos > 0.999, f"Cosine similarity too low: {cos}"
