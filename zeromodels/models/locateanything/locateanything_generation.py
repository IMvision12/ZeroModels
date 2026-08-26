import numpy as np
from keras import ops

# Faithful port of nvidia/LocateAnything-3B's generate_utils.py (the sampling /
# box-frame decoding is host-side logic on logits) + the modeling generate loop,
# adapted to a recompute (no-KV-cache) loop over the keras forward. Three modes:
# 'fast' (MTP only), 'slow' (pure AR), 'hybrid' (MTP with AR fallback, default).


def get_token_ids(tokenizer):
    return {
        "box_start_token_id": tokenizer.box_start_token_id,
        "box_end_token_id": tokenizer.box_end_token_id,
        "coord_start_token_id": tokenizer.coord_start_token_id,
        "coord_end_token_id": tokenizer.coord_end_token_id,
        "ref_start_token_id": tokenizer.ref_start_token_id,
        "ref_end_token_id": tokenizer.ref_end_token_id,
        "none_token_id": tokenizer.none_token_id,
        "null_token_id": tokenizer.null_token_id,
        "im_end_token_id": tokenizer.eos_token_id,
        "switch_token_id": tokenizer.switch_token_id,
        "default_mask_token_id": tokenizer.text_mask_token_id,
    }


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def topk(x, k, axis=-1):
    k = min(k, x.shape[axis])
    idx = np.argsort(-x, axis=axis)[..., :k]
    return np.take_along_axis(x, idx, axis=axis), idx


def apply_repetition_penalty(logits, generated, penalty):
    if penalty == 1.0:
        return logits
    out = logits.copy()
    for b in range(logits.shape[0]):
        toks = np.unique(generated[b])
        toks = toks[(toks >= 0) & (toks < logits.shape[-1])]
        col = out[b, :, toks]
        out[b, :, toks] = np.where(col > 0, col / penalty, col * penalty)
    return out


def top_p_logits(logits, top_p):
    sorted_idx = np.argsort(-logits, axis=-1)
    sorted_logits = np.take_along_axis(logits, sorted_idx, axis=-1)
    cumprobs = np.cumsum(softmax(sorted_logits, -1), axis=-1)
    remove = cumprobs > top_p
    remove[..., 1:] = remove[..., :-1]
    remove[..., 0] = False
    mask = np.zeros_like(logits, dtype=bool)
    np.put_along_axis(mask, sorted_idx, remove, axis=-1)
    return np.where(mask, np.finfo(logits.dtype).min, logits)


def top_k_logits(logits, top_k):
    top_k = min(top_k, logits.shape[-1])
    kth = np.sort(logits, axis=-1)[..., -top_k][..., None]
    return np.where(logits < kth, np.finfo(logits.dtype).min, logits)


def is_valid_box_frame(probs, tid, start_thresh=0.6, end_thresh=0.2):
    bs, be = tid["box_start_token_id"], tid["box_end_token_id"]
    null, im_end, none = (
        tid["null_token_id"],
        tid["im_end_token_id"],
        tid["none_token_id"],
    )
    if probs[0, bs] >= start_thresh and (
        probs[1, none] > 0.2
        and probs[2, be] > 0.2
        and probs[3, null] > 0.1
        and probs[4, null] > 0.1
    ):
        return "empty_box"
    end_score = probs[5, [be, null, im_end]].sum()
    return "legal_box" if end_score >= end_thresh else "illegal_box"


def decode_bbox_avg(
    probs, tid, keep_k=5, start_thresh=0.7, end_thresh=0.2, mode="hybrid"
):
    cs, ce = tid["coord_start_token_id"], tid["coord_end_token_id"]
    bs, be, none, null = (
        tid["box_start_token_id"],
        tid["box_end_token_id"],
        tid["none_token_id"],
        tid["null_token_id"],
    )
    box_type = is_valid_box_frame(probs, tid, start_thresh, end_thresh)
    if box_type == "empty_box":
        return np.array([bs, none, be, null, null, null], dtype="int64")
    if box_type == "illegal_box":
        return None
    # A real box leads with <box>. If the frame's top token at position 0 is not
    # box_start (e.g. a <ref> frame whose lower-ranked coord tokens would pass the
    # frame checks once the category word is demoted by the repetition penalty),
    # it is not a box -> defer to decode_ref instead of emitting a spurious box.
    if int(np.argmax(probs[0])) != bs:
        return None
    pos_probs, pos_ids = topk(probs[1:5], keep_k, -1)
    # A 4-coord box has coords at positions 1-4. If <box_end> is the top token at
    # any of them, the frame is a point/short box (e.g. <box><x><y></box>) -> defer
    # so handle_pattern emits it as point_box rather than padding spurious coords.
    if (pos_ids[:, 0] == be).any():
        return None
    mask = (pos_ids >= cs) & (pos_ids <= ce)
    if not mask.any(-1).all():
        return None
    fvi = mask.argmax(-1)[:, None]
    first_probs = np.take_along_axis(pos_probs, fvi, -1)[:, 0]
    first_ids = np.take_along_axis(pos_ids, fvi, -1)[:, 0]
    if mode == "hybrid":
        counts = mask.sum(-1)
        ids_max = np.where(mask, pos_ids, -999999).max(-1)
        ids_min = np.where(mask, pos_ids, 999999).min(-1)
        abnormal = (first_probs < 0.9) & (counts > 1) & ((ids_max - ids_min) > 60)
        coords = np.where(abnormal, 0, first_ids)
    else:
        coords = first_ids
    return np.concatenate([[bs], coords, [be]]).astype("int64")


def decode_ref(probs, tid, keep_k=5, start_thresh=0.6):
    rs, cs, ce = (
        tid["ref_start_token_id"],
        tid["coord_start_token_id"],
        tid["coord_end_token_id"],
    )
    if probs[0, rs] < start_thresh:
        return None
    _, pos_ids = topk(probs[1:], keep_k, -1)
    is_valid = ~((pos_ids >= cs) & (pos_ids <= ce))
    if not is_valid.any(-1).all():
        return None
    fvi = is_valid.argmax(-1)[:, None]
    text_ids = np.take_along_axis(pos_ids, fvi, -1)[:, 0]
    return np.concatenate([[rs], text_ids]).astype("int64")


def sample_tokens(logits, generated, tid, keep_k=5, mode="hybrid", **kw):
    temp = kw.get("temperature", 0)
    if kw.get("repetition_penalty", 1.0) != 1.0:
        logits = apply_repetition_penalty(logits, generated, kw["repetition_penalty"])
    if temp and temp > 0:
        logits = logits / temp
    if kw.get("top_p") is not None and kw["top_p"] < 1:
        logits = top_p_logits(logits, kw["top_p"])
    if kw.get("top_k") is not None:
        logits = top_k_logits(logits, kw["top_k"])
    probs = softmax(logits, -1)
    if temp and temp > 0:
        x0 = np.array(
            [
                [
                    np.random.choice(probs.shape[-1], p=probs[b, l])
                    for l in range(probs.shape[1])
                ]
                for b in range(probs.shape[0])
            ]
        )
    else:
        x0 = np.argmax(probs, -1)
    if logits.shape[1] == 1:
        return x0, None
    box = decode_bbox_avg(probs[0], tid, keep_k=kw.get("keep_k_avg", 4), mode=mode)
    if box is None:
        box = decode_ref(probs[0], tid)
    if box is None:
        box = np.zeros(1, dtype="int64")
    return x0, box[None]


def handle_pattern(x0, tid, mode="hybrid"):
    null, im_end = tid["null_token_id"], tid["im_end_token_id"]
    bs, be, none = (
        tid["box_start_token_id"],
        tid["box_end_token_id"],
        tid["none_token_id"],
    )
    cs, ce, re = (
        tid["coord_start_token_id"],
        tid["coord_end_token_id"],
        tid["ref_end_token_id"],
    )
    x0 = [int(t) for t in x0]
    if x0[0] in (null, im_end):
        return {"type": "im_end", "tokens": [im_end]}
    if x0[:2] == [bs, none]:
        return {"type": "empty_box", "tokens": [bs, none, be]}
    if x0[0] == bs:
        ci = 1
        for c in x0[1:5]:
            if cs <= c <= ce:
                ci += 1
            else:
                break
        if ci == 5 and x0[5] == be:
            return {"type": "coord_box", "tokens": x0}
        if ci == 3 and x0[3] == be:
            return {"type": "point_box", "tokens": x0[:4]}
        if mode == "fast":
            return {"type": "coord_box", "tokens": x0}
        return {"type": "error_box", "tokens": x0[:ci]}
    for i, t in enumerate(x0):
        if t == null:
            x0 = x0[:i]
            break
    if len(x0) >= 2 and x0[-1] == x0[-2] == re:
        x0 = x0[:-1]
    return {"type": "ref_object", "tokens": x0}


def generate_loop(
    model,
    input_ids,
    vision_embeds,
    tokenizer,
    n_future=6,
    generation_mode="hybrid",
    max_new_tokens=512,
    **kw,
):
    tid = get_token_ids(tokenizer)
    generated = np.asarray(
        ops.convert_to_numpy(input_ids), dtype="int64"
    )  # accept GPU/keras tensors
    if generated.ndim == 1:
        generated = generated[None]
    prompt_len = generated.shape[1]
    total = prompt_len + max_new_tokens
    use_mtp = generation_mode in ("fast", "hybrid")
    default_mask = tid["default_mask_token_id"]

    while generated.shape[1] < total:
        if use_mtp:
            dup = generated[:, -1:]
            masks = np.full((1, n_future - 1), default_mask, dtype="int64")
            seq_in = np.concatenate([generated, dup, masks], axis=1)
            pos = np.arange(seq_in.shape[1], dtype="int32")[None].copy()
            pos[0, -n_future:] -= 1
            out = model.forward_logits(
                {
                    "input_ids": seq_in,
                    "vision_embeds": vision_embeds,
                    "position_ids": pos,
                    "use_magi": True,
                }
            )
            logits = np.asarray(ops.convert_to_numpy(out))[:, -n_future:, :]
            x0, box = sample_tokens(
                logits, generated, tid, keep_k=5, mode=generation_mode, **kw
            )
            new = x0[0] if bool((box[0] == 0).all()) else box[0]
            pat = handle_pattern(new, tid, generation_mode)
            out_type, out_token = pat["type"], np.asarray(pat["tokens"], dtype="int64")
        else:
            pos = np.arange(generated.shape[1], dtype="int32")[None]
            out = model.forward_logits(
                {
                    "input_ids": generated,
                    "vision_embeds": vision_embeds,
                    "position_ids": pos,
                }
            )
            logits = np.asarray(ops.convert_to_numpy(out))[:, -1:, :]
            x0, _ = sample_tokens(logits, generated, tid, mode=generation_mode, **kw)
            out_token = x0[0]
            tv = int(out_token[0])
            out_type = "continue_ar"
            if generation_mode == "hybrid":
                if tv == tid["box_end_token_id"]:
                    out_type = "box_end_ar"
                elif (
                    tid["coord_start_token_id"] <= tv <= tid["coord_end_token_id"]
                    or tv == tid["none_token_id"]
                ):
                    out_type = "coord_ar"
                else:
                    out_type = "im_end"
            elif tv == tid["im_end_token_id"]:
                out_type = "im_end"

        generated = np.concatenate([generated, out_token[None]], axis=1)
        if out_type == "im_end":
            break
        if generation_mode == "hybrid":
            if out_type == "error_box":
                use_mtp = False
            elif out_type == "box_end_ar":
                use_mtp = True

    return generated[:, prompt_len:]


def generate_loop_cached(
    model,
    input_ids,
    vision_embeds,
    tokenizer,
    n_future=6,
    generation_mode="hybrid",
    max_new_tokens=512,
    **kw,
):
    """KV-cached variant of generate_loop. Each step forwards only the new tokens
    (the last accepted box re-run causally + dup + mask window) against the cache,
    instead of recomputing the whole sequence. Numerically identical to
    generate_loop; the magi block mask is the same formula for q_len < kv_len."""
    tid = get_token_ids(tokenizer)
    generated = np.asarray(ops.convert_to_numpy(input_ids), dtype="int64")
    if generated.ndim == 1:
        generated = generated[None]
    prompt_len = generated.shape[1]
    total = prompt_len + max_new_tokens
    use_mtp = generation_mode in ("fast", "hybrid")
    default_mask = tid["default_mask_token_id"]
    caches = None
    cache_len = 0

    while generated.shape[1] < total:
        length = generated.shape[1]
        vis = vision_embeds if caches is None else None  # merge vision on prefill only
        tail = generated[
            :, cache_len:
        ]  # uncached tokens (prompt, or last accepted box)
        if use_mtp:
            dup = generated[:, -1:]
            masks = np.full((1, n_future - 1), default_mask, dtype="int64")
            new_ids = np.concatenate([tail, dup, masks], axis=1)
            pos = np.arange(cache_len, length + n_future, dtype="int32")[None].copy()
            pos[0, -n_future:] -= 1
            logits, new_caches = model.forward_logits_cached(
                new_ids, pos, vision_embeds=vis, past_caches=caches, causal=False
            )
            logits = np.asarray(ops.convert_to_numpy(logits))[:, -n_future:, :]
            x0, box = sample_tokens(
                logits, generated, tid, keep_k=5, mode=generation_mode, **kw
            )
            new = x0[0] if bool((box[0] == 0).all()) else box[0]
            pat = handle_pattern(new, tid, generation_mode)
            out_type, out_token = pat["type"], np.asarray(pat["tokens"], dtype="int64")
        else:
            pos = np.arange(cache_len, length, dtype="int32")[None]
            logits, new_caches = model.forward_logits_cached(
                tail, pos, vision_embeds=vis, past_caches=caches, causal=True
            )
            logits = np.asarray(ops.convert_to_numpy(logits))[:, -1:, :]
            x0, _ = sample_tokens(logits, generated, tid, mode=generation_mode, **kw)
            out_token = x0[0]
            tv = int(out_token[0])
            out_type = "continue_ar"
            if generation_mode == "hybrid":
                if tv == tid["box_end_token_id"]:
                    out_type = "box_end_ar"
                elif (
                    tid["coord_start_token_id"] <= tv <= tid["coord_end_token_id"]
                    or tv == tid["none_token_id"]
                ):
                    out_type = "coord_ar"
                else:
                    out_type = "im_end"
            elif tv == tid["im_end_token_id"]:
                out_type = "im_end"

        # keep prefix + the re-run uncached tokens in the cache; drop dup/mask K/V
        caches = [(k[:, :, :length, :], v[:, :, :length, :]) for (k, v) in new_caches]
        cache_len = length
        generated = np.concatenate([generated, out_token[None]], axis=1)
        if out_type == "im_end":
            break
        if generation_mode == "hybrid":
            if out_type == "error_box":
                use_mtp = False
            elif out_type == "box_end_ar":
                use_mtp = True

    return generated[:, prompt_len:]
