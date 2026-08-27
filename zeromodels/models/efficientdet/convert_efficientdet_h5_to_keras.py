import re

import h5py
import numpy as np
from keras import ops

BACKBONE_PREFIX = "efficientnet-{bb}/efficientnet-{bb}"

EFFICIENTDET_RECIPES = {
    "efficientdet_d0": {
        "backbone_name": "efficientnet_b0",
        "image_size": 512,
        "fpn_num_filters": 64,
        "fpn_cell_repeats": 3,
        "box_class_repeats": 3,
    },
    "efficientdet_d1": {
        "backbone_name": "efficientnet_b1",
        "image_size": 640,
        "fpn_num_filters": 88,
        "fpn_cell_repeats": 4,
        "box_class_repeats": 3,
    },
    "efficientdet_d2": {
        "backbone_name": "efficientnet_b2",
        "image_size": 768,
        "fpn_num_filters": 112,
        "fpn_cell_repeats": 5,
        "box_class_repeats": 3,
    },
    "efficientdet_d3": {
        "backbone_name": "efficientnet_b3",
        "image_size": 896,
        "fpn_num_filters": 160,
        "fpn_cell_repeats": 6,
        "box_class_repeats": 4,
    },
    "efficientdet_d4": {
        "backbone_name": "efficientnet_b4",
        "image_size": 1024,
        "fpn_num_filters": 224,
        "fpn_cell_repeats": 7,
        "box_class_repeats": 4,
    },
    "efficientdet_d5": {
        "backbone_name": "efficientnet_b5",
        "image_size": 1280,
        "fpn_num_filters": 288,
        "fpn_cell_repeats": 7,
        "box_class_repeats": 4,
    },
    "efficientdet_d6": {
        "backbone_name": "efficientnet_b6",
        "image_size": 1280,
        "fpn_num_filters": 384,
        "fpn_cell_repeats": 8,
        "box_class_repeats": 5,
        "fpn_weight_method": "sum",
    },
    "efficientdet_d7": {
        "backbone_name": "efficientnet_b6",
        "image_size": 1536,
        "fpn_num_filters": 384,
        "fpn_cell_repeats": 8,
        "box_class_repeats": 5,
        "anchor_scale": 5.0,
        "fpn_weight_method": "sum",
    },
}

EFFICIENTDET_H5_URLS = {
    name: (
        "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/"
        f"coco2/{name.replace('_', '-')}.h5"
    )
    for name in EFFICIENTDET_RECIPES
}


def block_order(paths):
    seen = []
    for p in paths:
        mo = re.match(r"(blocks_\d+_\d+)", p)
        if mo and mo.group(1) not in seen:
            seen.append(mo.group(1))
    seen.sort(key=lambda s: tuple(int(x) for x in s.split("_")[1:]))
    return {name: i for i, name in enumerate(seen)}


def map_backbone_block(path, bb_prefix, block_n, has_expand):
    rest = re.sub(r"^blocks_\d+_\d+_", "", path)
    sub, var = rest.split("/", 1)
    if sub == "conv2d_1":  # expand conv
        tgt = "conv2d"
    elif sub == "conv2d_2":  # project conv
        tgt = "conv2d_1" if has_expand else "conv2d"
    elif sub == "dwconv2d":
        return f"{bb_prefix}/blocks_{block_n}/depthwise_conv2d/depthwise_kernel"
    elif sub == "se_conv_reduce":
        tgt = "se/conv2d"
    elif sub == "se_conv_expand":
        tgt = "se/conv2d_1"
    elif sub == "batchnorm_1":  # expand BN
        tgt = "tpu_batch_normalization"
    elif sub == "batchnorm_2":  # depthwise BN
        tgt = "tpu_batch_normalization_1" if has_expand else "tpu_batch_normalization"
    elif sub == "batchnorm_3":  # project BN
        tgt = (
            "tpu_batch_normalization_2"
            if has_expand
            else "tpu_batch_normalization_1"
        )
    else:
        raise ValueError(f"unmapped block sublayer {sub!r} in {path}")
    return f"{bb_prefix}/blocks_{block_n}/{tgt}/{var}"


def keras_path_to_h5(path, bb_prefix, order, expand_flags, num_levels):
    seg0 = path.split("/")[0]
    if seg0 == "decode_boxes":  # anchor constant, not in the checkpoint
        return None
    if path.startswith("conv_stem/"):
        return f"{bb_prefix}/stem/conv2d/{path.split('/', 1)[1]}"
    if path.startswith("batchnorm_1/"):
        return f"{bb_prefix}/stem/tpu_batch_normalization/{path.split('/', 1)[1]}"
    mo = re.match(r"(blocks_\d+_\d+)", path)
    if mo:
        block = mo.group(1)
        return map_backbone_block(path, bb_prefix, order[block], expand_flags[block])
    if seg0.startswith("resample_p"):
        return f"{seg0}/{path}"
    if seg0 in ("class_net", "box_net"):
        return f"{seg0}/{path}"
    if seg0 == "fpn_cells":
        m2 = re.match(r"fpn_cells/cell_(\d+)_fnode(\d+)/(.+)", path)
        r, i, rest = m2.group(1), int(m2.group(2)), m2.group(3)
        n = num_levels + i
        base = f"fpn_cells/fpn_cells/cell_{r}/fnode{i}"
        if rest.startswith("WSM"):
            return f"{base}/{rest}"
        if rest.startswith("conv/") or rest.startswith("bn/"):
            return f"{base}/op_after_combine{n}/{rest}"
        mr = re.match(r"resample_(\d+)_(\d+)/(.+)", rest)
        if mr:
            return f"{base}/resample_{mr.group(1)}_{mr.group(2)}_{n}/{mr.group(3)}"
        raise ValueError(f"unmapped fpn sublayer {rest!r} in {path}")
    raise ValueError(f"unmapped path {path!r}")


def transfer_efficientdet_weights(keras_model, h5_path, backbone=None):
    if backbone is None:
        backbone = keras_model.backbone_name.replace("_", "-")
    paths = [w.path for w in keras_model.weights]
    order = block_order(paths)
    expand_flags = {
        b: any(p.startswith(f"{b}_conv2d_1/") for p in paths) for b in order
    }
    num_levels = keras_model.max_level - keras_model.min_level + 1
    bb_prefix = BACKBONE_PREFIX.format(bb=backbone.replace("efficientnet-", ""))

    transferred = 0
    with h5py.File(h5_path, "r") as f:
        for w in keras_model.weights:
            key = keras_path_to_h5(
                w.path, bb_prefix, order, expand_flags, num_levels
            )
            if key is None:
                continue
            dataset = key + ":0"
            if dataset not in f:
                raise KeyError(f"{w.path} -> {dataset} not in {h5_path}")
            value = np.asarray(f[dataset])
            if tuple(value.shape) != tuple(w.shape):
                raise ValueError(
                    f"shape mismatch {w.path}: model {tuple(w.shape)} "
                    f"vs h5 {tuple(value.shape)}"
                )
            w.assign(ops.convert_to_tensor(value))
            transferred += 1
    return transferred
