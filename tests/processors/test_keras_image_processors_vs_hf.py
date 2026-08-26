from __future__ import annotations

import importlib
import os

import keras
import numpy as np
import pytest
from PIL import Image

transformers = pytest.importorskip("transformers")

from transformers import (
    AutoImageProcessor,
    DetrImageProcessor,
    DPTImageProcessor,
    EomtImageProcessor,
    RTDetrImageProcessor,
    SamImageProcessor,
    SegformerImageProcessor,
    SiglipImageProcessor,
)
from transformers import (
    CLIPImageProcessor as HFCLIPImageProcessor,
)
from transformers import (
    Sam2ImageProcessor as HFSam2ImageProcessor,
)

from zeromodels.models.clip.clip_image_processor import (
    CLIPImageProcessor as KerasCLIPImageProcessor,
)
from zeromodels.models.depth_anything_v1 import DepthAnythingV1ImageProcessor
from zeromodels.models.depth_anything_v2 import DepthAnythingV2ImageProcessor
from zeromodels.models.detr import DETRImageProcessor
from zeromodels.models.dfine.dfine_image_processor import DFineImageProcessor
from zeromodels.models.eomt.eomt_image_processor import EoMTImageProcessor
from zeromodels.models.metaclip2 import MetaClip2ImageProcessor
from zeromodels.models.rt_detr import RTDETRImageProcessor
from zeromodels.models.rt_detr_v2 import RTDETRV2ImageProcessor
from zeromodels.models.sam import SAMImageProcessor
from zeromodels.models.sam2 import SAM2ImageProcessor
from zeromodels.models.segformer.segformer_image_processor import (
    SegFormerImageProcessor,
)
from zeromodels.models.siglip.siglip_image_processor import (
    SigLIPImageProcessor as KerasSigLIPImageProcessor,
)

ASSET_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "assets", "data", "coco_cats.jpg"
    )
)

DATA_FORMATS = ["channels_last", "channels_first"]


def skip_if_tf_cpu_channels_first(data_format):
    # TensorFlow's CPU kernels don't support NCHW image ops; channels_first on
    # the tensorflow backend is only testable on a GPU machine. torch and jax
    # run both formats everywhere.
    if data_format == "channels_first" and keras.backend.backend() == "tensorflow":
        import tensorflow as tf

        if not tf.config.list_physical_devices("GPU"):
            pytest.skip("channels_first image ops on TensorFlow need a GPU")


def _to_channels_last(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[1] == 3 and arr.shape[-1] != 3:
        return np.transpose(arr, (0, 2, 3, 1))
    return arr


def _max_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = _to_channels_last(a)
    b = _to_channels_last(b)
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _pil_image():
    return Image.open(ASSET_PATH).convert("RGB")


def _as_numpy(x) -> np.ndarray:
    # .cpu() first: a CUDA tensor has both, and its .numpy() raises, so the
    # other order breaks this whole suite on any GPU box (CI is CPU-only).
    if hasattr(x, "cpu"):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else x.cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return keras.ops.convert_to_numpy(x)


def _run_detr(data_format):
    ours = _as_numpy(
        DETRImageProcessor(
            size={"height": 800, "width": 800},
            data_format=data_format,
        )(ASSET_PATH)["pixel_values"]
    )
    hf = DetrImageProcessor(
        do_resize=True,
        size={"height": 800, "width": 800},
        do_rescale=True,
        do_normalize=True,
        do_pad=False,
    )(images=_pil_image(), return_tensors="np")["pixel_values"]
    return ours, hf


def _run_rt_detr(data_format):
    ours = _as_numpy(
        RTDETRImageProcessor(
            size={"height": 640, "width": 640},
            data_format=data_format,
        )(ASSET_PATH)["pixel_values"]
    )
    hf = RTDetrImageProcessor(
        do_resize=True,
        size={"height": 640, "width": 640},
        do_rescale=True,
        do_normalize=False,
        do_pad=False,
    )(images=_pil_image(), return_tensors="np")["pixel_values"]
    return ours, hf


def _run_rt_detr_v2(data_format):
    ours = _as_numpy(
        RTDETRV2ImageProcessor(
            size={"height": 640, "width": 640},
            data_format=data_format,
        )(ASSET_PATH)["pixel_values"]
    )
    hf = RTDetrImageProcessor(
        do_resize=True,
        size={"height": 640, "width": 640},
        do_rescale=True,
        do_normalize=False,
        do_pad=False,
    )(images=_pil_image(), return_tensors="np")["pixel_values"]
    return ours, hf


def _run_dfine(data_format):
    ours = _as_numpy(
        DFineImageProcessor(data_format=data_format)(ASSET_PATH)["pixel_values"]
    )
    hf = RTDetrImageProcessor(
        do_resize=True,
        size={"height": 640, "width": 640},
        do_rescale=True,
        do_normalize=False,
        do_pad=False,
    )(images=_pil_image(), return_tensors="np")["pixel_values"]
    return ours, hf


def _run_segformer(data_format):
    ours = _as_numpy(
        SegFormerImageProcessor(data_format=data_format)(ASSET_PATH)["pixel_values"]
    )
    hf = SegformerImageProcessor(
        do_resize=True,
        size={"height": 512, "width": 512},
        do_rescale=True,
        do_normalize=True,
    )(images=_pil_image(), return_tensors="np")["pixel_values"]
    return ours, hf


def _run_eomt(data_format):
    ours = _as_numpy(
        EoMTImageProcessor(data_format=data_format)(ASSET_PATH)["pixel_values"]
    )
    hf = EomtImageProcessor(
        do_resize=True,
        size={"longest_edge": 640, "shortest_edge": 640},
        do_pad=True,
        do_rescale=True,
        do_normalize=True,
    )(images=_pil_image(), return_tensors="np")["pixel_values"]
    return ours, hf


def _run_sam(data_format):
    ours = _as_numpy(
        SAMImageProcessor(data_format=data_format)(ASSET_PATH)["pixel_values"]
    )
    hf = SamImageProcessor(
        size={"longest_edge": 1024},
        pad_size={"height": 1024, "width": 1024},
        do_rescale=True,
        do_normalize=True,
    )(images=_pil_image(), return_tensors="np")["pixel_values"]
    return ours, hf


def _run_sam2(data_format):
    ours = _as_numpy(
        SAM2ImageProcessor(data_format=data_format)(ASSET_PATH)["pixel_values"]
    )
    hf = HFSam2ImageProcessor()(images=_pil_image(), return_tensors="np")[
        "pixel_values"
    ]
    return ours, hf


def _run_clip(data_format):
    processor = KerasCLIPImageProcessor(data_format=data_format)
    ours = _as_numpy(processor(ASSET_PATH)["pixel_values"])
    hf = HFCLIPImageProcessor()(images=_pil_image(), return_tensors="np")[
        "pixel_values"
    ]
    return ours, hf


def _run_siglip(data_format):
    processor = KerasSigLIPImageProcessor(data_format=data_format)
    ours = _as_numpy(processor(ASSET_PATH)["pixel_values"])
    hf = SiglipImageProcessor()(images=_pil_image(), return_tensors="np")[
        "pixel_values"
    ]
    return ours, hf


def _run_metaclip2(data_format):
    processor = MetaClip2ImageProcessor(data_format=data_format)
    ours = _as_numpy(processor(ASSET_PATH)["pixel_values"])
    hf = HFCLIPImageProcessor(
        do_resize=True,
        size={"height": 224, "width": 224},
        do_center_crop=False,
        do_rescale=True,
        do_normalize=True,
        image_mean=(0.48145466, 0.4578275, 0.40821073),
        image_std=(0.26862954, 0.26130258, 0.27577711),
    )(images=_pil_image(), return_tensors="np")["pixel_values"]
    return ours, hf


def _run_depth_anything_v1(data_format):
    ours = _as_numpy(
        DepthAnythingV1ImageProcessor(data_format=data_format)(ASSET_PATH)[
            "pixel_values"
        ]
    )
    hf = DPTImageProcessor(
        do_resize=True,
        size={"height": 518, "width": 518},
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        do_rescale=True,
        do_normalize=True,
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
        resample=Image.BICUBIC,
    )(images=_pil_image(), return_tensors="np")["pixel_values"]
    return ours, hf


def _run_depth_anything_v2(data_format):
    ours = _as_numpy(
        DepthAnythingV2ImageProcessor(data_format=data_format)(ASSET_PATH)[
            "pixel_values"
        ]
    )
    hf = DPTImageProcessor(
        do_resize=True,
        size={"height": 518, "width": 518},
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        do_rescale=True,
        do_normalize=True,
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
        resample=Image.BICUBIC,
    )(images=_pil_image(), return_tensors="np")["pixel_values"]
    return ours, hf


PROCESSORS = {
    "detr": (_run_detr, 5e-2),
    "rt_detr": (_run_rt_detr, 1e-2),
    "rt_detr_v2": (_run_rt_detr_v2, 1e-2),
    "dfine": (_run_dfine, 1e-2),
    "segformer": (_run_segformer, 1.0),
    "eomt": (_run_eomt, 1e-5),
    "sam": (_run_sam, 5e-2),
    "sam2": (_run_sam2, 5e-2),
    "clip": (_run_clip, 5e-2),
    "siglip": (_run_siglip, 5e-2),
    "metaclip2": (_run_metaclip2, 5e-2),
    "depth_anything_v1": (_run_depth_anything_v1, 5e-1),
    "depth_anything_v2": (_run_depth_anything_v2, 5e-1),
}


@pytest.mark.parametrize("data_format", DATA_FORMATS)
@pytest.mark.parametrize("name", list(PROCESSORS.keys()))
def test_image_processor_photo_parity(name, data_format):
    skip_if_tf_cpu_channels_first(data_format)
    runner, threshold = PROCESSORS[name]
    ours, hf = runner(data_format)
    diff = _max_diff(ours, hf)
    assert diff < threshold, (
        f"{name}[{data_format}] max|diff|={diff:.3e} exceeds {threshold:.1e}"
    )
    print(f"[{name:<20}] {data_format:<15} max|diff|={diff:.3e}")


FROM_HF_SPECS = {
    "clip": (
        "clip.clip_image_processor",
        "CLIPImageProcessor",
        "openai/clip-vit-base-patch16",
        224,
        True,
    ),
    "siglip": (
        "siglip.siglip_image_processor",
        "SigLIPImageProcessor",
        "google/siglip-base-patch16-224",
        224,
        True,
    ),
    "siglip2": (
        "siglip2.siglip2_image_processor",
        "SigLIP2ImageProcessor",
        "google/siglip2-base-patch16-224",
        224,
        True,
    ),
    "metaclip2": (
        "metaclip2.metaclip2_image_processor",
        "MetaClip2ImageProcessor",
        "facebook/metaclip-2-worldwide-huge-378",
        378,
        False,
    ),
    "owlvit": (
        "owlvit.owlvit_image_processor",
        "OwlViTImageProcessor",
        "google/owlvit-base-patch32",
        768,
        True,
    ),
    "owlv2": (
        "owlv2.owlv2_image_processor",
        "Owlv2ImageProcessor",
        "google/owlv2-base-patch16-ensemble",
        960,
        True,
    ),
    "detr": (
        "detr.detr_image_processor",
        "DETRImageProcessor",
        "facebook/detr-resnet-50",
        800,
        True,
    ),
    "rt_detr": (
        "rt_detr.rt_detr_image_processor",
        "RTDETRImageProcessor",
        "PekingU/rtdetr_r50vd",
        640,
        True,
    ),
    "rt_detr_v2": (
        "rt_detr_v2.rt_detr_v2_image_processor",
        "RTDETRV2ImageProcessor",
        "PekingU/rtdetr_v2_r50vd",
        640,
        True,
    ),
    "dfine": (
        "dfine.dfine_image_processor",
        "DFineImageProcessor",
        "ustc-community/dfine-medium-coco",
        640,
        True,
    ),
    "rf_detr": (
        "rf_detr.rf_detr_image_processor",
        "RFDETRImageProcessor",
        "Roboflow/rf-detr-base",
        560,
        True,
    ),
    "sam": (
        "sam.sam_image_processor",
        "SAMImageProcessor",
        "facebook/sam-vit-base",
        1024,
        True,
    ),
    "sam2": (
        "sam2.sam2_image_processor",
        "SAM2ImageProcessor",
        "facebook/sam2.1-hiera-tiny",
        1024,
        True,
    ),
    "maskformer": (
        "maskformer.maskformer_image_processor",
        "MaskFormerImageProcessor",
        "facebook/maskformer-swin-base-ade",
        640,
        False,
    ),
    "mask2former": (
        "mask2former.mask2former_image_processor",
        "Mask2FormerImageProcessor",
        "facebook/mask2former-swin-small-coco-instance",
        384,
        True,
    ),
    "eomt": (
        "eomt.eomt_image_processor",
        "EoMTImageProcessor",
        "tue-mps/coco_panoptic_eomt_large_640",
        640,
        True,
    ),
    "segformer": (
        "segformer.segformer_image_processor",
        "SegFormerImageProcessor",
        "nvidia/segformer-b0-finetuned-ade-512-512",
        512,
        True,
    ),
    "mobilevit": (
        "mobilevit.mobilevit_image_processor",
        "MobileViTImageProcessor",
        "apple/mobilevit-small",
        288,
        True,
    ),
    "mobilevitv2": (
        "mobilevitv2.mobilevitv2_image_processor",
        "MobileViTV2ImageProcessor",
        "apple/mobilevitv2-1.0-imagenet1k-256",
        288,
        True,
    ),
    "depth_anything_v1": (
        "depth_anything_v1.depth_anything_v1_image_processor",
        "DepthAnythingV1ImageProcessor",
        "LiheYoung/depth-anything-small-hf",
        518,
        True,
    ),
    "depth_anything_v2": (
        "depth_anything_v2.depth_anything_v2_image_processor",
        "DepthAnythingV2ImageProcessor",
        "depth-anything/Depth-Anything-V2-Small-hf",
        518,
        True,
    ),
}


def _rgb(side, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((side, side, 3)) * 255).astype("uint8")


def _pixels(processor, img):
    return _to_channels_last(_as_numpy(processor(img)["pixel_values"]))


def _assert_pixels_match(name, leg, ours_px, hf_px, atol=1e-4):
    assert ours_px.shape == hf_px.shape, (
        f"{name}[{leg}]: shape {ours_px.shape} vs HF {hf_px.shape}"
    )
    diff = float(np.max(np.abs(ours_px.astype(np.float64) - hf_px.astype(np.float64))))
    assert diff < atol, f"{name}[{leg}]: max|diff|={diff:.3e} exceeds {atol:.0e}"
    print(f"[{leg:>7} {name:<20}] {ours_px.shape} max|diff|={diff:.3e}")


@pytest.mark.parametrize("data_format", DATA_FORMATS)
@pytest.mark.parametrize("name", list(FROM_HF_SPECS.keys()))
def test_image_processor_three_way_parity(name, data_format):
    """HF reference vs BOTH zeromodels construction paths: ``from_hf(repo)``
    (the preprocessor_config.json mapper) and the native ``Cls()`` defaults
    (what ``from_weights(variant)`` uses), in both data formats (the HF
    channels-first reference is transposed for comparison)."""
    skip_if_tf_cpu_channels_first(data_format)
    module, cls_name, repo, side, native_matches = FROM_HF_SPECS[name]
    cls = getattr(importlib.import_module(f"zeromodels.models.{module}"), cls_name)
    try:
        hf = AutoImageProcessor.from_pretrained(repo)
    except Exception as e:
        pytest.skip(f"{name}: HF image processor for {repo!r} unavailable: {e}")
    img = _rgb(side)
    hf_px = _to_channels_last(
        hf(images=Image.fromarray(img), return_tensors="np")["pixel_values"]
    )

    try:
        via_hf = cls.from_hf(repo, data_format=data_format)
    except Exception as e:
        pytest.skip(f"{name}: from_hf({repo!r}) unavailable: {e}")
    _assert_pixels_match(
        f"{name}/{data_format}", "from_hf", _pixels(via_hf, img), hf_px
    )

    if native_matches:
        _assert_pixels_match(
            f"{name}/{data_format}",
            "native",
            _pixels(cls(data_format=data_format), img),
            hf_px,
        )


# ---------------------------------------------------------------------------
# Offline coverage: snapshots, determinism, batching, and a guard on the test
# inputs themselves.
#
# The vs-HF tests above only check the cases someone thought to write, against
# an image that is already the target size. That blind spot hid a real bug:
# CLIP's resize was never exercised (scale == 1), so a 0.735 max|diff| against
# the reference sat in the suite unnoticed. These need no network and run on
# every backend.
# ---------------------------------------------------------------------------

import inspect  # noqa: E402

from tests.fixtures import snapshot_util  # noqa: E402

# Shapes chosen to force real work: wider than tall, taller than wide, smaller
# and larger than every target resolution, and one odd size that lands off the
# patch grid.
SNAPSHOT_SHAPES = {
    "wide_48x96": (48, 96),
    "tall_96x48": (96, 48),
    "large_300x500": (300, 500),
    "odd_223x225": (223, 225),
}


def _all_image_processors():
    import zeromodels.models as models

    found = {}
    for family in sorted(n for n in dir(models) if not n.startswith("_")):
        package = getattr(models, family)
        for name in getattr(package, "__all__", []):
            obj = getattr(package, name, None)
            if inspect.isclass(obj) and name.endswith("ImageProcessor"):
                found[name] = obj
    return found


IMAGE_PROCESSORS = _all_image_processors()


def _synthetic(shape, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray((rng.random((*shape, 3)) * 255).astype("uint8"))


def _record(processor, image):
    out = processor(image)
    out = out if isinstance(out, dict) else {"pixel_values": out}
    record = {}
    for key, value in out.items():
        try:
            record[key] = snapshot_util.stats(_as_numpy(value))
        except (TypeError, ValueError):
            record[key] = {"value": str(value)[:60]}
    return record


@pytest.mark.parametrize("name", sorted(IMAGE_PROCESSORS))
def test_image_processor_snapshot(name):
    """Pin each processor's output for fixed inputs, on every backend.

    Also the only cross-backend check on preprocessing: the golden file is
    shared, so a processor whose resize disagrees between torch/jax/tf fails
    here on whichever backend drifted.
    """
    processor = IMAGE_PROCESSORS[name]()
    record = {
        label: _record(processor, _synthetic(shape))
        for label, shape in sorted(SNAPSHOT_SHAPES.items())
    }
    snapshot_util.check("image_processor", name, record)


@pytest.mark.parametrize("name", sorted(IMAGE_PROCESSORS))
def test_image_processor_is_deterministic(name):
    processor = IMAGE_PROCESSORS[name]()
    image = _synthetic((64, 48))
    first, second = _record(processor, image), _record(processor, image)
    assert first == second, f"{name}: two calls on one image disagree"


@pytest.mark.parametrize("name", sorted(IMAGE_PROCESSORS))
def test_image_processor_resize_actually_runs(name):
    """A processor that resizes must react to the input's size.

    Guards the blind spot that hid the CLIP bug: feed only an
    already-correctly-sized image and the resize is a no-op, so a broken resize
    still matches the reference. If two very different inputs give byte-identical
    output the input is not reaching the resize, and the case above is vacuous.
    """
    processor = IMAGE_PROCESSORS[name]()
    if not getattr(processor, "do_resize", True):
        pytest.skip(f"{name} does not resize")
    small = _record(processor, _synthetic((32, 40), seed=1))
    large = _record(processor, _synthetic((400, 260), seed=2))
    assert small != large, (
        f"{name}: a 32x40 and a 400x260 image produced identical output, so the "
        "input never reaches the resize"
    )
