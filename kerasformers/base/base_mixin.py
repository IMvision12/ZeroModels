import contextlib
import importlib
import inspect
import json
import os
import sys
import urllib.request
import warnings
from typing import NamedTuple, Optional

import keras
from huggingface_hub import hf_hub_download

from kerasformers.base import base_attention
from kerasformers.conversion import download_weights
from kerasformers.conversion.hf_download_utils import (
    download_hf_state_dict,
)
from kerasformers.conversion.weight_transfer_util import (
    skip_mismatched_weights,
)

_HF_PREFIX = "hf:"


class CheckpointSource(NamedTuple):
    """Declares that this class loads its pretrained weights out of a DIFFERENT (bigger)
    class's hosted checkpoint, instead of reading the repo's weights file directly.

    One mechanism serves two families (see :meth:`WeightLoadingMixin.\
_load_from_checkpoint_source`):

    * encoder families (BERT / RoBERTa / DeBERTa): the hosted file is a SUPERSET
      (``*MaskedLM`` + pooler); the encoder and task heads copy their subset out by
      counter-stripped path suffix (``match="suffix"``). The superset class lives in the
      same module unless ``module`` says otherwise; ``build_kwargs`` build it as the full
      reference (e.g. ``{"add_pooler": True}``).
    * multimodal families (VLMs): the hosted file is the full ``*ConditionalGenerate``
      (declared in its ``kf_config.json``); a text head copies its backbone out by full
      post-root path (``match="path"``), dropping the vision / audio towers. ``module`` is
      the full class's import path; the source fires only when a repo declares that class.

    ``match`` picks the copy strategy (the two are NOT interchangeable: encoders collide on
    bare suffixes so they need the short counter-stripped key; VLM decoders repeat suffixes
    across layers so they need the full path).
    """

    source: str
    module: Optional[str] = None
    build_kwargs: Optional[dict] = None
    match: str = "suffix"


def warn_skipped(skipped):
    """Print a note about weights left at init due to ``skip_mismatch``."""
    if skipped:
        print(
            f"[from_weights] skip_mismatch: left {len(skipped)} weight(s) at their "
            f"initialized values due to shape mismatch (e.g. a resized head): "
            f"{skipped}"
        )


def _url_exists(url):
    """True if a range GET on ``url`` succeeds (uses HF_TOKEN for hf.co if set)."""
    headers = {"User-Agent": "kerasformers", "Range": "bytes=0-0"}
    token = os.environ.get("HF_TOKEN")
    if token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=30
        ) as response:
            return response.status in (200, 206)
    except Exception:
        return False


def resolve_weights_url(url):
    """Resolve an HF-repo weights URL to a downloadable file URL.

    A weights entry points at an HF repo: a bare repo
    (``https://huggingface.co/<org>/<repo>``) is resolved here to its
    ``model.weights.h5`` (single) or ``model.weights.json`` (sharded); a full
    file / ``resolve`` URL (e.g. ``.../resolve/main/model.weights.h5``) passes
    through unchanged.
    """
    if (
        "/resolve/" in url
        or "/blob/" in url
        or url.endswith((".weights.h5", ".weights.json"))
    ):
        return url
    repo = url.rstrip("/")
    for filename in ("model.weights.h5", "model.weights.json"):
        candidate = f"{repo}/resolve/main/{filename}"
        if _url_exists(candidate):
            return candidate
    raise ValueError(
        f"No 'model.weights.h5' or 'model.weights.json' found in HF repo '{repo}'."
    )


@contextlib.contextmanager
def build_dtype_scope(dtype):
    """Build the model under a global dtype policy (e.g. ``"bfloat16"``).

    A layer's own dtype policy does **not** propagate to sublayers it creates in
    ``__init__``: a nested ``Dense`` / ``Embedding`` reads the *global* policy at
    construction time, so passing ``dtype=`` to a model constructor still leaves
    its weights float32. Setting the global policy here makes the whole model,
    including the lazy build the converter triggers, allocate its variables in
    ``dtype``; a streamed fp32 / bf16 checkpoint is then cast to it on assign, so
    a bf16 model lands at its native ~2 bytes/param instead of an fp32 upcast.
    ``None`` restores the default (fp32) behaviour as a no-op.
    """
    if dtype is None:
        yield
        return
    previous = keras.config.dtype_policy()
    keras.config.set_dtype_policy(dtype)
    try:
        yield
    finally:
        keras.config.set_dtype_policy(previous)


@contextlib.contextmanager
def inference_scope():
    """Disable autograd on the torch backend (a no-op on JAX / TensorFlow).

    Weight loading (``build_for_transfer``'s dummy forward, converter transfer)
    and prefill are inference-only, but a torch forward outside this scope builds
    an autograd graph that saves every intermediate for backward. For an MXFP4
    checkpoint that means retaining each layer's ~GB dequantized expert bank, so
    a full-model build at load time can hold tens of GB it never frees and OOM a
    large checkpoint. Wrapping those forwards here keeps them graph-free.
    """
    if keras.backend.backend() == "torch":
        import torch

        with torch.no_grad():
            yield
    else:
        yield


class WeightLoadingMixin:
    """Unified pretrained-weight loading API shared by all kerasformers models.

    Mixed into :class:`BaseModel`. Kept as a plain mixin, **not** a
    ``keras.Model`` subclass, so the model base stays a clean ``keras.Model``
    subclass. Subclasses share a single entry point for loading pretrained
    weights, regardless of source:

    1. **Hub repo (self-describing)**: an ``"org/repo"`` id whose repo carries a
       ``kf_config.json``. The model is rebuilt from that flat config (via
       ``config_class``) and its weights are loaded from the same repo. Works for
       the official weights and for community fine-tunes of the same architecture.
    2. **Hub (``hf:`` conversion)**: an ``"hf:org/repo"`` id pulls the original
       ``config.json`` + safetensors and converts them via ``config_from_hf`` /
       ``transfer_from_hf`` (for repos published in the source library's format).

    A model that ships a typed :class:`~kerasformers.base.BaseConfig` sets
    ``config_class`` (and gets the ``Model(config)`` constructor + ``.config``);
    an ``hf:`` converter additionally provides ``config_from_hf`` and
    ``transfer_from_hf``.

    .. code-block:: python

        class OwlViTDetect(BaseModel):
            config_class = OwlViTConfig

            @classmethod
            def config_from_hf(cls, hf_config: dict): ...

            @classmethod
            def transfer_from_hf(cls, model, state_dict): ...

    Usage:

    .. code-block:: python

        m = OwlViTDetect.from_weights("kerasformers/owlvit-base-patch32")

        m = OwlViTDetect.from_weights("hf:google/owlvit-base-patch32")
        m = OwlViTDetect.from_weights("hf:alice/owlvit-finetune")

        m = OwlViTDetect(OwlViTConfig())  # build from a config, random weights
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = None
    # Typed BaseConfig subclass for this model (set by models that ship one).
    # When present, a repo's flat kf_config.json is parsed through it, the model
    # accepts a config object (``Model(config)``), and exposes ``self.config``.
    config_class = None
    # Default generation settings (e.g. Whisper's suppress_tokens) for models with
    # a ``generate(...)`` method. Written into a repo's kf_config.json under
    # ``generate_args`` and re-attached to the model on repo-id load.
    generate_args = None
    # Default ``load_dtype`` for :meth:`from_weights` when the caller passes none,
    # and the ``dtype`` recorded into a repo's kf_config.json. Defaults to float32
    # (the historical convention for the hosted vision / encoder / ASR weights);
    # models whose hosted weights are bf16 (gemma*, gpt-oss, qwen*, locateanything)
    # override to ``"bfloat16"`` so they load at native ~2 bytes/param instead of an
    # fp32 upcast. Pass ``load_dtype=...`` to override per call.
    default_load_dtype = "float32"

    # This class loads its weights out of a DIFFERENT (bigger) class's hosted checkpoint,
    # instead of reading the repo's weights file directly. Set a :class:`CheckpointSource`:
    #
    #     CHECKPOINT_SOURCE = CheckpointSource("BertMaskedLM", build_kwargs={"add_pooler": True})
    #     CHECKPOINT_SOURCE = CheckpointSource("DebertaV3Model")  # source == cls -> direct load
    #     CHECKPOINT_SOURCE = CheckpointSource(                    # VLM text head
    #         "Qwen3VLConditionalGenerate", module="kerasformers.models.qwen3_vl.qwen3_vl_model",
    #         match="path")
    #
    # ``from_weights`` builds the source, loads the hosted file into it, and copies THIS
    # class's subset out (by counter-stripped suffix for encoders, by full path for VLM text
    # heads). ``None`` = the normal single-class path. See :meth:`_load_from_checkpoint_source`.
    #
    # Two legacy attrs are still recognized and normalized to a CheckpointSource:
    # ``SHARED_CHECKPOINT = (name, build_kwargs)`` (encoder superset) and
    # ``FULL_CHECKPOINT_SOURCES = {declared_full_class: module}`` (VLM text head).
    CHECKPOINT_SOURCE = None
    SHARED_CHECKPOINT = None  # legacy alias for CHECKPOINT_SOURCE (encoder superset)

    @classmethod
    def from_weights(
        cls,
        identifier,
        load_weights=True,
        skip_mismatch=False,
        attn_implementation=None,
        quantization=None,
        load_dtype=None,
        cache_converted=False,
        **kwargs,
    ):
        """Build a model and (optionally) load pretrained weights.

        Args:
            identifier: One of two forms:

                * a kerasformers variant string (e.g. ``"resnet50_a1_in1k"``)
                  resolves against ``cls.BASE_MODEL_CONFIG`` /
                  ``cls.BASE_WEIGHT_CONFIG``.
                * ``"hf:<org>/<repo>"``: pulls config and weights from
                  the model Hub. Dispatches to :meth:`from_hf`, which
                  handles both transformers-style repos (CLIP, SigLIP,
                  DETR, …) and timm-style repos
                  (``hf:timm/resnet50.a1_in1k``).

            load_weights: If ``False``, only the architecture is built
                (random init). For ``hf:`` ids, ``config.json`` is still
                fetched to size the model; the weight files are not.
            skip_mismatch: If ``True``, weights whose checkpoint shape
                disagrees with the instantiated model are skipped during
                load and left at their default initialization. Useful for
                fine-tuning: pass ``num_classes=N, skip_mismatch=True`` to
                swap in a new classifier head while loading the rest of the
                backbone. Applied on both the repo-id ``kf_config.json`` load
                path and the ``hf:`` / variant converter transfer path
                (mismatched targets left at init).
            attn_implementation: ``"sdpa"`` (portable manual math, the default)
                or ``"flash"`` (``keras.ops.dot_product_attention`` with the
                flash kernel; needs a flash-capable GPU/TPU and fp16/bf16). Set
                before the model is built.
            quantization: ``None`` (default), ``"int8"``, ``"int4"`` or
                ``"fp8"`` (or a :class:`~kerasformers.quantization.\
QuantizationConfig` / scheme). When set, the model is quantized weight-only:
                Dense/Embedding weights become int8 / float8-e4m3 (~4x) or
                block-wise packed int4 (~8x), and the float weights are freed. fp8
                is torch/jax only. The model builds (in ``load_dtype``) then
                quantizes via :func:`kerasformers.quantization.quantize_model`.
            load_dtype: ``None`` resolves, highest priority first, to the repo's
                recorded ``weight_dtype`` (from a kerasformers Hub repo's
                ``kf_config.json`` -- the checkpoint's real precision), then to
                ``cls.default_load_dtype`` (fp32 for most models, bf16 for families
                that ship bf16 weights such as GPT-OSS) for the ``hf:`` / variant
                paths and repos predating ``weight_dtype``. Or pass a dtype string
                such as ``"bfloat16"`` / ``"float16"`` / ``"float32"``. Builds the
                device model under that global dtype policy, so a bf16 checkpoint
                loads at its native ~2 bytes/param instead of being upcast to fp32
                (≈half the device memory, cosine ~0.9998 vs fp32). The streamed
                checkpoint is cast to it on assign. Pass ``"float32"`` to force fp32
                on a bf16 model. Independent of ``quantization``, which runs after
                the build.
            cache_converted: If ``True`` (opt-in), cache the fully converted
                (and optionally quantized) model under
                ``$KERASFORMERS_HOME/converted`` on first load, and on later
                identical calls rebuild straight from that cache, skipping the
                download **and** the re-conversion. The cache stores weights-only
                + a rebuild recipe. Cache keys include the Hub commit SHA,
                converter/cache layout, backend/dtype and quantization recipe.
                Applies to models loaded as float (a built functional graph can't
                be re-quantized from a serialized skeleton, so quantized loads are
                not cached); a cache miss / failure silently falls back to the
                source path. Best on a persistent disk: set ``KERASFORMERS_HOME``
                on ephemeral boxes.
            **kwargs: Forwarded to the model constructor (or to
                ``from_hf`` when applicable).

        Returns:
            An initialized model instance.
        """
        if attn_implementation is not None:
            if attn_implementation not in base_attention.VALID_ATTN_IMPL:
                raise ValueError(
                    f"attn_implementation must be one of "
                    f"{base_attention.VALID_ATTN_IMPL}, got {attn_implementation!r}"
                )
            base_attention.ATTN_IMPLEMENTATION = attn_implementation

        # Resolve the build precision, highest priority first (mirrors the
        # ``dtype="auto"`` resolution in transformers: the repo's own record wins
        # over any library-side default):
        #   1. an explicit ``load_dtype=`` from the caller,
        #   2. the ``weight_dtype`` a kerasformers Hub repo records in its
        #      kf_config.json -- the checkpoint's actual dtype, so a repo saved in a
        #      precision other than its family default (a bf16 fine-tune of an fp32
        #      family, say) still loads natively instead of being cast,
        #   3. ``cls.default_load_dtype``, the family's native precision, for the
        #      variant / ``hf:`` paths and for repos predating ``weight_dtype``.
        # ``load_dtype="float32"`` forces fp32.
        if load_dtype is None:
            load_dtype = cls.hub_repo_weight_dtype(identifier)
        if load_dtype is None:
            load_dtype = cls.default_load_dtype

        # Converted-model cache: on a hit, rebuild from the local cache and skip
        # the download + conversion entirely. Only when loading weights (an
        # arch-only build has nothing to cache) and for cacheable model types.
        cache_directory = None
        if cache_converted and load_weights:
            from kerasformers.conversion import converted_cache

            if converted_cache.cache_supported(cls, quantization):
                cache_directory = converted_cache.cache_dir(
                    cls, identifier, quantization, load_dtype, kwargs
                )
                if converted_cache.is_cached(cache_directory):
                    cached = converted_cache.try_load_converted(
                        cache_directory, quantization, load_dtype
                    )
                    if cached is not None:
                        return cached

        # Build (and transfer) under the requested dtype policy so the device
        # model is allocated in e.g. bf16; post-hoc quantize runs outside it.
        # ``inference_scope`` keeps the load-time forwards graph-free on torch so
        # an MXFP4 build does not retain each layer's dequantized experts.
        with inference_scope(), build_dtype_scope(load_dtype):
            if identifier.startswith(_HF_PREFIX):
                hf_id = identifier[len(_HF_PREFIX) :]
                model = cls.from_hf(
                    hf_id,
                    load_weights=load_weights,
                    skip_mismatch=skip_mismatch,
                    quantization=quantization,
                    **kwargs,
                )
            elif "/" in identifier:
                # A bare Hub repo id ("kerasformers/detr-resnet-50"): rebuild from
                # the repo's own kf_config.json and load its weights, no variant
                # hardcoded in the package. Quantization / caching still run in the
                # post-build steps below, exactly as for the variant path.
                model = cls.from_hub_repo(
                    identifier,
                    load_weights=load_weights,
                    skip_mismatch=skip_mismatch,
                    **kwargs,
                )
            else:
                model = cls.from_variant(
                    identifier,
                    load_weights=load_weights,
                    skip_mismatch=skip_mismatch,
                    quantization=quantization,
                    **kwargs,
                )
        # A no-float load already quantized in place (and recorded the config);
        # only quantize here when that path didn't run (functional models, or the
        # release-`.h5` / timm paths).
        if (
            quantization is not None
            and getattr(model, "_quantization_config", None) is None
        ):
            from kerasformers.quantization import quantize_model

            model = quantize_model(model, quantization)

        # First-load cache write: store the converted result so a later identical
        # call rebuilds from it. Best-effort: never breaks the returned model.
        if cache_directory is not None:
            from kerasformers.conversion import converted_cache

            converted_cache.try_save_converted(
                model, cache_directory, quantization, load_dtype
            )
        return model

    @classmethod
    def _quantized_transfer(cls, model, state_dict, quantization, skip_mismatch=False):
        """Apply ``cls.transfer_from_hf``; stream into int storage when quantizing.

        Functional models are built at construction, so this runs a plain float
        transfer and the caller applies any quantization afterwards (gated on
        ``model._quantization_config``). A model whose sublayers build lazily (a
        VLM/ASR tower that does not auto-build) is materialized first via
        ``build_for_transfer`` so every weight exists before the stream.

        On a strict load (``skip_mismatch=False``) the build runs under
        :func:`~kerasformers.conversion.weight_transfer_util.zeros_init` so an
        unbuilt model skips the wasted random init of weights it is about to
        overwrite; with ``skip_mismatch`` a mismatched weight may be left at its
        initializer, so that optimization is disabled.
        """
        from kerasformers.conversion.weight_transfer_util import zeros_init

        build_init = contextlib.nullcontext() if skip_mismatch else zeros_init()
        with build_init:
            # A model whose sublayers build lazily (a VLM/ASR tower that does not
            # auto-build) transfers into weights that don't exist yet; materialize
            # them first. Functional models are already built and skip this.
            if hasattr(model, "build_for_transfer") and not model.built:
                model.build_for_transfer()
            cls.transfer_from_hf(model, state_dict)
        return False

    @classmethod
    def transfer_from_timm(cls, keras_model, state_dict):
        """Map a timm state-dict onto ``keras_model``'s weights.

        Default raises :class:`NotImplementedError`. Subclasses opt in
        by implementing the per-family timm-name → keras-weight mapping
        (typically delegating to a module-level
        ``transfer_<family>_weights`` function). Reached via
        :meth:`from_hf` when ``HF_MODEL_TYPE is None`` (i.e., the
        family loads from timm-style repos, not transformers-style
        ones).
        """
        raise NotImplementedError(
            f"{cls.__name__} does not support loading from timm-style HF "
            f"repos. Implement `transfer_from_timm` to enable it."
        )

    @staticmethod
    def load_weights_from_url(model, url, skip_mismatch=False):
        """Download release weights into an (already built) ``model``.

        Handles a single ``.weights.h5`` or a sharded ``.weights.json`` index
        (downloads each shard listed in ``weight_map`` from the same release),
        and a bare HF-repo URL (``https://huggingface.co/<org>/<repo>``), which is
        resolved to the repo's ``model.weights.h5`` / ``model.weights.json``.

        The download is delegated to :func:`~kerasformers.conversion.\
download_weights`: a Hugging Face repo is fetched through the HF cache
        (concurrent shards, resume, Xet, reuse across runs); any other URL
        streams directly.
        """
        url = resolve_weights_url(url)
        model.load_weights(download_weights(url), skip_mismatch=skip_mismatch)

    @classmethod
    def from_variant(
        cls,
        variant,
        load_weights=True,
        skip_mismatch=False,
        quantization=None,
        **kwargs,
    ):
        """Build a packaged variant and convert its weights from the source Hub.

        The variant is looked up in ``BASE_MODEL_CONFIG`` (architecture) and
        ``BASE_WEIGHT_CONFIG`` (an ``{"hf_id": ..., "safetensors": ...}`` entry
        naming the upstream checkpoint to convert on the fly). This is the
        bare-variant branch of :meth:`from_weights` (e.g.
        ``from_weights("qwen2-0.5b")``). Pre-converted kerasformers weights are
        loaded by Hub repo id instead (``from_weights("org/repo")`` ->
        :meth:`from_hub_repo`), not from here.
        """
        if cls.BASE_MODEL_CONFIG is None:
            raise NotImplementedError(
                f"{cls.__name__} must set BASE_MODEL_CONFIG to use from_weights()."
            )
        if variant not in cls.BASE_MODEL_CONFIG:
            available = sorted(cls.BASE_MODEL_CONFIG.keys())
            raise ValueError(
                f"Unknown variant '{variant}' for {cls.__name__}. "
                f"Available variants: {available}"
            )

        config = dict(cls.BASE_MODEL_CONFIG[variant])
        config.update(kwargs)
        model = cls(**config)

        if load_weights:
            if cls.BASE_WEIGHT_CONFIG is None or variant not in cls.BASE_WEIGHT_CONFIG:
                raise ValueError(
                    f"No weights configured for variant '{variant}'. Load "
                    f"pretrained weights by Hub repo id instead, e.g. "
                    f"{cls.__name__}.from_weights('kerasformers/{variant}'), or pass "
                    f"load_weights=False to build an untrained model."
                )
            entry = cls.BASE_WEIGHT_CONFIG[variant]
            if not isinstance(entry, dict) or not entry.get("hf_id"):
                raise ValueError(
                    f"Weights entry for variant '{variant}' must be a dict with an "
                    f"'hf_id' (the upstream checkpoint to convert). Pre-converted "
                    f"kerasformers weights load by Hub repo id, e.g. "
                    f"{cls.__name__}.from_weights('kerasformers/{variant}')."
                )
            hf_id = entry["hf_id"]
            gated = entry.get("gated", False)
            use_safetensors = entry.get("safetensors", False)

            if use_safetensors:
                # Read raw safetensors and run the model's hand-mapped transfer
                # (the same path as `hf:`): lighter than instantiating the HF
                # model, gives the exact checkpoint key layout the transfer
                # expects, and handles bf16 -> float32. Used by the Qwen
                # families, whose converters key off raw checkpoint tensors.
                state_dict = download_hf_state_dict(hf_id)
                completed = False
                try:
                    with skip_mismatched_weights(skip_mismatch) as skipped:
                        cls._quantized_transfer(
                            model, state_dict, quantization, skip_mismatch
                        )
                    completed = True
                finally:
                    close = getattr(state_dict, "close", None)
                    if callable(close):
                        close(completed=completed)
                warn_skipped(skipped)
            else:
                from kerasformers.conversion.hf_download_utils import (
                    load_and_convert_from_hf,
                )

                with skip_mismatched_weights(skip_mismatch) as skipped:
                    load_and_convert_from_hf(
                        model=model,
                        model_name=variant,
                        hf_model_id=hf_id,
                        transfer_fn=cls.transfer_from_hf,
                        is_gated=gated,
                    )
                warn_skipped(skipped)

        return model

    @classmethod
    def _accepts_hub_class(cls, declared):
        """Whether a repo whose ``kf_config.json`` declares model_class
        ``declared`` may be loaded by this class.

        Default: only the exact same class (strict). A family whose several task
        heads share one weights repo (CLIP / SigLIP: the full model plus its
        zero-shot / classify / embed / vision / text heads) sets
        :attr:`HUB_REPO_SIBLINGS` to the set of sibling class names, so any of
        them can load the repo whose config names the canonical full model.
        """
        return declared in getattr(cls, "HUB_REPO_SIBLINGS", frozenset())

    @classmethod
    def _config_from_kf_spec(cls, spec, variant):
        """Constructor kwargs from a parsed ``kf_config.json`` (or the legacy
        BASE_MODEL_CONFIG fallback). No model_class guard, no weight loading."""
        from kerasformers.conversion.kf_config import KF_METADATA_KEYS, retuple

        if spec is not None and isinstance(spec.get("config"), dict):
            # legacy nested format: constructor kwargs under a "config" *dict* key.
            # (Guard on dict: some models, e.g. MobileNet, have a flat field literally
            # named "config" whose value is an arch string, not the legacy wrapper.)
            return retuple(spec["config"])
        if spec is not None:
            # flat format: hyperparameters at the top level (transformers style).
            fields = {k: v for k, v in spec.items() if k not in KF_METADATA_KEYS}
            if cls.config_class is not None:
                return cls.config_class.from_dict(fields).constructor_kwargs()
            fields.pop("model_type", None)
            return retuple(fields)
        if cls.BASE_MODEL_CONFIG and variant in cls.BASE_MODEL_CONFIG:
            return dict(cls.BASE_MODEL_CONFIG[variant])
        raise ValueError(
            f"Cannot load variant '{variant}': the repo has no kf_config.json and "
            f"'{variant}' is not a known variant of {cls.__name__}. Pass a repo "
            f"that carries kf_config.json."
        )

    @staticmethod
    def _apply_generate_args(model, spec):
        """Attach a repo's kf_config ``generate_args`` to the built model, so its
        ``generate(...)`` picks up the repo's default generation settings. Merged
        over the class default, so a repo that overrides only some keys keeps the
        rest."""
        if spec is not None and spec.get("generate_args"):
            merged = dict(getattr(type(model), "generate_args", None) or {})
            merged.update(spec["generate_args"])
            model.generate_args = merged

    @staticmethod
    def _apply_quantization_config(model, quantization_config):
        """Run the matching :class:`KfQuantizer` so a natively-quantized repo (e.g.
        an mxfp4 GPT-OSS checkpoint) swaps in its packed layers *before* the weights
        load, and stamps ``model._quantization_config`` for save round-trips. The
        model stays quantization-agnostic; this is the sole thing that packs it.
        No-op when the repo carries no ``quantization_config`` block."""
        if quantization_config:
            from kerasformers.quantization import get_kf_quantizer

            get_kf_quantizer(quantization_config).preprocess_model(model)

    @staticmethod
    def hub_repo_weight_dtype(identifier):
        """The dtype a Hub repo's weights are stored in, or ``None``.

        Reads ``weight_dtype`` from the repo's ``kf_config.json`` (written by
        :func:`~kerasformers.conversion.kf_config.write_kf_config`) so
        :meth:`from_weights` can build at the checkpoint's real precision. Returns
        ``None`` for anything that is not a bare ``org/repo`` id (a bare variant or
        an ``hf:`` id, whose precision comes from the class default), for repos
        written before ``weight_dtype`` existed, and when the lookup fails -- so an
        offline or private repo falls back rather than raising.
        """
        if identifier.startswith(_HF_PREFIX) or "/" not in identifier:
            return None
        from kerasformers.conversion.kf_config import load_kf_config

        try:
            spec = load_kf_config(identifier.rstrip("/"))
        except Exception:
            return None
        return (spec or {}).get("weight_dtype")

    @classmethod
    def build_from_hub_repo(cls, repo_id, **kwargs):
        """Build (unloaded) from a repo's ``kf_config.json``, bypassing the
        model_class guard. For task heads that share a family's weights repo and
        load by copying from the full model rather than reading the h5 directly."""
        from kerasformers.conversion.kf_config import load_kf_config

        repo_id = repo_id.rstrip("/")
        variant = repo_id.rsplit("/", 1)[-1]
        spec = load_kf_config(repo_id)
        config = cls._config_from_kf_spec(spec, variant)
        config.update(kwargs)
        model = cls(**config)
        cls._apply_generate_args(model, spec)
        cls._apply_quantization_config(
            model, spec.get("quantization_config") if spec else None
        )
        return model

    @classmethod
    def _normalized_checkpoint_source(cls, declared):
        """Resolve this class's checkpoint source to a :class:`CheckpointSource`, or ``None``.

        Prefers the ``CHECKPOINT_SOURCE`` attr; falls back to the legacy ``SHARED_CHECKPOINT``
        (encoder superset) and ``FULL_CHECKPOINT_SOURCES`` (VLM full model, keyed by the repo's
        declared ``model_class``) spellings, so old declarations keep working.
        """
        cs = getattr(cls, "CHECKPOINT_SOURCE", None)
        if cs is not None:
            return cs
        shared = getattr(cls, "SHARED_CHECKPOINT", None)
        if shared is not None:
            name, build_kwargs = shared
            return CheckpointSource(name, None, build_kwargs, "suffix")
        full = getattr(cls, "FULL_CHECKPOINT_SOURCES", None) or {}
        if declared in full:
            return CheckpointSource(declared, full[declared], None, "path")
        return None

    @classmethod
    def _checkpoint_source_for(cls, declared, match):
        """The :class:`CheckpointSource` to apply for this load, or ``None``.

        ``match`` selects the family so the dispatch keeps its ordering: a ``"path"`` (VLM)
        source is resolved BEFORE the sibling guard and fires only when the repo's kf_config
        declares the full source class (and it is not this class); a ``"suffix"`` (encoder)
        source is resolved AFTER the guard and is unconditional.
        """
        cs = cls._normalized_checkpoint_source(declared)
        if cs is None or cs.match != match:
            return None
        if cs.match == "path" and (declared != cs.source or declared == cls.__name__):
            return None
        return cs

    @classmethod
    def _load_from_checkpoint_source(
        cls, repo_id, cs, load_weights=True, skip_mismatch=False, **kwargs
    ):
        """Load THIS class out of a bigger sibling's hosted checkpoint (see
        :class:`CheckpointSource`).

        ``match="path"``: build the full model and copy this head's backbone out of it (a VLM
        text head reading a multimodal checkpoint; the vision / audio weights go unused).

        ``match="suffix"``: the family ships one SUPERSET file (encoder + masked-LM + task
        heads). Keras' ``.weights.h5`` maps weights by structural index, so a direct
        cross-class load misplaces them; instead the superset is loaded into a reference of
        its OWN class (an exact structural match) and this class's weights are copied out by
        counter-stripped path suffix. When this class *is* the superset and builds identically
        (no reference kwargs), the file loads directly.
        """
        if cs.match == "path":
            full_cls = getattr(importlib.import_module(cs.module), cs.source)
            return cls._load_backbone_from_full(
                full_cls, repo_id, load_weights=load_weights, **kwargs
            )

        from kerasformers.conversion import copy_weights_by_path_suffix

        model = cls.build_from_hub_repo(repo_id, **kwargs)
        if not load_weights:
            return model
        if hasattr(model, "build_for_transfer") and not model.built:
            model.build_for_transfer()

        url = f"https://huggingface.co/{repo_id}"
        ref_name = cs.source
        ref_kwargs = cs.build_kwargs or {}
        if cls.__name__ == ref_name and not ref_kwargs:
            cls.load_weights_from_url(model, url, skip_mismatch)
            return model

        ref_module = (
            importlib.import_module(cs.module)
            if cs.module
            else sys.modules[cls.__module__]
        )
        ref_cls = getattr(ref_module, ref_name)
        ref = ref_cls.build_from_hub_repo(repo_id, **ref_kwargs)
        if hasattr(ref, "build_for_transfer") and not ref.built:
            ref.build_for_transfer()
        ref_cls.load_weights_from_url(ref, url, skip_mismatch)
        skipped = copy_weights_by_path_suffix(ref, model)
        del ref
        if skipped:
            warnings.warn(
                f"{cls.__name__}: [{', '.join(skipped)}] left randomly initialized "
                f"(the checkpoint has no weights for them). Fine-tune before use.",
                stacklevel=2,
            )
        return model

    @classmethod
    def from_hub_repo(cls, repo_id, load_weights=True, skip_mismatch=False, **kwargs):
        """Build + load a model from a Hub repo id carrying ``kf_config.json``.

        The repo describes itself: ``kf_config.json`` names the class and the
        constructor kwargs, and the weights live in the same repo
        (``model.weights.h5`` / sharded ``model.weights.json``). This is the
        loading path for ``from_weights("org/repo")``: nothing about the variant
        is hardcoded in the package, so community fine-tunes of the same
        architecture load the same way as the official weights.

        Falls back to ``BASE_MODEL_CONFIG[variant]`` (variant = repo basename)
        when the repo carries no ``kf_config.json`` yet, so a not-yet-backfilled
        official repo still loads.
        """
        from kerasformers.conversion.kf_config import load_kf_config

        repo_id = repo_id.rstrip("/")
        variant = repo_id.rsplit("/", 1)[-1]
        spec = load_kf_config(repo_id)
        declared = spec.get("model_class") if spec is not None else None
        if spec is not None:
            # A head can load its backbone out of a fuller (e.g. multimodal) sibling
            # checkpoint whose kf_config declares that full class: build it and copy this
            # head's weights out, dropping the rest (a text head reading a VLM checkpoint).
            # Checked before the sibling guard, since the head does not "accept" that class.
            cs = cls._checkpoint_source_for(declared, "path")
            if cs is not None:
                return cls._load_from_checkpoint_source(
                    repo_id,
                    cs,
                    load_weights=load_weights,
                    skip_mismatch=skip_mismatch,
                    **kwargs,
                )
            if (
                declared
                and declared != cls.__name__
                and not cls._accepts_hub_class(declared)
            ):
                raise ValueError(
                    f"'{repo_id}' kf_config.json declares model_class "
                    f"'{declared}', but {cls.__name__}.from_weights() was called. "
                    f"Load it with {declared}.from_weights('{repo_id}')."
                )

        # A family sharing one superset checkpoint (encoder + masked-LM + task heads) loads
        # this class's weights out of that checkpoint by path suffix -- so the whole family is
        # one hosted file. (Runs after the sibling guard above so cross-family loads raise.)
        cs = cls._checkpoint_source_for(declared, "suffix")
        if cs is not None:
            return cls._load_from_checkpoint_source(
                repo_id,
                cs,
                load_weights=load_weights,
                skip_mismatch=skip_mismatch,
                **kwargs,
            )

        config = cls._config_from_kf_spec(spec, variant)
        config.update(kwargs)
        model = cls(**config)
        cls._apply_generate_args(model, spec)
        # Swap in packed layers BEFORE building, so the packed weights load into them.
        cls._apply_quantization_config(
            model, spec.get("quantization_config") if spec else None
        )

        if load_weights:
            if hasattr(model, "build_for_transfer") and not model.built:
                model.build_for_transfer()
            cls.load_weights_from_url(
                model, f"https://huggingface.co/{repo_id}", skip_mismatch
            )
        return model

    @classmethod
    def from_hf(
        cls,
        hf_id,
        load_weights=True,
        variant=None,
        skip_mismatch=False,
        quantization=None,
        **kwargs,
    ):
        """Load a model from a model-hub repo.

        Two flavours, auto-detected by :attr:`HF_MODEL_TYPE`:

        1. **Transformers-style repos** (``HF_MODEL_TYPE`` set: CLIP,
           SigLIP, DETR, EoMT, …): pulls ``config.json``, validates
           ``model_type``, builds via :meth:`config_from_hf`, and
           dispatches to :meth:`transfer_from_hf`.
        2. **Timm-style repos** (``HF_MODEL_TYPE is None``: ResNet,
           ConvNeXt, EfficientNet, …): infers the kerasformers
           variant from the repo's trailing path segment, builds via
           :attr:`BASE_MODEL_CONFIG`, and dispatches to
           :meth:`transfer_from_timm`. No ``config.json`` is parsed
           (timm checkpoints don't carry a transformers-style
           ``model_type``).

        Args:
            hf_id: Model-hub id, e.g.
                ``"openai/clip-vit-base-patch16"`` (transformers-style)
                or ``"timm/resnet50.a1_in1k"`` (timm-style).
            load_weights: If ``False``, only the architecture is built.
            variant: For timm-style repos, override the inferred
                kerasformers variant id (e.g., for community fine-tunes
                whose repo name doesn't follow the timm convention).
                Ignored for transformers-style repos.
            **kwargs: Forwarded to the model constructor.

        Returns:
            An initialized model instance.
        """
        if cls.HF_MODEL_TYPE is None:
            if variant is None:
                tail = hf_id.split("/")[-1]
                stem = tail.replace(".", "_")
                for candidate in cls.BASE_MODEL_CONFIG or {}:
                    if stem == candidate or stem.startswith(candidate + "_"):
                        variant = candidate
                        break
                if variant is None:
                    raise ValueError(
                        f"Cannot infer kerasformers variant from hf_id "
                        f"'{hf_id}'. Pass `variant=` explicitly. Available "
                        f"variants: {sorted(cls.BASE_MODEL_CONFIG or {})}"
                    )
            model = cls.from_variant(variant, load_weights=False, **kwargs)
            if load_weights:
                state_dict = download_hf_state_dict(hf_id)
                completed = False
                try:
                    with skip_mismatched_weights(skip_mismatch) as skipped:
                        cls.transfer_from_timm(model, state_dict)
                    completed = True
                finally:
                    close = getattr(state_dict, "close", None)
                    if callable(close):
                        close(completed=completed)
                warn_skipped(skipped)
            return model

        with open(hf_hub_download(hf_id, "config.json"), "r") as f:
            hf_config = json.load(f)
        cls.assert_hf_model_type(hf_id, hf_config)
        kerasformers_kwargs = cls.config_from_hf(hf_config)
        kerasformers_kwargs.update(kwargs)
        model = cls(**kerasformers_kwargs)
        if load_weights:
            state_dict = download_hf_state_dict(hf_id)
            completed = False
            try:
                with skip_mismatched_weights(skip_mismatch) as skipped:
                    cls._quantized_transfer(
                        model, state_dict, quantization, skip_mismatch
                    )
                completed = True
            finally:
                close = getattr(state_dict, "close", None)
                if callable(close):
                    close(completed=completed)
            warn_skipped(skipped)
        return model

    @classmethod
    def assert_hf_model_type(cls, hf_id, hf_config):
        """Reject configs whose ``model_type`` doesn't match this class.

        Fails fast with a clear message instead of letting the user wait
        for a ``KeyError`` or shape mismatch deep inside weight transfer.
        Subclasses opt in by setting ``cls.HF_MODEL_TYPE``; the check is
        skipped when it's ``None``.
        """
        expected = cls.HF_MODEL_TYPE
        if expected is None:
            return
        if isinstance(expected, str):
            expected = (expected,)
        actual = hf_config.get("model_type")
        if actual not in expected:
            options = expected[0] if len(expected) == 1 else f"one of {list(expected)}"
            raise ValueError(
                f"{cls.__name__} can only load HF models whose "
                f"config.json model_type is {options}, but '{hf_id}' "
                f"has model_type={actual!r}. This kerasformers class is the "
                f"wrong destination for that checkpoint."
            )

    @classmethod
    def config_from_hf(cls, hf_config):
        """Map a ``config.json`` dict to ``cls.__init__`` kwargs.

        ``hf_config`` is the result of ``json.load(open("config.json"))``
        a plain dict, not a ``transformers`` config object. Subclasses
        must override this to support ``"hf:"`` loading.
        """
        raise NotImplementedError(f"{cls.__name__}.config_from_hf is not implemented.")

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        """Transfer weights from a source ``state_dict`` into ``keras_model``.

        ``hf_state_dict`` is a flat ``{name: numpy_array}`` mapping.
        Subclasses must override this to support ``"hf:"`` loading.
        """
        raise NotImplementedError(
            f"{cls.__name__}.transfer_from_hf is not implemented."
        )


class PreprocessorMixin(keras.layers.Layer):
    """Single base for every kerasformers preprocessing layer: tokenizers,
    processors, image processors, and feature extractors all inherit it.

    Preprocessing layers are stateless utility layers (no weights to build) that
    take *Python* inputs (strings, chat-message lists, raw images, raw audio)
    not tensors. ``__call__`` forwards straight to ``call`` so those inputs can be
    passed positionally (Keras's ``Layer.__call__`` rejects non-tensor positional
    args).

    The loading API, ``from_weights`` / ``from_variant`` / ``from_hf``, mirrors
    the model-side :class:`WeightLoadingMixin`, so a preprocessor loads with the
    *same* identifier as its model and can pull its files from a packaged
    variant id or from the HF Hub (an ``"hf:org/repo"`` id)::

        gen = Qwen2TextGenerate.from_weights("qwen2-7b-instruct")
        tok = Qwen2Tokenizer.from_weights("qwen2-7b-instruct")
        tok = CLIPTokenizer.from_weights("hf:openai/clip-vit-base-patch16")

    Subclasses (:class:`BaseTokenizer`, :class:`BaseProcessor`,
    :class:`BaseImageProcessor`, :class:`BaseAudioFeatureExtractor`) implement
    ``call`` and add their own state / ``get_config``: the base bakes in no
    defaults.
    """

    @classmethod
    def from_weights(cls, identifier, **kwargs):
        if identifier.startswith("hf:"):
            repo = identifier[len("hf:") :]
            if "/" not in repo:
                raise ValueError(
                    f"{cls.__name__}.from_weights('hf:{repo}'): the 'hf:' prefix "
                    f"expects a Hugging Face repo id of the form 'org/name' (e.g. "
                    f"'hf:openai/clip-vit-base-patch16'), but got {repo!r} with no "
                    f"'/'. If {repo!r} is a kerasformers release variant, drop the "
                    f"'hf:' prefix: {cls.__name__}.from_weights({repo!r})."
                )
            return cls.from_hf(repo, **kwargs)
        if "/" in identifier:
            return cls.from_hub_repo(identifier, **kwargs)
        return cls.from_variant(identifier, **kwargs)

    @classmethod
    def from_hub_repo(cls, repo_id, **kwargs):
        """Build the preprocessor from a repo's ``kf_preprocessor.json``.

        Mirrors the model's :meth:`WeightLoadingMixin.from_hub_repo`, so a
        processor loads with the same repo id as its model
        (``DETRImageProcessor.from_weights("kerasformers/detr-resnet-50")``).
        Transformers-style: a repo that carries no ``kf_preprocessor.json`` raises,
        rather than silently returning generic defaults; build the defaults
        explicitly with ``cls()`` if that is what you want.
        """
        from kerasformers.conversion.kf_config import (
            KF_METADATA_KEYS,
            load_kf_preprocessor,
        )

        spec = load_kf_preprocessor(repo_id.rstrip("/"))
        if spec is None:
            raise OSError(
                f"{repo_id} does not appear to have a file named "
                f"kf_preprocessor.json. Build {cls.__name__} with default settings "
                f"via {cls.__name__}() instead, or add a kf_preprocessor.json to "
                f"the repo."
            )
        if "config" in spec:
            config = dict(spec["config"])  # legacy nested format
        else:
            config = {k: v for k, v in spec.items() if k not in KF_METADATA_KEYS}
        config.update(kwargs)
        return cls(**config)

    @classmethod
    def from_variant(cls, variant, /, **kwargs):
        params = inspect.signature(cls).parameters
        if "variant" in params and "variant" not in kwargs:
            kwargs["variant"] = variant
        elif (
            "hf_id" in params
            and "hf_id" not in kwargs
            and "tokenizer_file" not in kwargs
        ):
            # Gated preprocessors take `hf_id`, not `variant`; map the packaged
            # variant to its gated Hub repo so `from_weights(variant)` works like
            # the model's own `from_weights(variant)`.
            hf_id = cls.release_variant_hf_id(variant)
            if hf_id is not None:
                kwargs["hf_id"] = hf_id
        return cls(**kwargs)

    @classmethod
    def release_variant_hf_id(cls, variant):
        # Look up the gated Hub repo for `variant` from the model's sibling
        # `config` module: scan for any `*_WEIGHTS_URLS` dict and return the
        # variant's `hf_id` (None -> the constructor raises "needs hf_id" as usual).
        import importlib

        package = cls.__module__.rsplit(".", 1)[0]
        family = package.rsplit(".", 1)[-1]
        try:
            config = importlib.import_module(f"{package}.{family}_config")
        except ModuleNotFoundError:
            return None
        for name in dir(config):
            if name.endswith("_WEIGHTS_URLS"):
                entry = getattr(config, name).get(variant)
                if isinstance(entry, dict) and entry.get("hf_id"):
                    return entry["hf_id"]
        return None

    @classmethod
    def from_hf(cls, repo, **kwargs):
        if "hf_id" not in inspect.signature(cls).parameters:
            raise NotImplementedError(
                f"{cls.__name__} cannot load from an 'hf:' repo: its constructor "
                f"takes no `hf_id`. Use a release variant, or override `from_hf` "
                f"to fetch the files from {repo!r}."
            )
        return cls(hf_id=repo, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} must implement `call`.")
