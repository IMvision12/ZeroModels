import inspect
from collections import OrderedDict

import keras
import numpy as np
from keras import ops

from zeromodels.base.base_mixin import inference_scope
from zeromodels.samplers import GreedySampler


class BaseGeneration:
    """Backend-agnostic autoregressive generation for decoder-only LMs (mirrors HF's ``GenerationMixin``).

    A mixin added to a decoder backbone (e.g. :class:`Qwen3Model`) to give it a
    fast ``generate``. It bundles the shared, optimized
    cross-backend decode engine with the decoder-only entry points (the prompt is the
    input token ids). :class:`BaseSeq2SeqGeneration` subclasses this and overrides
    ``generate`` / ``generate_step`` for encoder-decoder models (Whisper, Speech2Text),
    reusing the same engine.

    A model plugs in two hooks:

    - ``build_cache(token_ids, padding_mask, max_len) -> (cache, logits)`` -- the
      parallel prefill: populate a pre-allocated fixed-size KV cache (any opaque tensor
      the model defines) and return it plus the last-token logits.
    - ``call_with_cache(token_ids, cache, cache_update_index[, key_padding]) ->
      (logits, cache)`` -- one decode step that reads/writes the cache at the given
      index. ``key_padding`` is optional and passed only by the padding-aware path.

    Ragged (padded) batches: a mixed-length batch is generated correctly either way.
    By default (``SUPPORTS_PADDED_DECODE = False``) ``generate`` **buckets** the batch
    by real length and runs each equal-length group unpadded. A model that opts in with
    ``SUPPORTS_PADDED_DECODE = True`` is instead handed a **left-aligned** prompt (pads
    moved to the front, all rows' last real token flush at the right edge) and runs the
    whole batch as one call; it must then (1) use arange positions in ``build_cache``
    (correct under left-padding because RoPE is shift-invariant) and keep the diagonal
    of the prefill mask open so a leading-pad query row is never fully masked, and (2)
    accept a ``key_padding`` (``(batch, max_len)``, 1 = real / 0 = leading pad) in
    ``call_with_cache`` and add it to its decode key mask. Multimodal prefill keeps
    bucketing regardless (its per-row extra inputs are not realigned).

    Performance comes from a single fused decode loop (``keras.ops.while_loop`` over a
    constant-shape cache) wrapped in a per-backend compiled function -- ``jax.jit`` with
    stateless variable threading on JAX, ``tf.function(jit_compile=True)`` on
    TensorFlow, eager on Torch -- cached on the instance. Decoding strategy is a
    pluggable :class:`~zeromodels.samplers.Sampler` (greedy by default); for
    stochastic samplers the random noise is drawn once *outside* the loop (via a
    ``SeedGenerator``) and consumed with the Gumbel-max trick, so generation stays
    identical across backends. Output is a fixed ``(batch, max_new_tokens)`` padded with
    the eos id after a sequence finishes. A model may set the ``eos_token_id`` class
    attr for its default stop token(s); explicit ``generate`` arguments win over it.
    """

    eos_token_id = ()
    _generate_cache_maxsize = 8
    SUPPORTS_PADDED_DECODE = False

    def build_cache(self, token_ids, padding_mask, max_len):
        raise NotImplementedError(
            f"{type(self).__name__} must implement build_cache()."
        )

    def call_with_cache(self, token_ids, cache, cache_update_index, key_padding=None):
        raise NotImplementedError(
            f"{type(self).__name__} must implement call_with_cache()."
        )

    # LEGACY spelling of a match="path" CheckpointSource: repos whose ``zm_config.json``
    # declares one of these classes are loaded by building that (fuller, e.g. multimodal)
    # model and copying THIS head's backbone weights out of it, ignoring the rest (e.g. a
    # text head reading a vision-language checkpoint: the vision / audio weights become
    # unused keys). Maps a declared class name -> the module path it lives in. Prefer
    # ``CHECKPOINT_SOURCE = CheckpointSource(name, module=..., match="path")``; both are
    # normalized by ``WeightLoadingMixin._normalized_checkpoint_source``.
    FULL_CHECKPOINT_SOURCES = {}

    @classmethod
    def _load_backbone_from_full(cls, full_cls, repo_id, load_weights=True, **kwargs):
        """Build ``full_cls`` from ``repo_id`` and copy this head's backbone out of it.

        The head is constructed from the constructor kwargs it shares with the built full
        model, so extra config the head does not take (vision dims, M-RoPE sections) is
        dropped. Weights are matched by keras path suffix (see :meth:`_head_from_full`).
        """
        full = full_cls.from_weights(repo_id, load_weights=load_weights, **kwargs)
        return cls._head_from_full(full, copy_weights=load_weights)

    @classmethod
    def _head_from_full(cls, full, copy_weights=True):
        """Reconstruct this head off an already-built full (multimodal) model.

        Each head constructor param is read from the full model wherever it lives: on
        the full model directly (VLMs that duplicate the text dims at top level), else
        one level down in its text tower -- a ``language_model`` submodel or a
        ``text_config`` dict (VLMs that nest the decoder). Weights are matched by keras
        path suffix: an exact suffix, else the unique full-model weight ending in
        ``/<suffix>`` (the text backbone sits one level down, e.g. under
        ``language_model``); the full model's extra (vision / audio) weights are never
        referenced.
        """
        names = set()
        for klass in cls.__mro__:
            init = klass.__dict__.get("__init__")
            if init is None:
                continue
            for name, param in inspect.signature(init).parameters.items():
                if name in ("self", "name", "dtype", "trainable"):
                    continue
                if param.kind in (
                    param.POSITIONAL_OR_KEYWORD,
                    param.KEYWORD_ONLY,
                ):
                    names.add(name)
        missing = object()

        def resolve(name):
            if hasattr(full, name):
                return getattr(full, name)
            lm = getattr(full, "language_model", None)
            if lm is not None and hasattr(lm, name):
                return getattr(lm, name)
            tc = getattr(full, "text_config", None)
            if isinstance(tc, dict) and name in tc:
                return tc[name]
            return missing

        resolved = {n: resolve(n) for n in names}
        head = cls(**{n: v for n, v in resolved.items() if v is not missing})
        if copy_weights:
            head({"input_ids": np.array([[0, 1, 2, 3]], dtype="int64")})  # build
            by_suffix = {w.path.split("/", 1)[1]: w for w in full.weights}
            suffixes = list(by_suffix)
            for hw in head.weights:
                suffix = hw.path.split("/", 1)[1]
                if suffix in by_suffix:
                    src = by_suffix[suffix]
                else:
                    cand = [k for k in suffixes if k.endswith("/" + suffix)]
                    if len(cand) != 1:
                        raise KeyError(
                            f"cannot match head weight '{hw.path}' in "
                            f"{type(full).__name__} ({len(cand)} candidates for '{suffix}')"
                        )
                    src = by_suffix[cand[0]]
                hw.assign(ops.convert_to_tensor(src))
        return head

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=None,
        eos_token_id=None,
        sampler=None,
        seed=None,
        **prefill_inputs,
    ):
        max_new_tokens, eos, sampler, seed = self.resolve_generation_args(
            max_new_tokens, eos_token_id, sampler, seed
        )
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch = int(input_ids.shape[0])
        padding_mask = (
            None
            if attention_mask is None
            else ops.cast(ops.convert_to_tensor(attention_mask), "int32")
        )
        if padding_mask is not None:
            real_lens = np.asarray(ops.convert_to_numpy(padding_mask)).sum(axis=1)
            real_lens = real_lens.astype("int64")
            if int(real_lens.min()) < int(input_ids.shape[1]):
                if (
                    getattr(self, "SUPPORTS_PADDED_DECODE", False)
                    and not prefill_inputs
                ):
                    input_ids, padding_mask = self.left_align_prompts(
                        input_ids, padding_mask, real_lens
                    )
                else:
                    ids_np = np.asarray(ops.convert_to_numpy(input_ids))
                    out = np.zeros((batch, max_new_tokens), dtype="int32")
                    for length in sorted(set(real_lens.tolist())):
                        rows = np.nonzero(real_lens == length)[0]
                        group_ids = ops.convert_to_tensor(ids_np[rows][:, :length])
                        row_index = ops.convert_to_tensor(rows)
                        group_prefill = {
                            k: ops.take(v, row_index, axis=0)
                            for k, v in prefill_inputs.items()
                        }
                        group_out = self.generate(
                            group_ids,
                            attention_mask=None,
                            max_new_tokens=max_new_tokens,
                            eos_token_id=eos,
                            sampler=sampler,
                            seed=seed,
                            **group_prefill,
                        )
                        out[rows] = np.asarray(ops.convert_to_numpy(group_out))
                    return ops.convert_to_tensor(out)
        noise = self.draw_noise(sampler, max_new_tokens, batch, seed)
        if prefill_inputs:
            prompt_len = int(input_ids.shape[1])
            with inference_scope():
                cache, logits = self.build_cache(
                    input_ids,
                    padding_mask,
                    prompt_len + max_new_tokens,
                    **prefill_inputs,
                )
            return self.run_decode(
                cache, logits, prompt_len, noise, max_new_tokens, eos, sampler
            )

        sampler_key = (
            type(sampler).__name__,
            tuple(sorted(sampler.get_config().items())),
        )
        cache_key = (max_new_tokens, eos, attention_mask is not None, sampler_key)
        fn = self.cached_generate_function(cache_key, max_new_tokens, eos, sampler)
        return self.run_compiled(fn, (input_ids, padding_mask), noise)

    def left_align_prompts(self, input_ids, padding_mask, real_lens):
        ids = np.asarray(ops.convert_to_numpy(input_ids))
        mask = np.asarray(ops.convert_to_numpy(padding_mask))
        width = ids.shape[1]
        out_ids = np.zeros_like(ids)
        out_mask = np.zeros((ids.shape[0], width), dtype="int32")
        for i in range(ids.shape[0]):
            real = ids[i][mask[i] != 0]
            r = int(real.shape[0])
            if r:
                out_ids[i, width - r :] = real
                out_mask[i, width - r :] = 1
        return (
            ops.cast(ops.convert_to_tensor(out_ids), "int32"),
            ops.cast(ops.convert_to_tensor(out_mask), "int32"),
        )

    def generate_step(
        self, token_ids, padding_mask, noise, max_new_tokens, eos, sampler
    ):
        token_ids = ops.cast(ops.convert_to_tensor(token_ids), "int32")
        prompt_len = int(token_ids.shape[1])
        max_len = prompt_len + max_new_tokens
        cache, logits = self.build_cache(token_ids, padding_mask, max_len)
        key_padding = None
        if padding_mask is not None and getattr(self, "SUPPORTS_PADDED_DECODE", False):
            # Carry the (left-aligned) prompt mask across the whole cache so decode
            # keeps holding the leading pads out; generated slots (>= prompt_len) are
            # always real.
            batch = int(token_ids.shape[0])
            tail = ops.ones((batch, max_len - prompt_len), dtype="int32")
            key_padding = ops.concatenate(
                [ops.cast(padding_mask, "int32"), tail], axis=1
            )
        return self.decode_loop(
            cache, logits, prompt_len, noise, max_new_tokens, eos, sampler, key_padding
        )

    def decode_loop(
        self,
        cache,
        logits,
        prompt_len,
        noise,
        max_new_tokens,
        eos,
        sampler,
        key_padding=None,
    ):
        batch = int(logits.shape[0])
        first_tok = ops.cast(
            sampler.sample(logits, ops.take(noise, 0, axis=0)), "int32"
        )[:, None]
        first_eos = eos[0] if eos else 0
        if max_new_tokens <= 1:
            return first_tok

        done = ops.zeros((batch,), dtype="bool")
        for e in eos:
            done = ops.logical_or(done, first_tok[:, 0] == e)
        steps = max_new_tokens - 1
        buf = ops.full((steps, batch, 1), first_eos, dtype="int32")

        def cond(i, tok, cache, pos, done, buf):
            return ops.logical_and(i < steps, ops.logical_not(ops.all(done)))

        def body(i, tok, cache, pos, done, buf):
            if key_padding is None:
                logits, cache = self.call_with_cache(tok, cache, pos)
            else:
                logits, cache = self.call_with_cache(tok, cache, pos, key_padding)
            step_noise = ops.take(noise, i + 1, axis=0)
            nxt = ops.cast(sampler.sample(logits, step_noise), "int32")[:, None]
            nxt = ops.cast(ops.where(done[:, None], first_eos, nxt), "int32")
            for e in eos:
                done = ops.logical_or(done, nxt[:, 0] == e)
            buf = ops.slice_update(buf, (i, 0, 0), nxt[None])
            return (i + 1, nxt, cache, pos + 1, done, buf)

        init = (
            ops.convert_to_tensor(0, dtype="int32"),
            first_tok,
            cache,
            ops.convert_to_tensor(prompt_len, dtype="int32"),
            done,
            buf,
        )
        buf = ops.while_loop(cond, body, init, maximum_iterations=steps)[-1]
        tail = ops.transpose(buf[:, :, 0], (1, 0))  # (batch, steps)
        return ops.concatenate([first_tok, tail], axis=1)

    def make_generate_function(self, max_new_tokens, eos, sampler):
        backend = keras.backend.backend()
        if backend == "jax":
            import itertools

            import jax

            def compiled(runtime_args, noise, state):
                trainable, non_trainable = state
                mapping = itertools.chain(
                    zip(self.trainable_variables, trainable),
                    zip(self.non_trainable_variables, non_trainable),
                )
                with keras.StatelessScope(state_mapping=mapping):
                    return self.generate_step(
                        *runtime_args, noise, max_new_tokens, eos, sampler
                    )

            compiled = jax.jit(compiled)

            def run(runtime_args, noise):
                state = (
                    [v.value for v in self.trainable_variables],
                    [v.value for v in self.non_trainable_variables],
                )
                return compiled(runtime_args, noise, state)

            return run

        if backend == "tensorflow":
            import tensorflow as tf

            return tf.function(
                lambda runtime_args, noise: self.generate_step(
                    *runtime_args, noise, max_new_tokens, eos, sampler
                ),
                jit_compile=True,
            )

        def run(runtime_args, noise):
            return self.generate_step(
                *runtime_args, noise, max_new_tokens, eos, sampler
            )

        return run

    def resolve_generation_args(self, max_new_tokens, eos_token_id, sampler, seed):
        if max_new_tokens is None:
            max_new_tokens = 128
        if eos_token_id is None:
            eos_token_id = self.eos_token_id
        if sampler is None:
            sampler = GreedySampler()
        if seed is None:
            seed = 0
        eos = tuple(
            int(e)
            for e in (
                eos_token_id
                if isinstance(eos_token_id, (list, tuple))
                else [eos_token_id]
            )
        )
        return int(max_new_tokens), eos, sampler, int(seed)

    def draw_noise(self, sampler, max_new_tokens, batch, seed):
        if sampler.stochastic:
            return keras.random.uniform(
                (max_new_tokens, batch, int(self.vocab_size)),
                seed=keras.random.SeedGenerator(int(seed)),
            )
        return ops.zeros((max_new_tokens, batch, 1), dtype="float32")

    def cached_generate_function(self, cache_key, max_new_tokens, eos, sampler):
        fns = self.__dict__.get("_generate_functions")
        if fns is None:
            fns = self.__dict__["_generate_functions"] = OrderedDict()
        fn = fns.get(cache_key)
        if fn is not None:
            fns.move_to_end(cache_key)
            return fn
        fn = self.make_generate_function(max_new_tokens, eos, sampler)
        fns[cache_key] = fn
        if len(fns) > self._generate_cache_maxsize:
            fns.popitem(last=False)
        return fn

    def run_compiled(self, fn, runtime_args, noise):
        with inference_scope():
            out = fn(runtime_args, noise)
        return ops.convert_to_numpy(out)

    def run_decode(
        self, cache, logits, prompt_len, noise, max_new_tokens, eos, sampler
    ):
        # inference_scope disables autograd on torch (keeps the eager decode loop
        # graph-free); a no-op on JAX / TensorFlow, which don't tape gradients here.
        with inference_scope():
            out = self.decode_loop(
                cache, logits, prompt_len, noise, max_new_tokens, eos, sampler
            )
        return ops.convert_to_numpy(out)


class TextOnlyGeneration:
    """Text-only counterpart of a multimodal ``*ConditionalGenerate`` head.

    Pair this mixin (first, so it wins the MRO) with a ``*ConditionalGenerate`` whose
    backbone builds a plain decoder when given no vision / audio tower, e.g.::

        class Gemma4TextGenerate(TextOnlyGeneration, Gemma4ConditionalGenerate):
            config_class = Gemma4TextConfig

    It builds the model text-only and exposes a pure-text ``build_cache`` (the multimodal
    prefill inputs are dropped, so ``.generate()`` takes just token ids). Set
    ``config_class`` to the text sub-config: a multimodal checkpoint then loads as
    text-only, its vision / audio config simply ignored. A head may combine this mixin
    with a ``match="path"`` :attr:`CHECKPOINT_SOURCE` when the family ships no separate
    text-only repo (Gemma 3n): the mixin keeps the build text-only, the source extracts
    the text backbone out of the multimodal checkpoint. Families whose text and multimodal
    decoders truly differ (M-RoPE VLMs like Qwen-VL) instead give the text head its own
    distinct backbone.
    """

    def __init__(self, *args, **kwargs):
        # Some multimodal backbones nest the text decoder under a ``text_config`` dict
        # (towers keyed by ``vision_config`` / ``audio_config``); route the flat text
        # fields the loader passes into it, leaving the towers unset -> a plain decoder.
        # Flat backbones (a single ``*Model`` with optional-tower kwargs) pass through.
        if not args and "text_config" not in kwargs and self._nests_text_config():
            reserved = {
                k: kwargs.pop(k) for k in ("name", "dtype", "trainable") if k in kwargs
            }
            kwargs = {"text_config": kwargs, **reserved}
        # A text-only head never builds modality towers: drop any sibling ``*_config``
        # (``vision_config`` / ``audio_config`` / ...) so a full multimodal config -- e.g.
        # the one reconstructed when extracting the text backbone from a VLM checkpoint --
        # still yields a plain decoder.
        for key in [k for k in kwargs if k.endswith("_config") and k != "text_config"]:
            kwargs.pop(key)
        super().__init__(*args, **kwargs)

    @classmethod
    def _nests_text_config(cls):
        for klass in cls.__mro__:
            if klass is TextOnlyGeneration:
                continue
            init = klass.__dict__.get("__init__")
            if init and "text_config" in inspect.signature(init).parameters:
                return True
        return False

    def build_cache(self, token_ids, padding_mask, max_len):
        # text-only prefill: the multimodal inputs (pixel_values / input_features / ...)
        # default to None on the inherited build_cache.
        return super().build_cache(token_ids, padding_mask, max_len)
