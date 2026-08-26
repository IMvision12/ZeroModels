from zeromodels.base.base_mixin import PreprocessorMixin

HF_FEATURE_EXTRACTOR_ALIASES = {"feature_size": ("n_mels", "num_mel_bins")}


class BaseAudioFeatureExtractor(PreprocessorMixin):
    """Abstract base for zeromodels audio feature extractors.

    Subclasses implement ``call(raw_speech, ...)`` returning the spectrogram /
    feature tensor. The loading API (``from_weights`` / ``from_variant``) and
    the ``__call__`` -> ``call`` forwarder are inherited from
    :class:`PreprocessorMixin`; ``from_hf`` is overridden here to map the
    repo's ``preprocessor_config.json`` scalars (``sampling_rate``, ``n_fft``,
    ``hop_length``, ``chunk_length``, ``feature_size`` -> ``n_mels`` /
    ``num_mel_bins``, …) onto same-named constructor params. Explicit caller
    kwargs always win; a missing config falls back to the subclass defaults.
    Concrete subclasses define their own constructor kwargs (sampling rate,
    FFT size, mel bin count, chunk length, …) and ``get_config`` payload: the
    base bakes in no defaults.
    """

    def call(self, raw_speech, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement `call(raw_speech, ...)`."
        )

    @classmethod
    def from_hf(cls, repo, **kwargs):
        import inspect
        import json

        params = set(inspect.signature(cls).parameters)
        try:
            from huggingface_hub import hf_hub_download

            with open(
                hf_hub_download(repo, "preprocessor_config.json"), encoding="utf-8"
            ) as f:
                hf = json.load(f)
        except Exception:
            return cls(**kwargs)
        for key, value in hf.items():
            if value is None or isinstance(value, (list, dict)):
                continue
            for param in (key, *HF_FEATURE_EXTRACTOR_ALIASES.get(key, ())):
                if param in params:
                    kwargs.setdefault(param, value)
                    break
        return cls(**kwargs)
