import math

import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseAudioFeatureExtractor


def htk_mel_filter_bank(
    num_frequency_bins, num_mel_filters, min_frequency, max_frequency, sampling_rate
):
    """HTK triangular mel filter bank ``(num_frequency_bins, num_mel_filters)``,
    matching ``transformers.audio_utils.mel_filter_bank(mel_scale="htk", norm=None)``."""
    ln10 = math.log(10.0)
    mel_min = 2595.0 * math.log10(1.0 + min_frequency / 700.0)
    mel_max = 2595.0 * math.log10(1.0 + max_frequency / 700.0)
    mel_freqs = ops.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = 700.0 * (ops.exp(mel_freqs * ln10 / 2595.0) - 1.0)
    fft_freqs = ops.linspace(0.0, float(sampling_rate // 2), num_frequency_bins)
    diff = filter_freqs[1:] - filter_freqs[:-1]
    slopes = ops.expand_dims(filter_freqs, 0) - ops.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / diff[:-1]
    up_slopes = slopes[:, 2:] / diff[1:]
    return ops.maximum(0.0, ops.minimum(down_slopes, up_slopes))


def periodic_hann(window_length):
    """Periodic Hann window, matching ``window_function(name="hann")``."""
    n = ops.arange(window_length, dtype="float32")
    return 0.5 - 0.5 * ops.cos(2.0 * math.pi * n / window_length)


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma4AudioFeatureExtractor(BaseAudioFeatureExtractor):
    """Gemma 4 (USM) log-mel feature extractor, in pure Keras ops.

    Periodic-Hann STFT with semicausal time padding, an HTK mel filter bank, and a
    ``log(mel + mel_floor)`` compression. Returns
    ``{"input_features": (batch, frames, feature_size),
    "input_features_mask": (batch, frames)}`` with padded frames zeroed out.

    Args:
        feature_size: Number of mel channels. Defaults to ``128``.
        sampling_rate: Input sample rate in Hz. Defaults to ``16000``.
        frame_length_ms / hop_length_ms: STFT frame and hop in milliseconds.
        min_frequency / max_frequency: Mel filter bank frequency range in Hz.
        preemphasis / preemphasis_htk_flavor: Pre-emphasis coefficient and style.
        fft_overdrive: Double the FFT length when set.
        input_scale_factor: Scale applied to the waveform.
        mel_floor: Added inside the log to avoid ``log(0)``.
        per_bin_mean / per_bin_stddev: Optional per-channel normalization.
        max_length: Waveform truncation length in samples.
        pad_to_multiple_of: Pad the batch waveform length to this multiple.
    """

    def __init__(
        self,
        feature_size=128,
        sampling_rate=16000,
        padding_value=0.0,
        frame_length_ms=20.0,
        hop_length_ms=10.0,
        min_frequency=0.0,
        max_frequency=8000.0,
        preemphasis=0.0,
        preemphasis_htk_flavor=True,
        fft_overdrive=False,
        input_scale_factor=1.0,
        mel_floor=1e-3,
        per_bin_mean=None,
        per_bin_stddev=None,
        max_length=480000,
        pad_to_multiple_of=128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.frame_length_ms = frame_length_ms
        self.hop_length_ms = hop_length_ms
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.preemphasis = preemphasis
        self.preemphasis_htk_flavor = preemphasis_htk_flavor
        self.fft_overdrive = fft_overdrive
        self.input_scale_factor = input_scale_factor
        self.mel_floor = float(mel_floor)
        self.per_bin_mean = per_bin_mean
        self.per_bin_stddev = per_bin_stddev
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

        self.frame_length = int(round(sampling_rate * frame_length_ms / 1000.0))
        self.hop_length = int(round(sampling_rate * hop_length_ms / 1000.0))
        fft_length = 2 ** math.ceil(math.log2(self.frame_length))
        if fft_overdrive:
            fft_length *= 2
        self.fft_length = fft_length
        self.window = periodic_hann(self.frame_length)
        self.mel_filters = htk_mel_filter_bank(
            fft_length // 2 + 1,
            feature_size,
            min_frequency,
            max_frequency,
            sampling_rate,
        )
        self._mean = (
            None
            if per_bin_mean is None
            else ops.reshape(
                ops.convert_to_tensor(per_bin_mean, "float32"), (1, feature_size)
            )
        )
        self._stddev = (
            None
            if per_bin_stddev is None
            else ops.reshape(
                ops.convert_to_tensor(per_bin_stddev, "float32"), (1, feature_size)
            )
        )

    def _spectrogram(self, waveform, mask, num_frames):
        """``waveform`` / ``mask``: 1-D tensors already left-padded by
        ``frame_length // 2``. Returns ``(num_frames, feature_size)`` log-mel and a
        ``(num_frames,)`` validity mask."""
        if self.input_scale_factor != 1.0:
            waveform = waveform * self.input_scale_factor

        frame_size = self.frame_length + 1
        starts = ops.expand_dims(ops.arange(num_frames) * self.hop_length, 1)
        cols = ops.expand_dims(ops.arange(frame_size), 0)
        frames = ops.take(waveform, starts + cols, axis=0)  # (num_frames, frame_size)

        if self.preemphasis > 0.0:
            if self.preemphasis_htk_flavor:
                first = frames[:, :1] * (1.0 - self.preemphasis)
                rest = frames[:, 1:-1] - self.preemphasis * frames[:, :-2]
                frames = ops.concatenate([first, rest], axis=-1)
            else:
                frames = frames[:, 1:] - self.preemphasis * frames[:, :-1]
        else:
            frames = frames[:, :-1]

        frames = frames * self.window
        real, imag = ops.rfft(frames, fft_length=self.fft_length)
        magnitude = ops.sqrt(real * real + imag * imag)
        mel = ops.matmul(magnitude, self.mel_filters)
        log_mel = ops.log(mel + self.mel_floor)
        if self._mean is not None:
            log_mel = log_mel - self._mean
        if self._stddev is not None:
            log_mel = log_mel / self._stddev

        frame_end = ops.arange(num_frames) * self.hop_length + frame_size - 1
        frame_mask = ops.take(mask, frame_end, axis=0)
        return log_mel * ops.expand_dims(frame_mask, -1), frame_mask

    def call(self, raw_speech, sampling_rate=16000):
        if isinstance(raw_speech, np.ndarray) and raw_speech.ndim > 1:
            waves = list(raw_speech)
        elif (
            isinstance(raw_speech, (list, tuple))
            and len(raw_speech)
            and isinstance(raw_speech[0], (np.ndarray, list, tuple))
        ):
            waves = list(raw_speech)
        else:
            waves = [raw_speech]

        lengths = []
        arrays = []
        for w in waves:
            a = np.asarray(w, dtype="float32").reshape(-1)
            if self.max_length:
                a = a[: self.max_length]
            arrays.append(a)
            lengths.append(len(a))

        target = max(lengths) if lengths else 0
        if self.pad_to_multiple_of:
            target = (
                int(math.ceil(target / self.pad_to_multiple_of))
                * self.pad_to_multiple_of
            )
        pad_left = self.frame_length // 2
        num_frames = (
            target + pad_left - (self.frame_length + 1)
        ) // self.hop_length + 1

        features, masks = [], []
        for a, length in zip(arrays, lengths):
            wave = ops.convert_to_tensor(a, "float32")
            wave = ops.pad(wave, [[pad_left, target - length]])
            mask = ops.concatenate(
                [
                    ops.zeros((pad_left,), dtype="int32"),
                    ops.ones((length,), dtype="int32"),
                    ops.zeros((target - length,), dtype="int32"),
                ],
                axis=0,
            )
            feat, fmask = self._spectrogram(wave, mask, num_frames)
            features.append(feat)
            masks.append(fmask)

        return {
            "input_features": ops.stack(features, axis=0),
            "input_features_mask": ops.stack(masks, axis=0),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "feature_size": self.feature_size,
                "sampling_rate": self.sampling_rate,
                "padding_value": self.padding_value,
                "frame_length_ms": self.frame_length_ms,
                "hop_length_ms": self.hop_length_ms,
                "min_frequency": self.min_frequency,
                "max_frequency": self.max_frequency,
                "preemphasis": self.preemphasis,
                "preemphasis_htk_flavor": self.preemphasis_htk_flavor,
                "fft_overdrive": self.fft_overdrive,
                "input_scale_factor": self.input_scale_factor,
                "mel_floor": self.mel_floor,
                "per_bin_mean": self.per_bin_mean,
                "per_bin_stddev": self.per_bin_stddev,
                "max_length": self.max_length,
                "pad_to_multiple_of": self.pad_to_multiple_of,
            }
        )
        return config
