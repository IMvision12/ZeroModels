import math

import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseAudioFeatureExtractor


def create_fb_matrix(n_freqs, f_min, f_max, n_mels, sample_rate, fft_length, norm=None):
    """HTK triangular mel filter bank of shape ``(n_freqs, n_mels)``."""
    all_freqs = np.arange(n_freqs, dtype=np.float32) * (sample_rate / fft_length)
    m_min = 2595.0 * math.log10(1.0 + (f_min / 700.0))
    m_max = 2595.0 * math.log10(1.0 + (f_max / 700.0))
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = np.expand_dims(f_pts, 0) - np.expand_dims(all_freqs, 1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    fb = np.maximum(np.zeros(1, dtype=np.float32), np.minimum(down_slopes, up_slopes))
    if norm == "slaney":
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb *= np.expand_dims(enorm, 0)
    return fb.astype(np.float32)


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma3nAudioFeatureExtractor(BaseAudioFeatureExtractor):
    """Gemma 3n USM log-mel feature extractor, in pure Keras ops.

    Frames the waveform (frame ``frame_length + 1`` so HTK pre-emphasis has a
    left neighbour), applies HTK pre-emphasis and a periodic Hann window, takes
    the (optionally over-driven) rFFT magnitude, an HTK mel filter bank, and a
    ``log(max(mel, mel_floor))`` compression with optional per-bin
    mean / stddev normalization. Returns
    ``{"input_features": (batch, frames, feature_size),
    "input_features_mask": (batch, frames)}`` (``True`` marks padded frames).

    Args:
        feature_size: Number of mel channels. Defaults to ``128``.
        sampling_rate: Input sample rate in Hz. Defaults to ``16000``.
        frame_length_ms / hop_length_ms: STFT frame and hop in milliseconds.
        min_frequency / max_frequency: Mel filter bank range in Hz.
        preemphasis / preemphasis_htk_flavor: Pre-emphasis coefficient and style.
        fft_overdrive: Double the FFT length past the frame's next power of two.
        mel_floor: Added inside the log to avoid ``log(0)``.
        per_bin_mean / per_bin_stddev: Optional per-channel normalization.
    """

    def __init__(
        self,
        feature_size=128,
        sampling_rate=16000,
        padding_value=0.0,
        frame_length_ms=32.0,
        hop_length_ms=10.0,
        min_frequency=125.0,
        max_frequency=7600.0,
        preemphasis=0.97,
        preemphasis_htk_flavor=True,
        fft_overdrive=True,
        dither=0.0,
        input_scale_factor=1.0,
        mel_floor=1e-5,
        per_bin_mean=None,
        per_bin_stddev=None,
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
        self.dither = dither
        self.input_scale_factor = input_scale_factor
        self.mel_floor = float(mel_floor)

        self.frame_length = int(round(sampling_rate * frame_length_ms / 1000.0))
        self.hop_length = int(round(sampling_rate * hop_length_ms / 1000.0))
        fft_length = 2 ** math.ceil(math.log2(self.frame_length))
        if fft_overdrive:
            fft_length *= 2
        self.fft_length = fft_length

        hann = np.arange(self.frame_length, dtype=np.float32)
        self.window = (0.5 * (1 - np.cos(2 * np.pi * hann / self.frame_length))).astype(
            np.float32
        )
        self.mel_filters = create_fb_matrix(
            self.fft_length // 2 + 1,
            min_frequency,
            max_frequency,
            feature_size,
            sampling_rate,
            self.fft_length,
        )
        self.per_bin_mean = (
            np.array(per_bin_mean, dtype=np.float32).reshape(1, 1, feature_size)
            if per_bin_mean is not None
            else None
        )
        self.per_bin_stddev = (
            np.array(per_bin_stddev, dtype=np.float32).reshape(1, 1, feature_size)
            if per_bin_stddev is not None
            else None
        )

    def frame(self, waveform):
        # waveform: (frames_source,) -> (num_frames, frame_length + 1) via unfold.
        size = self.frame_length + 1
        n = int(waveform.shape[0])
        num_frames = (n - size) // self.hop_length + 1
        idx = (
            np.arange(num_frames)[:, None] * self.hop_length + np.arange(size)[None, :]
        )
        return ops.take(waveform, ops.convert_to_tensor(idx.astype("int32")), axis=0)

    def extract(self, waveform):
        w = ops.cast(ops.convert_to_tensor(waveform), "float32")
        if self.input_scale_factor != 1.0:
            w = w * self.input_scale_factor
        frames = self.frame(w)  # (num_frames, frame_length + 1)
        p = self.preemphasis
        if p > 0.0:
            if self.preemphasis_htk_flavor:
                first = frames[:, :1] * (1.0 - p)
                rest = frames[:, 1:-1] - p * frames[:, :-2]
                frames = ops.concatenate([first, rest], axis=-1)
            else:
                frames = frames[:, 1:] - p * frames[:, :-1]
        else:
            frames = frames[:, :-1]
        frames = frames * ops.convert_to_tensor(self.window)
        real, imag = ops.rfft(frames, fft_length=self.fft_length)
        magnitude = ops.sqrt(ops.square(real) + ops.square(imag))
        mel = ops.matmul(magnitude, ops.convert_to_tensor(self.mel_filters))
        log_mel = ops.log(ops.maximum(mel, self.mel_floor))
        if self.per_bin_mean is not None:
            log_mel = log_mel - ops.convert_to_tensor(self.per_bin_mean[0])
        if self.per_bin_stddev is not None:
            log_mel = log_mel / ops.convert_to_tensor(self.per_bin_stddev[0])
        return log_mel  # (num_frames, feature_size)

    def call(self, raw_speech, pad_to_multiple_of=128):
        if isinstance(raw_speech, np.ndarray) and raw_speech.ndim == 1:
            raw_speech = [raw_speech]
        elif not isinstance(raw_speech, (list, tuple)):
            raw_speech = [raw_speech]
        waveforms = [np.asarray(w, dtype=np.float32) for w in raw_speech]
        max_len = max(w.shape[0] for w in waveforms)
        if pad_to_multiple_of:
            max_len = math.ceil(max_len / pad_to_multiple_of) * pad_to_multiple_of
        feats, masks = [], []
        for w in waveforms:
            valid = w.shape[0]
            padded = np.full((max_len,), self.padding_value, dtype=np.float32)
            padded[:valid] = w
            mel = self.extract(padded)  # (num_frames, feature_size)
            num_frames = int(mel.shape[0])
            # HF convention: input_features_mask marks VALID frames (True = valid);
            # the fusion passes ~mask to the audio tower.
            frame_starts = np.arange(num_frames) * self.hop_length
            mask = frame_starts < valid
            feats.append(mel)
            masks.append(ops.convert_to_tensor(mask))
        return {
            "input_features": ops.stack(feats, axis=0),
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
                "dither": self.dither,
                "input_scale_factor": self.input_scale_factor,
                "mel_floor": self.mel_floor,
            }
        )
        return config
