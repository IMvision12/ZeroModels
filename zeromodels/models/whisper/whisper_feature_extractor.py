import math

import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseAudioFeatureExtractor


def build_mel_filter_bank(
    n_fft: int,
    n_mels: int,
    sample_rate: int = 16000,
    min_hz: float = 0.0,
    max_hz: float = 8000.0,
):
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    log_step = math.log(6.4) / 27.0

    def _hz_to_mel(f):
        linear = f / f_sp
        safe = ops.maximum(f, 1e-10) / min_log_hz
        log_part = min_log_mel + ops.log(safe) / log_step
        return ops.where(f >= min_log_hz, log_part, linear)

    def _mel_to_hz(m):
        linear = m * f_sp
        log_part = min_log_hz * ops.exp(log_step * (m - min_log_mel))
        return ops.where(m >= min_log_mel, log_part, linear)

    mel_min = _hz_to_mel(ops.convert_to_tensor(min_hz, dtype="float32"))
    mel_max = _hz_to_mel(ops.convert_to_tensor(max_hz, dtype="float32"))
    mel_points = ops.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    fft_freqs = ops.linspace(
        ops.convert_to_tensor(0.0, dtype="float32"),
        ops.convert_to_tensor(sample_rate / 2.0, dtype="float32"),
        n_fft // 2 + 1,
    )

    h0 = hz_points[:-2][:, None]
    h1 = hz_points[1:-1][:, None]
    h2 = hz_points[2:][:, None]
    f = fft_freqs[None, :]

    lower = (f - h0) / (h1 - h0)
    upper = (h2 - f) / (h2 - h1)
    filt = ops.maximum(0.0, ops.minimum(lower, upper))
    enorm = 2.0 / (h2 - h0)  # (n_mels, 1)
    return filt * enorm


@keras.saving.register_keras_serializable(package="zeromodels")
class WhisperFeatureExtractor(BaseAudioFeatureExtractor):
    """Mel spectrogram extractor.

    Pure Keras 3 implementation: all numeric operations go through
    ``keras.ops`` (``ops.stft``, ``ops.matmul``, ``ops.log``, ...) so the
    same code runs on TF / Torch / JAX backends. Input normalization
    (list → stacked ``(B, n_samples)``) uses numpy since it's plumbing.

    Defaults reproduce OpenAI Whisper:
      * 16 kHz, 30-second chunks (zero-padded)
      * STFT: n_fft=400, hop=160, Hann window, centered reflect pad
      * 80-channel Slaney mel filter bank, 0 to 8 kHz
      * log10 magnitude, clamped at ``max - 8.0``, then ``(x + 4) / 4``
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        chunk_length: int = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.chunk_length = chunk_length
        self.n_samples = sampling_rate * chunk_length
        self.nb_max_frames = self.n_samples // hop_length
        self.mel_filters = build_mel_filter_bank(n_fft, n_mels, sampling_rate)

    def _normalize_waves(self, raw_speech) -> np.ndarray:
        if isinstance(raw_speech, np.ndarray):
            waves = [raw_speech] if raw_speech.ndim == 1 else list(raw_speech)
        elif isinstance(raw_speech, (list, tuple)):
            waves = [np.asarray(w, dtype=np.float32) for w in raw_speech]
        else:
            arr = np.asarray(raw_speech, dtype=np.float32).squeeze()
            waves = [arr]

        out = np.zeros((len(waves), self.n_samples), dtype=np.float32)
        for i, w in enumerate(waves):
            w = np.asarray(w, dtype=np.float32)
            n = min(len(w), self.n_samples)
            out[i, :n] = w[:n]
        return out

    def _log_mel_spectrogram(self, batch):
        real, imag = ops.stft(
            batch,
            sequence_length=self.n_fft,
            sequence_stride=self.hop_length,
            fft_length=self.n_fft,
            window="hann",
            center=True,
        )
        power = real * real + imag * imag  # (B, n_frames, n_fft//2+1)
        mel = ops.matmul(
            power, ops.transpose(self.mel_filters, (1, 0))
        )  # (B, n_frames, n_mels)
        mel = mel[:, :-1, :]
        mel = ops.transpose(mel, (0, 2, 1))  # (B, n_mels, n_frames-1)

        inv_log10 = 1.0 / math.log(10.0)
        log_spec = ops.log(ops.maximum(mel, 1e-10)) * inv_log10
        max_per_sample = ops.max(log_spec, axis=(1, 2), keepdims=True)
        log_spec = ops.maximum(log_spec, max_per_sample - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def call(self, raw_speech, sampling_rate: int = 16000):
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"WhisperFeatureExtractor expects {self.sampling_rate} Hz input; "
                f"got {sampling_rate} Hz."
            )
        batch_np = self._normalize_waves(raw_speech)
        batch = ops.convert_to_tensor(batch_np, dtype="float32")
        return self._log_mel_spectrogram(batch)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sampling_rate": self.sampling_rate,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "n_mels": self.n_mels,
                "chunk_length": self.chunk_length,
            }
        )
        return config
