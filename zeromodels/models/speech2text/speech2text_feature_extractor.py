import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseAudioFeatureExtractor

_EPSILON = float(np.finfo(np.float32).eps)


def povey_window(frame_length: int) -> np.ndarray:
    n = np.arange(frame_length)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (frame_length - 1))
    return np.power(hann, 0.85).astype(np.float32)


def kaldi_mel_banks(
    num_bins: int,
    n_fft: int,
    sample_freq: int,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
) -> np.ndarray:
    def mel_scale(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    num_fft_bins = n_fft // 2
    nyquist = 0.5 * sample_freq
    if high_freq <= 0.0:
        high_freq = nyquist
    fft_bin_width = sample_freq / n_fft

    mel_low = mel_scale(low_freq)
    mel_high = mel_scale(high_freq)
    mel_delta = (mel_high - mel_low) / (num_bins + 1)

    bins = np.arange(num_bins)[:, None]
    left_mel = mel_low + bins * mel_delta
    center_mel = mel_low + (bins + 1) * mel_delta
    right_mel = mel_low + (bins + 2) * mel_delta

    mel = mel_scale(fft_bin_width * np.arange(num_fft_bins))[None, :]
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    banks = np.maximum(0.0, np.minimum(up_slope, down_slope))  # (num_bins, n_fft // 2)
    banks = np.pad(banks, ((0, 0), (0, 1)), mode="constant")  # + Nyquist column
    return banks.astype(np.float32)


@keras.saving.register_keras_serializable(package="zeromodels")
class Speech2TextFeatureExtractor(BaseAudioFeatureExtractor):
    """Kaldi-style log-mel filterbank (fbank) extractor for Speech2Text.

    Pure Keras 3 implementation - the spectrogram math goes through
    ``keras.ops`` so the same code runs on TF / Torch / JAX. Reproduces the
    reference ``Speech2TextFeatureExtractor`` (which wraps ``torchaudio``'s
    Kaldi fbank): scale the waveform to int16 range, frame at 25 ms / 10 ms with
    snip-edges, per-frame DC removal + 0.97 pre-emphasis + Povey window,
    512-point power spectrum, 80-channel HTK-mel filterbank, log, then
    per-utterance mean-variance normalization (CMVN).

    Args:
        sampling_rate: Input sample rate (Hz). Must be 16000.
        num_mel_bins: Number of mel filterbank channels (80).
        frame_length_ms / frame_shift_ms: Window / hop in milliseconds.
        preemphasis: Pre-emphasis coefficient.
        normalize_means / normalize_vars: Per-utterance CMVN toggles.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        num_mel_bins: int = 80,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        preemphasis: float = 0.97,
        normalize_means: bool = True,
        normalize_vars: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.num_mel_bins = num_mel_bins
        self.frame_length_ms = frame_length_ms
        self.frame_shift_ms = frame_shift_ms
        self.preemphasis = preemphasis
        self.normalize_means = normalize_means
        self.normalize_vars = normalize_vars

        self.frame_length = int(round(sampling_rate * frame_length_ms / 1000.0))
        self.frame_shift = int(round(sampling_rate * frame_shift_ms / 1000.0))
        self.n_fft = 1
        while self.n_fft < self.frame_length:
            self.n_fft *= 2

        self.window = povey_window(self.frame_length)
        self.mel_banks = kaldi_mel_banks(num_mel_bins, self.n_fft, sampling_rate)

    def _fbank(self, wave: np.ndarray):
        wave = np.asarray(wave, dtype=np.float32) * (2.0**15)
        n = wave.shape[0]
        if n < self.frame_length:
            wave = np.pad(wave, (0, self.frame_length - n))
            n = wave.shape[0]
        num_frames = 1 + (n - self.frame_length) // self.frame_shift

        x = ops.convert_to_tensor(wave, dtype="float32")
        idx = (
            np.arange(num_frames)[:, None] * self.frame_shift
            + np.arange(self.frame_length)[None, :]
        )
        frames = ops.take(x, ops.convert_to_tensor(idx, dtype="int32"), axis=0)

        # per-frame DC offset removal
        frames = frames - ops.mean(frames, axis=1, keepdims=True)
        # pre-emphasis with replicate-padded previous sample
        offset = ops.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
        frames = frames - self.preemphasis * offset
        # Povey window
        frames = frames * ops.convert_to_tensor(self.window, dtype="float32")
        # zero-pad to FFT size and take the power spectrum
        pad = self.n_fft - self.frame_length
        if pad > 0:
            frames = ops.pad(frames, [[0, 0], [0, pad]])
        real, imag = ops.rfft(frames)
        power = real * real + imag * imag  # (num_frames, n_fft // 2 + 1)
        # mel filterbank + log
        mel = ops.matmul(power, ops.transpose(self.mel_banks, (1, 0)))
        mel = ops.log(ops.maximum(mel, _EPSILON))
        return mel

    def _cmvn(self, mel):
        if self.normalize_means:
            mel = mel - ops.mean(mel, axis=0, keepdims=True)
        if self.normalize_vars:
            var = ops.var(mel, axis=0, keepdims=True)
            mel = mel / ops.sqrt(ops.maximum(var, 1e-10))
        return mel

    def _to_waves(self, raw_speech):
        if isinstance(raw_speech, np.ndarray):
            return [raw_speech] if raw_speech.ndim == 1 else list(raw_speech)
        if isinstance(raw_speech, (list, tuple)):
            return [np.asarray(w, dtype=np.float32) for w in raw_speech]
        return [np.asarray(raw_speech, dtype=np.float32).squeeze()]

    def call(self, raw_speech, sampling_rate: int = 16000):
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Speech2TextFeatureExtractor expects {self.sampling_rate} Hz "
                f"input; got {sampling_rate} Hz."
            )
        feats = [self._cmvn(self._fbank(w)) for w in self._to_waves(raw_speech)]
        max_t = max(int(f.shape[0]) for f in feats)
        padded = []
        for f in feats:
            t = int(f.shape[0])
            if t < max_t:
                f = ops.pad(f, [[0, max_t - t], [0, 0]])
            padded.append(f)
        return ops.stack(padded, axis=0)  # (B, T, num_mel_bins)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sampling_rate": self.sampling_rate,
                "num_mel_bins": self.num_mel_bins,
                "frame_length_ms": self.frame_length_ms,
                "frame_shift_ms": self.frame_shift_ms,
                "preemphasis": self.preemphasis,
                "normalize_means": self.normalize_means,
                "normalize_vars": self.normalize_vars,
            }
        )
        return config
