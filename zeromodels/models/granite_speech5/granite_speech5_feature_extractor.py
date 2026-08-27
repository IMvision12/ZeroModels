import math

import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseAudioFeatureExtractor


@keras.saving.register_keras_serializable(package="zeromodels")
class GraniteSpeech5FeatureExtractor(BaseAudioFeatureExtractor):
    """Log-mel(+delta) feature extractor for Granite Speech 5.0 (pure Keras 3).

    Reproduces HF ``GraniteSpeech5FeatureExtractor``:

    * 16 kHz ``torchaudio``-style ``MelSpectrogram`` (n_fft=512, win=400, hop=160,
      80 HTK mel bins, power spectrogram), centered reflect-padded STFT.
    * the waveform is right-padded so the frame count is even (no dropped frame);
      ``log10`` of the mel, floored at ``max - logmel_floor_db`` dB, then ``/4 + 1``.
    * each frame is concatenated with its time-delta (``[-1, 0, 1] / 2`` filter,
      replicate-padded), then consecutive frames are stacked in pairs -> width
      ``num_mel_bins * 4``.

    ``call`` returns ``input_features`` ``(B, frames // 2, num_mel_bins * 4)`` and an
    ``attention_mask`` over the (subsampled) valid frames per clip.
    """

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        sampling_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        num_mel_bins=80,
        delta_win_length=3,
        logmel_floor_db=8.0,
        frame_stacking=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.num_mel_bins = num_mel_bins
        self.delta_win_length = delta_win_length
        self.logmel_floor_db = logmel_floor_db
        self.frame_stacking = frame_stacking
        self.mel_filters = self.build_mel_filters()

    def build_mel_filters(self):
        f_min, f_max = 0.0, self.sampling_rate / 2.0
        n_freqs = self.n_fft // 2 + 1
        all_freqs = np.linspace(0, f_max, n_freqs)
        m_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
        m_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
        m_pts = np.linspace(m_min, m_max, self.num_mel_bins + 2)
        f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
        f_diff = np.diff(f_pts)
        slopes = f_pts[None, :] - all_freqs[:, None]
        down = -slopes[:, :-2] / f_diff[:-1][None, :]
        up = slopes[:, 2:] / f_diff[1:][None, :]
        fb = np.maximum(0.0, np.minimum(down, up))
        return fb.astype("float32")

    def normalize_waves(self, audios):
        if isinstance(audios, np.ndarray):
            waves = [audios] if audios.ndim == 1 else list(audios)
        elif isinstance(audios, (list, tuple)):
            waves = [np.asarray(w, dtype=np.float32).squeeze() for w in audios]
        else:
            waves = [np.asarray(audios, dtype=np.float32).squeeze()]
        lengths = [int(np.asarray(w).reshape(-1).shape[0]) for w in waves]
        max_len = max(lengths)
        out = np.zeros((len(waves), max_len), dtype=np.float32)
        for i, w in enumerate(waves):
            w = np.asarray(w, dtype=np.float32).reshape(-1)
            out[i, : len(w)] = w
        return out, lengths

    def log_mel(self, batch, num_frames):
        pad = (self.n_fft - self.win_length) // 2
        hann = ops.convert_to_tensor(
            np.hanning(self.win_length + 1)[:-1].astype("float32")
        )
        window = ops.pad(hann, [[pad, self.n_fft - self.win_length - pad]])
        real, imag = ops.stft(
            batch,
            sequence_length=self.n_fft,
            sequence_stride=self.hop_length,
            fft_length=self.n_fft,
            window=window,
            center=True,
        )
        power = real * real + imag * imag
        mel = ops.matmul(power, ops.convert_to_tensor(self.mel_filters))
        mel = mel[:, :num_frames, :]  # match HF: slice before the log/floor
        inv_log10 = 1.0 / math.log(10.0)
        logmel = ops.log(ops.maximum(mel, 1e-10)) * inv_log10
        mx = ops.max(logmel, axis=(1, 2), keepdims=True)
        logmel = ops.maximum(logmel, mx - self.logmel_floor_db)
        return logmel / 4.0 + 1.0

    def compute_deltas(self, logmel):
        """torchaudio ``compute_deltas``: a Savitzky-Golay derivative over time,
        with replicate padding. For ``win_length=3`` this is ``(x[t+1]-x[t-1])/2``."""
        n = (self.delta_win_length - 1) // 2
        denom = 2.0 * sum(k * k for k in range(1, n + 1))
        first = ops.repeat(logmel[:, :1], n, axis=1)
        last = ops.repeat(logmel[:, -1:], n, axis=1)
        padded = ops.concatenate([first, logmel, last], axis=1)
        length = int(logmel.shape[1])
        delta = None
        for k in range(-n, n + 1):
            if k == 0:
                continue
            term = k * padded[:, n + k : n + k + length, :]
            delta = term if delta is None else delta + term
        return delta / denom

    def call(self, raw_speech, sampling_rate=16000):
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"GraniteSpeech5FeatureExtractor expects {self.sampling_rate} Hz "
                f"input; got {sampling_rate} Hz."
            )
        batch_np, lengths = self.normalize_waves(raw_speech)
        # right-pad the waveform so the trailing frame-stacking group is filled
        mel_frames = batch_np.shape[1] // self.hop_length
        num_frames = self.frame_stacking * math.ceil(mel_frames / self.frame_stacking)
        num_samples_needed = (num_frames - 1) * self.hop_length + 1
        if batch_np.shape[1] < num_samples_needed:
            batch_np = np.pad(
                batch_np, ((0, 0), (0, num_samples_needed - batch_np.shape[1]))
            )

        batch = ops.convert_to_tensor(batch_np, dtype="float32")
        logmel = self.log_mel(batch, num_frames)
        deltas = self.compute_deltas(logmel)
        features = ops.concatenate([logmel, deltas], axis=-1)
        b = int(features.shape[0])
        features = ops.reshape(
            features,
            (
                b,
                num_frames // self.frame_stacking,
                self.frame_stacking * 2 * self.num_mel_bins,
            ),
        )

        enc_counts = [
            math.ceil((length // self.hop_length) / self.frame_stacking)
            for length in lengths
        ]
        max_enc = num_frames // self.frame_stacking
        mask = np.arange(max_enc)[None, :] < np.array(enc_counts)[:, None]
        return {
            "input_features": features,
            "attention_mask": ops.convert_to_tensor(mask.astype("int32")),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sampling_rate": self.sampling_rate,
                "n_fft": self.n_fft,
                "win_length": self.win_length,
                "hop_length": self.hop_length,
                "num_mel_bins": self.num_mel_bins,
                "delta_win_length": self.delta_win_length,
                "logmel_floor_db": self.logmel_floor_db,
                "frame_stacking": self.frame_stacking,
            }
        )
        return config
