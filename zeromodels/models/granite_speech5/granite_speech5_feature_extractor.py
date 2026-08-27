import math

import keras
from keras import ops

from zeromodels.base import BaseAudioFeatureExtractor


@keras.saving.register_keras_serializable(package="zeromodels")
class GraniteSpeech5FeatureExtractor(BaseAudioFeatureExtractor):
    """Log-mel(+delta) feature extractor for Granite Speech 5.0 (pure Keras 3 ops).

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
        self.window = self.build_window()

    def build_mel_filters(self):
        """HTK mel filterbank ``(n_fft // 2 + 1, num_mel_bins)`` as a Keras tensor."""
        f_max = self.sampling_rate / 2.0
        n_freqs = self.n_fft // 2 + 1
        all_freqs = ops.linspace(0.0, f_max, n_freqs)
        m_max = 2595.0 * ops.log10(1.0 + f_max / 700.0)
        m_pts = ops.linspace(0.0, m_max, self.num_mel_bins + 2)
        f_pts = 700.0 * (ops.power(10.0, m_pts / 2595.0) - 1.0)
        f_diff = f_pts[1:] - f_pts[:-1]
        slopes = f_pts[None, :] - all_freqs[:, None]
        down = -slopes[:, :-2] / f_diff[:-1][None, :]
        up = slopes[:, 2:] / f_diff[1:][None, :]
        return ops.maximum(0.0, ops.minimum(down, up))

    def build_window(self):
        """Periodic Hann window (``win_length``) zero-padded into ``n_fft``."""
        n = ops.arange(self.win_length, dtype="float32")
        hann = 0.5 - 0.5 * ops.cos(2.0 * math.pi * n / self.win_length)
        pad = (self.n_fft - self.win_length) // 2
        return ops.pad(hann, [[pad, self.n_fft - self.win_length - pad]])

    def normalize_waves(self, audios):
        """Parse any accepted input into a padded ``(batch, samples)`` tensor + the
        per-clip sample counts."""
        if isinstance(audios, (list, tuple)):
            clips = [audios] if not hasattr(audios[0], "__len__") else list(audios)
        elif len(ops.shape(ops.convert_to_tensor(audios))) > 1:
            clips = list(audios)
        else:
            clips = [audios]

        waves = [
            ops.reshape(ops.cast(ops.convert_to_tensor(c), "float32"), (-1,))
            for c in clips
        ]
        lengths = [int(ops.shape(w)[0]) for w in waves]
        max_len = max(lengths)
        batch = ops.stack(
            [ops.pad(w, [[0, max_len - length]]) for w, length in zip(waves, lengths)],
            axis=0,
        )
        return batch, lengths

    def log_mel(self, batch, num_frames):
        real, imag = ops.stft(
            batch,
            sequence_length=self.n_fft,
            sequence_stride=self.hop_length,
            fft_length=self.n_fft,
            window=self.window,
            center=True,
        )
        power = real * real + imag * imag
        mel = ops.matmul(power, self.mel_filters)
        mel = mel[:, :num_frames, :]  # match HF: slice before the log/floor
        logmel = ops.log10(ops.maximum(mel, 1e-10))
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
        length = int(ops.shape(logmel)[1])
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
        batch, lengths = self.normalize_waves(raw_speech)
        # right-pad the waveform so the trailing frame-stacking group is filled
        num_samples = int(ops.shape(batch)[1])
        mel_frames = num_samples // self.hop_length
        num_frames = self.frame_stacking * math.ceil(mel_frames / self.frame_stacking)
        num_samples_needed = (num_frames - 1) * self.hop_length + 1
        if num_samples < num_samples_needed:
            batch = ops.pad(batch, [[0, 0], [0, num_samples_needed - num_samples]])

        logmel = self.log_mel(batch, num_frames)
        deltas = self.compute_deltas(logmel)
        features = ops.concatenate([logmel, deltas], axis=-1)
        features = ops.reshape(
            features,
            (
                int(ops.shape(features)[0]),
                num_frames // self.frame_stacking,
                self.frame_stacking * 2 * self.num_mel_bins,
            ),
        )

        enc_counts = [
            math.ceil((length // self.hop_length) / self.frame_stacking)
            for length in lengths
        ]
        max_enc = num_frames // self.frame_stacking
        mask = ops.arange(max_enc)[None, :] < ops.convert_to_tensor(enc_counts)[:, None]
        return {
            "input_features": features,
            "attention_mask": ops.cast(mask, "int32"),
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
