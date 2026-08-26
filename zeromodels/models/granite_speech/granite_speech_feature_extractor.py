import math

import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseAudioFeatureExtractor


@keras.saving.register_keras_serializable(package="zeromodels")
class GraniteSpeechFeatureExtractor(BaseAudioFeatureExtractor):
    """Mel-spectrogram feature extractor for Granite Speech (pure Keras 3).

    Reproduces HF ``GraniteSpeechFeatureExtractor``:

    * 16 kHz, ``torchaudio``-style ``MelSpectrogram`` (n_fft=512, win=400,
      hop=160, 80 HTK mel bins, power spectrogram), centered reflect-padded STFT.
    * ``log10`` of the (transposed) mel, clamped at ``max - 8.0``, then ``/4 + 1``.
    * the last frame is dropped if the frame count is odd, and consecutive frames
      are stacked in pairs -> ``input_features`` of width ``2 * n_mels``.

    ``call`` also returns ``audio_embed_sizes`` (the projector output length per
    clip) and a boolean ``input_features_mask`` over the padded projector tokens,
    mirroring the HF processor contract.
    """

    model_input_names = ["input_features"]

    def __init__(
        self,
        sampling_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=80,
        projector_window_size=15,
        projector_downsample_rate=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.projector_window_size = projector_window_size
        self.projector_downsample_rate = projector_downsample_rate
        self.mel_filters = self.build_mel_filters()

    def build_mel_filters(self):
        f_min, f_max = 0.0, self.sampling_rate / 2.0
        n_freqs = self.n_fft // 2 + 1
        all_freqs = np.linspace(0, f_max, n_freqs)
        m_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
        m_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
        m_pts = np.linspace(m_min, m_max, self.n_mels + 2)
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
            arr = np.asarray(audios, dtype=np.float32).squeeze()
            waves = [arr]
        lengths = [int(np.asarray(w).shape[-1]) for w in waves]
        max_len = max(lengths)
        out = np.zeros((len(waves), max_len), dtype=np.float32)
        for i, w in enumerate(waves):
            w = np.asarray(w, dtype=np.float32).reshape(-1)
            out[i, : len(w)] = w
        return out, lengths

    def log_mel(self, batch):
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
        inv_log10 = 1.0 / math.log(10.0)
        logmel = ops.log(ops.maximum(mel, 1e-10)) * inv_log10
        mx = ops.max(logmel, axis=(1, 2), keepdims=True)
        logmel = ops.maximum(logmel, mx - 8.0)
        logmel = logmel / 4.0 + 1.0
        return logmel

    def stack_frames(self, logmel):
        n_frames = int(logmel.shape[1])
        if n_frames % 2 == 1:
            logmel = logmel[:, :-1, :]
            n_frames -= 1
        b = int(logmel.shape[0])
        return ops.reshape(logmel, (b, n_frames // 2, 2 * self.n_mels))

    def num_audio_features(self, audio_lengths):
        eff = self.projector_window_size // self.projector_downsample_rate
        sizes = []
        for raw in audio_lengths:
            mel_len = raw // self.hop_length + 1
            enc_len = mel_len // 2
            nblocks = math.ceil(enc_len / self.projector_window_size)
            sizes.append(nblocks * eff)
        return sizes

    def call(self, raw_speech, sampling_rate=16000):
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"GraniteSpeechFeatureExtractor expects {self.sampling_rate} Hz "
                f"input; got {sampling_rate} Hz."
            )
        batch_np, lengths = self.normalize_waves(raw_speech)
        batch = ops.convert_to_tensor(batch_np, dtype="float32")
        logmel = self.log_mel(batch)
        features = self.stack_frames(logmel)

        embed_sizes = self.num_audio_features(lengths)
        max_size = max(embed_sizes)
        mask = ops.convert_to_tensor(
            np.arange(max_size)[None, :] < np.array(embed_sizes)[:, None]
        )
        return {
            "input_features": features,
            "audio_embed_sizes": embed_sizes,
            "input_features_mask": mask,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sampling_rate": self.sampling_rate,
                "n_fft": self.n_fft,
                "win_length": self.win_length,
                "hop_length": self.hop_length,
                "n_mels": self.n_mels,
                "projector_window_size": self.projector_window_size,
                "projector_downsample_rate": self.projector_downsample_rate,
            }
        )
        return config
