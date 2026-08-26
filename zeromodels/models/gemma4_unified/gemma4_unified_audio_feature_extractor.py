import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseAudioFeatureExtractor


@keras.saving.register_keras_serializable(package="zeromodels")
class Gemma4UnifiedAudioFeatureExtractor(BaseAudioFeatureExtractor):
    """Gemma 4 unified audio feature extractor, in pure Keras ops.

    Encoder-free: instead of the "gemma4" USM log-mel pipeline, it simply chunks
    raw 16 kHz audio into fixed ``audio_samples_per_token`` (640, 40ms) frames.
    Each frame becomes one audio soft token whose features are its raw waveform
    samples (no mel, no downsampling). Each waveform is zero-padded up to a whole
    number of frames (so every one of its frames is valid); the batch is padded to
    the longest token count and padded tokens are marked in the mask. Returns
    ``{"input_features": (batch, num_tokens, audio_samples_per_token),
    "input_features_mask": (batch, num_tokens)}``.

    Args:
        feature_size: Samples per token. Defaults to ``640``.
        sampling_rate: Input sample rate in Hz. Defaults to ``16000``.
        audio_samples_per_token: Frame length in samples (== ``feature_size``).
        max_length: Optional waveform truncation length in samples.
    """

    def __init__(
        self,
        feature_size=640,
        sampling_rate=16000,
        audio_samples_per_token=640,
        max_length=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.audio_samples_per_token = audio_samples_per_token
        self.max_length = max_length

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

        spt = self.audio_samples_per_token
        frames_per_wave = []
        tensors = []
        for w in waves:
            wave = ops.reshape(ops.convert_to_tensor(w, "float32"), (-1,))
            if self.max_length:
                wave = wave[: self.max_length]
            length = int(ops.shape(wave)[0])
            pad = (-length) % spt
            if pad:
                wave = ops.pad(wave, [[0, pad]])
            n = (length + pad) // spt
            tensors.append(ops.reshape(wave, (n, spt)))
            frames_per_wave.append(n)

        target = max(frames_per_wave) if frames_per_wave else 0
        features, masks = [], []
        for frame, n in zip(tensors, frames_per_wave):
            frame = ops.pad(frame, [[0, target - n], [0, 0]])
            mask = ops.concatenate(
                [ops.ones((n,), dtype="bool"), ops.zeros((target - n,), dtype="bool")],
                axis=0,
            )
            features.append(frame)
            masks.append(mask)

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
                "audio_samples_per_token": self.audio_samples_per_token,
                "max_length": self.max_length,
            }
        )
        return config
