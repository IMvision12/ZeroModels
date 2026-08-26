import keras
import numpy as np
from keras import ops

from zeromodels.base import BaseAudioFeatureExtractor


@keras.saving.register_keras_serializable(package="zeromodels")
class MoonshineFeatureExtractor(BaseAudioFeatureExtractor):
    """Raw-waveform "feature extractor" for Moonshine.

    Moonshine ingests the 16 kHz waveform directly: its conv stem replaces the
    log-mel front end, so this extractor mirrors the reference
    ``Wav2Vec2FeatureExtractor`` configuration used by the Moonshine repo
    (``feature_size=1``, ``do_normalize=False``, ``padding_value=0.0``): it
    simply stacks a batch of waveforms, right-zero-padding shorter clips to the
    longest in the batch, and returns the ``(B, audio_length)`` float tensor.

    Args:
        sampling_rate: Expected input sample rate (Hz). Must be ``16000``.
        padding_value: Value used to pad shorter waveforms. Defaults to ``0.0``.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        padding_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value

    def _to_waves(self, raw_speech):
        if isinstance(raw_speech, np.ndarray):
            return [raw_speech] if raw_speech.ndim == 1 else list(raw_speech)
        if isinstance(raw_speech, (list, tuple)):
            return [np.asarray(w, dtype=np.float32) for w in raw_speech]
        return [np.asarray(raw_speech, dtype=np.float32).squeeze()]

    def call(self, raw_speech, sampling_rate: int = 16000):
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"MoonshineFeatureExtractor expects {self.sampling_rate} Hz "
                f"input; got {sampling_rate} Hz."
            )
        waves = [np.asarray(w, dtype=np.float32) for w in self._to_waves(raw_speech)]
        max_len = max(int(w.shape[0]) for w in waves)
        out = np.full((len(waves), max_len), self.padding_value, dtype=np.float32)
        for i, w in enumerate(waves):
            out[i, : w.shape[0]] = w
        return ops.convert_to_tensor(out, dtype="float32")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sampling_rate": self.sampling_rate,
                "padding_value": self.padding_value,
            }
        )
        return config
