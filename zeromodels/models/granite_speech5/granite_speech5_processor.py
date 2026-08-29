import keras
from keras import ops

from zeromodels.base import BaseProcessor

from .granite_speech5_feature_extractor import GraniteSpeech5FeatureExtractor
from .granite_speech5_tokenizer import GraniteSpeech5Tokenizer


@keras.saving.register_keras_serializable(package="zeromodels")
class GraniteSpeech5Processor(BaseProcessor):
    """Audio (+ optional target text) -> model inputs for Granite Speech 5.0 CTC.

    Composes the CTC tokenizer and the log-mel(+delta) feature extractor. ``call``
    runs the feature extractor on the audio (returning ``input_features`` and the
    encoder ``attention_mask``); when ``text`` is given it is tokenized into padded
    CTC ``labels``. ``batch_decode`` CTC-decodes the model's per-frame argmax ids.
    """

    TOKENIZER_CLS = GraniteSpeech5Tokenizer
    FEATURE_EXTRACTOR_CLS = GraniteSpeech5FeatureExtractor

    def __init__(
        self, tokenizer=None, feature_extractor=None, encoder_downsample=4, **kwargs
    ):
        super().__init__(**kwargs)
        self.feature_extractor = feature_extractor or self.FEATURE_EXTRACTOR_CLS()
        self.tokenizer = tokenizer or self.TOKENIZER_CLS()
        # Total time reduction the CTC encoder applies (2 subsampling blocks -> 4), used
        # to convert a CTC frame index to seconds for timestamp decoding.
        self.encoder_downsample = encoder_downsample

    @property
    def frame_seconds(self) -> float:
        """Audio duration one CTC output frame covers (feature hop x frame stacking x
        the encoder's time reduction)."""
        fe = self.feature_extractor
        return (
            (fe.hop_length / fe.sampling_rate)
            * fe.frame_stacking
            * self.encoder_downsample
        )

    def call(self, audio=None, text=None, sampling_rate=16000):
        if audio is None:
            raise ValueError("GraniteSpeech5Processor requires `audio`.")
        audio_inputs = self.feature_extractor(audio, sampling_rate=sampling_rate)
        out = {
            "input_features": ops.convert_to_tensor(audio_inputs["input_features"]),
            "attention_mask": ops.convert_to_tensor(audio_inputs["attention_mask"]),
        }
        if text is not None:
            texts = [text] if isinstance(text, str) else list(text)
            label_ids = [self.tokenizer.tokenize(t) for t in texts]
            max_len = max(len(x) for x in label_ids)
            pad_id = self.tokenizer.pad_token_id
            labels = [list(seq) + [pad_id] * (max_len - len(seq)) for seq in label_ids]
            out["labels"] = ops.convert_to_tensor(labels, dtype="int32")
        return out

    def batch_decode(
        self, token_ids, skip_special_tokens=True, return_timestamps=False
    ):
        if not return_timestamps:
            return self.tokenizer.batch_decode(
                token_ids, skip_special_tokens=skip_special_tokens
            )
        # Word-level timestamps: one dict per clip, mirroring Whisper's shape.
        token_ids = ops.convert_to_numpy(token_ids).tolist()
        fs = self.frame_seconds
        return [
            {
                "text": self.tokenizer.decode(
                    row, skip_special_tokens=skip_special_tokens
                ),
                "chunks": self.tokenizer.ctc_offsets(row, fs),
            }
            for row in token_ids
        ]

    def decode(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
