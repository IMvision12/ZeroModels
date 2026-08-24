import numpy as np
from keras import ops


def qwen_vl_input(
    batch_size=2, grid=(1, 4, 4), patch_dim=1176, image_token_id=4, merge=2
):
    """Multimodal input for the Qwen-VL models: pre-patchified image patches +
    input_ids with the right number of image-placeholder tokens per row."""
    t, gh, gw = grid
    n_patches = t * gh * gw
    n_merged = t * (gh // merge) * (gw // merge)
    seq = n_merged + 2
    ids = np.zeros((batch_size, seq), dtype="int32")
    ids[:, 0] = 10
    ids[:, 1 : 1 + n_merged] = image_token_id
    ids[:, -1] = 11
    return {
        "input_ids": ops.convert_to_tensor(ids),
        "pixel_values": ops.ones((batch_size * n_patches, patch_dim)),
        "image_grid_thw": ops.convert_to_tensor(
            np.tile(np.array(grid, dtype="int32"), (batch_size, 1))
        ),
    }


def qwen_text_input(batch_size=2, seq_len=6, vocab_size=128):
    """Token-id input for the pure-text Qwen LLMs (Qwen2 / Qwen3 / Qwen3.5)."""
    ids = np.tile(np.arange(seq_len, dtype="int32") % vocab_size, (batch_size, 1))
    return {"input_ids": ops.convert_to_tensor(ids)}


def causal_lm_input(batch_size=2, seq_len=6, vocab_size=128):
    """Token-id + attention-mask input for functional causal LMs (e.g. GPT-2)."""
    ids = np.tile(np.arange(seq_len, dtype="int32") % vocab_size, (batch_size, 1))
    return {
        "input_ids": ops.convert_to_tensor(ids),
        "attention_mask": ops.ones((batch_size, seq_len), dtype="int32"),
    }


def bert_input(batch_size=2, seq_len=16):
    """Token-id / mask / segment input for the BERT encoder models."""
    return {
        "input_ids": ops.ones((batch_size, seq_len), dtype="int32"),
        "attention_mask": ops.ones((batch_size, seq_len), dtype="int32"),
        "token_type_ids": ops.zeros((batch_size, seq_len), dtype="int32"),
    }


def bert_multiple_choice_input(batch_size=2, num_choices=3, seq_len=16):
    """Per-choice token input for BertMultipleChoice: (B, num_choices, seq)."""
    return {
        "input_ids": ops.ones((batch_size, num_choices, seq_len), dtype="int32"),
        "attention_mask": ops.ones((batch_size, num_choices, seq_len), dtype="int32"),
        "token_type_ids": ops.zeros((batch_size, num_choices, seq_len), dtype="int32"),
    }


def t5_input(batch_size=2, seq_len=16, decoder_seq_len=8):
    """Encoder + decoder token input for the T5 encoder-decoder models."""
    return {
        "input_ids": ops.ones((batch_size, seq_len), dtype="int32"),
        "attention_mask": ops.ones((batch_size, seq_len), dtype="int32"),
        "decoder_input_ids": ops.ones((batch_size, decoder_seq_len), dtype="int32"),
    }


def t5_encoder_input(batch_size=2, seq_len=16):
    """Token / mask input for the T5 encoder-only + task-head models (they build the
    decoder input from the encoder ids internally)."""
    return {
        "input_ids": ops.ones((batch_size, seq_len), dtype="int32"),
        "attention_mask": ops.ones((batch_size, seq_len), dtype="int32"),
    }


def modernbert_input(batch_size=2, seq_len=16):
    """Token-id / mask input for the ModernBERT encoders (no token-type ids)."""
    return {
        "input_ids": ops.ones((batch_size, seq_len), dtype="int32"),
        "attention_mask": ops.ones((batch_size, seq_len), dtype="int32"),
    }


def modernbert_multiple_choice_input(batch_size=2, num_choices=3, seq_len=16):
    """Per-choice token input for ModernBertMultipleChoice: (B, num_choices, seq)."""
    return {
        "input_ids": ops.ones((batch_size, num_choices, seq_len), dtype="int32"),
        "attention_mask": ops.ones((batch_size, num_choices, seq_len), dtype="int32"),
    }


def roberta_input(batch_size=2, seq_len=16):
    """Token / mask / segment input for the RoBERTa and XLM-R encoders. Token
    ids avoid the pad id (1) so the padding-offset position ids are exercised."""
    ids = np.full((batch_size, seq_len), 5, dtype="int32")
    ids[:, 0] = 0
    return {
        "input_ids": ops.convert_to_tensor(ids),
        "attention_mask": ops.ones((batch_size, seq_len), dtype="int32"),
        "token_type_ids": ops.zeros((batch_size, seq_len), dtype="int32"),
    }


def roberta_multiple_choice_input(batch_size=2, num_choices=3, seq_len=16):
    """Per-choice token input for RoBERTa/XLM-R multiple choice: (B, C, seq)."""
    ids = np.full((batch_size, num_choices, seq_len), 5, dtype="int32")
    ids[..., 0] = 0
    return {
        "input_ids": ops.convert_to_tensor(ids),
        "attention_mask": ops.ones((batch_size, num_choices, seq_len), dtype="int32"),
        "token_type_ids": ops.zeros((batch_size, num_choices, seq_len), dtype="int32"),
    }


def backbone_input(batch_size=2, spatial=32, channels=3):
    return ops.ones((batch_size, spatial, spatial, channels))


def detection_input(batch_size=2, spatial=32, channels=3):
    return ops.ones((batch_size, spatial, spatial, channels))


def segmentation_input(batch_size=2, spatial=32, channels=3):
    return ops.ones((batch_size, spatial, spatial, channels))


def clip_input(batch_size=2, image_size=64, max_seq_len=77):
    return {
        "images": ops.ones((batch_size, image_size, image_size, 3)),
        "token_ids": ops.ones((batch_size, max_seq_len), dtype="int32"),
        "padding_mask": ops.ones((batch_size, max_seq_len), dtype="int32"),
    }


def siglip_input(batch_size=2, image_size=64, max_seq_len=64):
    return {
        "images": ops.ones((batch_size, image_size, image_size, 3)),
        "token_ids": ops.ones((batch_size, max_seq_len), dtype="int32"),
        "padding_mask": ops.ones((batch_size, max_seq_len), dtype="int32"),
    }


def tips_v2_text_input(batch_size=2, max_seq_len=16):
    return {
        "token_ids": ops.ones((batch_size, max_seq_len), dtype="int32"),
        "padding_mask": ops.ones((batch_size, max_seq_len), dtype="int32"),
    }


def sam_input(batch_size=2, image_size=64, num_prompts=1, num_points=1):
    return {
        "pixel_values": ops.ones((batch_size, image_size, image_size, 3)),
        "input_points": ops.ones(
            (batch_size, num_prompts, num_points, 2), dtype="float32"
        ),
        "input_labels": ops.ones((batch_size, num_prompts, num_points), dtype="int32"),
    }


def owlvit_input(batch_size=2, image_size=64, max_seq_len=16, num_queries=2):
    return {
        "pixel_values": ops.ones((batch_size, image_size, image_size, 3)),
        "input_ids": ops.ones((batch_size * num_queries, max_seq_len), dtype="int32"),
    }


def whisper_input(batch_size=2, num_mel_bins=80, mel_length=40, decoder_seq_len=5):
    return {
        "input_features": ops.ones((batch_size, num_mel_bins, mel_length)),
        "decoder_input_ids": ops.zeros((batch_size, decoder_seq_len), dtype="int32"),
    }


def whisper_audio_input(batch_size=2, num_mel_bins=80, mel_length=40):
    return ops.ones((batch_size, num_mel_bins, mel_length))


def speech2text_input(batch_size=2, num_mel_bins=80, feat_length=40, decoder_seq_len=5):
    # Speech2Text fbank features are (batch, time, num_mel_bins) - the transpose
    # of Whisper's (batch, num_mel_bins, time) layout.
    return {
        "input_features": ops.ones((batch_size, feat_length, num_mel_bins)),
        "decoder_input_ids": ops.zeros((batch_size, decoder_seq_len), dtype="int32"),
    }


def moonshine_input(batch_size=2, audio_length=2000, decoder_seq_len=5):
    # Moonshine consumes the raw 16 kHz waveform directly (conv stem, no mel).
    return {
        "input_values": ops.ones((batch_size, audio_length)),
        "decoder_input_ids": ops.zeros((batch_size, decoder_seq_len), dtype="int32"),
    }


def granite_speech_input(
    batch_size=2, frames=15, input_dim=160, window=15, downsample=5, audio_token_id=4
):
    """Audio + text input for Granite Speech: one audio clip per text row, with the
    matching number of <|audio|> placeholder tokens."""
    import math

    n_audio = math.ceil(frames / window) * (window // downsample)
    seq = n_audio + 2
    ids = np.zeros((batch_size, seq), dtype="int32")
    ids[:, 0] = 10
    ids[:, 1 : 1 + n_audio] = audio_token_id
    ids[:, -1] = 11
    return {
        "input_ids": ops.convert_to_tensor(ids),
        "attention_mask": ops.convert_to_tensor(
            np.ones((batch_size, seq), dtype="int32")
        ),
        "input_features": ops.convert_to_tensor(
            np.full((batch_size, frames, input_dim), 0.01, dtype="float32")
        ),
        "input_features_mask": ops.convert_to_tensor(
            np.ones((batch_size, n_audio), dtype=bool)
        ),
    }


def grounding_dino_input(batch_size=2, image_size=224):
    """Image + grounding-text input for Grounding DINO: pixel values plus a short
    tokenized prompt whose special tokens ([CLS] . [SEP]) drive the block mask."""
    ids = np.tile(
        np.array([[101, 500, 1012, 800, 102, 102]], dtype="int32"), (batch_size, 1)
    )
    return {
        "pixel_values": ops.convert_to_tensor(
            np.random.RandomState(0)
            .rand(batch_size, image_size, image_size, 3)
            .astype("float32")
        ),
        "input_ids": ops.convert_to_tensor(ids),
        "attention_mask": ops.ones((batch_size, ids.shape[1]), dtype="int32"),
    }


def oneformer_input(batch_size=2, image_size=224, task_seq_len=7):
    return {
        "pixel_values": ops.ones((batch_size, image_size, image_size, 3)),
        "task_inputs": ops.ones((batch_size, task_seq_len)),
    }
