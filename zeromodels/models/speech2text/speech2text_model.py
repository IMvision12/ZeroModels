from typing import List, Optional, Union

import keras
import numpy as np
from keras import layers, ops

from zeromodels.base import BaseModel, BaseSeq2SeqGeneration

from .speech2text_config import Speech2TextConfig
from .speech2text_layers import (
    Speech2TextAttention,
    Speech2TextSinusoidalPositionEmbedding,
)

SPEECH2TEXT_HUB_SIBLINGS = frozenset(
    {"Speech2TextModel", "Speech2TextConditionalGenerate"}
)

_ACTIVATION_ALIASES = {
    "relu": keras.activations.relu,
    "gelu": lambda x: keras.activations.gelu(x, approximate=False),
    "gelu_new": lambda x: keras.activations.gelu(x, approximate=True),
    "silu": keras.activations.silu,
    "swish": keras.activations.silu,
}


def relu(x):
    return keras.activations.relu(x)


def resolve_activation(name):
    if callable(name):
        return name
    if name == "relu":
        return relu
    if name in _ACTIVATION_ALIASES:
        return _ACTIVATION_ALIASES[name]
    return keras.activations.get(name)


def glu(x):
    a, b = ops.split(x, 2, axis=-1)
    return a * ops.sigmoid(b)


def speech2text_conv_subsampler(
    features,
    hidden_dim,
    conv_channels,
    conv_kernel_sizes,
    num_conv_layers,
    name="encoder",
):
    x = features
    for i in range(num_conv_layers):
        out_ch = conv_channels if i < num_conv_layers - 1 else hidden_dim * 2
        k = conv_kernel_sizes[i]
        x = layers.ZeroPadding1D(padding=k // 2, name=f"{name}_conv_layers_{i}_pad")(x)
        x = layers.Conv1D(
            filters=out_ch,
            kernel_size=k,
            strides=2,
            padding="valid",
            name=f"{name}_conv_layers_{i}",
        )(x)
        x = layers.Lambda(glu, name=f"{name}_conv_layers_{i}_glu")(x)
    return x


def speech2text_encoder_block(
    x, hidden_dim, num_heads, mlp_dim, layer_idx, activation, layer_norm_eps
):
    prefix = f"encoder_layers_{layer_idx}"

    residual = x
    h = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{prefix}_self_attn_layer_norm"
    )(x)
    h = Speech2TextAttention(
        proj_dim=hidden_dim, num_heads=num_heads, name_prefix=f"{prefix}_self_attn"
    )(h)
    x = layers.Add()([residual, h])

    residual = x
    h = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{prefix}_final_layer_norm"
    )(x)
    h = layers.Dense(mlp_dim, name=f"{prefix}_fc1")(h)
    h = layers.Lambda(activation, name=f"{prefix}_fc1_act")(h)
    h = layers.Dense(hidden_dim, name=f"{prefix}_fc2")(h)
    x = layers.Add()([residual, h])
    return x


def speech2text_decoder_block(
    x,
    encoder_hidden_states,
    causal_mask,
    hidden_dim,
    num_heads,
    mlp_dim,
    layer_idx,
    activation,
    layer_norm_eps,
):
    prefix = f"decoder_layers_{layer_idx}"

    residual = x
    h = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{prefix}_self_attn_layer_norm"
    )(x)
    h = Speech2TextAttention(
        proj_dim=hidden_dim, num_heads=num_heads, name_prefix=f"{prefix}_self_attn"
    )(h, attention_mask=causal_mask)
    x = layers.Add()([residual, h])

    residual = x
    h = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{prefix}_encoder_attn_layer_norm"
    )(x)
    h = Speech2TextAttention(
        proj_dim=hidden_dim, num_heads=num_heads, name_prefix=f"{prefix}_encoder_attn"
    )(h, key_value_states=encoder_hidden_states)
    x = layers.Add()([residual, h])

    residual = x
    h = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{prefix}_final_layer_norm"
    )(x)
    h = layers.Dense(mlp_dim, name=f"{prefix}_fc1")(h)
    h = layers.Lambda(activation, name=f"{prefix}_fc1_act")(h)
    h = layers.Dense(hidden_dim, name=f"{prefix}_fc2")(h)
    x = layers.Add()([residual, h])
    return x


def make_causal_mask_from_ids(decoder_input_ids):
    seq_len = ops.shape(decoder_input_ids)[1]
    i = ops.arange(seq_len)[:, None]
    j = ops.arange(seq_len)[None, :]
    mask = ops.cast(j > i, "float32") * -1e9
    return mask[None, None, :, :]


def speech2text_encoder(
    hidden_dim,
    num_mel_bins,
    conv_channels,
    conv_kernel_sizes,
    num_conv_layers,
    max_source_positions,
    encoder_num_layers,
    encoder_attention_heads,
    encoder_ffn_dim,
    embed_scale,
    activation,
    layer_norm_eps,
    name="encoder",
):
    features = layers.Input(shape=(None, num_mel_bins), name="input_features")
    x = speech2text_conv_subsampler(
        features,
        hidden_dim,
        conv_channels,
        conv_kernel_sizes,
        num_conv_layers,
        name="encoder",
    )
    if embed_scale != 1.0:
        x = layers.Lambda(lambda t, s=embed_scale: t * s, name="encoder_embed_scale")(x)
    x = Speech2TextSinusoidalPositionEmbedding(
        num_positions=max_source_positions,
        hidden_dim=hidden_dim,
        name="encoder_embed_positions",
    )(x)
    for i in range(encoder_num_layers):
        x = speech2text_encoder_block(
            x,
            hidden_dim=hidden_dim,
            num_heads=encoder_attention_heads,
            mlp_dim=encoder_ffn_dim,
            layer_idx=i,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
        )
    x = layers.LayerNormalization(epsilon=layer_norm_eps, name="encoder_layer_norm")(x)
    return keras.Model(inputs=features, outputs=x, name=name)


def speech2text_decoder(
    hidden_dim,
    max_target_positions,
    vocab_size,
    decoder_num_layers,
    decoder_attention_heads,
    decoder_ffn_dim,
    embed_scale,
    activation,
    layer_norm_eps,
    pad_token_id,
    name="decoder",
):
    decoder_input_ids = layers.Input(
        shape=(None,), dtype="int32", name="decoder_input_ids"
    )
    encoder_hidden_states = layers.Input(
        shape=(None, hidden_dim), name="encoder_hidden_states"
    )

    tok_embed = layers.Embedding(
        input_dim=vocab_size, output_dim=hidden_dim, name="decoder_embed_tokens"
    )
    x = tok_embed(decoder_input_ids)
    if embed_scale != 1.0:
        x = layers.Lambda(lambda t, s=embed_scale: t * s, name="decoder_embed_scale")(x)
    x = Speech2TextSinusoidalPositionEmbedding(
        num_positions=max_target_positions,
        hidden_dim=hidden_dim,
        padding_idx=pad_token_id,
        name="decoder_embed_positions",
    )(x)

    causal_mask = layers.Lambda(
        make_causal_mask_from_ids,
        name="decoder_causal_mask",
        output_shape=lambda s: (1, 1, s[1], s[1]),
    )(decoder_input_ids)

    for i in range(decoder_num_layers):
        x = speech2text_decoder_block(
            x,
            encoder_hidden_states=encoder_hidden_states,
            causal_mask=causal_mask,
            hidden_dim=hidden_dim,
            num_heads=decoder_attention_heads,
            mlp_dim=decoder_ffn_dim,
            layer_idx=i,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
        )

    x = layers.LayerNormalization(epsilon=layer_norm_eps, name="decoder_layer_norm")(x)
    logits = layers.Dense(vocab_size, use_bias=False, name="lm_head")(x)

    return keras.Model(
        inputs={
            "decoder_input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
        },
        outputs=logits,
        name=name,
    )


@keras.saving.register_keras_serializable(package="zeromodels")
class Speech2TextModel(BaseModel):
    """Speech2Text (S2T) conv-Transformer encoder-decoder for ASR / ST.

    Wires :func:`speech2text_encoder` and :func:`speech2text_decoder`
    into one Functional graph callable with a single dict:

    >>> out = model({"input_features": fbank, "decoder_input_ids": ids})
    >>> out["encoder_hidden_states"]   # (B, T // 4, hidden_dim)
    >>> out["logits"]                  # (B, L, vocab_size)

    This is the teacher-forced path. For autoregressive transcription use
    :class:`Speech2TextConditionalGenerate`, which adds ``.generate(audio, processor)``.

    Construction:

    >>> Speech2TextModel.from_weights("s2t-small-librispeech-asr")
    >>> Speech2TextModel.from_weights("hf:facebook/s2t-small-librispeech-asr")

    .. note::
        Like Whisper, the audio input shape is dictated by the feature
        pipeline: 80-channel fbank features ``(num_mel_bins, T)`` fed as
        ``(B, T, num_mel_bins)``. There is no ``input_shape`` kwarg; feed
        :class:`Speech2TextFeatureExtractor` output directly.

    Args:
        hidden_dim: Hidden / embedding dimension (``d_model``).
        encoder_num_layers: Number of encoder transformer blocks.
        decoder_num_layers: Number of decoder transformer blocks.
        encoder_attention_heads: Encoder self-attn head count.
        decoder_attention_heads: Decoder self-/cross-attn head count.
        encoder_ffn_dim: Encoder MLP hidden dim.
        decoder_ffn_dim: Decoder MLP hidden dim.
        vocab_size: Token vocabulary size.
        num_mel_bins: Input fbank feature dimension (``input_feat_per_channel``).
        max_source_positions: Max encoder position (sinusoid table size).
        max_target_positions: Max decoder position.
        conv_channels: Intermediate channel width in the conv subsampler.
        conv_kernel_sizes: Per-layer conv kernel sizes.
        num_conv_layers: Number of conv+GLU subsampling layers.
        scale_embedding: Multiply embeddings by ``sqrt(hidden_dim)``.
        activation_function: FFN activation. ``"relu"`` for S2T.
        layer_norm_eps: Epsilon for every LayerNorm.
        pad_token_id: Padding id (offsets the sinusoidal positions).
        name: Model name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = Speech2TextConfig
    HUB_REPO_SIBLINGS = SPEECH2TEXT_HUB_SIBLINGS
    HF_MODEL_TYPE = "speech_to_text"
    # Default generation settings, written to kf_config.json under generate_args and
    # re-attached on repo-id load; Speech2TextConditionalGenerate.generate() reads them.
    generate_args = {"max_new_tokens": 200}

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "hidden_dim": hf_config["d_model"],
            "encoder_num_layers": hf_config["encoder_layers"],
            "decoder_num_layers": hf_config["decoder_layers"],
            "encoder_attention_heads": hf_config["encoder_attention_heads"],
            "decoder_attention_heads": hf_config["decoder_attention_heads"],
            "encoder_ffn_dim": hf_config["encoder_ffn_dim"],
            "decoder_ffn_dim": hf_config["decoder_ffn_dim"],
            "vocab_size": hf_config["vocab_size"],
            "num_mel_bins": hf_config.get("input_feat_per_channel", 80),
            "max_source_positions": hf_config.get("max_source_positions", 6000),
            "max_target_positions": hf_config.get("max_target_positions", 1024),
            "conv_channels": hf_config.get("conv_channels", 1024),
            "conv_kernel_sizes": tuple(hf_config.get("conv_kernel_sizes", [5, 5])),
            "num_conv_layers": hf_config.get("num_conv_layers", 2),
            "scale_embedding": hf_config.get("scale_embedding", True),
            "activation_function": hf_config.get("activation_function", "relu"),
            "pad_token_id": hf_config.get("pad_token_id", 1),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_speech2text_hf_to_keras import transfer_speech2text_weights

        transfer_speech2text_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        hidden_dim=256,
        encoder_num_layers=12,
        decoder_num_layers=6,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        vocab_size=10000,
        num_mel_bins=80,
        max_source_positions=6000,
        max_target_positions=1024,
        conv_channels=1024,
        conv_kernel_sizes=(5, 5),
        num_conv_layers=2,
        scale_embedding=True,
        activation_function="relu",
        layer_norm_eps=1e-5,
        pad_token_id=1,
        name="Speech2TextModel",
        **kwargs,
    ):
        conv_kernel_sizes = tuple(conv_kernel_sizes)
        activation_fn = resolve_activation(activation_function)
        embed_scale = float(hidden_dim) ** 0.5 if scale_embedding else 1.0

        encoder = speech2text_encoder(
            hidden_dim=hidden_dim,
            num_mel_bins=num_mel_bins,
            conv_channels=conv_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            num_conv_layers=num_conv_layers,
            max_source_positions=max_source_positions,
            encoder_num_layers=encoder_num_layers,
            encoder_attention_heads=encoder_attention_heads,
            encoder_ffn_dim=encoder_ffn_dim,
            embed_scale=embed_scale,
            activation=activation_fn,
            layer_norm_eps=layer_norm_eps,
            name=f"{name}_encoder",
        )
        decoder = speech2text_decoder(
            hidden_dim=hidden_dim,
            max_target_positions=max_target_positions,
            vocab_size=vocab_size,
            decoder_num_layers=decoder_num_layers,
            decoder_attention_heads=decoder_attention_heads,
            decoder_ffn_dim=decoder_ffn_dim,
            embed_scale=embed_scale,
            activation=activation_fn,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            name=f"{name}_decoder",
        )

        input_features = layers.Input(shape=(None, num_mel_bins), name="input_features")
        decoder_input_ids = layers.Input(
            shape=(None,), dtype="int32", name="decoder_input_ids"
        )
        encoder_hidden_states = encoder(input_features)
        logits = decoder(
            {
                "decoder_input_ids": decoder_input_ids,
                "encoder_hidden_states": encoder_hidden_states,
            }
        )

        super().__init__(
            inputs={
                "input_features": input_features,
                "decoder_input_ids": decoder_input_ids,
            },
            outputs={
                "encoder_hidden_states": encoder_hidden_states,
                "logits": logits,
            },
            name=name,
            **kwargs,
        )

        self.encoder = encoder
        self.decoder = decoder
        self.hidden_dim = hidden_dim
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.vocab_size = vocab_size
        self.num_mel_bins = num_mel_bins
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.conv_channels = conv_channels
        self.conv_kernel_sizes = conv_kernel_sizes
        self.num_conv_layers = num_conv_layers
        self.scale_embedding = scale_embedding
        self.activation_function = activation_function
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "encoder_num_layers": self.encoder_num_layers,
                "decoder_num_layers": self.decoder_num_layers,
                "encoder_attention_heads": self.encoder_attention_heads,
                "decoder_attention_heads": self.decoder_attention_heads,
                "encoder_ffn_dim": self.encoder_ffn_dim,
                "decoder_ffn_dim": self.decoder_ffn_dim,
                "vocab_size": self.vocab_size,
                "num_mel_bins": self.num_mel_bins,
                "max_source_positions": self.max_source_positions,
                "max_target_positions": self.max_target_positions,
                "conv_channels": self.conv_channels,
                "conv_kernel_sizes": self.conv_kernel_sizes,
                "num_conv_layers": self.num_conv_layers,
                "scale_embedding": self.scale_embedding,
                "activation_function": self.activation_function,
                "layer_norm_eps": self.layer_norm_eps,
                "pad_token_id": self.pad_token_id,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class Speech2TextConditionalGenerate(Speech2TextModel, BaseSeq2SeqGeneration):
    """Speech2Text speech-to-text model (transcription / speech translation).

    Composes the same encoder + decoder + ``lm_head`` Functional graph as
    :class:`Speech2TextModel` (loads the same weights), and adds
    :meth:`generate` - an end-to-end audio -> text method that runs the
    feature extractor, encoder, autoregressive greedy decoding, and
    detokenization via a :class:`Speech2TextProcessor`.

    .. code-block:: python

        model = Speech2TextConditionalGenerate.from_weights("s2t-small-librispeech-asr")
        processor = Speech2TextProcessor.from_weights("s2t-small-librispeech-asr")
        text = model.generate(audio, processor)
    """

    def generate(
        self,
        audio,
        processor,
        max_new_tokens: Optional[int] = None,
        sampling_rate: int = 16000,
        return_ids: bool = False,
    ) -> Union[List[str], List[List[int]]]:
        if max_new_tokens is None:
            max_new_tokens = (self.generate_args or {}).get("max_new_tokens", 200)
        inputs = processor(audio=audio, sampling_rate=sampling_rate)
        start_id = processor.decoder_start_token_id
        eos_id = processor.tokenizer.eos_token_id

        features = ops.convert_to_tensor(inputs["input_features"])
        batch = int(features.shape[0])
        decoder_start_ids = ops.full((batch, 1), start_id, dtype="int32")
        generated = super().generate(
            features,
            decoder_start_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
        )
        start_col = np.full((batch, 1), start_id, dtype=generated.dtype)
        ids = [list(row) for row in np.concatenate([start_col, generated], axis=1)]
        if return_ids:
            return ids
        return processor.batch_decode(ids, skip_special_tokens=True)

    def encode(self, encoder_inputs):
        return self.encoder(encoder_inputs)

    def _ensure_decode_layers(self):
        if getattr(self, "_dec_blocks", None) is not None:
            return
        d = self.decoder
        self._dec_embed = d.get_layer("decoder_embed_tokens")
        self._dec_pos = d.get_layer("decoder_embed_positions")
        self._dec_final_ln = d.get_layer("decoder_layer_norm")
        self._dec_head = d.get_layer("lm_head")
        self._dec_embed_scale = (
            float(self.hidden_dim) ** 0.5 if self.scale_embedding else 1.0
        )
        self._dec_act = resolve_activation(self.activation_function)
        attn = {
            layer.name_prefix: layer
            for layer in d.layers
            if isinstance(layer, Speech2TextAttention)
        }
        self._dec_blocks = [
            {
                "self_ln": d.get_layer(f"decoder_layers_{i}_self_attn_layer_norm"),
                "self_attn": attn[f"decoder_layers_{i}_self_attn"],
                "cross_ln": d.get_layer(f"decoder_layers_{i}_encoder_attn_layer_norm"),
                "cross_attn": attn[f"decoder_layers_{i}_encoder_attn"],
                "final_ln": d.get_layer(f"decoder_layers_{i}_final_layer_norm"),
                "fc1": d.get_layer(f"decoder_layers_{i}_fc1"),
                "fc2": d.get_layer(f"decoder_layers_{i}_fc2"),
            }
            for i in range(self.decoder_num_layers)
        ]

    @property
    def decode_num_heads(self):
        return self.decoder_attention_heads

    @property
    def decode_head_dim(self):
        return self.hidden_dim // self.decoder_attention_heads

    def decode_cross_kv(self, encoder_hidden_states):
        self._ensure_decode_layers()
        return [
            blk["cross_attn"].project(encoder_hidden_states) for blk in self._dec_blocks
        ]

    def decode_forward(self, ids, cache, start_pos):
        self._ensure_decode_layers()
        x = self._dec_embed(ids)
        if self._dec_embed_scale != 1.0:
            x = x * self._dec_embed_scale
        n = ids.shape[1]
        positions = self._dec_pos.offset + start_pos + ops.arange(n)
        x = x + ops.take(self._dec_pos.pos_embed, positions, axis=0)[None]

        new_cache = []
        for i, blk in enumerate(self._dec_blocks):
            self_k, self_v, cross_k, cross_v = cache[i]

            residual = x
            h = blk["self_ln"](x)
            h, self_k, self_v = self.cached_self_attention(
                blk["self_attn"], h, self_k, self_v, start_pos
            )
            x = residual + h

            residual = x
            h = blk["cross_ln"](x)
            h = self.cached_cross_attention(blk["cross_attn"], h, cross_k, cross_v)
            x = residual + h

            residual = x
            h = blk["final_ln"](x)
            x = residual + blk["fc2"](self._dec_act(blk["fc1"](h)))

            new_cache.append((self_k, self_v, cross_k, cross_v))

        x = self._dec_final_ln(x)
        return self._dec_head(x), tuple(new_cache)
