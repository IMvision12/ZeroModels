from typing import List, Optional, Union

import keras
import numpy as np
from keras import layers, ops

from kerasformers.base import BaseModel, BaseSeq2SeqGeneration

from .moonshine_config import MoonshineConfig
from .moonshine_layers import (
    MoonshineAttention,
    MoonshineRotaryEmbedding,
    apply_rotary_pos_emb,
)

MOONSHINE_HUB_SIBLINGS = frozenset({"MoonshineModel", "MoonshineConditionalGenerate"})


def moonshine_encoder_mlp(x, hidden_dim, mlp_dim, activation, prefix):
    h = layers.Dense(mlp_dim, name=f"{prefix}_fc1")(x)
    h = layers.Lambda(activation, name=f"{prefix}_fc1_act")(h)
    h = layers.Dense(hidden_dim, name=f"{prefix}_fc2")(h)
    return h


def moonshine_decoder_mlp(x, hidden_dim, mlp_dim, activation, prefix):
    h = layers.Dense(mlp_dim * 2, name=f"{prefix}_fc1")(x)

    def _gated(t):
        value, gate = ops.split(t, 2, axis=-1)
        return activation(gate) * value

    h = layers.Lambda(_gated, name=f"{prefix}_fc1_gate")(h)
    h = layers.Dense(hidden_dim, name=f"{prefix}_fc2")(h)
    return h


def moonshine_encoder_block(
    x,
    cos,
    sin,
    hidden_dim,
    num_heads,
    num_kv_heads,
    mlp_dim,
    layer_idx,
    activation,
    layer_norm_eps,
):
    prefix = f"encoder_layers_{layer_idx}"

    ln_1 = layers.LayerNormalization(
        epsilon=layer_norm_eps, center=False, name=f"{prefix}_input_layernorm"
    )(x)
    attn_out = MoonshineAttention(
        proj_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        name_prefix=f"{prefix}_self_attn",
    )(ln_1, cos=cos, sin=sin)
    x = layers.Add()([x, attn_out])

    ln_2 = layers.LayerNormalization(
        epsilon=layer_norm_eps, center=False, name=f"{prefix}_post_attention_layernorm"
    )(x)
    h = moonshine_encoder_mlp(ln_2, hidden_dim, mlp_dim, activation, prefix)
    x = layers.Add()([x, h])
    return x


def moonshine_decoder_block(
    x,
    encoder_hidden_states,
    causal_mask,
    cos,
    sin,
    hidden_dim,
    num_heads,
    num_kv_heads,
    mlp_dim,
    layer_idx,
    activation,
    layer_norm_eps,
):
    prefix = f"decoder_layers_{layer_idx}"

    ln_1 = layers.LayerNormalization(
        epsilon=layer_norm_eps, center=False, name=f"{prefix}_input_layernorm"
    )(x)
    self_attn_out = MoonshineAttention(
        proj_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        name_prefix=f"{prefix}_self_attn",
    )(ln_1, attention_mask=causal_mask, cos=cos, sin=sin)
    x = layers.Add()([x, self_attn_out])

    ln_2 = layers.LayerNormalization(
        epsilon=layer_norm_eps, center=False, name=f"{prefix}_post_attention_layernorm"
    )(x)
    cross_attn_out = MoonshineAttention(
        proj_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        name_prefix=f"{prefix}_encoder_attn",
    )(ln_2, key_value_states=encoder_hidden_states)
    x = layers.Add()([x, cross_attn_out])

    ln_3 = layers.LayerNormalization(
        epsilon=layer_norm_eps, center=False, name=f"{prefix}_final_layernorm"
    )(x)
    h = moonshine_decoder_mlp(ln_3, hidden_dim, mlp_dim, activation, prefix)
    x = layers.Add()([x, h])
    return x


def moonshine_encoder(
    hidden_dim,
    encoder_num_layers,
    encoder_attention_heads,
    encoder_num_kv_heads,
    encoder_ffn_dim,
    rotary_dim,
    max_positions,
    rope_theta,
    activation=keras.activations.gelu,
    layer_norm_eps=1e-5,
    name="encoder",
):
    input_values = layers.Input(shape=(None,), name="input_values")
    x = layers.Reshape((-1, 1), name="encoder_expand_in")(input_values)

    x = layers.Conv1D(
        filters=hidden_dim,
        kernel_size=127,
        strides=64,
        padding="valid",
        use_bias=False,
        name="encoder_conv1",
    )(x)
    x = layers.Lambda(keras.activations.tanh, name="encoder_conv1_act")(x)
    x = layers.GroupNormalization(
        groups=1, axis=-1, epsilon=1e-5, name="encoder_groupnorm"
    )(x)
    x = layers.Conv1D(
        filters=2 * hidden_dim,
        kernel_size=7,
        strides=3,
        padding="valid",
        name="encoder_conv2",
    )(x)
    x = layers.Lambda(keras.activations.gelu, name="encoder_conv2_act")(x)
    x = layers.Conv1D(
        filters=hidden_dim,
        kernel_size=3,
        strides=2,
        padding="valid",
        name="encoder_conv3",
    )(x)
    x = layers.Lambda(keras.activations.gelu, name="encoder_conv3_act")(x)

    cos, sin = MoonshineRotaryEmbedding(
        rotary_dim=rotary_dim,
        max_positions=max_positions,
        base=rope_theta,
        name="encoder_rotary_emb",
    )(x)

    for i in range(encoder_num_layers):
        x = moonshine_encoder_block(
            x,
            cos,
            sin,
            hidden_dim=hidden_dim,
            num_heads=encoder_attention_heads,
            num_kv_heads=encoder_num_kv_heads,
            mlp_dim=encoder_ffn_dim,
            layer_idx=i,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
        )

    x = layers.LayerNormalization(
        epsilon=layer_norm_eps, center=False, name="encoder_layer_norm"
    )(x)
    return keras.Model(inputs=input_values, outputs=x, name=name)


def make_causal_mask_from_ids(decoder_input_ids):
    seq_len = ops.shape(decoder_input_ids)[1]
    i = ops.arange(seq_len)[:, None]
    j = ops.arange(seq_len)[None, :]
    mask = ops.cast(j > i, "float32") * -1e9
    return mask[None, None, :, :]


def moonshine_decoder(
    hidden_dim,
    vocab_size,
    decoder_num_layers,
    decoder_attention_heads,
    decoder_num_kv_heads,
    decoder_ffn_dim,
    rotary_dim,
    max_positions,
    rope_theta,
    activation=keras.activations.silu,
    layer_norm_eps=1e-5,
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

    cos, sin = MoonshineRotaryEmbedding(
        rotary_dim=rotary_dim,
        max_positions=max_positions,
        base=rope_theta,
        name="decoder_rotary_emb",
    )(x)

    causal_mask = layers.Lambda(
        make_causal_mask_from_ids,
        name="decoder_causal_mask",
        output_shape=lambda s: (1, 1, s[1], s[1]),
    )(decoder_input_ids)

    for i in range(decoder_num_layers):
        x = moonshine_decoder_block(
            x,
            encoder_hidden_states=encoder_hidden_states,
            causal_mask=causal_mask,
            cos=cos,
            sin=sin,
            hidden_dim=hidden_dim,
            num_heads=decoder_attention_heads,
            num_kv_heads=decoder_num_kv_heads,
            mlp_dim=decoder_ffn_dim,
            layer_idx=i,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
        )

    x = layers.LayerNormalization(
        epsilon=layer_norm_eps, center=False, name="decoder_layer_norm"
    )(x)

    embed_weight = tok_embed.embeddings
    logits = layers.Lambda(
        lambda t, w=embed_weight: ops.matmul(t, ops.transpose(w, (1, 0))),
        name="proj_out",
    )(x)

    return keras.Model(
        inputs={
            "decoder_input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
        },
        outputs=logits,
        name=name,
    )


@keras.saving.register_keras_serializable(package="kerasformers")
class MoonshineModel(BaseModel):
    """Moonshine encoder-decoder transformer for automatic speech recognition.

    Wires :func:`moonshine_encoder` and :func:`moonshine_decoder` into a single
    Functional graph so the full model can be called with one dict:

    >>> out = model({"input_values": audio, "decoder_input_ids": ids})
    >>> out["encoder_hidden_states"]   # (B, T, hidden_dim)
    >>> out["logits"]                  # (B, L, vocab_size)

    This is the teacher-forced training path. For autoregressive inference use
    :class:`MoonshineConditionalGenerate`, which subclasses this and adds a
    ``.generate(audio, processor, ...)`` method.

    Construction:

    >>> MoonshineModel.from_weights("moonshine_tiny")               # on-the-fly convert
    >>> MoonshineModel.from_weights("hf:UsefulSensors/moonshine-tiny")
    >>> MoonshineModel.from_weights("hf:user/moonshine-finetune")

    Unlike Whisper / Speech2Text, the Moonshine encoder consumes the **raw
    16 kHz waveform** directly (a 3-conv + GroupNorm stem replaces the log-mel
    front end), so ``input_values`` is a ``(B, audio_length)`` float tensor:
    feed :class:`MoonshineFeatureExtractor` output, which only zero-pads a
    batch to a common length.

    Args:
        hidden_dim: Hidden / embedding dimension.
        encoder_num_layers: Number of encoder transformer blocks.
        decoder_num_layers: Number of decoder transformer blocks.
        encoder_attention_heads: Encoder self-attn query head count.
        decoder_attention_heads: Decoder self-/cross-attn query head count.
        encoder_num_kv_heads: Encoder key/value head count (GQA). Defaults to
            ``encoder_attention_heads``.
        decoder_num_kv_heads: Decoder key/value head count (GQA). Defaults to
            ``decoder_attention_heads``.
        encoder_ffn_dim: Encoder MLP intermediate dim.
        decoder_ffn_dim: Decoder MLP intermediate dim (the gated ``fc1``
            projects to ``2 * decoder_ffn_dim``).
        vocab_size: Token vocabulary size.
        max_position_embeddings: Size of the rotary position tables.
        partial_rotary_factor: Fraction of each head dimension that is rotated.
        rope_theta: RoPE base frequency.
        encoder_activation: Encoder MLP activation. ``"gelu"`` for Moonshine.
        decoder_activation: Decoder gated-MLP activation. ``"silu"`` for
            Moonshine.
        layer_norm_eps: Epsilon for every (bias-free) LayerNorm.
        name: Model name. Defaults to ``"MoonshineModel"``.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = MoonshineConfig
    HUB_REPO_SIBLINGS = MOONSHINE_HUB_SIBLINGS
    HF_MODEL_TYPE = "moonshine"
    generate_args = {
        "bos_token_id": 1,
        "pad_token_id": 2,
        "decoder_start_token_id": 1,
        "eos_token_id": 2,
        "max_length": 194,
    }

    @classmethod
    def config_from_hf(cls, hf_config):
        rope = hf_config.get("rope_parameters") or {}
        partial = rope.get(
            "partial_rotary_factor", hf_config.get("partial_rotary_factor", 0.9)
        )
        theta = rope.get("rope_theta", hf_config.get("rope_theta", 10000.0))
        enc_heads = hf_config["encoder_num_attention_heads"]
        dec_heads = hf_config["decoder_num_attention_heads"]
        return {
            "hidden_dim": hf_config["hidden_size"],
            "encoder_num_layers": hf_config["encoder_num_hidden_layers"],
            "decoder_num_layers": hf_config["decoder_num_hidden_layers"],
            "encoder_attention_heads": enc_heads,
            "decoder_attention_heads": dec_heads,
            "encoder_num_kv_heads": hf_config.get("encoder_num_key_value_heads")
            or enc_heads,
            "decoder_num_kv_heads": hf_config.get("decoder_num_key_value_heads")
            or dec_heads,
            "encoder_ffn_dim": hf_config["intermediate_size"],
            "decoder_ffn_dim": hf_config["intermediate_size"],
            "vocab_size": hf_config["vocab_size"],
            "max_position_embeddings": hf_config.get("max_position_embeddings", 194),
            "partial_rotary_factor": partial,
            "rope_theta": theta,
            "encoder_activation": hf_config.get("encoder_hidden_act", "gelu"),
            "decoder_activation": hf_config.get("decoder_hidden_act", "silu"),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from kerasformers.models.moonshine.convert_moonshine_hf_to_keras import (
            transfer_moonshine_weights,
        )

        transfer_moonshine_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        hidden_dim=288,
        encoder_num_layers=6,
        decoder_num_layers=6,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        encoder_num_kv_heads=None,
        decoder_num_kv_heads=None,
        encoder_ffn_dim=1152,
        decoder_ffn_dim=1152,
        vocab_size=32768,
        max_position_embeddings=194,
        partial_rotary_factor=0.9,
        rope_theta=10000.0,
        encoder_activation="gelu",
        decoder_activation="silu",
        layer_norm_eps=1e-5,
        name="MoonshineModel",
        **kwargs,
    ):
        if encoder_num_kv_heads is None:
            encoder_num_kv_heads = encoder_attention_heads
        if decoder_num_kv_heads is None:
            decoder_num_kv_heads = decoder_attention_heads
        enc_activation_fn = keras.activations.get(encoder_activation)
        dec_activation_fn = keras.activations.get(decoder_activation)
        enc_rotary_dim = int(
            (hidden_dim // encoder_attention_heads) * partial_rotary_factor
        )
        dec_rotary_dim = int(
            (hidden_dim // decoder_attention_heads) * partial_rotary_factor
        )

        encoder = moonshine_encoder(
            hidden_dim=hidden_dim,
            encoder_num_layers=encoder_num_layers,
            encoder_attention_heads=encoder_attention_heads,
            encoder_num_kv_heads=encoder_num_kv_heads,
            encoder_ffn_dim=encoder_ffn_dim,
            rotary_dim=enc_rotary_dim,
            max_positions=max_position_embeddings,
            rope_theta=rope_theta,
            activation=enc_activation_fn,
            layer_norm_eps=layer_norm_eps,
            name=f"{name}_encoder",
        )
        decoder = moonshine_decoder(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            decoder_num_layers=decoder_num_layers,
            decoder_attention_heads=decoder_attention_heads,
            decoder_num_kv_heads=decoder_num_kv_heads,
            decoder_ffn_dim=decoder_ffn_dim,
            rotary_dim=dec_rotary_dim,
            max_positions=max_position_embeddings,
            rope_theta=rope_theta,
            activation=dec_activation_fn,
            layer_norm_eps=layer_norm_eps,
            name=f"{name}_decoder",
        )

        input_values = layers.Input(shape=(None,), name="input_values")
        decoder_input_ids = layers.Input(
            shape=(None,), dtype="int32", name="decoder_input_ids"
        )
        encoder_hidden_states = encoder(input_values)
        logits = decoder(
            {
                "decoder_input_ids": decoder_input_ids,
                "encoder_hidden_states": encoder_hidden_states,
            }
        )

        super().__init__(
            inputs={
                "input_values": input_values,
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
        self.encoder_num_kv_heads = encoder_num_kv_heads
        self.decoder_num_kv_heads = decoder_num_kv_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_theta = rope_theta
        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation
        self.layer_norm_eps = layer_norm_eps

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "encoder_num_layers": self.encoder_num_layers,
                "decoder_num_layers": self.decoder_num_layers,
                "encoder_attention_heads": self.encoder_attention_heads,
                "decoder_attention_heads": self.decoder_attention_heads,
                "encoder_num_kv_heads": self.encoder_num_kv_heads,
                "decoder_num_kv_heads": self.decoder_num_kv_heads,
                "encoder_ffn_dim": self.encoder_ffn_dim,
                "decoder_ffn_dim": self.decoder_ffn_dim,
                "vocab_size": self.vocab_size,
                "max_position_embeddings": self.max_position_embeddings,
                "partial_rotary_factor": self.partial_rotary_factor,
                "rope_theta": self.rope_theta,
                "encoder_activation": self.encoder_activation,
                "decoder_activation": self.decoder_activation,
                "layer_norm_eps": self.layer_norm_eps,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class MoonshineConditionalGenerate(MoonshineModel, BaseSeq2SeqGeneration):
    """Moonshine speech-to-text model (transcription).

    Composes the same encoder + decoder + tied LM head Functional graph as
    :class:`MoonshineModel` (so it loads the same weights and is a drop-in
    replacement for teacher-forced training and forward passes), and adds
    :meth:`generate`, an end-to-end audio -> text method that pulls in a
    :class:`~kerasformers.models.moonshine.MoonshineProcessor` for feature
    extraction (raw-waveform batching), greedy decoding, and detokenization.

    This mirrors the reference pattern (``MoonshineModel`` is the bare
    encoder/decoder, ``MoonshineForConditionalGeneration`` adds the LM head +
    ``.generate()``).

    .. code-block:: python

        model = MoonshineConditionalGenerate.from_weights("moonshine_tiny")
        processor = MoonshineProcessor.from_weights("moonshine_tiny")
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
        ga = self.generate_args or {}
        start_id = ga.get("decoder_start_token_id", processor.decoder_start_token_id)
        eos_id = ga.get("eos_token_id", processor.tokenizer.eos_token_id)
        if max_new_tokens is None:
            max_length = ga.get("max_length")
            max_new_tokens = (max_length - 1) if max_length else 200
        inputs = processor(audio=audio, sampling_rate=sampling_rate)

        features = ops.convert_to_tensor(inputs["input_values"])
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
        self._rope = d.get_layer("decoder_rotary_emb")
        self._dec_final_ln = d.get_layer("decoder_layer_norm")
        self._dec_act = keras.activations.silu
        attn = {
            layer.name_prefix: layer
            for layer in d.layers
            if isinstance(layer, MoonshineAttention)
        }
        self._dec_blocks = [
            {
                "self_ln": d.get_layer(f"decoder_layers_{i}_input_layernorm"),
                "self_attn": attn[f"decoder_layers_{i}_self_attn"],
                "cross_ln": d.get_layer(f"decoder_layers_{i}_post_attention_layernorm"),
                "cross_attn": attn[f"decoder_layers_{i}_encoder_attn"],
                "final_ln": d.get_layer(f"decoder_layers_{i}_final_layernorm"),
                "fc1": d.get_layer(f"decoder_layers_{i}_fc1"),
                "fc2": d.get_layer(f"decoder_layers_{i}_fc2"),
            }
            for i in range(self.decoder_num_layers)
        ]

    @property
    def decode_num_heads(self):
        return self.decoder_num_kv_heads

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
        n = ids.shape[1]
        positions = start_pos + ops.arange(n)
        cos = ops.take(self._rope.cos_table, positions, axis=0)[None, None]
        sin = ops.take(self._rope.sin_table, positions, axis=0)[None, None]

        def rotary(t, _update_index):
            return apply_rotary_pos_emb(t, cos, sin)

        new_cache = []
        for i, blk in enumerate(self._dec_blocks):
            self_k, self_v, cross_k, cross_v = cache[i]

            residual = x
            h = blk["self_ln"](x)
            h, self_k, self_v = self.cached_self_attention(
                blk["self_attn"], h, self_k, self_v, start_pos, rotary=rotary
            )
            x = residual + h

            residual = x
            h = blk["cross_ln"](x)
            h = self.cached_cross_attention(blk["cross_attn"], h, cross_k, cross_v)
            x = residual + h

            residual = x
            h = blk["final_ln"](x)
            value, gate = ops.split(blk["fc1"](h), 2, axis=-1)
            h = blk["fc2"](self._dec_act(gate) * value)
            x = residual + h

            new_cache.append((self_k, self_v, cross_k, cross_v))

        x = self._dec_final_ln(x)
        logits = ops.matmul(x, ops.transpose(self._dec_embed.embeddings, (1, 0)))
        return logits, tuple(new_cache)
