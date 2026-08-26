import math

import keras
import numpy as np
from keras import layers, ops

from zeromodels.base import (
    BaseGeneration,
    BaseModel,
    CausalMask,
    MediaMerge,
    TiedHead,
)
from zeromodels.base.base_mixin import inference_scope

from .granite_speech_config import GraniteSpeechConfig
from .granite_speech_layers import (
    GraniteSpeechCTCEncoder,
    GraniteSpeechDecoderLayer,
    GraniteSpeechEncoderProjector,
    GraniteSpeechRMSNorm,
)

MASK_NEG = -1e9


@keras.saving.register_keras_serializable(package="zeromodels")
class GraniteSpeechAudioFeatures(layers.Layer):
    """Weightless wrapper running the (conformer) encoder + Q-Former projector in
    the functional graph.

    Holds ``encoder`` / ``projector`` by non-tracked references (they stay tracked
    on the model, so weight paths are unchanged) and defers their dynamic,
    grid/length-dependent work to eager runtime via ``compute_output_spec``; the
    projected audio embeddings are flattened to ``(num_audio_tokens, embed_dim)``
    for the shared :class:`~zeromodels.base.MediaMerge` scatter.
    """

    def __init__(self, encoder, projector, embed_dim, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "encoder", encoder)
        object.__setattr__(self, "projector", projector)
        self.embed_dim = embed_dim

    def call(self, input_features, input_features_mask=None):
        audio = self.projector(self.encoder(input_features))
        af = ops.reshape(audio, (-1, self.embed_dim))
        if input_features_mask is not None:
            # Keep only the valid projected audio embeddings (row-major), matching
            # the count of audio placeholder tokens. Runs eagerly (concrete shapes);
            # compute_output_spec keeps the dynamic count out of the graph trace.
            fm = ops.convert_to_numpy(
                ops.convert_to_tensor(input_features_mask)
            ).astype(bool)
            keep = ops.convert_to_tensor(np.nonzero(fm.reshape(-1))[0].astype("int32"))
            af = ops.take(af, keep, axis=0)
        return af

    def compute_output_spec(self, input_features, input_features_mask=None):
        return keras.KerasTensor((None, self.embed_dim), dtype=self.compute_dtype)


# GraniteSpeechModel (backbone) and GraniteSpeechConditionalGenerate (+ LM head + .generate)
# share the variant's weights repo, whose zm_config.json declares the canonical
# GraniteSpeechConditionalGenerate. GraniteSpeechPlus has its own repos + sibling set.
GRANITE_SPEECH_HUB_SIBLINGS = frozenset(
    {"GraniteSpeechModel", "GraniteSpeechConditionalGenerate"}
)


def rope_cos_sin(position_ids, head_dim, theta):
    inv_freq = 1.0 / ops.power(
        theta, ops.arange(0, head_dim, 2, dtype="float32") / head_dim
    )
    freqs = ops.cast(position_ids, "float32")[..., None] * inv_freq
    emb = ops.concatenate([freqs, freqs], axis=-1)
    return ops.cos(emb), ops.sin(emb)


@keras.saving.register_keras_serializable(package="zeromodels")
class GraniteSpeechTextModel(layers.Layer):
    """Granite causal decoder: ``embed -> num_layers x GraniteSpeechDecoderLayer
    -> RMSNorm``, with Granite's scalar multipliers (``embedding_multiplier`` on
    the embeddings, ``residual_multiplier`` inside each block, ``attention_multiplier``
    as the attention scaling). The token embedding lives here and is reused (tied)
    as the LM head by the generate model. ``call`` takes the multimodal-fused
    ``inputs_embeds`` (already embedding-multiplied) and threads an optional KV cache.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        mlp_dim,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim,
        norm_eps,
        attention_multiplier,
        residual_multiplier,
        tie_embeddings=True,
        lora_rank=0,
        lora_alpha=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.attention_multiplier = attention_multiplier
        self.residual_multiplier = residual_multiplier
        self.tie_embeddings = tie_embeddings
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        self.token_embedding = layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.decoder_layers = [
            GraniteSpeechDecoderLayer(
                embed_dim,
                mlp_dim,
                num_heads,
                num_kv_heads,
                head_dim,
                norm_eps,
                attention_multiplier,
                residual_multiplier,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                name=f"decoder_layer_{i}",
            )
            for i in range(num_layers)
        ]
        self.final_norm = GraniteSpeechRMSNorm(eps=norm_eps, name="final_norm")
        self.lm_head = (
            None
            if tie_embeddings
            else layers.Dense(vocab_size, use_bias=False, name="lm_head")
        )

    def build(self, input_shape):
        # lm_head is applied from the generate model's project(); build it here so
        # its weight is pathed under language_model (matching the HF
        # language_model.lm_head.weight key) rather than hoisted to the root model.
        if self.lm_head is not None and not self.lm_head.built:
            self.lm_head.build((None, self.embed_dim))
        self.built = True

    def call(
        self,
        inputs_embeds,
        cos,
        sin,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        apply_lora=False,
    ):
        hidden = inputs_embeds
        new_cache = [] if use_cache else None
        for i, layer in enumerate(self.decoder_layers):
            past = past_key_values[i] if past_key_values is not None else None
            out = layer(
                hidden,
                cos,
                sin,
                attention_mask=attention_mask,
                past_key_value=past,
                use_cache=use_cache,
                apply_lora=apply_lora,
            )
            if use_cache:
                hidden, kv = out
                new_cache.append(kv)
            else:
                hidden = out
        hidden = self.final_norm(hidden)
        return (hidden, new_cache) if use_cache else hidden

    def compute_output_spec(
        self,
        inputs_embeds,
        cos,
        sin,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        apply_lora=False,
    ):
        # In the functional graph this is called with use_cache=False (single tensor
        # out); isolate the decoder stack's eager int(shape) from the graph trace.
        return keras.KerasTensor(inputs_embeds.shape, dtype=inputs_embeds.dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "norm_eps": self.norm_eps,
                "attention_multiplier": self.attention_multiplier,
                "residual_multiplier": self.residual_multiplier,
                "tie_embeddings": self.tie_embeddings,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
            }
        )
        return config


def granite_speech_features(
    input_ids,
    attention_mask,
    input_features,
    input_features_mask,
    *,
    language_model,
    audio_features_layer,
    audio_merge,
    mask_layer,
    embedding_multiplier,
    head_dim,
    rope_theta,
    audio_token_id,
):
    # Embed text (audio slots zeroed), splice the projected audio embeddings onto
    # the audio-token slots, then run the Granite decoder with LoRA enabled (audio
    # is always present in a speech model).
    llm_ids = ops.where(ops.equal(input_ids, audio_token_id), 0, input_ids)
    inputs_embeds = language_model.token_embedding(llm_ids)
    audio = audio_features_layer(input_features, input_features_mask)
    inputs_embeds = audio_merge(inputs_embeds, input_ids, audio)
    inputs_embeds = inputs_embeds * embedding_multiplier
    position_ids = ops.cumsum(ops.ones_like(input_ids), axis=-1) - 1
    cos, sin = rope_cos_sin(position_ids, head_dim, rope_theta)
    mask = mask_layer(input_ids, attention_mask)
    return language_model(inputs_embeds, cos, sin, attention_mask=mask, apply_lora=True)


@keras.saving.register_keras_serializable(package="zeromodels")
class GraniteSpeechModel(BaseModel):
    """Granite Speech multimodal backbone: conformer audio encoder + BLIP-2
    Q-Former projector + Granite decoder, fused at audio-placeholder positions.

    Raw mel features run through a conformer CTC encoder and a windowed Q-Former
    projector to produce audio embeddings, which are scattered into the
    ``audio_token_id`` placeholder slots of the Granite text decoder (exactly like
    a vision-language model splices image embeddings). The decoder carries
    Granite's scalar multipliers and a query/value LoRA adapter that is enabled
    only when audio is present. The forward runs eagerly with ``keras.ops``. This
    base returns raw features (no LM head); use :class:`GraniteSpeechConditionalGenerate` for
    logits / text.

    GraniteSpeechPlus is the same architecture with ``cat_hidden_layers`` set: the
    CTC encoder concatenates the listed intermediate layer outputs with its final
    output before the projector (so ``projector_encoder_hidden_size`` =
    ``encoder_hidden_dim * (len(cat_hidden_layers) + 1)``).

    Output dict:

    .. code-block:: python

        out = model({
            "input_ids": ...,             # (B, L) int, audio placeholders
            "input_features": ...,        # (num_audios, frames, input_dim) mel
            "input_features_mask": ...,   # (num_audios, max_proj_len) bool, optional
        })
        out["last_hidden_state"]   # (B, L, embed_dim)

    The audio keys are optional (text-only is allowed).

    Construction:

    >>> GraniteSpeechModel.from_weights("granite_speech_3_3_2b")
    >>> GraniteSpeechModel.from_weights("hf:ibm-granite/granite-speech-3.3-2b")
    """

    HF_MODEL_TYPE = "granite_speech"
    default_load_dtype = "bfloat16"  # ibm-granite granite-speech checkpoints are bf16
    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = GraniteSpeechConfig
    HUB_REPO_SIBLINGS = GRANITE_SPEECH_HUB_SIBLINGS

    output_logits = False

    def __init__(
        self,
        vocab_size=49160,
        embed_dim=2048,
        mlp_dim=8192,
        num_layers=40,
        num_heads=32,
        num_kv_heads=8,
        norm_eps=1e-5,
        rope_theta=10000000.0,
        embedding_multiplier=12.0,
        residual_multiplier=0.22,
        attention_multiplier=0.015625,
        logits_scaling=8.0,
        tie_embeddings=True,
        eos_token_id=0,
        audio_token_id=49159,
        downsample_rate=5,
        window_size=15,
        has_lora_adapter=True,
        lora_rank=64,
        lora_alpha=32,
        encoder_input_dim=160,
        encoder_num_layers=16,
        encoder_hidden_dim=1024,
        encoder_feedforward_mult=4,
        encoder_num_heads=8,
        encoder_dim_head=128,
        encoder_output_dim=256,
        encoder_context_size=200,
        encoder_max_pos_emb=512,
        encoder_conv_kernel_size=15,
        encoder_conv_expansion_factor=2,
        projector_dim=1024,
        projector_num_layers=2,
        projector_num_heads=16,
        projector_intermediate_size=4096,
        projector_cross_attention_frequency=1,
        projector_layer_norm_eps=1e-12,
        cat_hidden_layers=None,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        head_dim = embed_dim // num_heads
        lora_rank_eff = lora_rank if has_lora_adapter else 0
        cat_hidden_layers = list(cat_hidden_layers) if cat_hidden_layers else None
        num_concat = (len(cat_hidden_layers) + 1) if cat_hidden_layers else 1
        projector_encoder_hidden_size = encoder_hidden_dim * num_concat

        encoder = GraniteSpeechCTCEncoder(
            input_dim=encoder_input_dim,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_num_layers,
            feedforward_mult=encoder_feedforward_mult,
            num_heads=encoder_num_heads,
            dim_head=encoder_dim_head,
            output_dim=encoder_output_dim,
            context_size=encoder_context_size,
            max_pos_emb=encoder_max_pos_emb,
            conv_expansion_factor=encoder_conv_expansion_factor,
            conv_kernel_size=encoder_conv_kernel_size,
            cat_hidden_layers=cat_hidden_layers,
            name="encoder",
        )
        projector = GraniteSpeechEncoderProjector(
            hidden_size=projector_dim,
            text_hidden_size=embed_dim,
            encoder_hidden_size=projector_encoder_hidden_size,
            num_layers=projector_num_layers,
            num_heads=projector_num_heads,
            intermediate_size=projector_intermediate_size,
            cross_attention_frequency=projector_cross_attention_frequency,
            layer_norm_eps=projector_layer_norm_eps,
            window_size=window_size,
            downsample_rate=downsample_rate,
            name="projector",
        )
        language_model = GraniteSpeechTextModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            norm_eps=norm_eps,
            attention_multiplier=attention_multiplier,
            residual_multiplier=residual_multiplier,
            tie_embeddings=tie_embeddings,
            lora_rank=lora_rank_eff,
            lora_alpha=lora_alpha,
            name="language_model",
        )
        audio_features_layer = GraniteSpeechAudioFeatures(
            encoder, projector, embed_dim, name="audio_features"
        )
        audio_merge = MediaMerge(audio_token_id, embed_dim, name="audio_merge")
        mask_layer = CausalMask(name="causal_mask")

        input_ids_in = layers.Input(shape=(None,), dtype="int32", name="input_ids")
        attn_in = layers.Input(shape=(None,), dtype="int32", name="attention_mask")
        feat_in = layers.Input(
            shape=(None, encoder_input_dim), dtype="float32", name="input_features"
        )
        feat_mask_in = layers.Input(
            shape=(None,), dtype="bool", name="input_features_mask"
        )
        inputs = {
            "input_ids": input_ids_in,
            "attention_mask": attn_in,
            "input_features": feat_in,
            "input_features_mask": feat_mask_in,
        }
        hidden = granite_speech_features(
            input_ids_in,
            attn_in,
            feat_in,
            feat_mask_in,
            language_model=language_model,
            audio_features_layer=audio_features_layer,
            audio_merge=audio_merge,
            mask_layer=mask_layer,
            embedding_multiplier=embedding_multiplier,
            head_dim=head_dim,
            rope_theta=rope_theta,
            audio_token_id=audio_token_id,
        )
        outputs = {"last_hidden_state": hidden}
        if self.output_logits:
            raw = (
                language_model.lm_head(hidden)
                if language_model.lm_head is not None
                else TiedHead(language_model.token_embedding, name="lm_head")(hidden)
            )
            outputs["logits"] = raw / logits_scaling

        super().__init__(
            inputs=inputs, outputs=outputs, name=name or type(self).__name__, **kwargs
        )

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.embedding_multiplier = embedding_multiplier
        self.residual_multiplier = residual_multiplier
        self.attention_multiplier = attention_multiplier
        self.logits_scaling = logits_scaling
        self.tie_embeddings = tie_embeddings
        self.eos_token_id = eos_token_id
        self.audio_token_id = audio_token_id
        self.downsample_rate = downsample_rate
        self.window_size = window_size
        self.has_lora_adapter = has_lora_adapter
        self.lora_rank = lora_rank_eff
        self.lora_alpha = lora_alpha
        self.encoder_input_dim = encoder_input_dim
        self.encoder_num_layers = encoder_num_layers
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_feedforward_mult = encoder_feedforward_mult
        self.encoder_num_heads = encoder_num_heads
        self.encoder_dim_head = encoder_dim_head
        self.encoder_output_dim = encoder_output_dim
        self.encoder_context_size = encoder_context_size
        self.encoder_max_pos_emb = encoder_max_pos_emb
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_expansion_factor = encoder_conv_expansion_factor
        self.projector_dim = projector_dim
        self.projector_num_layers = projector_num_layers
        self.projector_num_heads = projector_num_heads
        self.projector_intermediate_size = projector_intermediate_size
        self.projector_cross_attention_frequency = projector_cross_attention_frequency
        self.projector_layer_norm_eps = projector_layer_norm_eps
        self.cat_hidden_layers = cat_hidden_layers
        self.projector_encoder_hidden_size = projector_encoder_hidden_size
        self.encoder = encoder
        self.projector = projector
        self.language_model = language_model
        self.audio_features_layer = audio_features_layer
        self.audio_merge = audio_merge
        self.mask_layer = mask_layer

        # The audio encoder / projector / LoRA sublayers don't materialize during
        # functional graph construction (compute_output_spec skips their eager call);
        # a concrete dummy forward builds every weight before a stream is loaded.
        with inference_scope():
            self.build_dummy()

    def get_audio_features(self, input_features):
        encoder_out = self.encoder(input_features)
        return self.projector(encoder_out)

    def causal_mask(self, q_len, kv_len, offset, attention_mask=None):
        qi = ops.arange(q_len)[:, None] + offset
        ki = ops.arange(kv_len)[None, :]
        mask = ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            mask = mask + (1.0 - am)[:, None, None, :] * MASK_NEG
        return mask

    def merge_audio_embeddings(
        self, input_ids, inputs_embeds, audio_features, input_features_mask
    ):
        batch = int(input_ids.shape[0])
        seq = int(input_ids.shape[1])
        if input_features_mask is not None:
            fm = ops.convert_to_numpy(
                ops.convert_to_tensor(input_features_mask)
            ).astype(bool)
            af = ops.reshape(audio_features, (-1, self.embed_dim))
            keep = ops.convert_to_tensor(np.nonzero(fm.reshape(-1))[0].astype("int32"))
            audio_flat = ops.take(af, keep, axis=0)
        else:
            audio_flat = ops.reshape(audio_features, (-1, self.embed_dim))

        ids_flat = ops.convert_to_numpy(ops.reshape(input_ids, (-1,))).tolist()
        idx = [j for j, v in enumerate(ids_flat) if v == self.audio_token_id]
        embeds_flat = ops.reshape(inputs_embeds, (batch * seq, self.embed_dim))
        embeds_flat = ops.scatter_update(
            embeds_flat,
            ops.reshape(ops.convert_to_tensor(idx, dtype="int32"), (-1, 1)),
            ops.cast(audio_flat, embeds_flat.dtype),
        )
        return ops.reshape(embeds_flat, (batch, seq, self.embed_dim))

    def prepare_inputs(
        self, input_ids, input_features, input_features_mask, attention_mask
    ):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])

        llm_ids = ops.where(input_ids == self.audio_token_id, 0, input_ids)
        inputs_embeds = self.language_model.token_embedding(llm_ids)

        has_audio = input_features is not None
        if has_audio:
            input_features = ops.cast(ops.convert_to_tensor(input_features), "float32")
            audio_features = self.get_audio_features(input_features)
            inputs_embeds = self.merge_audio_embeddings(
                input_ids, inputs_embeds, audio_features, input_features_mask
            )
        inputs_embeds = inputs_embeds * self.embedding_multiplier

        position_ids = ops.broadcast_to(ops.arange(seq), (batch, seq))
        return inputs_embeds, position_ids, has_audio

    @classmethod
    def config_from_hf(cls, hf_config):
        text = hf_config["text_config"]
        enc = hf_config["encoder_config"]
        proj = hf_config["projector_config"]
        return {
            "vocab_size": text["vocab_size"],
            "embed_dim": text["hidden_size"],
            "mlp_dim": text["intermediate_size"],
            "num_layers": text["num_hidden_layers"],
            "num_heads": text["num_attention_heads"],
            "num_kv_heads": text["num_key_value_heads"],
            "norm_eps": text.get("rms_norm_eps", 1e-5),
            "rope_theta": text.get("rope_theta", 10000000.0),
            "embedding_multiplier": text.get("embedding_multiplier", 1.0),
            "residual_multiplier": text.get("residual_multiplier", 1.0),
            "attention_multiplier": text.get("attention_multiplier", 1.0),
            "logits_scaling": text.get("logits_scaling", 1.0),
            "tie_embeddings": text.get(
                "tie_word_embeddings", hf_config.get("tie_word_embeddings", True)
            ),
            "eos_token_id": text.get("eos_token_id", hf_config.get("eos_token_id", 0)),
            "audio_token_id": hf_config.get("audio_token_index", 49159),
            "downsample_rate": hf_config.get("downsample_rate", 5),
            "window_size": hf_config.get("window_size", 15),
            "has_lora_adapter": hf_config.get("has_lora_adapter", True),
            "encoder_input_dim": enc.get("input_dim", 160),
            "encoder_num_layers": enc["num_layers"],
            "encoder_hidden_dim": enc["hidden_dim"],
            "encoder_feedforward_mult": enc.get("feedforward_mult", 4),
            "encoder_num_heads": enc.get("num_heads", 8),
            "encoder_dim_head": enc.get("dim_head")
            or enc["hidden_dim"] // enc.get("num_heads", 8),
            "encoder_output_dim": enc.get("output_dim", 42),
            "encoder_context_size": enc.get("context_size", 200),
            "encoder_max_pos_emb": enc.get("max_pos_emb", 512),
            "encoder_conv_kernel_size": enc.get("conv_kernel_size", 15),
            "encoder_conv_expansion_factor": enc.get("conv_expansion_factor", 2),
            "projector_dim": proj["hidden_size"],
            "projector_num_layers": proj["num_hidden_layers"],
            "projector_num_heads": proj["num_attention_heads"],
            "projector_intermediate_size": proj["intermediate_size"],
            "projector_cross_attention_frequency": proj.get(
                "cross_attention_frequency", 1
            ),
            "projector_layer_norm_eps": proj.get("layer_norm_eps", 1e-12),
            "cat_hidden_layers": enc.get("cat_hidden_layers"),
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_granite_speech_hf_to_keras import transfer_granite_speech_weights

        transfer_granite_speech_weights(keras_model, hf_state_dict)

    def build_dummy(self):
        frames = 2 * self.window_size
        nblocks = math.ceil(frames / self.window_size)
        n_audio = nblocks * (self.window_size // self.downsample_rate)
        ids = np.array([[1] + [self.audio_token_id] * n_audio + [2]], dtype="int64")
        self(
            {
                "input_ids": ids,
                "attention_mask": np.ones_like(ids),
                "input_features": np.zeros(
                    (1, frames, self.encoder_input_dim), dtype="float32"
                ),
                "input_features_mask": np.ones((1, n_audio), dtype="bool"),
            }
        )

    def build_for_transfer(self):
        # This multimodal model builds lazily; the base (text-only) build_for_transfer
        # would leave the audio encoder / projector / LoRA weights uncreated. Run the
        # full audio+text dummy so every weight exists before a stream is loaded (used
        # by repo-id loading and the converted-weight cache).
        self.build_dummy()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "mlp_dim": self.mlp_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "norm_eps": self.norm_eps,
                "rope_theta": self.rope_theta,
                "embedding_multiplier": self.embedding_multiplier,
                "residual_multiplier": self.residual_multiplier,
                "attention_multiplier": self.attention_multiplier,
                "logits_scaling": self.logits_scaling,
                "tie_embeddings": self.tie_embeddings,
                "eos_token_id": self.eos_token_id,
                "audio_token_id": self.audio_token_id,
                "downsample_rate": self.downsample_rate,
                "window_size": self.window_size,
                "has_lora_adapter": self.has_lora_adapter,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "encoder_input_dim": self.encoder_input_dim,
                "encoder_num_layers": self.encoder_num_layers,
                "encoder_hidden_dim": self.encoder_hidden_dim,
                "encoder_feedforward_mult": self.encoder_feedforward_mult,
                "encoder_num_heads": self.encoder_num_heads,
                "encoder_dim_head": self.encoder_dim_head,
                "encoder_output_dim": self.encoder_output_dim,
                "encoder_context_size": self.encoder_context_size,
                "encoder_max_pos_emb": self.encoder_max_pos_emb,
                "encoder_conv_kernel_size": self.encoder_conv_kernel_size,
                "encoder_conv_expansion_factor": self.encoder_conv_expansion_factor,
                "projector_dim": self.projector_dim,
                "projector_num_layers": self.projector_num_layers,
                "projector_num_heads": self.projector_num_heads,
                "projector_intermediate_size": self.projector_intermediate_size,
                "projector_cross_attention_frequency": self.projector_cross_attention_frequency,
                "projector_layer_norm_eps": self.projector_layer_norm_eps,
                "cat_hidden_layers": self.cat_hidden_layers,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class GraniteSpeechConditionalGenerate(GraniteSpeechModel, BaseGeneration):
    """Granite Speech with an LM head + fast ``.generate()`` (audio+text -> text).

    Adds the (tied) vocabulary projection on top of :class:`GraniteSpeechModel`,
    with Granite's ``logits / logits_scaling``. ``call`` returns ``logits`` and
    ``last_hidden_state``. Fast generation comes from
    :class:`~zeromodels.base.BaseGeneration`: ``build_cache`` runs the audio
    encoder + projector + audio-token splice ONCE into a fixed KV cache (with the
    LoRA adapter enabled since audio is present), then ``call_with_cache`` does
    text-only decode steps (LoRA stays enabled across the turn). Pass audio
    exactly as for :class:`GraniteSpeechModel`:
    ``gen.generate(input_ids, input_features=..., input_features_mask=...)``.
    """

    output_logits = True

    def project(self, hidden):
        lm = self.language_model
        if lm.lm_head is not None:
            logits = lm.lm_head(hidden)
        else:
            logits = ops.matmul(hidden, ops.transpose(lm.token_embedding.embeddings))
        return logits / self.logits_scaling

    def build_cache(
        self,
        token_ids,
        padding_mask,
        max_len,
        input_features=None,
        input_features_mask=None,
    ):
        batch = int(token_ids.shape[0])
        prompt_len = int(token_ids.shape[1])
        nkv = self.language_model.num_kv_heads
        hd = self.language_model.head_dim
        inputs_embeds, position_ids, has_audio = self.prepare_inputs(
            token_ids, input_features, input_features_mask, padding_mask
        )
        cos, sin = rope_cos_sin(position_ids, self.head_dim, self.rope_theta)
        causal = self.causal_mask(
            prompt_len, prompt_len, offset=0, attention_mask=padding_mask
        )
        hidden, kv = self.language_model(
            inputs_embeds,
            cos,
            sin,
            attention_mask=causal,
            use_cache=True,
            apply_lora=has_audio,
        )
        layer_caches = []
        for k, v in kv:
            ck = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=k.dtype), (0, 0, 0, 0), k
            )
            cv = ops.slice_update(
                ops.zeros((batch, nkv, max_len, hd), dtype=v.dtype), (0, 0, 0, 0), v
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        kv_cache = ops.stack(layer_caches, axis=1)
        self._decode_apply_lora = bool(has_audio)
        logits = self.project(hidden[:, -1, :])
        return kv_cache, logits

    def call_with_cache(self, token_ids, cache, cache_update_index):
        kv_cache = cache
        apply_lora = getattr(self, "_decode_apply_lora", False)
        batch = int(token_ids.shape[0])
        max_len = int(kv_cache.shape[4])
        pos = ops.broadcast_to(ops.reshape(cache_update_index, (1, 1)), (batch, 1))
        cos, sin = rope_cos_sin(pos, self.head_dim, self.rope_theta)
        key_mask = ops.cast(
            ops.where(ops.arange(max_len) <= cache_update_index, 0.0, MASK_NEG),
            "float32",
        )[None, None, None, :]
        h = self.language_model.token_embedding(token_ids) * self.embedding_multiplier
        layer_caches = []
        for i, layer in enumerate(self.language_model.decoder_layers):
            h, ck, cv = layer.decode_step(
                h,
                cos,
                sin,
                kv_cache[:, i, 0],
                kv_cache[:, i, 1],
                cache_update_index,
                key_mask,
                apply_lora=apply_lora,
            )
            layer_caches.append(ops.stack([ck, cv], axis=1))
        kv_cache = ops.stack(layer_caches, axis=1)
        logits = self.project(self.language_model.final_norm(h))[:, 0, :]
        return logits, kv_cache
