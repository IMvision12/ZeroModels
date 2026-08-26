import keras
from keras import layers, ops

from zeromodels.base import (
    BaseModel,
    BaseSeq2SeqGeneration,
    CheckpointSource,
)
from zeromodels.base.base_mixin import inference_scope

from .t5_config import T5Config
from .t5_layers import (
    MASK_NEG,
    T5DecoderBlock,
    T5EncoderBlock,
    T5LayerNorm,
    compute_relative_bias,
)

# All T5 classes (backbone + generative head + encoder + task heads) share the variant's
# weights repo, whose zm_config.json declares the canonical T5Model. The single hosted
# checkpoint is the full encoder-decoder; each class loads its own subset by leaf suffix.
T5_HUB_SIBLINGS = frozenset(
    {
        "T5Model",
        "T5ConditionalGenerate",
        "T5EncoderModel",
        "T5SequenceClassify",
        "T5TokenClassify",
        "T5QnA",
    }
)

_ARCH_FIELDS = (
    "vocab_size",
    "embed_dim",
    "key_value_dim",
    "mlp_dim",
    "num_layers",
    "num_decoder_layers",
    "num_heads",
    "relative_attention_num_buckets",
    "relative_attention_max_distance",
    "hidden_act",
    "layer_norm_eps",
    "dropout",
    "tie_word_embeddings",
    "pad_token_id",
    "eos_token_id",
    "decoder_start_token_id",
)


_T5_ARCH_DEFAULTS = {
    "vocab_size": 32128,
    "embed_dim": 768,
    "key_value_dim": 64,
    "mlp_dim": 3072,
    "num_layers": 12,
    "num_decoder_layers": 12,
    "num_heads": 12,
    "relative_attention_num_buckets": 32,
    "relative_attention_max_distance": 128,
    "hidden_act": "relu",
    "layer_norm_eps": 1e-6,
    "dropout": 0.1,
    "tie_word_embeddings": True,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "decoder_start_token_id": 0,
}


def resolve_t5_arch(kwargs):
    # Pop the T5 arch fields out of **kwargs (task heads take them positionally-free),
    # applying defaults; leaves non-arch kwargs (name, dtype, ...) in place.
    for k in ("model", "hf_id", "url", "num_classes"):
        kwargs.pop(k, None)
    return {k: kwargs.pop(k, v) for k, v in _T5_ARCH_DEFAULTS.items()}


def t5_config_from_hf(hf_config):
    return {
        "vocab_size": hf_config["vocab_size"],
        "embed_dim": hf_config["d_model"],
        "key_value_dim": hf_config["d_kv"],
        "mlp_dim": hf_config["d_ff"],
        "num_layers": hf_config["num_layers"],
        "num_decoder_layers": hf_config.get("num_decoder_layers")
        or hf_config["num_layers"],
        "num_heads": hf_config["num_heads"],
        "relative_attention_num_buckets": hf_config.get(
            "relative_attention_num_buckets", 32
        ),
        "relative_attention_max_distance": hf_config.get(
            "relative_attention_max_distance", 128
        ),
        "hidden_act": hf_config.get("dense_act_fn", "relu"),
        "layer_norm_eps": hf_config.get("layer_norm_epsilon", 1e-6),
        "dropout": hf_config.get("dropout_rate", 0.1),
        "tie_word_embeddings": hf_config.get("tie_word_embeddings", True),
        "pad_token_id": hf_config.get("pad_token_id", 0),
        "eos_token_id": hf_config.get("eos_token_id", 1),
        "decoder_start_token_id": hf_config.get("decoder_start_token_id", 0),
    }


@keras.saving.register_keras_serializable(package="zeromodels")
class T5PositionBias(layers.Layer):
    """Weightless T5 attention bias: learned relative-position bias, plus optional
    causal masking (decoder self-attention) and key padding.

    Wraps :func:`compute_relative_bias` so its dynamic ``arange`` stays out of the
    functional-graph trace (``compute_output_spec`` declares the shape; the arange
    runs only at eager runtime). Holds the relative-bias :class:`Embedding` by a
    non-tracked reference so it stays a direct model attribute (unchanged weight
    path); this layer adds no weight of its own.
    """

    def __init__(
        self,
        rel_bias,
        num_heads,
        num_buckets,
        max_distance,
        bidirectional,
        causal,
        **kwargs,
    ):
        super().__init__(**kwargs)
        object.__setattr__(self, "rel_bias", rel_bias)
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.causal = causal

    def call(self, hidden, attention_mask=None):
        seq = ops.shape(hidden)[1]
        bias = compute_relative_bias(
            self.rel_bias,
            seq,
            seq,
            self.bidirectional,
            self.num_buckets,
            self.max_distance,
        )
        if self.causal:
            qi = ops.arange(seq)[:, None]
            ki = ops.arange(seq)[None, :]
            bias = (
                bias
                + ops.cast(ops.where(ki <= qi, 0.0, MASK_NEG), "float32")[None, None]
            )
        if attention_mask is not None:
            am = ops.cast(ops.convert_to_tensor(attention_mask), "float32")
            bias = bias + (1.0 - am)[:, None, None, :] * MASK_NEG
        return bias

    def compute_output_spec(self, hidden, attention_mask=None):
        seq = hidden.shape[1]
        return keras.KerasTensor(
            (hidden.shape[0], self.num_heads, seq, seq), dtype="float32"
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "num_buckets": self.num_buckets,
                "max_distance": self.max_distance,
                "bidirectional": self.bidirectional,
                "causal": self.causal,
            }
        )
        return config


def t5_shift_right(input_ids, start_token):
    # Graph-safe right shift: prepend the decoder start token, drop the last token.
    start = ops.full_like(input_ids[:, :1], start_token)
    return ops.concatenate([start, input_ids[:, :-1]], axis=1)


def make_encoder_layers(
    embed_dim,
    key_value_dim,
    mlp_dim,
    num_layers,
    num_heads,
    hidden_act,
    layer_norm_eps,
    num_buckets,
    max_distance,
):
    rel_bias = layers.Embedding(num_buckets, num_heads, name="encoder_rel_bias")
    bias = T5PositionBias(
        rel_bias,
        num_heads,
        num_buckets,
        max_distance,
        bidirectional=True,
        causal=False,
        name="encoder_bias",
    )
    blocks = [
        T5EncoderBlock(
            embed_dim,
            key_value_dim,
            num_heads,
            mlp_dim,
            hidden_act,
            layer_norm_eps,
            prefix=f"enc_{i}",
            name=f"encoder_block_{i}",
        )
        for i in range(num_layers)
    ]
    final_norm = T5LayerNorm(layer_norm_eps, name="encoder_final_layer_norm")
    return rel_bias, bias, blocks, final_norm


def make_decoder_layers(
    embed_dim,
    key_value_dim,
    mlp_dim,
    num_decoder_layers,
    num_heads,
    hidden_act,
    layer_norm_eps,
    num_buckets,
    max_distance,
):
    rel_bias = layers.Embedding(num_buckets, num_heads, name="decoder_rel_bias")
    bias = T5PositionBias(
        rel_bias,
        num_heads,
        num_buckets,
        max_distance,
        bidirectional=False,
        causal=True,
        name="decoder_bias",
    )
    blocks = [
        T5DecoderBlock(
            embed_dim,
            key_value_dim,
            num_heads,
            mlp_dim,
            hidden_act,
            layer_norm_eps,
            prefix=f"dec_{i}",
            name=f"decoder_block_{i}",
        )
        for i in range(num_decoder_layers)
    ]
    final_norm = T5LayerNorm(layer_norm_eps, name="decoder_final_layer_norm")
    return rel_bias, bias, blocks, final_norm


def _t5_backbone(cfg, decoder=True):
    # Create the shared embedding + encoder (and optionally decoder) layers from a
    # resolved arch config. Shared by every T5 class so their weight paths match.
    shared = layers.Embedding(cfg["vocab_size"], cfg["embed_dim"], name="shared")
    e = make_encoder_layers(
        cfg["embed_dim"],
        cfg["key_value_dim"],
        cfg["mlp_dim"],
        cfg["num_layers"],
        cfg["num_heads"],
        cfg["hidden_act"],
        cfg["layer_norm_eps"],
        cfg["relative_attention_num_buckets"],
        cfg["relative_attention_max_distance"],
    )
    d = None
    if decoder:
        d = make_decoder_layers(
            cfg["embed_dim"],
            cfg["key_value_dim"],
            cfg["mlp_dim"],
            cfg["num_decoder_layers"],
            cfg["num_heads"],
            cfg["hidden_act"],
            cfg["layer_norm_eps"],
            cfg["relative_attention_num_buckets"],
            cfg["relative_attention_max_distance"],
        )
    return shared, e, d


def t5_encode_features(
    input_ids,
    attention_mask,
    *,
    shared,
    encoder_bias,
    encoder_blocks,
    encoder_final_norm,
):
    hidden = shared(input_ids)
    bias = encoder_bias(hidden, attention_mask)
    for block in encoder_blocks:
        hidden = block(hidden, bias)
    return encoder_final_norm(hidden)


def t5_decode_features(
    decoder_input_ids,
    encoder_hidden,
    decoder_attention_mask,
    encoder_attention_mask,
    *,
    shared,
    decoder_bias,
    decoder_blocks,
    decoder_final_norm,
):
    hidden = shared(decoder_input_ids)
    self_bias = decoder_bias(hidden, decoder_attention_mask)
    if encoder_attention_mask is not None:
        am = ops.cast(ops.convert_to_tensor(encoder_attention_mask), "float32")
        cross_bias = (1.0 - am)[:, None, None, :] * MASK_NEG
    else:
        cross_bias = 0.0
    for block in decoder_blocks:
        hidden = block(hidden, self_bias, encoder_hidden, cross_bias)
    return decoder_final_norm(hidden)


class _T5Encoder:
    """Shared encoder helpers (mixed into T5Model / T5EncoderModel).

    ``encode`` runs the functional encoder body imperatively (used by generation
    and the task heads' internal reuse); it delegates to :func:`t5_encode_features`
    with the model's own layers.
    """

    def encode(self, input_ids, attention_mask=None):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        return t5_encode_features(
            input_ids,
            attention_mask,
            shared=self.shared,
            encoder_bias=self.encoder_bias,
            encoder_blocks=self.encoder_blocks,
            encoder_final_norm=self.encoder_final_layer_norm,
        )

    def _assign_backbone(self, shared, e, d, cfg):
        # Attach the backbone layers + arch attrs after super().__init__ (functional
        # models can't hold attributes before the graph is built).
        self.shared = shared
        (
            self.encoder_rel_bias,
            self.encoder_bias,
            self.encoder_blocks,
            self.encoder_final_layer_norm,
        ) = e
        if d is not None:
            (
                self.decoder_rel_bias,
                self.decoder_bias,
                self.decoder_blocks,
                self.decoder_final_layer_norm,
            ) = d
        for k, v in cfg.items():
            setattr(self, k, v)


@keras.saving.register_keras_serializable(package="zeromodels")
class T5Model(BaseModel, _T5Encoder):
    """Original T5 encoder-decoder backbone (no LM head).

    A shared token embedding feeds a bidirectional encoder and a causal decoder that
    cross-attends to the encoder output. Attention uses learned relative position bias
    (shared within each stack), T5-style RMSNorm, pre-LayerNorm residuals, and bias-free
    projections. A functional model; returns the decoder ``last_hidden_state`` and the
    encoder output. Use :class:`T5ConditionalGenerate` for logits / text.

    References:
    - [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)

    Args:
        vocab_size: Token vocabulary size.
        embed_dim: Model dimension (``d_model``).
        key_value_dim: Per-head q/k/v dimension (``d_kv``).
        mlp_dim: Feed-forward intermediate size (``d_ff``).
        num_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        num_heads: Number of attention heads.
        relative_attention_num_buckets: Relative-position-bias bucket count.
        relative_attention_max_distance: Maximum distance for the relative bias.
        hidden_act: Feed-forward activation.
        layer_norm_eps: RMSNorm epsilon.
        dropout: Dropout rate (inference: unused).
        tie_word_embeddings: Whether :class:`T5ConditionalGenerate` ties + scales the LM head.
        pad_token_id: Padding token id (also the decoder start token).
        eos_token_id: End-of-sequence token id.
        decoder_start_token_id: First decoder input token.
    """

    HF_MODEL_TYPE = "t5"
    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = T5Config
    HUB_REPO_SIBLINGS = T5_HUB_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource("T5Model")

    output_logits = False

    def __init__(
        self,
        vocab_size=32128,
        embed_dim=768,
        key_value_dim=64,
        mlp_dim=3072,
        num_layers=12,
        num_decoder_layers=12,
        num_heads=12,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        hidden_act="relu",
        layer_norm_eps=1e-6,
        dropout=0.1,
        tie_word_embeddings=True,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        cfg = {k: v for k, v in locals().items() if k in _T5_ARCH_DEFAULTS}
        shared, e, d = _t5_backbone(cfg, decoder=True)
        e_rb, e_bias, e_blocks, e_norm = e
        d_rb, d_bias, d_blocks, d_norm = d
        lm_head = None
        if self.output_logits and not tie_word_embeddings:
            lm_head = layers.Dense(vocab_size, use_bias=False, name="lm_head")

        input_ids_in = layers.Input(shape=(None,), dtype="int32", name="input_ids")
        attn_in = layers.Input(shape=(None,), dtype="int32", name="attention_mask")
        dec_ids_in = layers.Input(
            shape=(None,), dtype="int32", name="decoder_input_ids"
        )
        inputs = {
            "input_ids": input_ids_in,
            "attention_mask": attn_in,
            "decoder_input_ids": dec_ids_in,
        }
        encoder_hidden = t5_encode_features(
            input_ids_in,
            attn_in,
            shared=shared,
            encoder_bias=e_bias,
            encoder_blocks=e_blocks,
            encoder_final_norm=e_norm,
        )
        decoder_hidden = t5_decode_features(
            dec_ids_in,
            encoder_hidden,
            None,
            attn_in,
            shared=shared,
            decoder_bias=d_bias,
            decoder_blocks=d_blocks,
            decoder_final_norm=d_norm,
        )
        outputs = {
            "last_hidden_state": decoder_hidden,
            "encoder_last_hidden_state": encoder_hidden,
        }
        if self.output_logits:
            if lm_head is not None:
                outputs["logits"] = lm_head(decoder_hidden)
            else:
                scaled = decoder_hidden * (embed_dim**-0.5)
                outputs["logits"] = T5TiedHead(shared, name="lm_head")(scaled)

        super().__init__(
            inputs=inputs, outputs=outputs, name=name or type(self).__name__, **kwargs
        )
        self._assign_backbone(shared, e, d, cfg)
        self.lm_head = lm_head

        # Blocks isolate their attention behind compute_output_spec, so their weights
        # don't materialize during graph construction; a dummy forward builds them.
        with inference_scope():
            self._materialize()

    def _materialize(self):
        dummy = ops.zeros((1, 4), dtype="int32")
        self(
            {
                "input_ids": dummy,
                "attention_mask": ops.ones((1, 4), dtype="int32"),
                "decoder_input_ids": dummy,
            }
        )

    def shift_right(self, input_ids):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        return t5_shift_right(input_ids, self.decoder_start_token_id)

    def decode(
        self,
        decoder_input_ids,
        encoder_hidden_states,
        decoder_attention_mask=None,
        encoder_attention_mask=None,
    ):
        decoder_input_ids = ops.cast(ops.convert_to_tensor(decoder_input_ids), "int32")
        return t5_decode_features(
            decoder_input_ids,
            encoder_hidden_states,
            decoder_attention_mask,
            encoder_attention_mask,
            shared=self.shared,
            decoder_bias=self.decoder_bias,
            decoder_blocks=self.decoder_blocks,
            decoder_final_norm=self.decoder_final_layer_norm,
        )

    @classmethod
    def config_from_hf(cls, hf_config):
        return t5_config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_t5_hf_to_keras import transfer_t5_weights

        transfer_t5_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update({k: getattr(self, k) for k in _ARCH_FIELDS})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class T5TiedHead(layers.Layer):
    """Weightless LM head projecting with the (already-scaled) shared embedding.

    Reads the live shared-embedding weight in ``call`` (never baking a graph-time
    copy) and holds it by a non-tracked reference so the embedding keeps its single
    weight and path; the head adds none. The caller scales the hidden state by
    ``embed_dim ** -0.5`` before this projection (original-T5 tied-head convention).
    """

    def __init__(self, embedding, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "embedding", embedding)

    def call(self, hidden):
        kernel = ops.transpose(ops.cast(self.embedding.embeddings, hidden.dtype))
        return ops.matmul(hidden, kernel)

    def compute_output_spec(self, hidden):
        shape = list(hidden.shape)
        shape[-1] = self.embedding.input_dim
        return keras.KerasTensor(shape, dtype=hidden.dtype)


@keras.saving.register_keras_serializable(package="zeromodels")
class T5ConditionalGenerate(T5Model, BaseSeq2SeqGeneration):
    """T5 backbone + a (tied, scaled) language-model head and text-to-text generation.

    ``logits`` are ``(batch, target_seq, vocab_size)``. For original T5 the LM head is
    the transposed shared embedding and the decoder output is scaled by
    ``embed_dim ** -0.5`` first (``tie_word_embeddings``). ``generate`` runs the encoder
    once and greedily decodes with the shared embedding as the head. Constructor ``Args``
    are inherited from :class:`T5Model`.
    """

    output_logits = True

    def project(self, hidden):
        if self.tie_word_embeddings:
            hidden = hidden * (self.embed_dim**-0.5)
            return ops.matmul(hidden, ops.transpose(self.shared.embeddings))
        return self.lm_head(hidden)

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=None,
        eos_token_id=None,
        **kwargs,
    ):
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        if attention_mask is not None:
            attention_mask = ops.cast(ops.convert_to_tensor(attention_mask), "int32")
        max_new_tokens = max_new_tokens or 64
        eos = eos_token_id if eos_token_id is not None else self.eos_token_id
        batch = int(input_ids.shape[0])

        encoder_hidden_states = self.encode(input_ids, attention_mask)
        generated = ops.full((batch, 1), self.decoder_start_token_id, dtype="int32")
        done = ops.zeros((batch,), dtype="bool")
        for _ in range(max_new_tokens):
            decoder_hidden = self.decode(
                generated, encoder_hidden_states, None, attention_mask
            )
            next_logits = self.project(decoder_hidden)[:, -1, :]
            next_ids = ops.cast(ops.argmax(next_logits, axis=-1), "int32")
            next_ids = ops.cast(ops.where(done, eos, next_ids), "int32")
            generated = ops.concatenate([generated, next_ids[:, None]], axis=1)
            done = ops.logical_or(done, ops.equal(next_ids, eos))
            if bool(ops.all(done)):
                break
        return generated


@keras.saving.register_keras_serializable(package="zeromodels")
class T5EncoderModel(BaseModel, _T5Encoder):
    """T5 encoder stack only (no decoder, no LM head).

    A shared token embedding feeds the bidirectional encoder; returns the encoder
    ``last_hidden_state`` ``(batch, seq, embed_dim)`` for embedding / feature use.
    Constructor ``Args`` mirror :class:`T5Model` (decoder fields are accepted and ignored).
    """

    HF_MODEL_TYPE = "t5"
    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = T5Config
    HUB_REPO_SIBLINGS = T5_HUB_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource("T5Model")

    def __init__(
        self,
        vocab_size=32128,
        embed_dim=768,
        key_value_dim=64,
        mlp_dim=3072,
        num_layers=12,
        num_decoder_layers=12,
        num_heads=12,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        hidden_act="relu",
        layer_norm_eps=1e-6,
        dropout=0.1,
        tie_word_embeddings=True,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        name=None,
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        cfg = {k: v for k, v in locals().items() if k in _T5_ARCH_DEFAULTS}
        shared, e, _ = _t5_backbone(cfg, decoder=False)
        e_rb, e_bias, e_blocks, e_norm = e
        input_ids_in = layers.Input(shape=(None,), dtype="int32", name="input_ids")
        attn_in = layers.Input(shape=(None,), dtype="int32", name="attention_mask")
        inputs = {"input_ids": input_ids_in, "attention_mask": attn_in}
        hidden = t5_encode_features(
            input_ids_in,
            attn_in,
            shared=shared,
            encoder_bias=e_bias,
            encoder_blocks=e_blocks,
            encoder_final_norm=e_norm,
        )
        super().__init__(
            inputs=inputs,
            outputs={"last_hidden_state": hidden},
            name=name or type(self).__name__,
            **kwargs,
        )
        self._assign_backbone(shared, e, None, cfg)

        with inference_scope():
            self(
                {
                    "input_ids": ops.zeros((1, 4), dtype="int32"),
                    "attention_mask": ops.ones((1, 4), dtype="int32"),
                }
            )

    @classmethod
    def config_from_hf(cls, hf_config):
        return t5_config_from_hf(hf_config)

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_t5_hf_to_keras import transfer_t5_weights

        transfer_t5_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update({k: getattr(self, k) for k in _ARCH_FIELDS})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class T5SequenceClassify(T5Model):
    """T5 sequence classifier (HF ``T5ForSequenceClassification``).

    The full encoder-decoder backbone (decoder input = right-shifted ``input_ids``), then
    the decoder hidden state at the last EOS position feeds a tanh classification head.
    Returns logits ``(batch, num_classes)``. Constructor ``Args`` extend :class:`T5Model`
    with ``num_classes`` and ``classifier_dropout``.
    """

    output_logits = False

    def __init__(self, num_classes=2, classifier_dropout=0.0, name=None, **kwargs):
        cfg = resolve_t5_arch(kwargs)
        shared, e, d = _t5_backbone(cfg, decoder=True)
        e_rb, e_bias, e_blocks, e_norm = e
        d_rb, d_bias, d_blocks, d_norm = d
        classifier_dense = layers.Dense(cfg["embed_dim"], name="classifier_dense")
        classifier_out_proj = layers.Dense(num_classes, name="classifier_out_proj")

        input_ids_in = layers.Input(shape=(None,), dtype="int32", name="input_ids")
        attn_in = layers.Input(shape=(None,), dtype="int32", name="attention_mask")
        encoder_hidden = t5_encode_features(
            input_ids_in,
            attn_in,
            shared=shared,
            encoder_bias=e_bias,
            encoder_blocks=e_blocks,
            encoder_final_norm=e_norm,
        )
        decoder_hidden = t5_decode_features(
            t5_shift_right(input_ids_in, cfg["decoder_start_token_id"]),
            encoder_hidden,
            None,
            attn_in,
            shared=shared,
            decoder_bias=d_bias,
            decoder_blocks=d_blocks,
            decoder_final_norm=d_norm,
        )
        pooled = T5EosPool(cfg["eos_token_id"], name="eos_pool")(
            decoder_hidden, input_ids_in
        )
        logits = classifier_out_proj(ops.tanh(classifier_dense(pooled)))
        super(T5Model, self).__init__(
            inputs={"input_ids": input_ids_in, "attention_mask": attn_in},
            outputs=logits,
            name=name or type(self).__name__,
        )
        self._assign_backbone(shared, e, d, cfg)
        self.num_classes = num_classes
        self.classifier_dropout = classifier_dropout
        self.classifier_dense = classifier_dense
        self.classifier_out_proj = classifier_out_proj
        with inference_scope():
            self(
                {
                    "input_ids": ops.zeros((1, 4), dtype="int32"),
                    "attention_mask": ops.ones((1, 4), dtype="int32"),
                }
            )

    @classmethod
    def config_from_hf(cls, hf_config):
        config = t5_config_from_hf(hf_config)
        config["num_classes"] = (
            len(hf_config["id2label"])
            if "id2label" in hf_config
            else hf_config.get("num_labels", 2)
        )
        return config

    def get_config(self):
        config = super().get_config()
        config["num_classes"] = self.num_classes
        config["classifier_dropout"] = self.classifier_dropout
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class T5TokenClassify(T5EncoderModel):
    """T5 token classifier (HF ``T5ForTokenClassification``).

    The ENCODER only, then a per-token linear classifier. Returns logits
    ``(batch, seq, num_classes)``. Constructor ``Args`` extend :class:`T5EncoderModel`
    with ``num_classes`` and ``classifier_dropout``.
    """

    def __init__(self, num_classes=2, classifier_dropout=0.0, name=None, **kwargs):
        cfg = resolve_t5_arch(kwargs)
        shared, e, _ = _t5_backbone(cfg, decoder=False)
        e_rb, e_bias, e_blocks, e_norm = e
        classifier = layers.Dense(num_classes, name="classifier")

        input_ids_in = layers.Input(shape=(None,), dtype="int32", name="input_ids")
        attn_in = layers.Input(shape=(None,), dtype="int32", name="attention_mask")
        hidden = t5_encode_features(
            input_ids_in,
            attn_in,
            shared=shared,
            encoder_bias=e_bias,
            encoder_blocks=e_blocks,
            encoder_final_norm=e_norm,
        )
        logits = classifier(hidden)
        super(T5EncoderModel, self).__init__(
            inputs={"input_ids": input_ids_in, "attention_mask": attn_in},
            outputs=logits,
            name=name or type(self).__name__,
        )
        self._assign_backbone(shared, e, None, cfg)
        self.num_classes = num_classes
        self.classifier_dropout = classifier_dropout
        self.classifier = classifier
        with inference_scope():
            self(
                {
                    "input_ids": ops.zeros((1, 4), dtype="int32"),
                    "attention_mask": ops.ones((1, 4), dtype="int32"),
                }
            )

    @classmethod
    def config_from_hf(cls, hf_config):
        config = t5_config_from_hf(hf_config)
        config["num_classes"] = (
            len(hf_config["id2label"])
            if "id2label" in hf_config
            else hf_config.get("num_labels", 2)
        )
        return config

    def get_config(self):
        config = super().get_config()
        config["num_classes"] = self.num_classes
        config["classifier_dropout"] = self.classifier_dropout
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class T5QnA(T5Model):
    """T5 extractive question answering (HF ``T5ForQuestionAnswering``).

    The full encoder-decoder backbone (decoder input = right-shifted ``input_ids``) and a
    linear ``qa_outputs`` head over the decoder output, returning ``start_logits`` and
    ``end_logits`` ``(batch, seq)`` each. Constructor ``Args`` are inherited from
    :class:`T5Model`.
    """

    output_logits = False

    def __init__(self, name=None, **kwargs):
        cfg = resolve_t5_arch(kwargs)
        shared, e, d = _t5_backbone(cfg, decoder=True)
        e_rb, e_bias, e_blocks, e_norm = e
        d_rb, d_bias, d_blocks, d_norm = d
        qa_outputs = layers.Dense(2, name="qa_outputs")

        input_ids_in = layers.Input(shape=(None,), dtype="int32", name="input_ids")
        attn_in = layers.Input(shape=(None,), dtype="int32", name="attention_mask")
        encoder_hidden = t5_encode_features(
            input_ids_in,
            attn_in,
            shared=shared,
            encoder_bias=e_bias,
            encoder_blocks=e_blocks,
            encoder_final_norm=e_norm,
        )
        decoder_hidden = t5_decode_features(
            t5_shift_right(input_ids_in, cfg["decoder_start_token_id"]),
            encoder_hidden,
            None,
            attn_in,
            shared=shared,
            decoder_bias=d_bias,
            decoder_blocks=d_blocks,
            decoder_final_norm=d_norm,
        )
        start_logits, end_logits = T5QnASplit(name="qa_split")(
            qa_outputs(decoder_hidden)
        )
        super(T5Model, self).__init__(
            inputs={"input_ids": input_ids_in, "attention_mask": attn_in},
            outputs={"start_logits": start_logits, "end_logits": end_logits},
            name=name or type(self).__name__,
        )
        self._assign_backbone(shared, e, d, cfg)
        self.qa_outputs = qa_outputs
        with inference_scope():
            self(
                {
                    "input_ids": ops.zeros((1, 4), dtype="int32"),
                    "attention_mask": ops.ones((1, 4), dtype="int32"),
                }
            )


@keras.saving.register_keras_serializable(package="zeromodels")
class T5EosPool(layers.Layer):
    """Select the decoder hidden state at each row's last EOS token position."""

    def __init__(self, eos_token_id, **kwargs):
        super().__init__(**kwargs)
        self.eos_token_id = eos_token_id

    def call(self, sequence_output, input_ids):
        seq = ops.shape(input_ids)[1]
        eos_positions = ops.where(
            ops.equal(input_ids, self.eos_token_id), ops.arange(seq)[None, :], -1
        )
        last_eos = ops.max(eos_positions, axis=1)
        return ops.take_along_axis(sequence_output, last_eos[:, None, None], axis=1)[
            :, 0, :
        ]

    def compute_output_spec(self, sequence_output, input_ids):
        return keras.KerasTensor(
            (sequence_output.shape[0], sequence_output.shape[-1]),
            dtype=sequence_output.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config["eos_token_id"] = self.eos_token_id
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class T5QnASplit(layers.Layer):
    """Split the 2-way QnA projection into (start_logits, end_logits)."""

    def call(self, logits):
        start_logits, end_logits = ops.split(logits, 2, axis=-1)
        return ops.squeeze(start_logits, -1), ops.squeeze(end_logits, -1)

    def compute_output_spec(self, logits):
        shape = logits.shape[:-1]
        return (
            keras.KerasTensor(shape, dtype=logits.dtype),
            keras.KerasTensor(shape, dtype=logits.dtype),
        )
