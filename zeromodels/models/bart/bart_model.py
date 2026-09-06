import keras
from keras import layers, ops

from zeromodels.base import BaseModel, BaseSeq2SeqGeneration
from zeromodels.base.base_mixin import inference_scope

from .bart_config import BartConfig
from .bart_layers import BartAttention, BartLearnedPositionalEmbedding

BART_HUB_SIBLINGS = frozenset(
    {"BartModel", "BartConditionalGenerate", "BartSequenceClassify", "BartQnA"}
)

MASK_NEG = -1e9

_BART_ARCH_DEFAULTS = {
    "vocab_size": 50265,
    "hidden_dim": 1024,
    "encoder_num_layers": 12,
    "decoder_num_layers": 12,
    "encoder_attention_heads": 16,
    "decoder_attention_heads": 16,
    "encoder_ffn_dim": 4096,
    "decoder_ffn_dim": 4096,
    "max_position_embeddings": 1024,
    "activation_function": "gelu",
    "scale_embedding": False,
    "layer_norm_eps": 1e-5,
    "classifier_dropout": 0.0,
    "num_labels": 3,
    "pad_token_id": 1,
    "bos_token_id": 0,
    "eos_token_id": 2,
    "decoder_start_token_id": 2,
}
_ARCH_FIELDS = tuple(_BART_ARCH_DEFAULTS)


def resolve_bart_arch(kwargs):
    cfg = dict(_BART_ARCH_DEFAULTS)
    for k in ("model", "hf_id", "url", "num_classes"):
        kwargs.pop(k, None)
    for k in list(kwargs):
        if k in cfg:
            cfg[k] = kwargs.pop(k)
    return cfg


def resolve_activation(name):
    if callable(name):
        return name
    if name == "gelu":
        return lambda x: keras.activations.gelu(x, approximate=False)
    if name in ("gelu_new", "gelu_fast", "gelu_pytorch_tanh"):
        return lambda x: keras.activations.gelu(x, approximate=True)
    if name == "relu":
        return keras.activations.relu
    return keras.activations.get(name)


def additive_pad_mask(attention_mask):
    """``(B, T)`` 1/0 mask -> additive ``(B, 1, 1, T)`` (0 keep, -1e9 block)."""
    m = ops.cast(attention_mask, "float32")
    return (1.0 - m)[:, None, None, :] * MASK_NEG


def make_causal_mask(ids):
    seq_len = ops.shape(ids)[1]
    i = ops.arange(seq_len)[:, None]
    j = ops.arange(seq_len)[None, :]
    return (ops.cast(j > i, "float32") * MASK_NEG)[None, None]


def shift_right_ids(input_ids, decoder_start_token_id):
    start = ops.full(
        (ops.shape(input_ids)[0], 1), decoder_start_token_id, dtype=input_ids.dtype
    )
    return ops.concatenate([start, input_ids[:, :-1]], axis=1)


def pad_mask_layer(attn_mask):
    return layers.Lambda(
        additive_pad_mask, output_shape=lambda s: (s[0], 1, 1, s[1])
    )(attn_mask)


def causal_mask_layer(ids):
    return layers.Lambda(make_causal_mask, output_shape=lambda s: (1, 1, s[1], s[1]))(
        ids
    )


def shift_right_layer(input_ids, decoder_start_token_id):
    return layers.Lambda(
        lambda t: shift_right_ids(t, decoder_start_token_id), output_shape=lambda s: s
    )(input_ids)


def make_blocks(kind, hidden_dim, num_layers, num_heads, ffn_dim, eps):
    cross = kind == "decoder"
    blocks = []
    for i in range(num_layers):
        p = f"{kind}_layers_{i}"
        blk = {
            "self_attn": BartAttention(
                hidden_dim, num_heads, name_prefix=f"{p}_self_attn"
            ),
            "self_ln": layers.LayerNormalization(
                epsilon=eps, name=f"{p}_self_attn_layer_norm"
            ),
            "fc1": layers.Dense(ffn_dim, name=f"{p}_fc1"),
            "fc2": layers.Dense(hidden_dim, name=f"{p}_fc2"),
            "final_ln": layers.LayerNormalization(
                epsilon=eps, name=f"{p}_final_layer_norm"
            ),
        }
        if cross:
            blk["cross_attn"] = BartAttention(
                hidden_dim, num_heads, name_prefix=f"{p}_encoder_attn"
            )
            blk["cross_ln"] = layers.LayerNormalization(
                epsilon=eps, name=f"{p}_encoder_attn_layer_norm"
            )
        blocks.append(blk)
    return blocks


def make_backbone(cfg):
    hd, eps = cfg["hidden_dim"], cfg["layer_norm_eps"]
    return {
        "shared": layers.Embedding(cfg["vocab_size"], hd, name="shared"),
        "enc_pos": BartLearnedPositionalEmbedding(
            cfg["max_position_embeddings"], hd, name="encoder_embed_positions"
        ),
        "enc_ln_embed": layers.LayerNormalization(
            epsilon=eps, name="encoder_layernorm_embedding"
        ),
        "enc_blocks": make_blocks(
            "encoder", hd, cfg["encoder_num_layers"],
            cfg["encoder_attention_heads"], cfg["encoder_ffn_dim"], eps,
        ),
        "dec_pos": BartLearnedPositionalEmbedding(
            cfg["max_position_embeddings"], hd, name="decoder_embed_positions"
        ),
        "dec_ln_embed": layers.LayerNormalization(
            epsilon=eps, name="decoder_layernorm_embedding"
        ),
        "dec_blocks": make_blocks(
            "decoder", hd, cfg["decoder_num_layers"],
            cfg["decoder_attention_heads"], cfg["decoder_ffn_dim"], eps,
        ),
        "tied_head": None,  # created lazily by heads that need it
        "embed_scale": float(hd) ** 0.5 if cfg["scale_embedding"] else 1.0,
        "act": resolve_activation(cfg["activation_function"]),
    }


def encode_features(input_ids, pad_mask, bb):
    x = bb["shared"](input_ids)
    if bb["embed_scale"] != 1.0:
        x = x * bb["embed_scale"]
    x = bb["enc_pos"](x)
    x = bb["enc_ln_embed"](x)
    for b in bb["enc_blocks"]:
        residual = x
        h = b["self_attn"](x, attention_mask=pad_mask)
        x = b["self_ln"](residual + h)
        residual = x
        h = b["fc2"](bb["act"](b["fc1"](x)))
        x = b["final_ln"](residual + h)
    return x


def decode_features(decoder_input_ids, encoder_hidden, cross_pad_mask, cmask, bb):
    x = bb["shared"](decoder_input_ids)
    if bb["embed_scale"] != 1.0:
        x = x * bb["embed_scale"]
    x = bb["dec_pos"](x)
    x = bb["dec_ln_embed"](x)
    for b in bb["dec_blocks"]:
        residual = x
        h = b["self_attn"](x, attention_mask=cmask)
        x = b["self_ln"](residual + h)
        residual = x
        h = b["cross_attn"](
            x, key_value_states=encoder_hidden, attention_mask=cross_pad_mask
        )
        x = b["cross_ln"](residual + h)
        residual = x
        h = b["fc2"](bb["act"](b["fc1"](x)))
        x = b["final_ln"](residual + h)
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class BartTiedHead(layers.Layer):
    """LM head projecting with the shared embedding plus ``final_logits_bias``.

    ``logits = hidden @ shared.embeddingsᵀ + final_logits_bias``; reads the live
    shared weight (never a graph-time copy) and owns only the ``(1, vocab)`` bias.
    """

    def __init__(self, embedding, vocab_size, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "embedding", embedding)
        self.vocab_size = vocab_size

    def build(self, input_shape):
        self.final_logits_bias = self.add_weight(
            shape=(1, self.vocab_size),
            initializer="zeros",
            trainable=True,
            name="final_logits_bias",
        )
        super().build(input_shape)

    def call(self, hidden):
        kernel = ops.transpose(ops.cast(self.embedding.embeddings, hidden.dtype))
        return ops.matmul(hidden, kernel) + ops.cast(
            self.final_logits_bias, hidden.dtype
        )

    def compute_output_spec(self, hidden):
        shape = list(hidden.shape)
        shape[-1] = self.vocab_size
        return keras.KerasTensor(shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"vocab_size": self.vocab_size})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class BartEosPool(layers.Layer):
    """Pool the decoder hidden state at the last ``</s>`` (eos) token of ``input_ids``."""

    def __init__(self, eos_token_id, **kwargs):
        super().__init__(**kwargs)
        self.eos_token_id = eos_token_id

    def call(self, hidden_states, input_ids):
        seq = ops.shape(input_ids)[1]
        is_eos = ops.cast(ops.equal(input_ids, self.eos_token_id), "int32")
        positions = ops.arange(seq)[None]
        last_eos = ops.argmax(is_eos * (positions + 1), axis=-1)  # last eos per row
        onehot = ops.cast(ops.one_hot(last_eos, seq), hidden_states.dtype)
        return ops.sum(hidden_states * onehot[..., None], axis=1)

    def compute_output_shape(self, hidden_states_shape, input_ids_shape):
        return (hidden_states_shape[0], hidden_states_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update({"eos_token_id": self.eos_token_id})
        return config


class _BartBackboneMixin:
    """Shared backbone plumbing: store the layer objects + arch fields on ``self``."""

    def assign_backbone(self, bb, cfg):
        self.shared = bb["shared"]
        self.encoder_embed_positions = bb["enc_pos"]
        self.encoder_layernorm_embedding = bb["enc_ln_embed"]
        self.encoder_blocks = bb["enc_blocks"]
        self.decoder_embed_positions = bb["dec_pos"]
        self.decoder_layernorm_embedding = bb["dec_ln_embed"]
        self.decoder_blocks = bb["dec_blocks"]
        self.tied_head = bb["tied_head"]
        self.act = bb["act"]
        self.embed_scale = bb["embed_scale"]
        for k, v in cfg.items():
            setattr(self, k, v)

    @classmethod
    def config_from_hf(cls, hf_config):
        cfg = {
            "vocab_size": hf_config["vocab_size"],
            "hidden_dim": hf_config["d_model"],
            "encoder_num_layers": hf_config["encoder_layers"],
            "decoder_num_layers": hf_config["decoder_layers"],
            "encoder_attention_heads": hf_config["encoder_attention_heads"],
            "decoder_attention_heads": hf_config["decoder_attention_heads"],
            "encoder_ffn_dim": hf_config["encoder_ffn_dim"],
            "decoder_ffn_dim": hf_config["decoder_ffn_dim"],
            "max_position_embeddings": hf_config.get("max_position_embeddings", 1024),
            "activation_function": hf_config.get("activation_function", "gelu"),
            "scale_embedding": hf_config.get("scale_embedding", False),
            "pad_token_id": hf_config.get("pad_token_id", 1),
            "bos_token_id": hf_config.get("bos_token_id", 0),
            "eos_token_id": hf_config.get("eos_token_id", 2),
            "decoder_start_token_id": hf_config.get("decoder_start_token_id", 2),
        }
        if "id2label" in hf_config:
            cfg["num_labels"] = len(hf_config["id2label"])
        elif "num_labels" in hf_config:
            cfg["num_labels"] = hf_config["num_labels"]
        return cfg

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_bart_hf_to_keras import transfer_bart_weights

        transfer_bart_weights(keras_model, hf_state_dict)

    def get_config(self):
        config = super().get_config()
        config.update({k: getattr(self, k) for k in _ARCH_FIELDS})
        config["name"] = self.name
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class BartModel(_BartBackboneMixin, BaseModel):
    """BART encoder-decoder backbone (no task head).

    A shared byte-level-BPE token embedding feeds a bidirectional post-norm encoder
    and an autoregressive decoder that cross-attends to it: learned positional
    embeddings (offset by 2), a ``layernorm_embedding`` after embed + positions, and
    GELU feed-forwards. Returns the decoder ``last_hidden_state`` and the encoder
    output; use :class:`BartConditionalGenerate` for logits / summaries.

        model = BartModel.from_weights("hf:facebook/bart-base")
        out = model({"input_ids": ids, "attention_mask": mask,
                     "decoder_input_ids": dec_ids})

    Args (defaults are ``facebook/bart-large``):
        vocab_size, hidden_dim, encoder_num_layers, decoder_num_layers,
        encoder_attention_heads, decoder_attention_heads, encoder_ffn_dim,
        decoder_ffn_dim, max_position_embeddings, activation_function,
        scale_embedding, layer_norm_eps, classifier_dropout, num_labels,
        pad_token_id, bos_token_id, eos_token_id, decoder_start_token_id.
    """

    HF_MODEL_TYPE = "bart"
    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    config_class = BartConfig
    HUB_REPO_SIBLINGS = BART_HUB_SIBLINGS
    generate_args = {"max_new_tokens": 128}
    output_logits = False

    def __init__(self, name=None, **kwargs):
        cfg = resolve_bart_arch(kwargs)
        bb = make_backbone(cfg)
        if self.output_logits:
            bb["tied_head"] = BartTiedHead(
                bb["shared"], cfg["vocab_size"], name="lm_head"
            )

        input_ids = layers.Input(shape=(None,), dtype="int32", name="input_ids")
        attn_mask = layers.Input(shape=(None,), dtype="int32", name="attention_mask")
        dec_ids = layers.Input(shape=(None,), dtype="int32", name="decoder_input_ids")

        pad_mask = pad_mask_layer(attn_mask)
        cmask = causal_mask_layer(dec_ids)
        encoder_hidden = encode_features(input_ids, pad_mask, bb)
        decoder_hidden = decode_features(dec_ids, encoder_hidden, pad_mask, cmask, bb)
        outputs = {
            "last_hidden_state": decoder_hidden,
            "encoder_last_hidden_state": encoder_hidden,
        }
        if self.output_logits:
            outputs["logits"] = bb["tied_head"](decoder_hidden)

        super().__init__(
            inputs={
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "decoder_input_ids": dec_ids,
            },
            outputs=outputs,
            name=name or type(self).__name__,
            **kwargs,
        )
        self.assign_backbone(bb, cfg)
        self._materialize()

    def _materialize(self):
        with inference_scope():
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
        return shift_right_ids(input_ids, self.decoder_start_token_id)


@keras.saving.register_keras_serializable(package="zeromodels")
class BartConditionalGenerate(BartModel, BaseSeq2SeqGeneration):
    """BART with the tied LM head + ``final_logits_bias`` and fast ``.generate()``.

    Same backbone / weights as :class:`BartModel`; adds ``logits`` and cached
    seq2seq decoding for summarization / translation / generation.

        gen = BartConditionalGenerate.from_weights("hf:facebook/bart-large-cnn")
        out = gen.generate(input_ids, decoder_start_ids)
    """

    output_logits = True

    def project(self, hidden):
        return self.tied_head(hidden)

    @property
    def decode_num_heads(self):
        return self.decoder_attention_heads

    @property
    def decode_head_dim(self):
        return self.hidden_dim // self.decoder_attention_heads

    def encode(self, encoder_inputs):
        if isinstance(encoder_inputs, dict):
            input_ids = encoder_inputs["input_ids"]
            attention_mask = encoder_inputs.get("attention_mask")
        else:
            input_ids, attention_mask = encoder_inputs, None
        input_ids = ops.cast(ops.convert_to_tensor(input_ids), "int32")
        if attention_mask is None:
            attention_mask = ops.ones_like(input_ids)
        pad_mask = additive_pad_mask(attention_mask)
        bb = self._backbone_dict()
        return encode_features(input_ids, pad_mask, bb)

    def _backbone_dict(self):
        return {
            "shared": self.shared,
            "enc_pos": self.encoder_embed_positions,
            "enc_ln_embed": self.encoder_layernorm_embedding,
            "enc_blocks": self.encoder_blocks,
            "dec_pos": self.decoder_embed_positions,
            "dec_ln_embed": self.decoder_layernorm_embedding,
            "dec_blocks": self.decoder_blocks,
            "act": self.act,
            "embed_scale": self.embed_scale,
        }

    def decode_cross_kv(self, encoder_hidden_states):
        return [
            b["cross_attn"].project(encoder_hidden_states) for b in self.decoder_blocks
        ]

    def decode_forward(self, ids, cache, start_pos):
        x = self.shared(ids)
        if self.embed_scale != 1.0:
            x = x * self.embed_scale
        n = ids.shape[1]
        pos = self.decoder_embed_positions
        positions = pos.offset + start_pos + ops.arange(n)
        x = x + ops.take(pos.weight, positions, axis=0)[None]
        x = self.decoder_layernorm_embedding(x)

        new_cache = []
        for i, b in enumerate(self.decoder_blocks):
            self_k, self_v, cross_k, cross_v = cache[i]

            residual = x
            h, self_k, self_v = self.cached_self_attention(
                b["self_attn"], x, self_k, self_v, start_pos
            )
            x = b["self_ln"](residual + h)

            residual = x
            h = self.cached_cross_attention(b["cross_attn"], x, cross_k, cross_v)
            x = b["cross_ln"](residual + h)

            residual = x
            h = b["fc2"](self.act(b["fc1"](x)))
            x = b["final_ln"](residual + h)

            new_cache.append((self_k, self_v, cross_k, cross_v))

        return self.project(x), tuple(new_cache)


@keras.saving.register_keras_serializable(package="zeromodels")
class BartSequenceClassify(BartModel):
    """BART sequence classifier (``bart-large-mnli``, zero-shot NLI).

    Feeds the text to both towers (decoder input = right-shifted ``input_ids``),
    pools the decoder hidden state at the final ``</s>`` token, and runs a
    ``dense -> tanh -> out_proj`` head. ``logits`` are ``(batch, num_labels)``.
    """

    output_logits = False

    def __init__(self, name=None, **kwargs):
        cfg = resolve_bart_arch(kwargs)
        bb = make_backbone(cfg)
        dense = layers.Dense(cfg["hidden_dim"], name="classification_head_dense")
        out_proj = layers.Dense(cfg["num_labels"], name="classification_head_out_proj")

        input_ids = layers.Input(shape=(None,), dtype="int32", name="input_ids")
        attn_mask = layers.Input(shape=(None,), dtype="int32", name="attention_mask")
        pad_mask = pad_mask_layer(attn_mask)
        encoder_hidden = encode_features(input_ids, pad_mask, bb)
        dec_ids = shift_right_layer(input_ids, cfg["decoder_start_token_id"])
        cmask = causal_mask_layer(dec_ids)
        decoder_hidden = decode_features(dec_ids, encoder_hidden, pad_mask, cmask, bb)
        pooled = BartEosPool(cfg["eos_token_id"], name="eos_pool")(
            decoder_hidden, input_ids
        )
        logits = out_proj(ops.tanh(dense(pooled)))

        super(BartModel, self).__init__(
            inputs={"input_ids": input_ids, "attention_mask": attn_mask},
            outputs=logits,
            name=name or type(self).__name__,
        )
        self.assign_backbone(bb, cfg)
        self.classification_head_dense = dense
        self.classification_head_out_proj = out_proj
        with inference_scope():
            self(
                {
                    "input_ids": ops.zeros((1, 4), dtype="int32"),
                    "attention_mask": ops.ones((1, 4), dtype="int32"),
                }
            )


@keras.saving.register_keras_serializable(package="zeromodels")
class BartQnA(BartModel):
    """BART extractive question answering: span start / end logits on the decoder output."""

    output_logits = False

    def __init__(self, name=None, **kwargs):
        cfg = resolve_bart_arch(kwargs)
        bb = make_backbone(cfg)
        qa_outputs = layers.Dense(2, name="qa_outputs")

        input_ids = layers.Input(shape=(None,), dtype="int32", name="input_ids")
        attn_mask = layers.Input(shape=(None,), dtype="int32", name="attention_mask")
        pad_mask = pad_mask_layer(attn_mask)
        encoder_hidden = encode_features(input_ids, pad_mask, bb)
        dec_ids = shift_right_layer(input_ids, cfg["decoder_start_token_id"])
        cmask = causal_mask_layer(dec_ids)
        decoder_hidden = decode_features(dec_ids, encoder_hidden, pad_mask, cmask, bb)
        span = qa_outputs(decoder_hidden)
        start_logits = span[..., 0]
        end_logits = span[..., 1]

        super(BartModel, self).__init__(
            inputs={"input_ids": input_ids, "attention_mask": attn_mask},
            outputs={"start_logits": start_logits, "end_logits": end_logits},
            name=name or type(self).__name__,
        )
        self.assign_backbone(bb, cfg)
        self.qa_outputs = qa_outputs
        with inference_scope():
            self(
                {
                    "input_ids": ops.zeros((1, 4), dtype="int32"),
                    "attention_mask": ops.ones((1, 4), dtype="int32"),
                }
            )
