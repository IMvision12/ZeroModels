import keras
from keras import layers, ops

from zeromodels.base import BaseModel, CheckpointSource

from .deberta_v2_config import (
    DebertaV2Config,
)
from .deberta_v2_layers import (
    DebertaV2ConvLayer,
    DebertaV2DisentangledSelfAttention,
    DebertaV2Embeddings,
    DebertaV2FlattenChoices,
    DebertaV2RelativeEmbedding,
    DebertaV2UnflattenChoices,
    make_log_bucket_position,
)

# All classes (encoder + masked-LM + task heads) share the variant's weights repo,
# whose zm_config.json declares the canonical DebertaV2Model encoder (model.weights.h5).
DEBERTA_V2_HUB_SIBLINGS = frozenset(
    {
        "DebertaV2Model",
        "DebertaV2MaskedLM",
        "DebertaV2SequenceClassify",
        "DebertaV2TokenClassify",
        "DebertaV2QnA",
        "DebertaV2MultipleChoice",
    }
)


def deberta_v2_encoder_layer(
    x,
    attention_mask,
    relative_pos,
    rel_embeddings,
    *,
    embed_dim,
    num_heads,
    mlp_dim,
    position_buckets,
    pos_att_type,
    share_att_key,
    hidden_act,
    layer_norm_eps,
    dropout,
    attention_dropout,
    layer_idx,
):
    prefix = f"blocks_{layer_idx}"

    attn = DebertaV2DisentangledSelfAttention(
        embed_dim,
        num_heads,
        position_buckets,
        pos_att_type=pos_att_type,
        share_att_key=share_att_key,
        attention_dropout=attention_dropout,
        block_prefix=prefix,
        name=f"{prefix}_attention_self",
    )(
        x,
        attention_mask=attention_mask,
        relative_pos=relative_pos,
        rel_embeddings=rel_embeddings,
    )
    attn = layers.Dense(embed_dim, name=f"{prefix}_attention_output_dense")(attn)
    attn = layers.Dropout(dropout)(attn)
    attn = layers.Add(name=f"{prefix}_attention_output_add")([attn, x])
    attn = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{prefix}_attention_output_layernorm"
    )(attn)

    inter = layers.Dense(mlp_dim, name=f"{prefix}_intermediate_dense")(attn)
    inter = layers.Activation(hidden_act, name=f"{prefix}_intermediate_act")(inter)
    out = layers.Dense(embed_dim, name=f"{prefix}_output_dense")(inter)
    out = layers.Dropout(dropout)(out)
    out = layers.Add(name=f"{prefix}_output_add")([out, attn])
    out = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{prefix}_output_layernorm"
    )(out)
    return out


def deberta_v2_backbone(
    input_ids,
    attention_mask,
    token_type_ids,
    *,
    vocab_size,
    embed_dim,
    num_layers,
    num_heads,
    mlp_dim,
    max_relative_positions,
    position_buckets,
    pos_att_type,
    share_att_key,
    norm_rel_ebd,
    conv_kernel_size,
    conv_act,
    hidden_act,
    layer_norm_eps,
    dropout,
    attention_dropout,
):
    embeddings = DebertaV2Embeddings(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        layer_norm_eps=layer_norm_eps,
        dropout=dropout,
        name="embeddings",
    )([input_ids, attention_mask, token_type_ids])

    m = ops.cast(attention_mask, "int32")
    ext = ops.expand_dims(ops.expand_dims(m, 1), 1)
    att_mask = ext * ops.transpose(ext, (0, 1, 3, 2))

    pos = ops.cumsum(ops.ones_like(input_ids), axis=1) - 1
    rel = ops.expand_dims(pos, 2) - ops.expand_dims(pos, 1)
    relative_pos = make_log_bucket_position(
        rel, position_buckets, max_relative_positions
    )

    rel_embeddings = DebertaV2RelativeEmbedding(
        2 * position_buckets, embed_dim, name="rel_embeddings"
    )(embeddings)
    if norm_rel_ebd:
        rel_embeddings = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="rel_embeddings_layernorm"
        )(rel_embeddings)

    x = embeddings
    for i in range(num_layers):
        layer_out = deberta_v2_encoder_layer(
            x,
            att_mask,
            relative_pos,
            rel_embeddings,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            position_buckets=position_buckets,
            pos_att_type=pos_att_type,
            share_att_key=share_att_key,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            attention_dropout=attention_dropout,
            layer_idx=i,
        )
        if i == 0 and conv_kernel_size:
            layer_out = DebertaV2ConvLayer(
                embed_dim,
                conv_kernel_size,
                conv_act,
                layer_norm_eps=layer_norm_eps,
                dropout=dropout,
                name="conv",
            )(embeddings, residual=layer_out, attention_mask=attention_mask)
        x = layer_out
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class DebertaV2Model(BaseModel):
    """Instantiates the DeBERTa-v2 encoder backbone.

    DeBERTa-v2 extends DeBERTa's disentangled attention with log-bucketed
    relative positions, a shared key/query projection for the relative terms, a
    LayerNorm on the relative embeddings, and (for the xlarge sizes) a single
    convolution layer after the first transformer block. Position information is
    injected through attention (no absolute-position or token-type embeddings).

    Takes a dict of ``input_ids`` / ``attention_mask`` / ``token_type_ids`` (all
    ``(B, seq)`` int; ``token_type_ids`` is accepted for API parity but unused)
    and returns ``{"last_hidden_state": (B, seq, embed_dim)}``. No pooler.

    References:
    - [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)

    Args:
        vocab_size: Integer, token vocabulary size. Defaults to `128100`.
        embed_dim: Integer, model dimension. Defaults to `1536`.
        num_layers: Integer, number of encoder layers. Defaults to `24`.
        num_heads: Integer, number of attention heads. Defaults to `24`.
        mlp_dim: Integer, feed-forward hidden dimension. Defaults to `6144`.
        max_position_embeddings: Integer, maximum sequence length. Defaults to `512`.
        max_relative_positions: Integer, max absolute relative position. Defaults to `512`.
        position_buckets: Integer, relative-position bucket count. Defaults to `256`.
        pos_att_type: List of disentangled-attention terms. Defaults to `["p2c", "c2p"]`.
        norm_rel_ebd: Boolean, LayerNorm the relative embeddings. Defaults to `True`.
        conv_kernel_size: Integer, conv kernel size (0/None disables). Defaults to `3`.
        conv_act: String, conv activation. Defaults to `"gelu"`.
        hidden_act: String, feed-forward activation. Defaults to `"gelu"`.
        norm_eps: Float, LayerNorm epsilon. Defaults to `1e-7`. The deprecated
            alias `layer_norm_eps` is still accepted.
        pad_token_id: Integer, padding token id. Defaults to `0`.
        dropout: Float, hidden dropout rate. Defaults to `0.0`.
        attention_dropout: Float, attention dropout rate. Defaults to `0.0`.
        name: String, model name. Defaults to `"DebertaV2Model"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "deberta-v2"
    config_class = DebertaV2Config
    HUB_REPO_SIBLINGS = DEBERTA_V2_HUB_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource("DebertaV2MaskedLM")

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_deberta_v2_hf_to_keras import transfer_deberta_v2_weights

        transfer_deberta_v2_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        max_rel = hf_config.get("max_relative_positions", -1)
        if max_rel is None or max_rel < 1:
            max_rel = hf_config["max_position_embeddings"]
        norm = (hf_config.get("norm_rel_ebd") or "").lower()
        # HF stores pos_att_type as a pipe-separated string ("p2c|c2p") or a list;
        # list("p2c|c2p") would split into characters, so handle the string case.
        pos_att_type = hf_config.get("pos_att_type") or ["p2c", "c2p"]
        if isinstance(pos_att_type, str):
            pos_att_type = pos_att_type.split("|")
        return {
            "vocab_size": hf_config["vocab_size"],
            "embed_dim": hf_config["hidden_size"],
            "num_layers": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "mlp_dim": hf_config["intermediate_size"],
            "max_position_embeddings": hf_config["max_position_embeddings"],
            "max_relative_positions": max_rel,
            "position_buckets": hf_config.get("position_buckets") or 256,
            "pos_att_type": list(pos_att_type),
            "share_att_key": hf_config.get("share_att_key", True),
            "norm_rel_ebd": "layer_norm" in norm,
            "conv_kernel_size": hf_config.get("conv_kernel_size") or 0,
            "conv_act": hf_config.get("conv_act") or "gelu",
            "hidden_act": hf_config.get("hidden_act", "gelu"),
            "norm_eps": hf_config.get("layer_norm_eps", 1e-7),
            "pad_token_id": hf_config.get("pad_token_id", 0),
        }

    def __init__(
        self,
        vocab_size=128100,
        embed_dim=1536,
        num_layers=24,
        num_heads=24,
        mlp_dim=6144,
        max_position_embeddings=512,
        max_relative_positions=512,
        position_buckets=256,
        pos_att_type=("p2c", "c2p"),
        share_att_key=True,
        norm_rel_ebd=True,
        conv_kernel_size=3,
        conv_act="gelu",
        hidden_act="gelu",
        norm_eps=1e-7,
        pad_token_id=0,
        dropout=0.0,
        attention_dropout=0.0,
        name="DebertaV2Model",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes"):
            kwargs.pop(k, None)
        norm_eps = kwargs.pop("layer_norm_eps", norm_eps)
        pos_att_type = list(pos_att_type)

        inputs = {
            "input_ids": layers.Input(shape=(None,), dtype="int32", name="input_ids"),
            "attention_mask": layers.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
            "token_type_ids": layers.Input(
                shape=(None,), dtype="int32", name="token_type_ids"
            ),
        }
        sequence_output = deberta_v2_backbone(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"],
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_relative_positions=max_relative_positions,
            position_buckets=position_buckets,
            pos_att_type=pos_att_type,
            share_att_key=share_att_key,
            norm_rel_ebd=norm_rel_ebd,
            conv_kernel_size=conv_kernel_size,
            conv_act=conv_act,
            hidden_act=hidden_act,
            layer_norm_eps=norm_eps,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        super().__init__(
            inputs=inputs,
            outputs={"last_hidden_state": sequence_output},
            name=name,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_position_embeddings = max_position_embeddings
        self.max_relative_positions = max_relative_positions
        self.position_buckets = position_buckets
        self.pos_att_type = pos_att_type
        self.share_att_key = share_att_key
        self.norm_rel_ebd = norm_rel_ebd
        self.conv_kernel_size = conv_kernel_size
        self.conv_act = conv_act
        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.attention_dropout = attention_dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "max_position_embeddings": self.max_position_embeddings,
                "max_relative_positions": self.max_relative_positions,
                "position_buckets": self.position_buckets,
                "pos_att_type": self.pos_att_type,
                "share_att_key": self.share_att_key,
                "norm_rel_ebd": self.norm_rel_ebd,
                "conv_kernel_size": self.conv_kernel_size,
                "conv_act": self.conv_act,
                "hidden_act": self.hidden_act,
                "norm_eps": self.norm_eps,
                "pad_token_id": self.pad_token_id,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class DebertaV2MaskedLM(BaseModel):
    """DeBERTa-v2 with the masked-language-modeling head.

    Wraps a :class:`DebertaV2Model` backbone and attaches DeBERTa's MLM head, a
    dense transform with ``gelu`` + LayerNorm, then a vocabulary projection:
    producing token logits ``(B, seq, vocab_size)``. The head weights are part of
    the pretrained checkpoint, so ``from_weights`` restores a ready-to-use
    fill-mask model.

    References:
    - [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)

    Args:
        See :class:`DebertaV2Model` for the backbone arguments.
        name: String, model name. Defaults to `"DebertaV2MaskedLM"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "deberta-v2"
    config_class = DebertaV2Config
    HUB_REPO_SIBLINGS = DEBERTA_V2_HUB_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource("DebertaV2MaskedLM")

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_deberta_v2_hf_to_keras import transfer_deberta_v2_weights

        transfer_deberta_v2_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return DebertaV2Model.config_from_hf(hf_config)

    def __init__(self, name="DebertaV2MaskedLM", **kwargs):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes"):
            kwargs.pop(k, None)
        cfg = {
            "vocab_size": 128100,
            "embed_dim": 1536,
            "num_layers": 24,
            "num_heads": 24,
            "mlp_dim": 6144,
            "max_position_embeddings": 512,
            "max_relative_positions": 512,
            "position_buckets": 256,
            "pos_att_type": ("p2c", "c2p"),
            "share_att_key": True,
            "norm_rel_ebd": True,
            "conv_kernel_size": 3,
            "conv_act": "gelu",
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-7,
            "pad_token_id": 0,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            **kwargs,
        }
        backbone = DebertaV2Model(**cfg, name=f"{name}_backbone")

        x = backbone.output["last_hidden_state"]
        x = layers.Dense(cfg["embed_dim"], name="lm_head_dense")(x)
        x = layers.Activation("gelu", name="lm_head_act")(x)
        x = layers.LayerNormalization(
            epsilon=cfg["layer_norm_eps"], name="lm_head_layernorm"
        )(x)
        logits = layers.Dense(cfg["vocab_size"], name="lm_head_decoder")(x)

        super().__init__(inputs=backbone.input, outputs=logits, name=name)
        self._cfg = cfg

    def get_config(self):
        config = super().get_config()
        config.update({**self._cfg, "name": self.name})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class DebertaV2SequenceClassify(BaseModel):
    """DeBERTa-v2 sentence/sequence classifier.

    Wraps a :class:`DebertaV2Model` backbone and attaches DeBERTa's context pooler
    (dense + ``gelu`` on the first token) plus a dense classifier, producing
    ``num_classes`` logits ``(B, num_classes)``. The pretrained checkpoint has no
    task head, so ``from_weights`` restores the backbone and leaves the pooler +
    classifier randomly initialized (ready for fine-tuning) unless a fine-tuned
    release is configured.

    References:
    - [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)

    Args:
        See :class:`DebertaV2Model` for the backbone arguments.
        num_classes: Integer, number of output classes. Defaults to `2`.
        pooler_dropout: Float, dropout inside the context pooler. Defaults to `0.0`.
        classifier_dropout: Float, dropout before the classifier. Defaults to `0.0`.
        classifier_activation: String/callable, head activation. Defaults to `"linear"`.
        name: String, model name. Defaults to `"DebertaV2SequenceClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "deberta-v2"
    config_class = DebertaV2Config
    HUB_REPO_SIBLINGS = DEBERTA_V2_HUB_SIBLINGS

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_deberta_v2_hf_to_keras import transfer_deberta_v2_weights

        transfer_deberta_v2_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        config = DebertaV2Model.config_from_hf(hf_config)
        config["num_classes"] = (
            len(hf_config["id2label"])
            if "id2label" in hf_config
            else hf_config.get("num_labels", 2)
        )
        return config

    CHECKPOINT_SOURCE = CheckpointSource("DebertaV2MaskedLM")

    def __init__(
        self,
        num_classes=2,
        pooler_dropout=0.0,
        classifier_dropout=0.0,
        classifier_activation="linear",
        name="DebertaV2SequenceClassify",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url"):
            kwargs.pop(k, None)
        cfg = {
            "vocab_size": 128100,
            "embed_dim": 1536,
            "num_layers": 24,
            "num_heads": 24,
            "mlp_dim": 6144,
            "max_position_embeddings": 512,
            "max_relative_positions": 512,
            "position_buckets": 256,
            "pos_att_type": ("p2c", "c2p"),
            "share_att_key": True,
            "norm_rel_ebd": True,
            "conv_kernel_size": 3,
            "conv_act": "gelu",
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-7,
            "pad_token_id": 0,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            **kwargs,
        }
        backbone = DebertaV2Model(**cfg, name=f"{name}_backbone")

        x = backbone.output["last_hidden_state"][:, 0]
        x = layers.Dropout(pooler_dropout)(x)
        x = layers.Dense(cfg["embed_dim"], name="pooler_dense")(x)
        x = layers.Activation("gelu", name="pooler_act")(x)
        x = layers.Dropout(classifier_dropout)(x)
        logits = layers.Dense(
            num_classes, activation=classifier_activation, name="classifier"
        )(x)

        super().__init__(inputs=backbone.input, outputs=logits, name=name)
        self._cfg = cfg
        self.num_classes = num_classes
        self.pooler_dropout = pooler_dropout
        self.classifier_dropout = classifier_dropout
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                **self._cfg,
                "num_classes": self.num_classes,
                "pooler_dropout": self.pooler_dropout,
                "classifier_dropout": self.classifier_dropout,
                "classifier_activation": self.classifier_activation,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class DebertaV2TokenClassify(BaseModel):
    """DeBERTa-v2 token classifier (e.g. NER / POS tagging).

    Wraps a :class:`DebertaV2Model` backbone and attaches dropout plus a per-token
    dense head, producing ``num_classes`` logits ``(B, seq, num_classes)``. The
    head is randomly initialized from the pretrained checkpoint and meant for
    fine-tuning.

    References:
    - [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)

    Args:
        See :class:`DebertaV2Model` for the backbone arguments.
        num_classes: Integer, number of token classes. Defaults to `2`.
        classifier_dropout: Float, dropout before the classifier. Defaults to `0.0`.
        classifier_activation: String/callable, head activation. Defaults to `"linear"`.
        name: String, model name. Defaults to `"DebertaV2TokenClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "deberta-v2"
    config_class = DebertaV2Config
    HUB_REPO_SIBLINGS = DEBERTA_V2_HUB_SIBLINGS

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_deberta_v2_hf_to_keras import transfer_deberta_v2_weights

        transfer_deberta_v2_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        config = DebertaV2Model.config_from_hf(hf_config)
        config["num_classes"] = (
            len(hf_config["id2label"])
            if "id2label" in hf_config
            else hf_config.get("num_labels", 2)
        )
        return config

    CHECKPOINT_SOURCE = CheckpointSource("DebertaV2MaskedLM")

    def __init__(
        self,
        num_classes=2,
        classifier_dropout=0.0,
        classifier_activation="linear",
        name="DebertaV2TokenClassify",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url"):
            kwargs.pop(k, None)
        cfg = {
            "vocab_size": 128100,
            "embed_dim": 1536,
            "num_layers": 24,
            "num_heads": 24,
            "mlp_dim": 6144,
            "max_position_embeddings": 512,
            "max_relative_positions": 512,
            "position_buckets": 256,
            "pos_att_type": ("p2c", "c2p"),
            "share_att_key": True,
            "norm_rel_ebd": True,
            "conv_kernel_size": 3,
            "conv_act": "gelu",
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-7,
            "pad_token_id": 0,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            **kwargs,
        }
        backbone = DebertaV2Model(**cfg, name=f"{name}_backbone")

        x = backbone.output["last_hidden_state"]
        x = layers.Dropout(classifier_dropout)(x)
        logits = layers.Dense(
            num_classes, activation=classifier_activation, name="classifier"
        )(x)

        super().__init__(inputs=backbone.input, outputs=logits, name=name)
        self._cfg = cfg
        self.num_classes = num_classes
        self.classifier_dropout = classifier_dropout
        self.classifier_activation = classifier_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                **self._cfg,
                "num_classes": self.num_classes,
                "classifier_dropout": self.classifier_dropout,
                "classifier_activation": self.classifier_activation,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class DebertaV2QnA(BaseModel):
    """DeBERTa-v2 extractive question-answering head.

    Wraps a :class:`DebertaV2Model` backbone and attaches a dense span head that
    maps each token to two logits, split into ``start_logits`` and ``end_logits``
    (each ``(B, seq)``). The head is randomly initialized from the pretrained
    checkpoint and meant for fine-tuning (or loaded from a fine-tuned ``hf:``
    repo such as a SQuAD model).

    References:
    - [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)

    Args:
        See :class:`DebertaV2Model` for the backbone arguments.
        name: String, model name. Defaults to `"DebertaV2QnA"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "deberta-v2"
    config_class = DebertaV2Config
    HUB_REPO_SIBLINGS = DEBERTA_V2_HUB_SIBLINGS

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_deberta_v2_hf_to_keras import transfer_deberta_v2_weights

        transfer_deberta_v2_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return DebertaV2Model.config_from_hf(hf_config)

    CHECKPOINT_SOURCE = CheckpointSource("DebertaV2MaskedLM")

    def __init__(self, name="DebertaV2QnA", **kwargs):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes"):
            kwargs.pop(k, None)
        cfg = {
            "vocab_size": 128100,
            "embed_dim": 1536,
            "num_layers": 24,
            "num_heads": 24,
            "mlp_dim": 6144,
            "max_position_embeddings": 512,
            "max_relative_positions": 512,
            "position_buckets": 256,
            "pos_att_type": ("p2c", "c2p"),
            "share_att_key": True,
            "norm_rel_ebd": True,
            "conv_kernel_size": 3,
            "conv_act": "gelu",
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-7,
            "pad_token_id": 0,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            **kwargs,
        }
        backbone = DebertaV2Model(**cfg, name=f"{name}_backbone")

        x = backbone.output["last_hidden_state"]
        span = layers.Dense(2, name="qa_outputs")(x)
        outputs = {"start_logits": span[:, :, 0], "end_logits": span[:, :, 1]}

        super().__init__(inputs=backbone.input, outputs=outputs, name=name)
        self._cfg = cfg

    def get_config(self):
        config = super().get_config()
        config.update({**self._cfg, "name": self.name})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class DebertaV2MultipleChoice(BaseModel):
    """DeBERTa-v2 multiple-choice head (e.g. SWAG).

    Takes a dict of ``(B, num_choices, seq)`` int tensors, flattens the choices
    into the batch, runs the :class:`DebertaV2Model` backbone, scores each choice
    with DeBERTa's context pooler (dense + ``gelu`` on the first token) plus a
    shared dense layer, and reshapes back to per-example ``(B, num_choices)``
    logits. The head is randomly initialized and meant for fine-tuning.

    References:
    - [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)

    Args:
        See :class:`DebertaV2Model` for the backbone arguments.
        num_choices: Integer, number of choices per example. Defaults to `4`.
        pooler_dropout: Float, dropout inside the context pooler. Defaults to `0.0`.
        classifier_dropout: Float, dropout before the choice scorer. Defaults to `0.0`.
        name: String, model name. Defaults to `"DebertaV2MultipleChoice"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "deberta-v2"
    config_class = DebertaV2Config
    HUB_REPO_SIBLINGS = DEBERTA_V2_HUB_SIBLINGS

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_deberta_v2_hf_to_keras import transfer_deberta_v2_weights

        transfer_deberta_v2_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return DebertaV2Model.config_from_hf(hf_config)

    CHECKPOINT_SOURCE = CheckpointSource("DebertaV2MaskedLM")

    def __init__(
        self,
        num_choices=4,
        pooler_dropout=0.0,
        classifier_dropout=0.0,
        name="DebertaV2MultipleChoice",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes"):
            kwargs.pop(k, None)
        cfg = {
            "vocab_size": 128100,
            "embed_dim": 1536,
            "num_layers": 24,
            "num_heads": 24,
            "mlp_dim": 6144,
            "max_position_embeddings": 512,
            "max_relative_positions": 512,
            "position_buckets": 256,
            "pos_att_type": ("p2c", "c2p"),
            "share_att_key": True,
            "norm_rel_ebd": True,
            "conv_kernel_size": 3,
            "conv_act": "gelu",
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-7,
            "pad_token_id": 0,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            **kwargs,
        }

        input_ids = layers.Input(
            shape=(num_choices, None), dtype="int32", name="input_ids"
        )
        attention_mask = layers.Input(
            shape=(num_choices, None), dtype="int32", name="attention_mask"
        )
        token_type_ids = layers.Input(
            shape=(num_choices, None), dtype="int32", name="token_type_ids"
        )

        backbone = DebertaV2Model(**cfg, name=f"{name}_backbone")
        flatten = DebertaV2FlattenChoices(name="flatten_choices")
        seq = backbone(
            {
                "input_ids": flatten(input_ids),
                "attention_mask": flatten(attention_mask),
                "token_type_ids": flatten(token_type_ids),
            }
        )["last_hidden_state"]
        x = seq[:, 0]
        x = layers.Dropout(pooler_dropout)(x)
        x = layers.Dense(cfg["embed_dim"], name="pooler_dense")(x)
        x = layers.Activation("gelu", name="pooler_act")(x)
        x = layers.Dropout(classifier_dropout)(x)
        x = layers.Dense(1, name="classifier")(x)
        logits = DebertaV2UnflattenChoices(num_choices, name="unflatten_choices")(x)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        super().__init__(inputs=inputs, outputs=logits, name=name)
        self._cfg = cfg
        self.num_choices = num_choices
        self.pooler_dropout = pooler_dropout
        self.classifier_dropout = classifier_dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                **self._cfg,
                "num_choices": self.num_choices,
                "pooler_dropout": self.pooler_dropout,
                "classifier_dropout": self.classifier_dropout,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
