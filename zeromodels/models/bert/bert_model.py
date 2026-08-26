import keras
from keras import layers, ops

from zeromodels.base import BaseModel, CheckpointSource

from .bert_config import BertConfig
from .bert_layers import (
    BertEmbeddings,
    BertFlattenChoices,
    BertSelfAttention,
    BertUnflattenChoices,
)

MASK_NEG = -1e9

# All BERT classes (encoder + masked-LM + task heads) share the variant's weights repo.
# The single hosted model.weights.h5 is the FULL checkpoint (encoder + pooler + MLM head),
# saved from BertMaskedLM(add_pooler=True); each class loads its own subset (Keras ignores
# the weights it does not have). The zm_config.json declares the canonical BertModel.
BERT_HUB_SIBLINGS = frozenset(
    {
        "BertModel",
        "BertMaskedLM",
        "BertSequenceClassify",
        "BertTokenClassify",
        "BertNextSentencePredict",
        "BertQnA",
        "BertMultipleChoice",
    }
)


def bert_encoder_layer(
    x,
    attention_mask,
    *,
    embed_dim,
    num_heads,
    mlp_dim,
    hidden_act,
    layer_norm_eps,
    dropout,
    attention_dropout,
    layer_idx,
):
    """One BERT transformer block: self-attention + feed-forward.

    Both sub-blocks use post-LayerNorm residuals, matching the original BERT
    (``LayerNorm(x + Sublayer(x))``).

    Args:
        x: Input token states ``(B, seq, embed_dim)``.
        attention_mask: Additive mask ``(B, 1, 1, seq)`` (0 keep, large-negative drop).
        embed_dim: Model dimension.
        num_heads: Number of attention heads.
        mlp_dim: Feed-forward hidden dimension.
        hidden_act: Feed-forward activation.
        layer_norm_eps: Epsilon for the two LayerNorms.
        dropout: Hidden dropout rate.
        attention_dropout: Attention-weight dropout rate.
        layer_idx: Encoder-layer index (used for unique layer names).

    Returns:
        Token states ``(B, seq, embed_dim)``.
    """
    prefix = f"blocks_{layer_idx}"

    attn = BertSelfAttention(
        embed_dim,
        num_heads,
        attention_dropout=attention_dropout,
        block_prefix=prefix,
        name=f"{prefix}_attention_self",
    )(x, attention_mask=attention_mask)
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


def bert_backbone(
    input_ids,
    attention_mask,
    token_type_ids,
    *,
    vocab_size,
    embed_dim,
    num_layers,
    num_heads,
    mlp_dim,
    max_position_embeddings,
    type_vocab_size,
    hidden_act,
    layer_norm_eps,
    dropout,
    attention_dropout,
    add_pooler,
):
    """BERT embeddings + transformer encoder (+ optional pooler).

    Returns ``(sequence_output, pooled_output)`` where ``pooled_output`` is
    ``None`` when ``add_pooler`` is False.
    """
    embeddings = BertEmbeddings(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=type_vocab_size,
        layer_norm_eps=layer_norm_eps,
        dropout=dropout,
        name="embeddings",
    )([input_ids, token_type_ids])

    mask = ops.cast(attention_mask, "float32")
    mask = ops.expand_dims(ops.expand_dims(mask, 1), 1)
    mask = (1.0 - mask) * MASK_NEG

    x = embeddings
    for i in range(num_layers):
        x = bert_encoder_layer(
            x,
            mask,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            attention_dropout=attention_dropout,
            layer_idx=i,
        )

    sequence_output = x
    pooled_output = None
    if add_pooler:
        first_token = sequence_output[:, 0]
        pooled_output = layers.Dense(embed_dim, activation="tanh", name="pooler_dense")(
            first_token
        )
    return sequence_output, pooled_output


@keras.saving.register_keras_serializable(package="zeromodels")
class BertModel(BaseModel):
    """Instantiates the BERT encoder backbone.

    BERT embeds tokens with summed word / absolute-position / token-type
    embeddings, then applies a stack of bidirectional transformer encoder
    layers (multi-head self-attention + feed-forward, each with a post-LayerNorm
    residual). An optional pooler applies a ``tanh`` dense projection to the
    first ([CLS]) token.

    The model takes a dict of ``input_ids``, ``attention_mask`` and
    ``token_type_ids`` (all ``(B, seq)`` int tensors, as produced by
    :class:`BertTokenizer`) and returns a dict with ``last_hidden_state``
    ``(B, seq, embed_dim)`` and, when ``add_pooler=True``, ``pooler_output``
    ``(B, embed_dim)``.

    References:
    - [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

    Args:
        vocab_size: Integer, token vocabulary size. Defaults to `30522`.
        embed_dim: Integer, model / embedding dimension. Defaults to `768`.
        num_layers: Integer, number of transformer encoder layers.
            Defaults to `12`.
        num_heads: Integer, number of attention heads. Defaults to `12`.
        mlp_dim: Integer, feed-forward hidden dimension. Defaults to `3072`.
        max_position_embeddings: Integer, size of the position-embedding table.
            Defaults to `512`.
        type_vocab_size: Integer, number of token-type ids. Defaults to `2`.
        hidden_act: String, feed-forward / pooler-free activation. Defaults to `"gelu"`.
        norm_eps: Float, LayerNorm epsilon. Defaults to `1e-12`. Accepts
            `layer_norm_eps` as a deprecated alias.
        pad_token_id: Integer, padding token id. Defaults to `0`.
        dropout: Float, hidden dropout rate. Defaults to `0.0`.
        attention_dropout: Float, attention-weight dropout rate. Defaults to `0.0`.
        add_pooler: Boolean, whether to add the [CLS] pooler. Defaults to `True`.
        name: String, model name. Defaults to `"BertModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "bert"
    config_class = BertConfig
    HUB_REPO_SIBLINGS = BERT_HUB_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource(
        "BertMaskedLM", build_kwargs={"add_pooler": True}
    )

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_bert_hf_to_keras import transfer_bert_weights

        transfer_bert_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "vocab_size": hf_config["vocab_size"],
            "embed_dim": hf_config["hidden_size"],
            "num_layers": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "mlp_dim": hf_config["intermediate_size"],
            "max_position_embeddings": hf_config["max_position_embeddings"],
            "type_vocab_size": hf_config["type_vocab_size"],
            "hidden_act": hf_config.get("hidden_act", "gelu"),
            "norm_eps": hf_config.get("layer_norm_eps", 1e-12),
            "pad_token_id": hf_config.get("pad_token_id", 0),
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "add_pooler": True,
        }

    def __init__(
        self,
        vocab_size=30522,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_act="gelu",
        norm_eps=1e-12,
        pad_token_id=0,
        dropout=0.0,
        attention_dropout=0.0,
        add_pooler=True,
        name="BertModel",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes"):
            kwargs.pop(k, None)
        norm_eps = kwargs.pop("layer_norm_eps", norm_eps)

        inputs = {
            "input_ids": layers.Input(shape=(None,), dtype="int32", name="input_ids"),
            "attention_mask": layers.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
            "token_type_ids": layers.Input(
                shape=(None,), dtype="int32", name="token_type_ids"
            ),
        }
        sequence_output, pooled_output = bert_backbone(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"],
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            hidden_act=hidden_act,
            layer_norm_eps=norm_eps,
            dropout=dropout,
            attention_dropout=attention_dropout,
            add_pooler=add_pooler,
        )

        outputs = {"last_hidden_state": sequence_output}
        if pooled_output is not None:
            outputs["pooler_output"] = pooled_output

        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.add_pooler = add_pooler

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
                "type_vocab_size": self.type_vocab_size,
                "hidden_act": self.hidden_act,
                "norm_eps": self.norm_eps,
                "pad_token_id": self.pad_token_id,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
                "add_pooler": self.add_pooler,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class BertMaskedLM(BaseModel):
    """BERT with the masked-language-modeling head.

    Wraps a :class:`BertModel` backbone (no pooler) and attaches BERT's MLM head
    a dense transform with ``hidden_act`` + LayerNorm, then a vocabulary
    projection, producing token logits ``(B, seq, vocab_size)``. The head's
    weights are part of the pretrained checkpoint, so ``from_weights`` restores
    a ready-to-use fill-mask model.

    References:
    - [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

    Args:
        See :class:`BertModel` for the backbone arguments.
        name: String, model name. Defaults to `"BertMaskedLM"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "bert"
    config_class = BertConfig
    HUB_REPO_SIBLINGS = BERT_HUB_SIBLINGS

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_bert_hf_to_keras import transfer_bert_weights

        transfer_bert_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return BertModel.config_from_hf(hf_config)

    CHECKPOINT_SOURCE = CheckpointSource(
        "BertMaskedLM", build_kwargs={"add_pooler": True}
    )

    def __init__(
        self,
        vocab_size=30522,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_act="gelu",
        norm_eps=1e-12,
        pad_token_id=0,
        dropout=0.0,
        attention_dropout=0.0,
        add_pooler=False,
        name="BertMaskedLM",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes"):
            kwargs.pop(k, None)
        norm_eps = kwargs.pop("layer_norm_eps", norm_eps)

        backbone = BertModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            hidden_act=hidden_act,
            norm_eps=norm_eps,
            pad_token_id=pad_token_id,
            dropout=dropout,
            attention_dropout=attention_dropout,
            add_pooler=add_pooler,
            name=f"{name}_backbone",
        )

        x = backbone.output["last_hidden_state"]
        x = layers.Dense(embed_dim, name="mlm_transform_dense")(x)
        x = layers.Activation(hidden_act, name="mlm_transform_act")(x)
        x = layers.LayerNormalization(epsilon=norm_eps, name="mlm_transform_layernorm")(
            x
        )
        logits = layers.Dense(vocab_size, name="mlm_decoder")(x)

        # add_pooler=True also carries the pretrained [CLS] pooler, so a single hosted
        # checkpoint (saved from here) serves BertModel (encoder + pooler) and
        # BertMaskedLM (encoder + MLM head) alike -- each class loads its own subset.
        # Default False keeps the plain fill-mask output; loaded models use this default.
        if add_pooler:
            outputs = {
                "logits": logits,
                "pooler_output": backbone.output["pooler_output"],
            }
        else:
            outputs = logits

        super().__init__(inputs=backbone.input, outputs=outputs, name=name, **kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.add_pooler = add_pooler

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
                "type_vocab_size": self.type_vocab_size,
                "hidden_act": self.hidden_act,
                "norm_eps": self.norm_eps,
                "pad_token_id": self.pad_token_id,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
                "add_pooler": self.add_pooler,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class BertSequenceClassify(BaseModel):
    """BERT sentence/sequence classifier.

    Wraps a :class:`BertModel` backbone (with pooler) and attaches dropout plus a
    dense head on the pooled [CLS] representation, producing ``num_classes``
    logits ``(B, num_classes)``. The pretrained checkpoint has no task head, so
    ``from_weights`` restores the backbone + pooler and leaves the classifier
    randomly initialized (ready for fine-tuning) unless a fine-tuned release is
    configured.

    References:
    - [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

    Args:
        See :class:`BertModel` for the backbone arguments.
        num_classes: Integer, number of output classes. Defaults to `2`.
        classifier_dropout: Float, dropout before the classifier. Defaults to `0.0`.
        classifier_activation: String/callable, head activation (`"linear"` for
            logits). Defaults to `"linear"`.
        name: String, model name. Defaults to `"BertSequenceClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "bert"
    config_class = BertConfig
    HUB_REPO_SIBLINGS = BERT_HUB_SIBLINGS

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_bert_hf_to_keras import transfer_bert_weights

        transfer_bert_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        config = BertModel.config_from_hf(hf_config)
        config["num_classes"] = (
            len(hf_config["id2label"])
            if "id2label" in hf_config
            else hf_config.get("num_labels", 2)
        )
        return config

    CHECKPOINT_SOURCE = CheckpointSource(
        "BertMaskedLM", build_kwargs={"add_pooler": True}
    )

    def __init__(
        self,
        vocab_size=30522,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_act="gelu",
        norm_eps=1e-12,
        pad_token_id=0,
        dropout=0.0,
        attention_dropout=0.0,
        num_classes=2,
        classifier_dropout=0.0,
        classifier_activation="linear",
        name="BertSequenceClassify",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url", "add_pooler"):
            kwargs.pop(k, None)
        norm_eps = kwargs.pop("layer_norm_eps", norm_eps)

        backbone = BertModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            hidden_act=hidden_act,
            norm_eps=norm_eps,
            pad_token_id=pad_token_id,
            dropout=dropout,
            attention_dropout=attention_dropout,
            add_pooler=True,
            name=f"{name}_backbone",
        )

        x = backbone.output["pooler_output"]
        x = layers.Dropout(classifier_dropout)(x)
        logits = layers.Dense(
            num_classes, activation=classifier_activation, name="classifier"
        )(x)

        super().__init__(inputs=backbone.input, outputs=logits, name=name, **kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.num_classes = num_classes
        self.classifier_dropout = classifier_dropout
        self.classifier_activation = classifier_activation

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
                "type_vocab_size": self.type_vocab_size,
                "hidden_act": self.hidden_act,
                "norm_eps": self.norm_eps,
                "pad_token_id": self.pad_token_id,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
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
class BertTokenClassify(BaseModel):
    """BERT token classifier (e.g. NER / POS tagging).

    Wraps a :class:`BertModel` backbone (no pooler) and attaches dropout plus a
    dense head applied per token, producing ``num_classes`` logits
    ``(B, seq, num_classes)``. As with :class:`BertSequenceClassify`, the head is
    randomly initialized from the pretrained checkpoint and meant for
    fine-tuning.

    References:
    - [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

    Args:
        See :class:`BertModel` for the backbone arguments.
        num_classes: Integer, number of token classes. Defaults to `2`.
        classifier_dropout: Float, dropout before the classifier. Defaults to `0.0`.
        classifier_activation: String/callable, head activation. Defaults to `"linear"`.
        name: String, model name. Defaults to `"BertTokenClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "bert"
    config_class = BertConfig
    HUB_REPO_SIBLINGS = BERT_HUB_SIBLINGS

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_bert_hf_to_keras import transfer_bert_weights

        transfer_bert_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        config = BertModel.config_from_hf(hf_config)
        config["num_classes"] = (
            len(hf_config["id2label"])
            if "id2label" in hf_config
            else hf_config.get("num_labels", 2)
        )
        return config

    CHECKPOINT_SOURCE = CheckpointSource(
        "BertMaskedLM", build_kwargs={"add_pooler": True}
    )

    def __init__(
        self,
        vocab_size=30522,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_act="gelu",
        norm_eps=1e-12,
        pad_token_id=0,
        dropout=0.0,
        attention_dropout=0.0,
        num_classes=2,
        classifier_dropout=0.0,
        classifier_activation="linear",
        name="BertTokenClassify",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url", "add_pooler"):
            kwargs.pop(k, None)
        norm_eps = kwargs.pop("layer_norm_eps", norm_eps)

        backbone = BertModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            hidden_act=hidden_act,
            norm_eps=norm_eps,
            pad_token_id=pad_token_id,
            dropout=dropout,
            attention_dropout=attention_dropout,
            add_pooler=False,
            name=f"{name}_backbone",
        )

        x = backbone.output["last_hidden_state"]
        x = layers.Dropout(classifier_dropout)(x)
        logits = layers.Dense(
            num_classes, activation=classifier_activation, name="classifier"
        )(x)

        super().__init__(inputs=backbone.input, outputs=logits, name=name, **kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.num_classes = num_classes
        self.classifier_dropout = classifier_dropout
        self.classifier_activation = classifier_activation

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
                "type_vocab_size": self.type_vocab_size,
                "hidden_act": self.hidden_act,
                "norm_eps": self.norm_eps,
                "pad_token_id": self.pad_token_id,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
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
class BertNextSentencePredict(BaseModel):
    """BERT next-sentence-prediction head.

    Wraps a :class:`BertModel` backbone (with pooler) and attaches BERT's
    next-sentence head: a dense projection of the pooled [CLS] token to two
    logits (``isNext`` / ``notNext``), ``(B, 2)``. These head weights are part
    of the pretrained checkpoint, so loading a base BERT via ``hf:`` restores a
    working NSP model.

    References:
    - [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

    Args:
        See :class:`BertModel` for the backbone arguments.
        name: String, model name. Defaults to `"BertNextSentencePredict"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "bert"
    config_class = BertConfig
    HUB_REPO_SIBLINGS = BERT_HUB_SIBLINGS

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_bert_hf_to_keras import transfer_bert_weights

        transfer_bert_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return BertModel.config_from_hf(hf_config)

    CHECKPOINT_SOURCE = CheckpointSource(
        "BertMaskedLM", build_kwargs={"add_pooler": True}
    )

    def __init__(
        self,
        vocab_size=30522,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_act="gelu",
        norm_eps=1e-12,
        pad_token_id=0,
        dropout=0.0,
        attention_dropout=0.0,
        name="BertNextSentencePredict",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes", "add_pooler"):
            kwargs.pop(k, None)
        norm_eps = kwargs.pop("layer_norm_eps", norm_eps)

        backbone = BertModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            hidden_act=hidden_act,
            norm_eps=norm_eps,
            pad_token_id=pad_token_id,
            dropout=dropout,
            attention_dropout=attention_dropout,
            add_pooler=True,
            name=f"{name}_backbone",
        )

        x = backbone.output["pooler_output"]
        logits = layers.Dense(2, name="nsp_classifier")(x)

        super().__init__(inputs=backbone.input, outputs=logits, name=name, **kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
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
                "type_vocab_size": self.type_vocab_size,
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
class BertQnA(BaseModel):
    """BERT extractive question-answering head.

    Wraps a :class:`BertModel` backbone (no pooler) and attaches a dense span
    head that maps each token to two logits, split into ``start_logits`` and
    ``end_logits`` (each ``(B, seq)``). The head is randomly initialized from
    the pretrained checkpoint and meant for fine-tuning (or loaded from a
    fine-tuned ``hf:`` repo such as a SQuAD model).

    References:
    - [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

    Args:
        See :class:`BertModel` for the backbone arguments.
        name: String, model name. Defaults to `"BertQnA"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "bert"
    config_class = BertConfig
    HUB_REPO_SIBLINGS = BERT_HUB_SIBLINGS

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_bert_hf_to_keras import transfer_bert_weights

        transfer_bert_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return BertModel.config_from_hf(hf_config)

    CHECKPOINT_SOURCE = CheckpointSource(
        "BertMaskedLM", build_kwargs={"add_pooler": True}
    )

    def __init__(
        self,
        vocab_size=30522,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_act="gelu",
        norm_eps=1e-12,
        pad_token_id=0,
        dropout=0.0,
        attention_dropout=0.0,
        name="BertQnA",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes", "add_pooler"):
            kwargs.pop(k, None)
        norm_eps = kwargs.pop("layer_norm_eps", norm_eps)

        backbone = BertModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            hidden_act=hidden_act,
            norm_eps=norm_eps,
            pad_token_id=pad_token_id,
            dropout=dropout,
            attention_dropout=attention_dropout,
            add_pooler=False,
            name=f"{name}_backbone",
        )

        x = backbone.output["last_hidden_state"]
        span = layers.Dense(2, name="qa_outputs")(x)
        outputs = {"start_logits": span[:, :, 0], "end_logits": span[:, :, 1]}

        super().__init__(inputs=backbone.input, outputs=outputs, name=name, **kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
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
                "type_vocab_size": self.type_vocab_size,
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
class BertMultipleChoice(BaseModel):
    """BERT multiple-choice head (e.g. SWAG).

    Takes a dict of ``(B, num_choices, seq)`` int tensors, flattens the choices
    into the batch, runs the :class:`BertModel` backbone (with pooler), and
    scores each choice with a shared dense layer, reshaping back to per-example
    ``(B, num_choices)`` logits. The head is randomly initialized and meant for
    fine-tuning (or loaded from a fine-tuned ``hf:`` repo).

    References:
    - [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

    Args:
        See :class:`BertModel` for the backbone arguments.
        classifier_dropout: Float, dropout before the choice scorer. Defaults to `0.0`.
        name: String, model name. Defaults to `"BertMultipleChoice"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "bert"
    config_class = BertConfig
    HUB_REPO_SIBLINGS = BERT_HUB_SIBLINGS

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_bert_hf_to_keras import transfer_bert_weights

        transfer_bert_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return BertModel.config_from_hf(hf_config)

    CHECKPOINT_SOURCE = CheckpointSource(
        "BertMaskedLM", build_kwargs={"add_pooler": True}
    )

    def __init__(
        self,
        vocab_size=30522,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_act="gelu",
        norm_eps=1e-12,
        pad_token_id=0,
        dropout=0.0,
        attention_dropout=0.0,
        num_choices=4,
        classifier_dropout=0.0,
        name="BertMultipleChoice",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes", "add_pooler"):
            kwargs.pop(k, None)
        norm_eps = kwargs.pop("layer_norm_eps", norm_eps)

        input_ids = layers.Input(
            shape=(num_choices, None), dtype="int32", name="input_ids"
        )
        attention_mask = layers.Input(
            shape=(num_choices, None), dtype="int32", name="attention_mask"
        )
        token_type_ids = layers.Input(
            shape=(num_choices, None), dtype="int32", name="token_type_ids"
        )

        backbone = BertModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            hidden_act=hidden_act,
            norm_eps=norm_eps,
            pad_token_id=pad_token_id,
            dropout=dropout,
            attention_dropout=attention_dropout,
            add_pooler=True,
            name=f"{name}_backbone",
        )

        flatten = BertFlattenChoices(name="flatten_choices")
        pooled = backbone(
            {
                "input_ids": flatten(input_ids),
                "attention_mask": flatten(attention_mask),
                "token_type_ids": flatten(token_type_ids),
            }
        )["pooler_output"]
        x = layers.Dropout(classifier_dropout)(pooled)
        x = layers.Dense(1, name="classifier")(x)
        logits = BertUnflattenChoices(num_choices, name="unflatten_choices")(x)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        super().__init__(inputs=inputs, outputs=logits, name=name, **kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.num_choices = num_choices
        self.classifier_dropout = classifier_dropout

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
                "type_vocab_size": self.type_vocab_size,
                "hidden_act": self.hidden_act,
                "norm_eps": self.norm_eps,
                "pad_token_id": self.pad_token_id,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
                "num_choices": self.num_choices,
                "classifier_dropout": self.classifier_dropout,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
