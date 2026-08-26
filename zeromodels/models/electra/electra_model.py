import keras
from keras import layers, ops

from zeromodels.base import BaseModel, CheckpointSource

from .electra_config import ElectraConfig
from .electra_layers import (
    ElectraEmbeddings,
    ElectraFlattenChoices,
    ElectraSelfAttention,
    ElectraUnflattenChoices,
)

MASK_NEG = -1e9

# ELECTRA ships two checkpoint families per size. The DISCRIMINATOR repo (kf_config
# declares ElectraModel) hosts the encoder and serves ElectraModel + the classify / QA /
# multiple-choice heads (each loads the encoder subset via CHECKPOINT_SOURCE, its own head
# random-init for fine-tuning). The GENERATOR repo (kf_config declares ElectraMaskedLM)
# hosts the masked-LM (encoder + generator head + tied decoder) and serves ElectraMaskedLM.
ELECTRA_DISCRIMINATOR_SIBLINGS = frozenset(
    {
        "ElectraModel",
        "ElectraSequenceClassify",
        "ElectraTokenClassify",
        "ElectraQnA",
        "ElectraMultipleChoice",
    }
)


BACKBONE_ATTRS = (
    "vocab_size",
    "embedding_size",
    "embed_dim",
    "num_layers",
    "num_heads",
    "mlp_dim",
    "max_position_embeddings",
    "type_vocab_size",
    "hidden_act",
    "layer_norm_eps",
    "pad_token_id",
    "dropout",
    "attention_dropout",
)


def store_backbone_attrs(model, ns):
    """Copy the shared backbone constructor args (from a caller's ``locals()``) onto
    the model so ``get_config`` can round-trip them. Every ELECTRA class takes the
    same backbone arguments, so this keeps the six ``__init__``s free of the repeated
    block of assignments; head-specific attrs are set by the caller afterwards."""
    for k in BACKBONE_ATTRS:
        setattr(model, k, ns[k])


def electra_encoder_layer(
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
    """One ELECTRA transformer block: self-attention + feed-forward.

    Both sub-blocks use post-LayerNorm residuals, matching the original BERT-style
    ELECTRA encoder (``LayerNorm(x + Sublayer(x))``).
    """
    prefix = f"blocks_{layer_idx}"

    attn = ElectraSelfAttention(
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


def electra_backbone(
    input_ids,
    attention_mask,
    token_type_ids,
    *,
    vocab_size,
    embedding_size,
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
):
    """ELECTRA embeddings (+ optional projection) + transformer encoder.

    Embeds at ``embedding_size`` and, when that differs from ``embed_dim``,
    projects up to the hidden size with ``embeddings_project`` before the encoder.
    Returns the final sequence output ``(B, seq, embed_dim)``. There is no pooler.
    """
    embeddings = ElectraEmbeddings(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=type_vocab_size,
        layer_norm_eps=layer_norm_eps,
        dropout=dropout,
        name="embeddings",
    )([input_ids, token_type_ids])

    if embedding_size != embed_dim:
        # Named without the substring "embedding" so the weight converter treats it as a
        # Dense (transposed) rather than an (untransposed) embedding table.
        x = layers.Dense(embed_dim, name="embed_project")(embeddings)
    else:
        x = embeddings

    mask = ops.cast(attention_mask, "float32")
    mask = ops.expand_dims(ops.expand_dims(mask, 1), 1)
    mask = (1.0 - mask) * MASK_NEG

    for i in range(num_layers):
        x = electra_encoder_layer(
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
    return x


def electra_config_from_hf(hf_config):
    return {
        "vocab_size": hf_config["vocab_size"],
        "embedding_size": hf_config.get("embedding_size", hf_config["hidden_size"]),
        "embed_dim": hf_config["hidden_size"],
        "num_layers": hf_config["num_hidden_layers"],
        "num_heads": hf_config["num_attention_heads"],
        "mlp_dim": hf_config["intermediate_size"],
        "max_position_embeddings": hf_config["max_position_embeddings"],
        "type_vocab_size": hf_config["type_vocab_size"],
        "hidden_act": hf_config.get("hidden_act", "gelu"),
        "layer_norm_eps": hf_config.get("layer_norm_eps", 1e-12),
        "pad_token_id": hf_config.get("pad_token_id", 0),
    }


def _num_classes_from_hf(hf_config):
    return (
        len(hf_config["id2label"])
        if "id2label" in hf_config
        else hf_config.get("num_labels", 2)
    )


def encoder_inputs(seq_shape=(None,)):
    return {
        "input_ids": layers.Input(shape=seq_shape, dtype="int32", name="input_ids"),
        "attention_mask": layers.Input(
            shape=seq_shape, dtype="int32", name="attention_mask"
        ),
        "token_type_ids": layers.Input(
            shape=seq_shape, dtype="int32", name="token_type_ids"
        ),
    }


@keras.saving.register_keras_serializable(package="zeromodels")
class ElectraModel(BaseModel):
    """Instantiates the ELECTRA encoder backbone.

    ELECTRA embeds tokens (word / absolute-position / token-type) at
    ``embedding_size``, projects up to ``embed_dim`` when the two differ, then
    applies a stack of bidirectional transformer encoder layers (multi-head
    self-attention + feed-forward, each with a post-LayerNorm residual). There is
    no pooler.

    The model takes a dict of ``input_ids``, ``attention_mask`` and
    ``token_type_ids`` (all ``(B, seq)`` int tensors, as produced by
    :class:`ElectraTokenizer`) and returns a dict with ``last_hidden_state``
    ``(B, seq, embed_dim)``.

    References:
    - [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)

    Args:
        vocab_size: Integer, token vocabulary size. Defaults to `30522`.
        embedding_size: Integer, token-embedding dimension. Defaults to `128`.
        embed_dim: Integer, hidden size. Defaults to `256`.
        num_layers: Integer, number of encoder layers. Defaults to `12`.
        num_heads: Integer, number of attention heads. Defaults to `4`.
        mlp_dim: Integer, feed-forward hidden dimension. Defaults to `1024`.
        max_position_embeddings: Integer, position-table size. Defaults to `512`.
        type_vocab_size: Integer, number of token-type ids. Defaults to `2`.
        hidden_act: String, feed-forward activation. Defaults to `"gelu"`.
        layer_norm_eps: Float, LayerNorm epsilon. Defaults to `1e-12`.
        pad_token_id: Integer, padding token id. Defaults to `0`.
        dropout: Float, hidden dropout rate. Defaults to `0.0`.
        attention_dropout: Float, attention-weight dropout rate. Defaults to `0.0`.
        name: String, model name. Defaults to `"ElectraModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "electra"
    config_class = ElectraConfig
    HUB_REPO_SIBLINGS = ELECTRA_DISCRIMINATOR_SIBLINGS

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_electra_hf_to_keras import transfer_electra_weights

        transfer_electra_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return electra_config_from_hf(hf_config)

    def __init__(
        self,
        vocab_size=30522,
        embedding_size=128,
        embed_dim=256,
        num_layers=12,
        num_heads=4,
        mlp_dim=1024,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        pad_token_id=0,
        dropout=0.0,
        attention_dropout=0.0,
        name="ElectraModel",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)

        inputs = encoder_inputs()
        sequence_output = electra_backbone(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"],
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        super().__init__(
            inputs=inputs,
            outputs={"last_hidden_state": sequence_output},
            name=name,
            **kwargs,
        )
        store_backbone_attrs(self, locals())

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embedding_size": self.embedding_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "max_position_embeddings": self.max_position_embeddings,
                "type_vocab_size": self.type_vocab_size,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
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
class ElectraMaskedLM(BaseModel):
    """ELECTRA generator with the masked-language-modeling head.

    Wraps an :class:`ElectraModel` backbone and attaches ELECTRA's generator head:
    a dense projection to ``embedding_size`` with a GELU + LayerNorm, then a
    vocabulary decoder tied to the word embeddings, producing token logits
    ``(B, seq, vocab_size)``. Loaded from the generator repo (which carries these
    weights), so ``from_weights`` restores a ready-to-use fill-mask model.

    References:
    - [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)

    Args:
        See :class:`ElectraModel` for the backbone arguments.
        name: String, model name. Defaults to `"ElectraMaskedLM"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "electra"
    config_class = ElectraConfig
    HUB_REPO_SIBLINGS = frozenset({"ElectraMaskedLM"})

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_electra_hf_to_keras import transfer_electra_weights

        transfer_electra_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return electra_config_from_hf(hf_config)

    def __init__(
        self,
        vocab_size=30522,
        embedding_size=128,
        embed_dim=256,
        num_layers=12,
        num_heads=4,
        mlp_dim=1024,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        pad_token_id=0,
        dropout=0.0,
        attention_dropout=0.0,
        name="ElectraMaskedLM",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)

        backbone = ElectraModel(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            dropout=dropout,
            attention_dropout=attention_dropout,
            name=f"{name}_backbone",
        )

        x = backbone.output["last_hidden_state"]
        x = layers.Dense(embedding_size, name="generator_dense")(x)
        x = layers.Activation("gelu", name="generator_act")(x)
        x = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="generator_layernorm"
        )(x)
        logits = layers.Dense(vocab_size, name="generator_lm_head")(x)

        super().__init__(inputs=backbone.input, outputs=logits, name=name, **kwargs)
        store_backbone_attrs(self, locals())

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embedding_size": self.embedding_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "max_position_embeddings": self.max_position_embeddings,
                "type_vocab_size": self.type_vocab_size,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
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
class ElectraSequenceClassify(BaseModel):
    """ELECTRA sentence/sequence classifier.

    Wraps an :class:`ElectraModel` backbone and attaches ELECTRA's classification
    head on the first ([CLS]) token: dropout, a dense projection with GELU,
    dropout, then a linear output layer, producing ``num_classes`` logits
    ``(B, num_classes)``. The head is randomly initialized from the discriminator
    checkpoint (ready for fine-tuning) and loads trained weights from a `hf:`
    fine-tune.

    References:
    - [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)

    Args:
        See :class:`ElectraModel` for the backbone arguments.
        num_classes: Integer, number of output classes. Defaults to `2`.
        classifier_dropout: Float, dropout in the head. Defaults to `0.0`.
        name: String, model name. Defaults to `"ElectraSequenceClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "electra"
    config_class = ElectraConfig
    HUB_REPO_SIBLINGS = ELECTRA_DISCRIMINATOR_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource("ElectraModel")

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_electra_hf_to_keras import transfer_electra_weights

        transfer_electra_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        config = electra_config_from_hf(hf_config)
        config["num_classes"] = _num_classes_from_hf(hf_config)
        return config

    def __init__(
        self,
        vocab_size=30522,
        embedding_size=128,
        embed_dim=256,
        num_layers=12,
        num_heads=4,
        mlp_dim=1024,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        pad_token_id=0,
        dropout=0.0,
        attention_dropout=0.0,
        num_classes=2,
        classifier_dropout=0.0,
        name="ElectraSequenceClassify",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url"):
            kwargs.pop(k, None)

        backbone = ElectraModel(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            dropout=dropout,
            attention_dropout=attention_dropout,
            name=f"{name}_backbone",
        )

        x = backbone.output["last_hidden_state"][:, 0]
        x = layers.Dropout(classifier_dropout)(x)
        x = layers.Dense(embed_dim, name="classifier_dense")(x)
        x = layers.Activation("gelu", name="classifier_act")(x)
        x = layers.Dropout(classifier_dropout)(x)
        logits = layers.Dense(num_classes, name="classifier_out_proj")(x)

        super().__init__(inputs=backbone.input, outputs=logits, name=name, **kwargs)
        store_backbone_attrs(self, locals())
        self.num_classes = num_classes
        self.classifier_dropout = classifier_dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embedding_size": self.embedding_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "max_position_embeddings": self.max_position_embeddings,
                "type_vocab_size": self.type_vocab_size,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
                "pad_token_id": self.pad_token_id,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
                "num_classes": self.num_classes,
                "classifier_dropout": self.classifier_dropout,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class ElectraTokenClassify(BaseModel):
    """ELECTRA token classifier (e.g. NER / POS tagging).

    Wraps an :class:`ElectraModel` backbone and attaches dropout plus a dense head
    applied per token, producing ``num_classes`` logits ``(B, seq, num_classes)``.
    The head is randomly initialized from the discriminator checkpoint and meant
    for fine-tuning.

    References:
    - [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)

    Args:
        See :class:`ElectraModel` for the backbone arguments.
        num_classes: Integer, number of token classes. Defaults to `2`.
        classifier_dropout: Float, dropout before the classifier. Defaults to `0.0`.
        name: String, model name. Defaults to `"ElectraTokenClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "electra"
    config_class = ElectraConfig
    HUB_REPO_SIBLINGS = ELECTRA_DISCRIMINATOR_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource("ElectraModel")

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_electra_hf_to_keras import transfer_electra_weights

        transfer_electra_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        config = electra_config_from_hf(hf_config)
        config["num_classes"] = _num_classes_from_hf(hf_config)
        return config

    def __init__(
        self,
        vocab_size=30522,
        embedding_size=128,
        embed_dim=256,
        num_layers=12,
        num_heads=4,
        mlp_dim=1024,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        pad_token_id=0,
        dropout=0.0,
        attention_dropout=0.0,
        num_classes=2,
        classifier_dropout=0.0,
        name="ElectraTokenClassify",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url"):
            kwargs.pop(k, None)

        backbone = ElectraModel(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            dropout=dropout,
            attention_dropout=attention_dropout,
            name=f"{name}_backbone",
        )

        x = backbone.output["last_hidden_state"]
        x = layers.Dropout(classifier_dropout)(x)
        logits = layers.Dense(num_classes, name="classifier")(x)

        super().__init__(inputs=backbone.input, outputs=logits, name=name, **kwargs)
        store_backbone_attrs(self, locals())
        self.num_classes = num_classes
        self.classifier_dropout = classifier_dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embedding_size": self.embedding_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "max_position_embeddings": self.max_position_embeddings,
                "type_vocab_size": self.type_vocab_size,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
                "pad_token_id": self.pad_token_id,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
                "num_classes": self.num_classes,
                "classifier_dropout": self.classifier_dropout,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class ElectraQnA(BaseModel):
    """ELECTRA extractive question-answering head.

    Wraps an :class:`ElectraModel` backbone and attaches a dense span head that
    maps each token to two logits, split into ``start_logits`` and ``end_logits``
    (each ``(B, seq)``). The head is randomly initialized from the discriminator
    checkpoint and meant for fine-tuning (or loaded from a fine-tuned `hf:` repo).

    References:
    - [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)

    Args:
        See :class:`ElectraModel` for the backbone arguments.
        name: String, model name. Defaults to `"ElectraQnA"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "electra"
    config_class = ElectraConfig
    HUB_REPO_SIBLINGS = ELECTRA_DISCRIMINATOR_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource("ElectraModel")

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_electra_hf_to_keras import transfer_electra_weights

        transfer_electra_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return electra_config_from_hf(hf_config)

    def __init__(
        self,
        vocab_size=30522,
        embedding_size=128,
        embed_dim=256,
        num_layers=12,
        num_heads=4,
        mlp_dim=1024,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        pad_token_id=0,
        dropout=0.0,
        attention_dropout=0.0,
        name="ElectraQnA",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)

        backbone = ElectraModel(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            dropout=dropout,
            attention_dropout=attention_dropout,
            name=f"{name}_backbone",
        )

        x = backbone.output["last_hidden_state"]
        span = layers.Dense(2, name="qa_outputs")(x)
        outputs = {"start_logits": span[:, :, 0], "end_logits": span[:, :, 1]}

        super().__init__(inputs=backbone.input, outputs=outputs, name=name, **kwargs)
        store_backbone_attrs(self, locals())

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embedding_size": self.embedding_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "max_position_embeddings": self.max_position_embeddings,
                "type_vocab_size": self.type_vocab_size,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
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
class ElectraMultipleChoice(BaseModel):
    """ELECTRA multiple-choice head (e.g. SWAG).

    Takes a dict of ``(B, num_choices, seq)`` int tensors, flattens the choices
    into the batch, runs the :class:`ElectraModel` backbone, summarizes each choice
    from its first token with a dense + GELU + dropout, scores it with a shared
    dense layer, and reshapes back to per-example ``(B, num_choices)`` logits. The
    head is randomly initialized and meant for fine-tuning.

    References:
    - [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)

    Args:
        See :class:`ElectraModel` for the backbone arguments.
        classifier_dropout: Float, dropout in the summary head. Defaults to `0.1`.
        name: String, model name. Defaults to `"ElectraMultipleChoice"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "electra"
    config_class = ElectraConfig
    HUB_REPO_SIBLINGS = ELECTRA_DISCRIMINATOR_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource("ElectraModel")

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_electra_hf_to_keras import transfer_electra_weights

        transfer_electra_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return electra_config_from_hf(hf_config)

    def __init__(
        self,
        vocab_size=30522,
        embedding_size=128,
        embed_dim=256,
        num_layers=12,
        num_heads=4,
        mlp_dim=1024,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        pad_token_id=0,
        dropout=0.0,
        attention_dropout=0.0,
        num_choices=4,
        classifier_dropout=0.1,
        name="ElectraMultipleChoice",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)

        input_ids = layers.Input(
            shape=(num_choices, None), dtype="int32", name="input_ids"
        )
        attention_mask = layers.Input(
            shape=(num_choices, None), dtype="int32", name="attention_mask"
        )
        token_type_ids = layers.Input(
            shape=(num_choices, None), dtype="int32", name="token_type_ids"
        )

        backbone = ElectraModel(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            dropout=dropout,
            attention_dropout=attention_dropout,
            name=f"{name}_backbone",
        )

        flatten = ElectraFlattenChoices(name="flatten_choices")
        sequence_output = backbone(
            {
                "input_ids": flatten(input_ids),
                "attention_mask": flatten(attention_mask),
                "token_type_ids": flatten(token_type_ids),
            }
        )["last_hidden_state"]
        pooled = sequence_output[:, 0]
        pooled = layers.Dense(embed_dim, name="summary")(pooled)
        pooled = layers.Activation("gelu", name="summary_act")(pooled)
        pooled = layers.Dropout(classifier_dropout)(pooled)
        score = layers.Dense(1, name="classifier")(pooled)
        logits = ElectraUnflattenChoices(num_choices, name="unflatten_choices")(score)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        super().__init__(inputs=inputs, outputs=logits, name=name, **kwargs)
        store_backbone_attrs(self, locals())
        self.num_choices = num_choices
        self.classifier_dropout = classifier_dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embedding_size": self.embedding_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "max_position_embeddings": self.max_position_embeddings,
                "type_vocab_size": self.type_vocab_size,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
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
