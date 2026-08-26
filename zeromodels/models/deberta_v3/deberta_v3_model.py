import keras
from keras import layers

from zeromodels.base import BaseModel, CheckpointSource
from zeromodels.models.deberta_v2.deberta_v2_layers import (
    DebertaV2FlattenChoices,
    DebertaV2UnflattenChoices,
)
from zeromodels.models.deberta_v2.deberta_v2_model import (
    DebertaV2Model,
    deberta_v2_backbone,
)

from .deberta_v3_config import (
    DebertaV3Config,
)

# All classes (encoder + masked-LM + task heads) share the variant's weights repo,
# whose zm_config.json declares the canonical DebertaV3Model encoder (model.weights.h5).
DEBERTA_V3_HUB_SIBLINGS = frozenset(
    {
        "DebertaV3Model",
        "DebertaV3MaskedLM",
        "DebertaV3SequenceClassify",
        "DebertaV3TokenClassify",
        "DebertaV3QnA",
        "DebertaV3MultipleChoice",
    }
)


@keras.saving.register_keras_serializable(package="zeromodels")
class DebertaV3Model(BaseModel):
    """Instantiates the DeBERTa-v3 encoder backbone.

    DeBERTa-v3 keeps DeBERTa-v2's architecture (log-bucketed disentangled
    attention, shared key/query relative projections, LayerNorm-ed relative
    embeddings) and differs only in pretraining (ELECTRA-style replaced-token
    detection) and tokenizer vocabulary. It reuses the v2 backbone with no
    convolution layer.

    Takes ``input_ids`` / ``attention_mask`` / ``token_type_ids`` and returns
    ``{"last_hidden_state": (B, seq, embed_dim)}``.

    References:
    - [DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training](https://arxiv.org/abs/2111.09543)

    Args:
        vocab_size: Integer, token vocabulary size. Defaults to `128100`.
        embed_dim: Integer, model dimension. Defaults to `768`.
        num_layers: Integer, number of encoder layers. Defaults to `12`.
        num_heads: Integer, number of attention heads. Defaults to `12`.
        mlp_dim: Integer, feed-forward hidden dimension. Defaults to `3072`.
        max_position_embeddings: Integer, maximum sequence length. Defaults to `512`.
        max_relative_positions: Integer, max absolute relative position. Defaults to `512`.
        position_buckets: Integer, relative-position bucket count. Defaults to `256`.
        pos_att_type: List of disentangled-attention terms. Defaults to `["p2c", "c2p"]`.
        norm_rel_ebd: Boolean, LayerNorm the relative embeddings. Defaults to `True`.
        conv_kernel_size: Integer, conv kernel size (0/None disables). Defaults to `0`.
        conv_act: String, conv activation. Defaults to `"gelu"`.
        hidden_act: String, feed-forward activation. Defaults to `"gelu"`.
        norm_eps: Float, LayerNorm epsilon. Defaults to `1e-7`. The deprecated
            alias `layer_norm_eps` is still accepted.
        pad_token_id: Integer, padding token id. Defaults to `0`.
        dropout: Float, hidden dropout rate. Defaults to `0.0`.
        attention_dropout: Float, attention dropout rate. Defaults to `0.0`.
        name: String, model name. Defaults to `"DebertaV3Model"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "deberta-v2"
    config_class = DebertaV3Config
    HUB_REPO_SIBLINGS = DEBERTA_V3_HUB_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource("DebertaV3Model")

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from zeromodels.models.deberta_v2.convert_deberta_v2_hf_to_keras import (
            transfer_deberta_v2_weights,
        )

        transfer_deberta_v2_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return DebertaV2Model.config_from_hf(hf_config)

    def __init__(
        self,
        vocab_size=128100,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        max_position_embeddings=512,
        max_relative_positions=512,
        position_buckets=256,
        pos_att_type=("p2c", "c2p"),
        share_att_key=True,
        norm_rel_ebd=True,
        conv_kernel_size=0,
        conv_act="gelu",
        hidden_act="gelu",
        norm_eps=1e-7,
        pad_token_id=0,
        dropout=0.0,
        attention_dropout=0.0,
        name="DebertaV3Model",
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
        # deberta_v2_backbone() takes every backbone kwarg except the model-level
        # max_position_embeddings / pad_token_id.
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
class DebertaV3MaskedLM(BaseModel):
    """DeBERTa-v3 with the masked-language-modeling head.

    Wraps a :class:`DebertaV3Model` backbone and attaches DeBERTa's MLM head, a
    dense transform with ``gelu`` + LayerNorm, then a vocabulary projection:
    producing token logits ``(B, seq, vocab_size)``. DeBERTa-v3 is ELECTRA/RTD-
    pretrained, so the checkpoint has no MLM head; ``from_weights`` restores the
    backbone and leaves the MLM head randomly initialized (ready for fine-tuning),
    mirroring Hugging Face when loading DeBERTa-v3 for masked LM.

    References:
    - [DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training](https://arxiv.org/abs/2111.09543)

    Args:
        See :class:`DebertaV3Model` for the backbone arguments.
        name: String, model name. Defaults to `"DebertaV3MaskedLM"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "deberta-v2"
    config_class = DebertaV3Config
    HUB_REPO_SIBLINGS = DEBERTA_V3_HUB_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource("DebertaV3Model")

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from zeromodels.models.deberta_v2.convert_deberta_v2_hf_to_keras import (
            transfer_deberta_v2_weights,
        )

        transfer_deberta_v2_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return DebertaV2Model.config_from_hf(hf_config)

    def __init__(self, name="DebertaV3MaskedLM", **kwargs):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes"):
            kwargs.pop(k, None)
        cfg = {
            "vocab_size": 128100,
            "embed_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "mlp_dim": 3072,
            "max_position_embeddings": 512,
            "max_relative_positions": 512,
            "position_buckets": 256,
            "pos_att_type": ("p2c", "c2p"),
            "share_att_key": True,
            "norm_rel_ebd": True,
            "conv_kernel_size": 0,
            "conv_act": "gelu",
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-7,
            "pad_token_id": 0,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            **kwargs,
        }
        backbone = DebertaV3Model(**cfg, name=f"{name}_backbone")
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
class DebertaV3SequenceClassify(BaseModel):
    """DeBERTa-v3 sentence/sequence classifier.

    Wraps a :class:`DebertaV3Model` backbone and attaches DeBERTa's context pooler
    (dense + ``gelu`` on the first token) plus a dense classifier, producing
    ``num_classes`` logits ``(B, num_classes)``. The pretrained checkpoint has no
    task head, so ``from_weights`` restores the backbone and leaves the pooler +
    classifier randomly initialized (ready for fine-tuning) unless a fine-tuned
    release is configured.

    References:
    - [DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training](https://arxiv.org/abs/2111.09543)

    Args:
        See :class:`DebertaV3Model` for the backbone arguments.
        num_classes: Integer, number of output classes. Defaults to `2`.
        pooler_dropout: Float, dropout inside the context pooler. Defaults to `0.0`.
        classifier_dropout: Float, dropout before the classifier. Defaults to `0.0`.
        classifier_activation: String/callable, head activation. Defaults to `"linear"`.
        name: String, model name. Defaults to `"DebertaV3SequenceClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "deberta-v2"
    config_class = DebertaV3Config
    HUB_REPO_SIBLINGS = DEBERTA_V3_HUB_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource("DebertaV3Model")

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from zeromodels.models.deberta_v2.convert_deberta_v2_hf_to_keras import (
            transfer_deberta_v2_weights,
        )

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

    def __init__(
        self,
        num_classes=2,
        pooler_dropout=0.0,
        classifier_dropout=0.0,
        classifier_activation="linear",
        name="DebertaV3SequenceClassify",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url"):
            kwargs.pop(k, None)
        cfg = {
            "vocab_size": 128100,
            "embed_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "mlp_dim": 3072,
            "max_position_embeddings": 512,
            "max_relative_positions": 512,
            "position_buckets": 256,
            "pos_att_type": ("p2c", "c2p"),
            "share_att_key": True,
            "norm_rel_ebd": True,
            "conv_kernel_size": 0,
            "conv_act": "gelu",
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-7,
            "pad_token_id": 0,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            **kwargs,
        }
        backbone = DebertaV3Model(**cfg, name=f"{name}_backbone")

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
class DebertaV3TokenClassify(BaseModel):
    """DeBERTa-v3 token classifier (e.g. NER / POS tagging).

    Wraps a :class:`DebertaV3Model` backbone and attaches dropout plus a per-token
    dense head, producing ``num_classes`` logits ``(B, seq, num_classes)``. The
    head is randomly initialized from the pretrained checkpoint and meant for
    fine-tuning.

    References:
    - [DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training](https://arxiv.org/abs/2111.09543)

    Args:
        See :class:`DebertaV3Model` for the backbone arguments.
        num_classes: Integer, number of token classes. Defaults to `2`.
        classifier_dropout: Float, dropout before the classifier. Defaults to `0.0`.
        classifier_activation: String/callable, head activation. Defaults to `"linear"`.
        name: String, model name. Defaults to `"DebertaV3TokenClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "deberta-v2"
    config_class = DebertaV3Config
    HUB_REPO_SIBLINGS = DEBERTA_V3_HUB_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource("DebertaV3Model")

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from zeromodels.models.deberta_v2.convert_deberta_v2_hf_to_keras import (
            transfer_deberta_v2_weights,
        )

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

    def __init__(
        self,
        num_classes=2,
        classifier_dropout=0.0,
        classifier_activation="linear",
        name="DebertaV3TokenClassify",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url"):
            kwargs.pop(k, None)
        cfg = {
            "vocab_size": 128100,
            "embed_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "mlp_dim": 3072,
            "max_position_embeddings": 512,
            "max_relative_positions": 512,
            "position_buckets": 256,
            "pos_att_type": ("p2c", "c2p"),
            "share_att_key": True,
            "norm_rel_ebd": True,
            "conv_kernel_size": 0,
            "conv_act": "gelu",
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-7,
            "pad_token_id": 0,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            **kwargs,
        }
        backbone = DebertaV3Model(**cfg, name=f"{name}_backbone")
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
class DebertaV3QnA(BaseModel):
    """DeBERTa-v3 extractive question-answering head.

    Wraps a :class:`DebertaV3Model` backbone and attaches a dense span head that
    maps each token to two logits, split into ``start_logits`` and ``end_logits``
    (each ``(B, seq)``). The head is randomly initialized from the pretrained
    checkpoint and meant for fine-tuning (or loaded from a fine-tuned ``hf:``
    repo such as a SQuAD model).

    References:
    - [DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training](https://arxiv.org/abs/2111.09543)

    Args:
        See :class:`DebertaV3Model` for the backbone arguments.
        name: String, model name. Defaults to `"DebertaV3QnA"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "deberta-v2"
    config_class = DebertaV3Config
    HUB_REPO_SIBLINGS = DEBERTA_V3_HUB_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource("DebertaV3Model")

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from zeromodels.models.deberta_v2.convert_deberta_v2_hf_to_keras import (
            transfer_deberta_v2_weights,
        )

        transfer_deberta_v2_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return DebertaV2Model.config_from_hf(hf_config)

    def __init__(self, name="DebertaV3QnA", **kwargs):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes"):
            kwargs.pop(k, None)
        cfg = {
            "vocab_size": 128100,
            "embed_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "mlp_dim": 3072,
            "max_position_embeddings": 512,
            "max_relative_positions": 512,
            "position_buckets": 256,
            "pos_att_type": ("p2c", "c2p"),
            "share_att_key": True,
            "norm_rel_ebd": True,
            "conv_kernel_size": 0,
            "conv_act": "gelu",
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-7,
            "pad_token_id": 0,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            **kwargs,
        }
        backbone = DebertaV3Model(**cfg, name=f"{name}_backbone")
        span = layers.Dense(2, name="qa_outputs")(backbone.output["last_hidden_state"])
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
class DebertaV3MultipleChoice(BaseModel):
    """DeBERTa-v3 multiple-choice head (e.g. SWAG).

    Takes a dict of ``(B, num_choices, seq)`` int tensors, flattens the choices
    into the batch, runs the :class:`DebertaV3Model` backbone, scores each choice
    with DeBERTa's context pooler (dense + ``gelu`` on the first token) plus a
    shared dense layer, and reshapes back to per-example ``(B, num_choices)``
    logits. The head is randomly initialized and meant for fine-tuning.

    References:
    - [DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training](https://arxiv.org/abs/2111.09543)

    Args:
        See :class:`DebertaV3Model` for the backbone arguments.
        num_choices: Integer, number of choices per example. Defaults to `4`.
        pooler_dropout: Float, dropout inside the context pooler. Defaults to `0.0`.
        classifier_dropout: Float, dropout before the choice scorer. Defaults to `0.0`.
        name: String, model name. Defaults to `"DebertaV3MultipleChoice"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "deberta-v2"
    config_class = DebertaV3Config
    HUB_REPO_SIBLINGS = DEBERTA_V3_HUB_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource("DebertaV3Model")

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from zeromodels.models.deberta_v2.convert_deberta_v2_hf_to_keras import (
            transfer_deberta_v2_weights,
        )

        transfer_deberta_v2_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return DebertaV2Model.config_from_hf(hf_config)

    def __init__(
        self,
        num_choices=4,
        pooler_dropout=0.0,
        classifier_dropout=0.0,
        name="DebertaV3MultipleChoice",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes"):
            kwargs.pop(k, None)
        cfg = {
            "vocab_size": 128100,
            "embed_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "mlp_dim": 3072,
            "max_position_embeddings": 512,
            "max_relative_positions": 512,
            "position_buckets": 256,
            "pos_att_type": ("p2c", "c2p"),
            "share_att_key": True,
            "norm_rel_ebd": True,
            "conv_kernel_size": 0,
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
        backbone = DebertaV3Model(**cfg, name=f"{name}_backbone")
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
