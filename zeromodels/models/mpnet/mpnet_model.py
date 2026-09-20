import keras
from keras import layers, ops

from zeromodels.base import BaseModel, CheckpointSource

from .mpnet_config import MPNetConfig
from .mpnet_layers import (
    MPNetEmbeddings,
    MPNetFlattenChoices,
    MPNetRelativeAttentionBias,
    MPNetSelfAttention,
    MPNetUnflattenChoices,
)

MASK_NEG = -1e9

# All classes (encoder + masked-LM + task heads) share the variant's weights repo,
# whose zm_config.json declares the canonical MPNetModel encoder (model.weights.h5).
MPNET_HUB_SIBLINGS = frozenset(
    {
        "MPNetModel",
        "MPNetMaskedLM",
        "MPNetSequenceClassify",
        "MPNetTokenClassify",
        "MPNetQnA",
        "MPNetMultipleChoice",
    }
)

_BACKBONE_KW = {
    "vocab_size": 30527,
    "embed_dim": 768,
    "num_layers": 12,
    "num_heads": 12,
    "mlp_dim": 3072,
    "max_position_embeddings": 512,
    "relative_attention_num_buckets": 32,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-12,
    "pad_token_id": 1,
    "dropout": 0.0,
    "attention_dropout": 0.0,
}


def mpnet_encoder_layer(
    x,
    attention_mask,
    position_bias,
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
    """One MPNet transformer block: self-attention + feed-forward.

    Both sub-blocks use post-LayerNorm residuals. The attention output projection is
    inside :class:`MPNetSelfAttention` (HF's ``attention.attn.o``), so this adds only
    the residual and LayerNorm (HF's ``attention.LayerNorm``).

    Args:
        x: Input token states ``(B, seq, embed_dim)``.
        attention_mask: Additive mask ``(B, 1, 1, seq)`` (0 keep, large-negative drop).
        position_bias: Shared relative bias ``(1, num_heads, seq, seq)``.
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

    attn = MPNetSelfAttention(
        embed_dim,
        num_heads,
        attention_dropout=attention_dropout,
        block_prefix=prefix,
        name=f"{prefix}_attention_attn",
    )(x, attention_mask=attention_mask, position_bias=position_bias)
    attn = layers.Dropout(dropout)(attn)
    attn = layers.Add(name=f"{prefix}_attention_add")([attn, x])
    attn = layers.LayerNormalization(
        epsilon=layer_norm_eps, name=f"{prefix}_attention_layernorm"
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


def mpnet_backbone(
    input_ids,
    attention_mask,
    *,
    vocab_size,
    embed_dim,
    num_layers,
    num_heads,
    mlp_dim,
    max_position_embeddings,
    relative_attention_num_buckets,
    pad_token_id,
    hidden_act,
    layer_norm_eps,
    dropout,
    attention_dropout,
    add_pooler,
):
    """MPNet embeddings + transformer encoder (+ optional pooler).

    The relative position bias is built once here and shared by every layer, matching
    HF's ``MPNetEncoder.forward``.

    Returns ``(sequence_output, pooled_output)`` where ``pooled_output`` is ``None``
    when ``add_pooler`` is False.
    """
    embeddings = MPNetEmbeddings(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=pad_token_id,
        layer_norm_eps=layer_norm_eps,
        dropout=dropout,
        name="embeddings",
    )(input_ids)

    position_bias = MPNetRelativeAttentionBias(
        num_buckets=relative_attention_num_buckets,
        num_heads=num_heads,
        name="encoder_relative_attention_bias",
    )(input_ids)

    mask = ops.cast(attention_mask, "float32")
    mask = ops.expand_dims(ops.expand_dims(mask, 1), 1)
    mask = (1.0 - mask) * MASK_NEG

    x = embeddings
    for i in range(num_layers):
        x = mpnet_encoder_layer(
            x,
            mask,
            position_bias,
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


def mpnet_classification_head(sequence_output, embed_dim, num_classes, dropout_rate):
    """MPNet's sentence-level head: ``<s>`` token -> dropout -> dense/tanh -> out_proj.

    Matches HF ``MPNetClassificationHead`` (used by sequence classification), which
    pools the first token itself rather than reading the encoder's pooler.
    """
    x = sequence_output[:, 0]
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(embed_dim, activation="tanh", name="classifier_dense")(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


@keras.saving.register_keras_serializable(package="zeromodels")
class MPNetModel(BaseModel):
    """Instantiates the MPNet encoder backbone.

    MPNet embeds tokens with summed word and absolute-position embeddings (there are
    **no** token-type embeddings), then applies a stack of bidirectional transformer
    encoder layers. Each attention layer adds a **shared relative position bias**,
    computed once from ``relative_attention_num_buckets`` log-spaced buckets, on top of
    the usual scaled dot-product scores. Position ids are offset past the padding id as
    in RoBERTa. An optional pooler applies a ``tanh`` dense projection to the first
    (``<s>``) token.

    The model takes a dict of ``input_ids`` and ``attention_mask`` (both ``(B, seq)``
    int tensors, as produced by :class:`MPNetTokenizer`) and returns a dict with
    ``last_hidden_state`` ``(B, seq, embed_dim)`` and, when ``add_pooler=True``,
    ``pooler_output`` ``(B, embed_dim)``.

    References:
    - [MPNet: Masked and Permuted Pre-training for Language Understanding](https://arxiv.org/abs/2004.09297)

    Args:
        vocab_size: Integer, token vocabulary size. Defaults to `30527`.
        embed_dim: Integer, model / embedding dimension. Defaults to `768`.
        num_layers: Integer, number of transformer encoder layers. Defaults to `12`.
        num_heads: Integer, number of attention heads. Defaults to `12`.
        mlp_dim: Integer, feed-forward hidden dimension. Defaults to `3072`.
        max_position_embeddings: Integer, size of the position-embedding table.
            Defaults to `512`.
        relative_attention_num_buckets: Integer, relative-bias bucket count.
            Defaults to `32`.
        hidden_act: String, feed-forward activation. Defaults to `"gelu"`.
        layer_norm_eps: Float, LayerNorm epsilon. Defaults to `1e-12`.
        pad_token_id: Integer, padding token id (also the position offset).
            Defaults to `1`.
        dropout: Float, hidden dropout rate. Defaults to `0.0`.
        attention_dropout: Float, attention-weight dropout rate. Defaults to `0.0`.
        add_pooler: Boolean, whether to add the ``<s>`` pooler. Defaults to `True`.
        name: String, model name. Defaults to `"MPNetModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "mpnet"
    config_class = MPNetConfig
    HUB_REPO_SIBLINGS = MPNET_HUB_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource(
        "MPNetMaskedLM", build_kwargs={"add_pooler": True}
    )

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_mpnet_hf_to_keras import transfer_mpnet_weights

        transfer_mpnet_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "vocab_size": hf_config["vocab_size"],
            "embed_dim": hf_config["hidden_size"],
            "num_layers": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "mlp_dim": hf_config["intermediate_size"],
            "max_position_embeddings": hf_config["max_position_embeddings"],
            "relative_attention_num_buckets": hf_config.get(
                "relative_attention_num_buckets", 32
            ),
            "hidden_act": hf_config.get("hidden_act", "gelu"),
            "layer_norm_eps": hf_config.get("layer_norm_eps", 1e-12),
            "pad_token_id": hf_config.get("pad_token_id", 1),
        }

    def __init__(
        self,
        vocab_size=30527,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        max_position_embeddings=512,
        relative_attention_num_buckets=32,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        pad_token_id=1,
        dropout=0.0,
        attention_dropout=0.0,
        add_pooler=True,
        name="MPNetModel",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)

        # MPNet has no token-type embeddings, so the encoder takes these two inputs only.
        inputs = {
            "input_ids": layers.Input(shape=(None,), dtype="int32", name="input_ids"),
            "attention_mask": layers.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
        }
        sequence_output, pooled_output = mpnet_backbone(
            inputs["input_ids"],
            inputs["attention_mask"],
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            relative_attention_num_buckets=relative_attention_num_buckets,
            pad_token_id=pad_token_id,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
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
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
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
                "relative_attention_num_buckets": self.relative_attention_num_buckets,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
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
class MPNetMaskedLM(BaseModel):
    """MPNet with the masked-and-permuted language-modeling head.

    Wraps an :class:`MPNetModel` backbone (no pooler) and attaches MPNet's LM head -- a
    dense transform with ``gelu`` then LayerNorm, followed by a vocabulary projection --
    producing token logits ``(B, seq, vocab_size)``. The head's weights are part of the
    pretrained checkpoint, so loading restores a ready-to-use fill-mask model.

    References:
    - [MPNet: Masked and Permuted Pre-training for Language Understanding](https://arxiv.org/abs/2004.09297)

    Args:
        See :class:`MPNetModel` for the backbone arguments.
        add_pooler: Boolean, whether to also carry the backbone's ``<s>`` pooler and
            return it alongside the logits. Defaults to `False`.
        name: String, model name. Defaults to `"MPNetMaskedLM"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "mpnet"
    config_class = MPNetConfig
    HUB_REPO_SIBLINGS = MPNET_HUB_SIBLINGS

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_mpnet_hf_to_keras import transfer_mpnet_weights

        transfer_mpnet_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return MPNetModel.config_from_hf(hf_config)

    def __init__(self, add_pooler=False, name="MPNetMaskedLM", **kwargs):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)
        cfg = {**_BACKBONE_KW, **kwargs}
        self._cfg = dict(cfg)

        backbone = MPNetModel(**cfg, add_pooler=add_pooler, name=f"{name}_backbone")
        x = backbone.output["last_hidden_state"]
        x = layers.Dense(cfg["embed_dim"], name="lm_head_dense")(x)
        x = layers.Activation("gelu", name="lm_head_act")(x)
        x = layers.LayerNormalization(
            epsilon=cfg["layer_norm_eps"], name="lm_head_layernorm"
        )(x)
        logits = layers.Dense(cfg["vocab_size"], name="lm_head_decoder")(x)

        # add_pooler=True carries the pooler so a single hosted checkpoint serves
        # MPNetModel (encoder + pooler), MPNetMaskedLM (encoder + MLM head) and the task
        # heads alike; each class loads its own subset. Default False keeps the plain
        # fill-mask output.
        if add_pooler:
            outputs = {
                "logits": logits,
                "pooler_output": backbone.output["pooler_output"],
            }
        else:
            outputs = logits

        super().__init__(inputs=backbone.input, outputs=outputs, name=name)

        self.add_pooler = add_pooler

    def get_config(self):
        config = super().get_config()
        config.update({**self._cfg, "add_pooler": self.add_pooler, "name": self.name})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class MPNetSequenceClassify(BaseModel):
    """MPNet sentence/sequence classifier.

    Wraps an :class:`MPNetModel` backbone (no pooler) and attaches MPNet's
    classification head -- dropout, a ``tanh`` dense, dropout, then a projection to
    ``num_classes`` -- applied to the first (``<s>``) token, producing logits
    ``(B, num_classes)``. Note MPNet's head pools the token itself rather than reading
    the encoder pooler, matching Hugging Face. The pretrained checkpoint has no task
    head, so the classifier stays randomly initialized and ready for fine-tuning.

    References:
    - [MPNet: Masked and Permuted Pre-training for Language Understanding](https://arxiv.org/abs/2004.09297)

    Args:
        See :class:`MPNetModel` for the backbone arguments.
        num_classes: Integer, number of output classes. Defaults to `2`.
        classifier_dropout: Float, dropout inside the head. Defaults to `0.0`.
        classifier_activation: String/callable, head activation (`"linear"` for
            logits). Defaults to `"linear"`.
        name: String, model name. Defaults to `"MPNetSequenceClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "mpnet"
    config_class = MPNetConfig
    HUB_REPO_SIBLINGS = MPNET_HUB_SIBLINGS

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_mpnet_hf_to_keras import transfer_mpnet_weights

        transfer_mpnet_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        config = MPNetModel.config_from_hf(hf_config)
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
        name="MPNetSequenceClassify",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "add_pooler"):
            kwargs.pop(k, None)
        cfg = {**_BACKBONE_KW, **kwargs}
        self._cfg = dict(cfg)

        backbone = MPNetModel(**cfg, add_pooler=False, name=f"{name}_backbone")
        x = mpnet_classification_head(
            backbone.output["last_hidden_state"],
            cfg["embed_dim"],
            num_classes,
            classifier_dropout,
        )
        logits = layers.Dense(
            num_classes, activation=classifier_activation, name="classifier_out_proj"
        )(x)

        super().__init__(inputs=backbone.input, outputs=logits, name=name)

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
class MPNetTokenClassify(BaseModel):
    """MPNet token classifier (e.g. NER / POS tagging).

    Wraps an :class:`MPNetModel` backbone (no pooler) and attaches dropout plus a dense
    head applied per token, producing logits ``(B, seq, num_classes)``. The head is
    randomly initialized from the pretrained checkpoint and meant for fine-tuning.

    References:
    - [MPNet: Masked and Permuted Pre-training for Language Understanding](https://arxiv.org/abs/2004.09297)

    Args:
        See :class:`MPNetModel` for the backbone arguments.
        num_classes: Integer, number of token classes. Defaults to `2`.
        classifier_dropout: Float, dropout before the classifier. Defaults to `0.0`.
        classifier_activation: String/callable, head activation. Defaults to `"linear"`.
        name: String, model name. Defaults to `"MPNetTokenClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "mpnet"
    config_class = MPNetConfig
    HUB_REPO_SIBLINGS = MPNET_HUB_SIBLINGS

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_mpnet_hf_to_keras import transfer_mpnet_weights

        transfer_mpnet_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return MPNetSequenceClassify.config_from_hf(hf_config)

    def __init__(
        self,
        num_classes=2,
        classifier_dropout=0.0,
        classifier_activation="linear",
        name="MPNetTokenClassify",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "add_pooler"):
            kwargs.pop(k, None)
        cfg = {**_BACKBONE_KW, **kwargs}
        self._cfg = dict(cfg)

        backbone = MPNetModel(**cfg, add_pooler=False, name=f"{name}_backbone")
        x = layers.Dropout(classifier_dropout)(backbone.output["last_hidden_state"])
        logits = layers.Dense(
            num_classes, activation=classifier_activation, name="classifier"
        )(x)

        super().__init__(inputs=backbone.input, outputs=logits, name=name)

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
class MPNetQnA(BaseModel):
    """MPNet extractive question answering.

    Wraps an :class:`MPNetModel` backbone (no pooler) and attaches a dense head
    producing two logits per token, split into ``start_logits`` / ``end_logits``
    (each ``(B, seq)``). The head is randomly initialized from the pretrained
    checkpoint and meant for fine-tuning.

    References:
    - [MPNet: Masked and Permuted Pre-training for Language Understanding](https://arxiv.org/abs/2004.09297)

    Args:
        See :class:`MPNetModel` for the backbone arguments.
        name: String, model name. Defaults to `"MPNetQnA"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "mpnet"
    config_class = MPNetConfig
    HUB_REPO_SIBLINGS = MPNET_HUB_SIBLINGS

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_mpnet_hf_to_keras import transfer_mpnet_weights

        transfer_mpnet_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return MPNetModel.config_from_hf(hf_config)

    def __init__(self, name="MPNetQnA", **kwargs):
        for k in ("model", "hf_id", "url", "num_classes", "add_pooler"):
            kwargs.pop(k, None)
        cfg = {**_BACKBONE_KW, **kwargs}
        self._cfg = dict(cfg)

        backbone = MPNetModel(**cfg, add_pooler=False, name=f"{name}_backbone")
        span = layers.Dense(2, name="qa_outputs")(backbone.output["last_hidden_state"])
        start_logits = span[..., 0]
        end_logits = span[..., 1]

        super().__init__(
            inputs=backbone.input,
            outputs={"start_logits": start_logits, "end_logits": end_logits},
            name=name,
        )

    def get_config(self):
        config = super().get_config()
        config.update({**self._cfg, "name": self.name})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class MPNetMultipleChoice(BaseModel):
    """MPNet multiple-choice classifier (e.g. SWAG / RACE).

    Takes ``(B, num_choices, seq)`` inputs, folds the choice axis into the batch, runs
    the :class:`MPNetModel` backbone (with pooler), scores each choice with a single
    dense unit on the pooled ``<s>`` token, then folds the choice axis back out to give
    logits ``(B, num_choices)``. The head is randomly initialized from the pretrained
    checkpoint and meant for fine-tuning.

    References:
    - [MPNet: Masked and Permuted Pre-training for Language Understanding](https://arxiv.org/abs/2004.09297)

    Args:
        See :class:`MPNetModel` for the backbone arguments.
        num_choices: Integer, number of choices per example. Defaults to `2`.
        classifier_dropout: Float, dropout before the scorer. Defaults to `0.0`.
        name: String, model name. Defaults to `"MPNetMultipleChoice"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "mpnet"
    config_class = MPNetConfig
    HUB_REPO_SIBLINGS = MPNET_HUB_SIBLINGS

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_mpnet_hf_to_keras import transfer_mpnet_weights

        transfer_mpnet_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return MPNetModel.config_from_hf(hf_config)

    def __init__(
        self,
        num_choices=2,
        classifier_dropout=0.0,
        name="MPNetMultipleChoice",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes", "add_pooler"):
            kwargs.pop(k, None)
        cfg = {**_BACKBONE_KW, **kwargs}
        self._cfg = dict(cfg)

        inputs = {
            "input_ids": layers.Input(
                shape=(num_choices, None), dtype="int32", name="input_ids"
            ),
            "attention_mask": layers.Input(
                shape=(num_choices, None), dtype="int32", name="attention_mask"
            ),
        }
        flat = MPNetFlattenChoices(name="flatten_choices")
        backbone = MPNetModel(**cfg, add_pooler=True, name=f"{name}_backbone")
        pooled = backbone(
            {
                "input_ids": flat(inputs["input_ids"]),
                "attention_mask": flat(inputs["attention_mask"]),
            }
        )["pooler_output"]
        x = layers.Dropout(classifier_dropout)(pooled)
        scores = layers.Dense(1, name="classifier")(x)
        logits = MPNetUnflattenChoices(num_choices, name="unflatten_choices")(scores)

        super().__init__(inputs=inputs, outputs=logits, name=name)

        self.num_choices = num_choices
        self.classifier_dropout = classifier_dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                **self._cfg,
                "num_choices": self.num_choices,
                "classifier_dropout": self.classifier_dropout,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
