import math

import keras
from keras import layers, ops

from zeromodels.base import BaseModel, CheckpointSource

from .modernbert_config import ModernBertConfig
from .modernbert_layers import (
    ModernBertAttention,
    ModernBertEmbeddings,
    ModernBertFlattenChoices,
    ModernBertMLP,
    ModernBertUnflattenChoices,
)

# All ModernBERT classes (encoder + masked-LM + task heads) share the variant's weights
# repo. The single hosted model.weights.h5 is the FULL masked-LM checkpoint (encoder +
# prediction head + decoder), saved from ModernBertMaskedLM; each class loads its own
# subset (the shared head_dense / head_norm feed the classify heads too, matching HF; the
# random classifier / span head stay untrained). The zm_config.json declares the canonical
# ModernBertModel.
MODERNBERT_HUB_SIBLINGS = frozenset(
    {
        "ModernBertModel",
        "ModernBertMaskedLM",
        "ModernBertSequenceClassify",
        "ModernBertTokenClassify",
        "ModernBertQnA",
        "ModernBertMultipleChoice",
    }
)


def rope_cos_sin(positions, head_dim, theta):
    """Rotary ``cos`` / ``sin`` tables ``(B, 1, seq, head_dim)`` for a given theta.

    ``positions`` is a ``(B, seq)`` float tensor of absolute positions; the
    inverse frequencies are derived from ``head_dim`` and ``theta`` (a constant),
    so only the position-dependent part flows from the input.
    """
    dim_range = ops.cast(ops.arange(0, head_dim, 2), "float32")
    inv_freq = ops.exp(dim_range * (-math.log(theta) / head_dim))
    freqs = ops.expand_dims(positions, -1) * ops.reshape(inv_freq, (1, 1, -1))
    emb = ops.concatenate([freqs, freqs], axis=-1)
    cos = ops.expand_dims(ops.cos(emb), 1)
    sin = ops.expand_dims(ops.sin(emb), 1)
    return cos, sin


def modernbert_head(x, embed_dim, classifier_activation, norm_eps):
    """ModernBERT prediction head: dense (no bias) + activation + LayerNorm."""
    x = layers.Dense(embed_dim, use_bias=False, name="head_dense")(x)
    x = layers.Activation(classifier_activation, name="head_act")(x)
    x = layers.LayerNormalization(epsilon=norm_eps, center=False, name="head_norm")(x)
    return x


def pool_hidden(sequence_output, attention_mask, classifier_pooling):
    """Pool token states to one vector per example (``"cls"`` or masked ``"mean"``)."""
    if classifier_pooling == "cls":
        return sequence_output[:, 0]
    mask = ops.expand_dims(ops.cast(attention_mask, "float32"), -1)
    return ops.sum(sequence_output * mask, axis=1) / ops.sum(mask, axis=1)


def modernbert_encoder_layer(
    x,
    attention_mask,
    rope_cos,
    rope_sin,
    *,
    embed_dim,
    num_heads,
    mlp_dim,
    hidden_act,
    norm_eps,
    dropout,
    attention_dropout,
    layer_idx,
):
    """One ModernBERT block: pre-norm rotary attention + pre-norm GeGLU MLP.

    Both sub-blocks use pre-LayerNorm residuals (``x + Sublayer(Norm(x))``). The
    first layer skips the attention LayerNorm (the embeddings are already
    normalized), matching Hugging Face's ``attn_norm = Identity`` for layer 0.
    """
    prefix = f"blocks_{layer_idx}"

    if layer_idx == 0:
        attn_in = x
    else:
        attn_in = layers.LayerNormalization(
            epsilon=norm_eps, center=False, name=f"{prefix}_attn_norm"
        )(x)
    attn = ModernBertAttention(
        embed_dim,
        num_heads,
        attention_dropout=attention_dropout,
        block_prefix=f"{prefix}_attn",
        name=f"{prefix}_attn",
    )(attn_in, attention_mask=attention_mask, rope_cos=rope_cos, rope_sin=rope_sin)
    x = layers.Add(name=f"{prefix}_attn_add")([x, attn])

    mlp_in = layers.LayerNormalization(
        epsilon=norm_eps, center=False, name=f"{prefix}_mlp_norm"
    )(x)
    mlp = ModernBertMLP(
        embed_dim,
        mlp_dim,
        hidden_act=hidden_act,
        dropout=dropout,
        block_prefix=f"{prefix}_mlp",
        name=f"{prefix}_mlp",
    )(mlp_in)
    x = layers.Add(name=f"{prefix}_mlp_add")([x, mlp])
    return x


def modernbert_backbone(
    input_ids,
    attention_mask,
    *,
    vocab_size,
    embed_dim,
    num_layers,
    num_heads,
    mlp_dim,
    hidden_act,
    norm_eps,
    local_attention,
    global_attn_every_n_layers,
    global_rope_theta,
    local_rope_theta,
    dropout,
    attention_dropout,
):
    """ModernBERT embeddings + alternating local/global encoder + final norm.

    Global layers (every ``global_attn_every_n_layers``) use full attention with
    ``global_rope_theta``; the rest use a sliding window of ``local_attention``
    tokens with ``local_rope_theta``. Returns the final-normalized sequence
    output ``(B, seq, embed_dim)``.
    """
    head_dim = embed_dim // num_heads
    x = ModernBertEmbeddings(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        norm_eps=norm_eps,
        dropout=dropout,
        name="embeddings",
    )(input_ids)

    m = ops.cast(attention_mask, "float32")
    positions = ops.cast(ops.cumsum(ops.ones_like(input_ids), axis=1) - 1, "float32")

    global_cos, global_sin = rope_cos_sin(positions, head_dim, global_rope_theta)
    local_cos, local_sin = rope_cos_sin(positions, head_dim, local_rope_theta)

    global_mask = ops.expand_dims(ops.expand_dims((1.0 - m) * -1e9, 1), 1)
    window = local_attention // 2
    dist = ops.abs(ops.expand_dims(positions, -1) - ops.expand_dims(positions, 1))
    within = ops.cast(dist <= window, "float32")
    keep_local = within * ops.expand_dims(m, 1)
    local_mask = ops.expand_dims((1.0 - keep_local) * -1e9, 1)

    for i in range(num_layers):
        is_global = (i % global_attn_every_n_layers) == 0
        x = modernbert_encoder_layer(
            x,
            global_mask if is_global else local_mask,
            global_cos if is_global else local_cos,
            global_sin if is_global else local_sin,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            hidden_act=hidden_act,
            norm_eps=norm_eps,
            dropout=dropout,
            attention_dropout=attention_dropout,
            layer_idx=i,
        )

    return layers.LayerNormalization(epsilon=norm_eps, center=False, name="final_norm")(
        x
    )


@keras.saving.register_keras_serializable(package="zeromodels")
class ModernBertModel(BaseModel):
    """Instantiates the ModernBERT encoder backbone.

    ModernBERT embeds tokens (no absolute-position or token-type embeddings) and
    applies a stack of pre-LayerNorm transformer blocks with rotary position
    embeddings, GeGLU feed-forwards, and attention that alternates between a
    global (full) layer every ``global_attn_every_n_layers`` and local
    sliding-window layers, followed by a final LayerNorm. There is no pooler.

    The model takes a dict of ``input_ids`` and ``attention_mask`` (both
    ``(B, seq)`` int tensors, as produced by :class:`ModernBertTokenizer`) and
    returns a dict with ``last_hidden_state`` ``(B, seq, embed_dim)``.

    References:
    - [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663)

    Args:
        vocab_size: Integer, token vocabulary size. Defaults to `50368`.
        embed_dim: Integer, model / embedding dimension. Defaults to `768`.
        num_layers: Integer, number of transformer encoder layers. Defaults to `22`.
        num_heads: Integer, number of attention heads. Defaults to `12`.
        mlp_dim: Integer, GeGLU hidden dimension. Defaults to `1152`.
        max_position_embeddings: Integer, maximum supported sequence length.
            Defaults to `8192`.
        hidden_act: String, GeGLU activation. Defaults to `"gelu"`.
        norm_eps: Float, LayerNorm epsilon. Defaults to `1e-5`.
        local_attention: Integer, total sliding-window size of local layers.
            Defaults to `128`.
        global_attn_every_n_layers: Integer, period of global (full-attention)
            layers. Defaults to `3`.
        global_rope_theta: Float, RoPE base for global layers. Defaults to `160000.0`.
        local_rope_theta: Float, RoPE base for local layers. Defaults to `10000.0`.
        pad_token_id: Integer, padding token id. Defaults to `50283`.
        dropout: Float, hidden dropout rate. Defaults to `0.0`.
        attention_dropout: Float, attention-weight dropout rate. Defaults to `0.0`.
        name: String, model name. Defaults to `"ModernBertModel"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "modernbert"
    config_class = ModernBertConfig
    HUB_REPO_SIBLINGS = MODERNBERT_HUB_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource("ModernBertMaskedLM")

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_modernbert_hf_to_keras import transfer_modernbert_weights

        transfer_modernbert_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "vocab_size": hf_config["vocab_size"],
            "embed_dim": hf_config["hidden_size"],
            "num_layers": hf_config["num_hidden_layers"],
            "num_heads": hf_config["num_attention_heads"],
            "mlp_dim": hf_config["intermediate_size"],
            "max_position_embeddings": hf_config["max_position_embeddings"],
            "hidden_act": hf_config.get("hidden_activation", "gelu"),
            "norm_eps": hf_config.get("norm_eps", 1e-5),
            "local_attention": hf_config.get("local_attention", 128),
            "global_attn_every_n_layers": hf_config.get(
                "global_attn_every_n_layers", 3
            ),
            "global_rope_theta": hf_config.get("global_rope_theta", 160000.0),
            "local_rope_theta": hf_config.get("local_rope_theta", 10000.0),
            "pad_token_id": hf_config.get("pad_token_id", 50283),
        }

    def __init__(
        self,
        vocab_size=50368,
        embed_dim=768,
        num_layers=22,
        num_heads=12,
        mlp_dim=1152,
        max_position_embeddings=8192,
        hidden_act="gelu",
        norm_eps=1e-5,
        local_attention=128,
        global_attn_every_n_layers=3,
        global_rope_theta=160000.0,
        local_rope_theta=10000.0,
        pad_token_id=50283,
        dropout=0.0,
        attention_dropout=0.0,
        name="ModernBertModel",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes"):
            kwargs.pop(k, None)

        inputs = {
            "input_ids": layers.Input(shape=(None,), dtype="int32", name="input_ids"),
            "attention_mask": layers.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
        }
        sequence_output = modernbert_backbone(
            inputs["input_ids"],
            inputs["attention_mask"],
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            hidden_act=hidden_act,
            norm_eps=norm_eps,
            local_attention=local_attention,
            global_attn_every_n_layers=global_attn_every_n_layers,
            global_rope_theta=global_rope_theta,
            local_rope_theta=local_rope_theta,
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
        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        self.local_attention = local_attention
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.global_rope_theta = global_rope_theta
        self.local_rope_theta = local_rope_theta
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
                "hidden_act": self.hidden_act,
                "norm_eps": self.norm_eps,
                "local_attention": self.local_attention,
                "global_attn_every_n_layers": self.global_attn_every_n_layers,
                "global_rope_theta": self.global_rope_theta,
                "local_rope_theta": self.local_rope_theta,
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
class ModernBertMaskedLM(BaseModel):
    """ModernBERT with the masked-language-modeling head.

    Wraps a :class:`ModernBertModel` backbone and attaches ModernBERT's MLM head,
    a prediction head (dense + activation + LayerNorm) then a vocabulary decoder,
    producing token logits ``(B, seq, vocab_size)``. The head and decoder weights
    are part of the pretrained checkpoint, so ``from_weights`` restores a
    ready-to-use fill-mask model.

    References:
    - [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663)

    Args:
        See :class:`ModernBertModel` for the backbone arguments.
        classifier_activation: String/callable, prediction-head activation.
            Defaults to `"gelu"`.
        name: String, model name. Defaults to `"ModernBertMaskedLM"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "modernbert"
    config_class = ModernBertConfig
    HUB_REPO_SIBLINGS = MODERNBERT_HUB_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource("ModernBertMaskedLM")

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_modernbert_hf_to_keras import transfer_modernbert_weights

        transfer_modernbert_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return ModernBertModel.config_from_hf(hf_config)

    def __init__(
        self,
        vocab_size=50368,
        embed_dim=768,
        num_layers=22,
        num_heads=12,
        mlp_dim=1152,
        max_position_embeddings=8192,
        hidden_act="gelu",
        norm_eps=1e-5,
        local_attention=128,
        global_attn_every_n_layers=3,
        global_rope_theta=160000.0,
        local_rope_theta=10000.0,
        pad_token_id=50283,
        dropout=0.0,
        attention_dropout=0.0,
        classifier_activation="gelu",
        name="ModernBertMaskedLM",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes"):
            kwargs.pop(k, None)

        backbone = ModernBertModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            hidden_act=hidden_act,
            norm_eps=norm_eps,
            local_attention=local_attention,
            global_attn_every_n_layers=global_attn_every_n_layers,
            global_rope_theta=global_rope_theta,
            local_rope_theta=local_rope_theta,
            pad_token_id=pad_token_id,
            dropout=dropout,
            attention_dropout=attention_dropout,
            name=f"{name}_backbone",
        )

        x = backbone.output["last_hidden_state"]
        x = modernbert_head(x, embed_dim, classifier_activation, norm_eps)
        logits = layers.Dense(vocab_size, name="mlm_decoder")(x)

        super().__init__(inputs=backbone.input, outputs=logits, name=name, **kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        self.local_attention = local_attention
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.global_rope_theta = global_rope_theta
        self.local_rope_theta = local_rope_theta
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.attention_dropout = attention_dropout
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
                "hidden_act": self.hidden_act,
                "norm_eps": self.norm_eps,
                "local_attention": self.local_attention,
                "global_attn_every_n_layers": self.global_attn_every_n_layers,
                "global_rope_theta": self.global_rope_theta,
                "local_rope_theta": self.local_rope_theta,
                "pad_token_id": self.pad_token_id,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
                "classifier_activation": self.classifier_activation,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class ModernBertSequenceClassify(BaseModel):
    """ModernBERT sentence/sequence classifier.

    Wraps a :class:`ModernBertModel` backbone, pools the token states
    (``classifier_pooling`` ``"cls"`` or ``"mean"``), applies the prediction head
    + dropout, and a dense classifier, producing ``num_classes`` logits
    ``(B, num_classes)``. The classifier is randomly initialized from the
    pretrained checkpoint and meant for fine-tuning.

    References:
    - [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663)

    Args:
        See :class:`ModernBertModel` for the backbone arguments.
        num_classes: Integer, number of output classes. Defaults to `2`.
        classifier_pooling: String, ``"cls"`` or ``"mean"``. Defaults to `"mean"`.
        classifier_dropout: Float, dropout before the classifier. Defaults to `0.0`.
        classifier_activation: String/callable, prediction-head activation.
            Defaults to `"gelu"`.
        name: String, model name. Defaults to `"ModernBertSequenceClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "modernbert"
    config_class = ModernBertConfig
    HUB_REPO_SIBLINGS = MODERNBERT_HUB_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource("ModernBertMaskedLM")

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_modernbert_hf_to_keras import transfer_modernbert_weights

        transfer_modernbert_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        config = ModernBertModel.config_from_hf(hf_config)
        config["num_classes"] = (
            len(hf_config["id2label"])
            if "id2label" in hf_config
            else hf_config.get("num_labels", 2)
        )
        config["classifier_pooling"] = hf_config.get("classifier_pooling", "mean")
        config["classifier_activation"] = hf_config.get("classifier_activation", "gelu")
        return config

    def __init__(
        self,
        vocab_size=50368,
        embed_dim=768,
        num_layers=22,
        num_heads=12,
        mlp_dim=1152,
        max_position_embeddings=8192,
        hidden_act="gelu",
        norm_eps=1e-5,
        local_attention=128,
        global_attn_every_n_layers=3,
        global_rope_theta=160000.0,
        local_rope_theta=10000.0,
        pad_token_id=50283,
        dropout=0.0,
        attention_dropout=0.0,
        num_classes=2,
        classifier_pooling="mean",
        classifier_dropout=0.0,
        classifier_activation="gelu",
        name="ModernBertSequenceClassify",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url"):
            kwargs.pop(k, None)

        backbone = ModernBertModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            hidden_act=hidden_act,
            norm_eps=norm_eps,
            local_attention=local_attention,
            global_attn_every_n_layers=global_attn_every_n_layers,
            global_rope_theta=global_rope_theta,
            local_rope_theta=local_rope_theta,
            pad_token_id=pad_token_id,
            dropout=dropout,
            attention_dropout=attention_dropout,
            name=f"{name}_backbone",
        )

        sequence_output = backbone.output["last_hidden_state"]
        pooled = pool_hidden(
            sequence_output, backbone.input["attention_mask"], classifier_pooling
        )
        pooled = modernbert_head(pooled, embed_dim, classifier_activation, norm_eps)
        pooled = layers.Dropout(classifier_dropout)(pooled)
        logits = layers.Dense(num_classes, name="classifier")(pooled)

        super().__init__(inputs=backbone.input, outputs=logits, name=name, **kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        self.local_attention = local_attention
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.global_rope_theta = global_rope_theta
        self.local_rope_theta = local_rope_theta
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.num_classes = num_classes
        self.classifier_pooling = classifier_pooling
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
                "hidden_act": self.hidden_act,
                "norm_eps": self.norm_eps,
                "local_attention": self.local_attention,
                "global_attn_every_n_layers": self.global_attn_every_n_layers,
                "global_rope_theta": self.global_rope_theta,
                "local_rope_theta": self.local_rope_theta,
                "pad_token_id": self.pad_token_id,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
                "num_classes": self.num_classes,
                "classifier_pooling": self.classifier_pooling,
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
class ModernBertTokenClassify(BaseModel):
    """ModernBERT token classifier (e.g. NER / POS tagging).

    Wraps a :class:`ModernBertModel` backbone and applies the prediction head +
    dropout + a dense head per token, producing ``num_classes`` logits
    ``(B, seq, num_classes)``. The classifier is randomly initialized and meant
    for fine-tuning.

    References:
    - [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663)

    Args:
        See :class:`ModernBertModel` for the backbone arguments.
        num_classes: Integer, number of token classes. Defaults to `2`.
        classifier_dropout: Float, dropout before the classifier. Defaults to `0.0`.
        classifier_activation: String/callable, prediction-head activation.
            Defaults to `"gelu"`.
        name: String, model name. Defaults to `"ModernBertTokenClassify"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "modernbert"
    config_class = ModernBertConfig
    HUB_REPO_SIBLINGS = MODERNBERT_HUB_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource("ModernBertMaskedLM")

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_modernbert_hf_to_keras import transfer_modernbert_weights

        transfer_modernbert_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        config = ModernBertModel.config_from_hf(hf_config)
        config["num_classes"] = (
            len(hf_config["id2label"])
            if "id2label" in hf_config
            else hf_config.get("num_labels", 2)
        )
        config["classifier_activation"] = hf_config.get("classifier_activation", "gelu")
        return config

    def __init__(
        self,
        vocab_size=50368,
        embed_dim=768,
        num_layers=22,
        num_heads=12,
        mlp_dim=1152,
        max_position_embeddings=8192,
        hidden_act="gelu",
        norm_eps=1e-5,
        local_attention=128,
        global_attn_every_n_layers=3,
        global_rope_theta=160000.0,
        local_rope_theta=10000.0,
        pad_token_id=50283,
        dropout=0.0,
        attention_dropout=0.0,
        num_classes=2,
        classifier_dropout=0.0,
        classifier_activation="gelu",
        name="ModernBertTokenClassify",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url"):
            kwargs.pop(k, None)

        backbone = ModernBertModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            hidden_act=hidden_act,
            norm_eps=norm_eps,
            local_attention=local_attention,
            global_attn_every_n_layers=global_attn_every_n_layers,
            global_rope_theta=global_rope_theta,
            local_rope_theta=local_rope_theta,
            pad_token_id=pad_token_id,
            dropout=dropout,
            attention_dropout=attention_dropout,
            name=f"{name}_backbone",
        )

        x = backbone.output["last_hidden_state"]
        x = modernbert_head(x, embed_dim, classifier_activation, norm_eps)
        x = layers.Dropout(classifier_dropout)(x)
        logits = layers.Dense(num_classes, name="classifier")(x)

        super().__init__(inputs=backbone.input, outputs=logits, name=name, **kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        self.local_attention = local_attention
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.global_rope_theta = global_rope_theta
        self.local_rope_theta = local_rope_theta
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
                "hidden_act": self.hidden_act,
                "norm_eps": self.norm_eps,
                "local_attention": self.local_attention,
                "global_attn_every_n_layers": self.global_attn_every_n_layers,
                "global_rope_theta": self.global_rope_theta,
                "local_rope_theta": self.local_rope_theta,
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
class ModernBertQnA(BaseModel):
    """ModernBERT extractive question-answering head.

    Wraps a :class:`ModernBertModel` backbone, applies the prediction head +
    dropout, then a dense span head mapping each token to two logits, split into
    ``start_logits`` and ``end_logits`` (each ``(B, seq)``). The head is randomly
    initialized and meant for fine-tuning (or loaded from a fine-tuned ``hf:``
    repo).

    References:
    - [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663)

    Args:
        See :class:`ModernBertModel` for the backbone arguments.
        classifier_dropout: Float, dropout before the span head. Defaults to `0.0`.
        classifier_activation: String/callable, prediction-head activation.
            Defaults to `"gelu"`.
        name: String, model name. Defaults to `"ModernBertQnA"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "modernbert"
    config_class = ModernBertConfig
    HUB_REPO_SIBLINGS = MODERNBERT_HUB_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource("ModernBertMaskedLM")

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_modernbert_hf_to_keras import transfer_modernbert_weights

        transfer_modernbert_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        config = ModernBertModel.config_from_hf(hf_config)
        config["classifier_activation"] = hf_config.get("classifier_activation", "gelu")
        return config

    def __init__(
        self,
        vocab_size=50368,
        embed_dim=768,
        num_layers=22,
        num_heads=12,
        mlp_dim=1152,
        max_position_embeddings=8192,
        hidden_act="gelu",
        norm_eps=1e-5,
        local_attention=128,
        global_attn_every_n_layers=3,
        global_rope_theta=160000.0,
        local_rope_theta=10000.0,
        pad_token_id=50283,
        dropout=0.0,
        attention_dropout=0.0,
        classifier_dropout=0.0,
        classifier_activation="gelu",
        name="ModernBertQnA",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes"):
            kwargs.pop(k, None)

        backbone = ModernBertModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            hidden_act=hidden_act,
            norm_eps=norm_eps,
            local_attention=local_attention,
            global_attn_every_n_layers=global_attn_every_n_layers,
            global_rope_theta=global_rope_theta,
            local_rope_theta=local_rope_theta,
            pad_token_id=pad_token_id,
            dropout=dropout,
            attention_dropout=attention_dropout,
            name=f"{name}_backbone",
        )

        x = backbone.output["last_hidden_state"]
        x = modernbert_head(x, embed_dim, classifier_activation, norm_eps)
        x = layers.Dropout(classifier_dropout)(x)
        span = layers.Dense(2, name="classifier")(x)
        outputs = {"start_logits": span[:, :, 0], "end_logits": span[:, :, 1]}

        super().__init__(inputs=backbone.input, outputs=outputs, name=name, **kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        self.local_attention = local_attention
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.global_rope_theta = global_rope_theta
        self.local_rope_theta = local_rope_theta
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.attention_dropout = attention_dropout
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
                "hidden_act": self.hidden_act,
                "norm_eps": self.norm_eps,
                "local_attention": self.local_attention,
                "global_attn_every_n_layers": self.global_attn_every_n_layers,
                "global_rope_theta": self.global_rope_theta,
                "local_rope_theta": self.local_rope_theta,
                "pad_token_id": self.pad_token_id,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
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
class ModernBertMultipleChoice(BaseModel):
    """ModernBERT multiple-choice head (e.g. SWAG).

    Takes a dict of ``(B, num_choices, seq)`` int tensors, flattens the choices
    into the batch, runs the :class:`ModernBertModel` backbone, pools each choice
    (``classifier_pooling``), scores it with the prediction head + a shared dense
    layer, and reshapes back to per-example ``(B, num_choices)`` logits. The head
    is randomly initialized and meant for fine-tuning.

    References:
    - [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663)

    Args:
        See :class:`ModernBertModel` for the backbone arguments.
        classifier_pooling: String, ``"cls"`` or ``"mean"``. Defaults to `"mean"`.
        classifier_dropout: Float, dropout before the choice scorer. Defaults to `0.0`.
        classifier_activation: String/callable, prediction-head activation.
            Defaults to `"gelu"`.
        name: String, model name. Defaults to `"ModernBertMultipleChoice"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "modernbert"
    config_class = ModernBertConfig
    HUB_REPO_SIBLINGS = MODERNBERT_HUB_SIBLINGS
    CHECKPOINT_SOURCE = CheckpointSource("ModernBertMaskedLM")

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_modernbert_hf_to_keras import transfer_modernbert_weights

        transfer_modernbert_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        config = ModernBertModel.config_from_hf(hf_config)
        config["classifier_pooling"] = hf_config.get("classifier_pooling", "mean")
        config["classifier_activation"] = hf_config.get("classifier_activation", "gelu")
        return config

    def __init__(
        self,
        vocab_size=50368,
        embed_dim=768,
        num_layers=22,
        num_heads=12,
        mlp_dim=1152,
        max_position_embeddings=8192,
        hidden_act="gelu",
        norm_eps=1e-5,
        local_attention=128,
        global_attn_every_n_layers=3,
        global_rope_theta=160000.0,
        local_rope_theta=10000.0,
        pad_token_id=50283,
        dropout=0.0,
        attention_dropout=0.0,
        num_choices=4,
        classifier_pooling="mean",
        classifier_dropout=0.0,
        classifier_activation="gelu",
        name="ModernBertMultipleChoice",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "mlm_url", "num_classes"):
            kwargs.pop(k, None)

        input_ids = layers.Input(
            shape=(num_choices, None), dtype="int32", name="input_ids"
        )
        attention_mask = layers.Input(
            shape=(num_choices, None), dtype="int32", name="attention_mask"
        )

        backbone = ModernBertModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            max_position_embeddings=max_position_embeddings,
            hidden_act=hidden_act,
            norm_eps=norm_eps,
            local_attention=local_attention,
            global_attn_every_n_layers=global_attn_every_n_layers,
            global_rope_theta=global_rope_theta,
            local_rope_theta=local_rope_theta,
            pad_token_id=pad_token_id,
            dropout=dropout,
            attention_dropout=attention_dropout,
            name=f"{name}_backbone",
        )

        flatten = ModernBertFlattenChoices(name="flatten_choices")
        flat_mask = flatten(attention_mask)
        sequence_output = backbone(
            {"input_ids": flatten(input_ids), "attention_mask": flat_mask}
        )["last_hidden_state"]
        pooled = pool_hidden(sequence_output, flat_mask, classifier_pooling)
        pooled = modernbert_head(pooled, embed_dim, classifier_activation, norm_eps)
        pooled = layers.Dropout(classifier_dropout)(pooled)
        score = layers.Dense(1, name="classifier")(pooled)
        logits = ModernBertUnflattenChoices(num_choices, name="unflatten_choices")(
            score
        )

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        super().__init__(inputs=inputs, outputs=logits, name=name, **kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        self.local_attention = local_attention
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.global_rope_theta = global_rope_theta
        self.local_rope_theta = local_rope_theta
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.num_choices = num_choices
        self.classifier_pooling = classifier_pooling
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
                "hidden_act": self.hidden_act,
                "norm_eps": self.norm_eps,
                "local_attention": self.local_attention,
                "global_attn_every_n_layers": self.global_attn_every_n_layers,
                "global_rope_theta": self.global_rope_theta,
                "local_rope_theta": self.local_rope_theta,
                "pad_token_id": self.pad_token_id,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
                "num_choices": self.num_choices,
                "classifier_pooling": self.classifier_pooling,
                "classifier_dropout": self.classifier_dropout,
                "classifier_activation": self.classifier_activation,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
