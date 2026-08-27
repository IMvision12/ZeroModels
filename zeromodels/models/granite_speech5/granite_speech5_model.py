import keras
from keras import layers, ops

from zeromodels.base import BaseModel

from .granite_speech5_config import GraniteSpeech5Config
from .granite_speech5_layers import DownsampleMask, GraniteSpeech5Block

BACKBONE_ATTRS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "head_dim",
    "num_mel_bins",
    "hidden_act",
    "max_position_embeddings",
    "context_size",
    "conv_kernel_size",
    "conv_expansion_factor",
    "subsample_layers",
    "attention_bias",
    "pad_token_id",
)


def store_backbone_attrs(model, ns):
    for k in BACKBONE_ATTRS:
        setattr(model, k, ns[k])


def encoder_inputs(feature_size):
    return {
        "input_features": layers.Input(
            shape=(None, feature_size), dtype="float32", name="input_features"
        ),
        "attention_mask": layers.Input(
            shape=(None,), dtype="int32", name="attention_mask"
        ),
    }


def granite_speech5_encoder(
    input_features,
    attention_mask,
    *,
    vocab_size,
    hidden_size,
    intermediate_size,
    num_hidden_layers,
    num_attention_heads,
    head_dim,
    max_position_embeddings,
    context_size,
    conv_kernel_size,
    conv_expansion_factor,
    subsample_layers,
    attention_bias,
):
    """Conformer CTC encoder graph.

    ``input_linear`` lifts the stacked log-mel(+delta) features to ``hidden_size``,
    then ``num_hidden_layers`` conformer blocks run (the ``subsample_layers`` ones
    halve time). The padding mask is halved after each subsampling block, and at the
    midpoint layer the self-conditioned CTC posteriors (``out`` -> softmax ->
    ``out_mid``) are added back into the hidden states.

    Returns ``(last_hidden_state, output_attention_mask, out_dense)``; ``out_dense``
    is returned so the CTC head can tie to it.
    """
    hidden = layers.Dense(hidden_size, name="input_linear")(input_features)
    keep = ops.cast(attention_mask, hidden.dtype)[:, :, None]
    hidden = hidden * keep

    out_dense = layers.Dense(vocab_size, name="out")
    out_mid_dense = layers.Dense(hidden_size, name="out_mid")

    cur_mask = attention_mask
    subsample_layers = set(subsample_layers)
    for i in range(num_hidden_layers):
        hidden = GraniteSpeech5Block(
            hidden_size,
            intermediate_size,
            num_attention_heads,
            head_dim,
            context_size,
            max_position_embeddings,
            conv_expansion_factor,
            conv_kernel_size,
            attention_bias=attention_bias,
            subsample=(i in subsample_layers),
            name=f"layers_{i}",
        )(hidden, attention_mask=cur_mask)

        if i in subsample_layers:
            cur_mask = DownsampleMask(name=f"downsample_mask_{i}")(cur_mask)

        if i + 1 == num_hidden_layers // 2:
            mid = out_dense(hidden)
            hidden = hidden + out_mid_dense(ops.softmax(mid, axis=-1))

    return hidden, cur_mask, out_dense


def config_from_hf(hf_config):
    enc = hf_config.get("encoder_config", hf_config)
    return {
        "vocab_size": hf_config.get("vocab_size", enc["vocab_size"]),
        "hidden_size": enc["hidden_size"],
        "intermediate_size": enc["intermediate_size"],
        "num_hidden_layers": enc["num_hidden_layers"],
        "num_attention_heads": enc["num_attention_heads"],
        "head_dim": enc.get(
            "head_dim", enc["hidden_size"] // enc["num_attention_heads"]
        ),
        "num_mel_bins": enc.get("num_mel_bins", 80),
        "hidden_act": enc.get("hidden_act", "silu"),
        "max_position_embeddings": enc.get("max_position_embeddings", 512),
        "context_size": enc.get("context_size", 128),
        "conv_kernel_size": enc.get("conv_kernel_size", 7),
        "conv_expansion_factor": enc.get("conv_expansion_factor", 2),
        "subsample_layers": tuple(enc.get("subsample_layers", (0, 1))),
        "attention_bias": enc.get("attention_bias", True),
        "pad_token_id": hf_config.get("pad_token_id", 0),
    }


@keras.saving.register_keras_serializable(package="zeromodels")
class GraniteSpeech5Model(BaseModel):
    """Granite Speech 5.0 conformer CTC audio encoder (backbone).

    A self-conditioned CTC conformer: block-wise self-attention with Shaw's
    relative positions, a depthwise-convolution module per block, two early
    time-subsampling blocks (each halving the frame rate), and a mid-layer CTC
    self-conditioning injection. Takes a dict of ``input_features``
    ``(B, time, num_mel_bins * 4)`` and ``attention_mask`` ``(B, time)`` (as
    produced by :class:`GraniteSpeech5FeatureExtractor`) and returns
    ``last_hidden_state`` ``(B, time // 4, hidden_size)`` plus the subsampled
    ``output_attention_mask``.

    References:
    - [Granite Speech CTC encoder](https://huggingface.co/papers/2505.08699)

    Args:
        vocab_size: Integer, CTC vocabulary size. Defaults to `16384`.
        hidden_size: Integer, conformer hidden width. Defaults to `1024`.
        intermediate_size: Integer, feed-forward width. Defaults to `4096`.
        num_hidden_layers: Integer, number of conformer blocks. Defaults to `16`.
        num_attention_heads: Integer, attention heads. Defaults to `8`.
        head_dim: Integer, per-head dimension. Defaults to `128`.
        num_mel_bins: Integer, mel bins (input width is `num_mel_bins * 4`). Defaults to `80`.
        hidden_act: String, feed-forward activation. Defaults to `"silu"`.
        max_position_embeddings: Integer, relative-position span. Defaults to `512`.
        context_size: Integer, block-attention window. Defaults to `128`.
        conv_kernel_size: Integer, depthwise-conv kernel. Defaults to `7`.
        conv_expansion_factor: Integer, conv channel expansion. Defaults to `2`.
        subsample_layers: Tuple, block indices that subsample time. Defaults to `(0, 1)`.
        attention_bias: Boolean, feed-forward bias. Defaults to `True`.
        pad_token_id: Integer, CTC blank id. Defaults to `0`.
        name: String, model name. Defaults to `"GraniteSpeech5Model"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "granite_speech5_ctc"
    config_class = GraniteSpeech5Config

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_granite_speech5_hf_to_keras import (
            transfer_granite_speech5_weights,
        )

        transfer_granite_speech5_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return config_from_hf(hf_config)

    def __init__(
        self,
        vocab_size=16384,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=16,
        num_attention_heads=8,
        head_dim=128,
        num_mel_bins=80,
        hidden_act="silu",
        max_position_embeddings=512,
        context_size=128,
        conv_kernel_size=7,
        conv_expansion_factor=2,
        subsample_layers=(0, 1),
        attention_bias=True,
        pad_token_id=0,
        name="GraniteSpeech5Model",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)

        inputs = encoder_inputs(num_mel_bins * 4)
        hidden, out_mask, _ = granite_speech5_encoder(
            inputs["input_features"],
            inputs["attention_mask"],
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            context_size=context_size,
            conv_kernel_size=conv_kernel_size,
            conv_expansion_factor=conv_expansion_factor,
            subsample_layers=subsample_layers,
            attention_bias=attention_bias,
        )
        super().__init__(
            inputs=inputs,
            outputs={
                "last_hidden_state": hidden,
                "output_attention_mask": ops.cast(out_mask, "float32"),
            },
            name=name,
            **kwargs,
        )
        store_backbone_attrs(self, locals())

    def get_config(self):
        config = super().get_config()
        config.update({k: getattr(self, k) for k in BACKBONE_ATTRS})
        config["name"] = self.name
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="zeromodels")
class GraniteSpeech5CTC(BaseModel):
    """Granite Speech 5.0 conformer encoder with a CTC head.

    Runs the :class:`GraniteSpeech5Model` conformer encoder and projects the final
    hidden states to CTC token logits ``(B, time // 4, vocab_size)`` with the head
    tied to the encoder's mid-layer self-conditioning projection (``encoder.out``).
    ``generate`` performs greedy CTC decoding (per-frame argmax, padded frames set
    to the blank id); collapse the repeats / drop the blank with
    :class:`GraniteSpeech5Tokenizer`.

    References:
    - [Granite Speech CTC encoder](https://huggingface.co/papers/2505.08699)

    Args:
        See :class:`GraniteSpeech5Model` for the encoder arguments.
        name: String, model name. Defaults to `"GraniteSpeech5CTC"`.

    Returns:
        A Keras `Model` instance.
    """

    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "granite_speech5_ctc"
    config_class = GraniteSpeech5Config

    @classmethod
    def transfer_from_hf(cls, keras_model, state_dict):
        from .convert_granite_speech5_hf_to_keras import (
            transfer_granite_speech5_weights,
        )

        transfer_granite_speech5_weights(keras_model, state_dict)

    @classmethod
    def config_from_hf(cls, hf_config):
        return config_from_hf(hf_config)

    def __init__(
        self,
        vocab_size=16384,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=16,
        num_attention_heads=8,
        head_dim=128,
        num_mel_bins=80,
        hidden_act="silu",
        max_position_embeddings=512,
        context_size=128,
        conv_kernel_size=7,
        conv_expansion_factor=2,
        subsample_layers=(0, 1),
        attention_bias=True,
        pad_token_id=0,
        name="GraniteSpeech5CTC",
        **kwargs,
    ):
        for k in ("model", "hf_id", "url", "num_classes"):
            kwargs.pop(k, None)

        inputs = encoder_inputs(num_mel_bins * 4)
        hidden, out_mask, out_dense = granite_speech5_encoder(
            inputs["input_features"],
            inputs["attention_mask"],
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            context_size=context_size,
            conv_kernel_size=conv_kernel_size,
            conv_expansion_factor=conv_expansion_factor,
            subsample_layers=subsample_layers,
            attention_bias=attention_bias,
        )
        # CTC head is tied to the encoder's self-conditioning projection.
        logits = out_dense(hidden)
        super().__init__(
            inputs=inputs,
            outputs={
                "logits": logits,
                "output_attention_mask": ops.cast(out_mask, "float32"),
            },
            name=name,
            **kwargs,
        )
        store_backbone_attrs(self, locals())

    def generate(self, inputs):
        """Greedy CTC decoding: per-frame argmax with padded frames set to blank."""
        outputs = self(inputs)
        ids = ops.argmax(outputs["logits"], axis=-1)
        mask = ops.cast(outputs["output_attention_mask"], "bool")
        return ops.where(mask, ids, self.pad_token_id)

    def get_config(self):
        config = super().get_config()
        config.update({k: getattr(self, k) for k in BACKBONE_ATTRS})
        config["name"] = self.name
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
