import keras
from keras import layers, ops, utils

from zeromodels.base import BaseModel
from zeromodels.base.base_model import hf_num_classes
from zeromodels.conversion import copy_weights_by_path_suffix
from zeromodels.models.table_transformer.table_transformer_layers import (
    TableTransformerExpandQueryEmbedding,
    TableTransformerFlattenFeatures,
    TableTransformerMultiHeadAttention,
    TableTransformerPositionEmbeddingSine,
)
from zeromodels.utils import standardize_input_shape

from .table_transformer_config import TableTransformerConfig


def table_transformer_encoder_layer(
    x,
    pos_embed,
    hidden_dim,
    num_heads,
    dim_feedforward,
    dropout_rate=0.1,
    block_prefix="encoder_layers_0",
):
    self_attn = TableTransformerMultiHeadAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        block_prefix=f"{block_prefix}_self_attn",
        name=f"{block_prefix}_self_attn",
    )
    residual = x
    h = layers.LayerNormalization(
        epsilon=1e-5,
        name=f"{block_prefix}_self_attn_layer_norm",
    )(x)
    q = k = layers.Add(name=f"{block_prefix}_sa_qk_add")([h, pos_embed])
    attn_output = self_attn(q, k, h)
    attn_output = layers.Dropout(dropout_rate, name=f"{block_prefix}_sa_drop")(
        attn_output
    )
    x = layers.Add(name=f"{block_prefix}_sa_residual")([residual, attn_output])

    residual = x
    h = layers.LayerNormalization(
        epsilon=1e-5,
        name=f"{block_prefix}_final_layer_norm",
    )(x)
    ff_output = layers.Dense(
        dim_feedforward,
        activation="relu",
        name=f"{block_prefix}_fc1",
    )(h)
    ff_output = layers.Dropout(dropout_rate, name=f"{block_prefix}_ff_drop")(ff_output)
    ff_output = layers.Dense(hidden_dim, name=f"{block_prefix}_fc2")(ff_output)
    x = layers.Add(name=f"{block_prefix}_ff_residual")([residual, ff_output])

    return x


def table_transformer_decoder_layer(
    x,
    memory,
    pos_embed,
    query_pos,
    hidden_dim,
    num_heads,
    dim_feedforward,
    dropout_rate=0.1,
    block_prefix="decoder_layers_0",
):
    self_attn = TableTransformerMultiHeadAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        block_prefix=f"{block_prefix}_self_attn",
        name=f"{block_prefix}_self_attn",
    )

    residual = x
    h = layers.LayerNormalization(
        epsilon=1e-5,
        name=f"{block_prefix}_self_attn_layer_norm",
    )(x)
    q = k = layers.Add(name=f"{block_prefix}_sa_qk_add")([h, query_pos])
    attn_output = self_attn(q, k, h)
    attn_output = layers.Dropout(dropout_rate, name=f"{block_prefix}_sa_drop")(
        attn_output
    )
    x = layers.Add(name=f"{block_prefix}_sa_residual")([residual, attn_output])

    cross_attn = TableTransformerMultiHeadAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        block_prefix=f"{block_prefix}_encoder_attn",
        name=f"{block_prefix}_encoder_attn",
    )

    residual = x
    h = layers.LayerNormalization(
        epsilon=1e-5,
        name=f"{block_prefix}_encoder_attn_layer_norm",
    )(x)
    q_cross = layers.Add(name=f"{block_prefix}_ca_q_add")([h, query_pos])
    k_cross = layers.Add(name=f"{block_prefix}_ca_k_add")([memory, pos_embed])
    cross_output = cross_attn(q_cross, k_cross, memory)
    cross_output = layers.Dropout(dropout_rate, name=f"{block_prefix}_ca_drop")(
        cross_output
    )
    x = layers.Add(name=f"{block_prefix}_ca_residual")([residual, cross_output])

    residual = x
    h = layers.LayerNormalization(
        epsilon=1e-5,
        name=f"{block_prefix}_final_layer_norm",
    )(x)
    ff_output = layers.Dense(
        dim_feedforward,
        activation="relu",
        name=f"{block_prefix}_fc1",
    )(h)
    ff_output = layers.Dropout(dropout_rate, name=f"{block_prefix}_ff_drop")(ff_output)
    ff_output = layers.Dense(hidden_dim, name=f"{block_prefix}_fc2")(ff_output)
    x = layers.Add(name=f"{block_prefix}_ff_residual")([residual, ff_output])

    return x


def table_transformer_backbone(
    input_tensor,
    data_format="channels_last",
    channels_axis=-1,
):
    depths = [2, 2, 2, 2]
    filters_list = [64, 128, 256, 512]

    x = input_tensor
    x = layers.ZeroPadding2D(padding=3, data_format=data_format)(x)
    x = layers.Conv2D(
        64,
        7,
        strides=2,
        padding="valid",
        use_bias=False,
        data_format=data_format,
        name="backbone_conv1",
    )(x)
    x = layers.BatchNormalization(
        axis=channels_axis,
        epsilon=1e-5,
        momentum=0.1,
        name="backbone_bn1",
    )(x)
    x = layers.ReLU()(x)
    x = layers.ZeroPadding2D(padding=1, data_format=data_format)(x)
    x = layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="valid",
        data_format=data_format,
    )(x)

    stage_outputs = []
    for stage_idx, depth in enumerate(depths):
        filters = filters_list[stage_idx]
        for block_idx in range(depth):
            prefix = f"backbone_layer{stage_idx + 1}_{block_idx}"
            strides = 2 if block_idx == 0 and stage_idx > 0 else 1
            residual = x

            if strides > 1:
                x = layers.ZeroPadding2D(padding=1, data_format=data_format)(x)
                x = layers.Conv2D(
                    filters,
                    3,
                    strides=strides,
                    padding="valid",
                    use_bias=False,
                    data_format=data_format,
                    name=f"{prefix}_conv1",
                )(x)
            else:
                x = layers.Conv2D(
                    filters,
                    3,
                    strides=1,
                    padding="same",
                    use_bias=False,
                    data_format=data_format,
                    name=f"{prefix}_conv1",
                )(x)
            x = layers.BatchNormalization(
                axis=channels_axis,
                epsilon=1e-5,
                momentum=0.1,
                name=f"{prefix}_bn1",
            )(x)
            x = layers.ReLU()(x)

            x = layers.Conv2D(
                filters,
                3,
                strides=1,
                padding="same",
                use_bias=False,
                data_format=data_format,
                name=f"{prefix}_conv2",
            )(x)
            x = layers.BatchNormalization(
                axis=channels_axis,
                epsilon=1e-5,
                momentum=0.1,
                name=f"{prefix}_bn2",
            )(x)

            in_channels = residual.shape[channels_axis]
            if strides != 1 or in_channels != filters:
                residual = layers.Conv2D(
                    filters,
                    1,
                    strides=strides,
                    padding="valid",
                    use_bias=False,
                    data_format=data_format,
                    name=f"{prefix}_downsample_conv",
                )(residual)
                residual = layers.BatchNormalization(
                    axis=channels_axis,
                    epsilon=1e-5,
                    momentum=0.1,
                    name=f"{prefix}_downsample_bn",
                )(residual)

            x = layers.Add()([x, residual])
            x = layers.ReLU()(x)
        stage_outputs.append(x)

    return tuple(stage_outputs)


def table_transformer_encoder(
    backbone_features,
    hidden_dim,
    num_heads,
    num_encoder_layers,
    dim_feedforward,
    dropout_rate,
):
    data_format = keras.config.image_data_format()

    projected = layers.Conv2D(
        hidden_dim,
        1,
        padding="valid",
        data_format=data_format,
        name="input_projection",
    )(backbone_features)

    pos_embed = TableTransformerPositionEmbeddingSine(
        hidden_dim=hidden_dim,
        name="position_embedding",
    )(projected)

    src = TableTransformerFlattenFeatures(hidden_dim, name="flatten_src")(projected)
    pos = TableTransformerFlattenFeatures(hidden_dim, name="flatten_pos")(pos_embed)

    encoder_output = src
    for i in range(num_encoder_layers):
        encoder_output = table_transformer_encoder_layer(
            encoder_output,
            pos,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            block_prefix=f"encoder_layers_{i}",
        )
    encoder_output = layers.LayerNormalization(
        epsilon=1e-5,
        name="encoder_layernorm",
    )(encoder_output)

    return encoder_output, pos


def table_transformer_decoder(
    encoder_output,
    pos,
    hidden_dim,
    num_heads,
    num_decoder_layers,
    dim_feedforward,
    dropout_rate,
    num_queries,
):
    query_embed = TableTransformerExpandQueryEmbedding(
        num_queries,
        hidden_dim,
        name="query_position_embeddings",
    )(encoder_output)

    decoder_output = ops.zeros_like(query_embed)
    for i in range(num_decoder_layers):
        decoder_output = table_transformer_decoder_layer(
            decoder_output,
            encoder_output,
            pos,
            query_embed,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            block_prefix=f"decoder_layers_{i}",
        )

    last_hidden_state = layers.LayerNormalization(
        epsilon=1e-5,
        name="decoder_layernorm",
    )(decoder_output)

    return last_hidden_state


def table_transformer_functional(
    inputs,
    hidden_dim,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    dim_feedforward,
    dropout_rate,
    num_queries,
):
    data_format = keras.config.image_data_format()
    channels_axis = -1 if data_format == "channels_last" else 1

    backbone_features = table_transformer_backbone(
        inputs,
        data_format=data_format,
        channels_axis=channels_axis,
    )
    encoder_output, pos = table_transformer_encoder(
        backbone_features[-1],
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout_rate=dropout_rate,
    )
    last_hidden_state = table_transformer_decoder(
        encoder_output,
        pos,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout_rate=dropout_rate,
        num_queries=num_queries,
    )
    return last_hidden_state


@keras.saving.register_keras_serializable(package="zeromodels")
class TableTransformerModel(BaseModel):
    """Table Transformer backbone + transformer encoder/decoder (no heads).

    Matches the reference ``TableTransformerModel``: outputs the decoder
    ``last_hidden_state`` with shape ``(B, num_queries, hidden_dim)``. Wraps the
    functional graph built by :func:`table_transformer_functional`, a ResNet-18
    backbone, a stack of pre-norm transformer encoder layers with sine 2D
    position embeddings plus a final encoder LayerNorm, and a stack of pre-norm
    transformer decoder layers with learned object queries plus a final decoder
    LayerNorm. Classification and bbox heads are pruned from the output graph;
    use :class:`TableTransformerDetect` for full detection outputs.

    Reference:
        - `PubTables-1M <https://arxiv.org/abs/2110.00061>`_
        - `End-to-End Object Detection with Transformers
          <https://arxiv.org/abs/2005.12872>`_

    Args:
        hidden_dim: Transformer model dimension (channel width of both encoder
            and decoder, and of the input projection that reduces the backbone's
            512-channel feature map). Defaults to ``256``.
        num_heads: Number of attention heads in every transformer
            self-attention and cross-attention layer. Defaults to ``8``.
        num_encoder_layers: Number of stacked transformer encoder layers.
            Defaults to ``6``.
        num_decoder_layers: Number of stacked transformer decoder layers.
            Defaults to ``6``.
        dim_feedforward: FFN intermediate dimension inside each encoder /
            decoder layer. Defaults to ``2048``.
        dropout_rate: Dropout probability used in attention and FFN sub-layers.
            Defaults to ``0.1``.
        num_queries: Number of learned object queries, also the number of
            detections produced per image. Defaults to ``15``.
        image_size: Input image specification. Accepts an integer ``N`` (builds
            an ``N x N x 3`` square input), a 2-tuple ``(H, W)`` (assumes 3
            channels), or a 3-tuple ordered to match the active
            ``keras.config.image_data_format()``: ``(H, W, C)`` for
            ``channels_last`` or ``(C, H, W)`` for ``channels_first``. Defaults
            to ``800``.
        input_tensor: Optional pre-existing Keras tensor to use as the model
            input instead of creating a new :class:`Input`. Defaults to ``None``.
        name: Model name. Defaults to ``"TableTransformerModel"``.
        **kwargs: Additional keyword arguments forwarded to :class:`BaseModel`.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "table-transformer"

    def __init__(
        self,
        hidden_dim=256,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout_rate=0.1,
        num_queries=15,
        image_size=800,
        input_tensor=None,
        name="TableTransformerModel",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        image_size = standardize_input_shape(image_size, data_format)

        if input_tensor is None:
            img_input = layers.Input(shape=image_size)
        else:
            if not utils.is_keras_tensor(input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=image_size)
            else:
                img_input = input_tensor

        last_hidden_state = table_transformer_functional(
            img_input,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            num_queries=num_queries,
        )

        super().__init__(
            inputs=img_input, outputs=last_hidden_state, name=name, **kwargs
        )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.num_queries = num_queries
        self.image_size = image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "num_encoder_layers": self.num_encoder_layers,
                "num_decoder_layers": self.num_decoder_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout_rate": self.dropout_rate,
                "num_queries": self.num_queries,
                "image_size": self.image_size,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "hidden_dim": hf_config["d_model"],
            "num_heads": hf_config["encoder_attention_heads"],
            "num_encoder_layers": hf_config["encoder_layers"],
            "num_decoder_layers": hf_config["decoder_layers"],
            "dim_feedforward": hf_config["encoder_ffn_dim"],
            "dropout_rate": hf_config["dropout"],
            "num_queries": hf_config["num_queries"],
        }

    @classmethod
    def from_hf(cls, hf_id, load_weights=True, skip_mismatch=False, **kwargs):
        model = super().from_hf(hf_id, load_weights=False, **kwargs)
        if load_weights:
            src = TableTransformerDetect.from_hf(hf_id, skip_mismatch=skip_mismatch)
            unmatched = copy_weights_by_path_suffix(src, model)
            if unmatched and not skip_mismatch:
                raise ValueError(
                    f"{cls.__name__}.from_hf: {len(unmatched)} weight(s) not "
                    f"matched from the {type(src).__name__} checkpoint: "
                    f"{unmatched[:5]}"
                )
            del src
        return model


@keras.saving.register_keras_serializable(package="zeromodels")
class TableTransformerDetect(BaseModel):
    """Table Transformer object detection model (transformer + heads).

    The same architecture serves both Table Transformer tasks; only
    ``num_queries`` and ``num_classes`` differ between the checkpoints:

    - table **detection** (``microsoft/table-transformer-detection``):
      ``num_queries=15``, ``num_classes=3`` (table, table rotated, no-object).
    - table **structure recognition**
      (``microsoft/table-transformer-structure-recognition`` and the v1.1
      variants): ``num_queries=125``, ``num_classes=7`` (table, column, row,
      column header, projected row header, spanning cell, no-object).

    Output dict:

    .. code-block:: python

        out = model(images)
        out["logits"]      # (B, num_queries, num_classes): class logits
        out["pred_boxes"]  # (B, num_queries, 4): sigmoid cxcywh in [0, 1]

    Reference:
    - [PubTables-1M](https://arxiv.org/abs/2110.00061)
    - [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

    Loads pretrained weights via ``TableTransformerDetect.from_weights(...)``.
    See ``BaseModel.from_weights`` for the loading API.
    """

    BASE_MODEL_CONFIG = None
    config_class = TableTransformerConfig
    # Weights load by Hub repo id, e.g.
    # from_weights("zeromodels/table-transformer-detection"), via zm_config.json
    # on the repo (no url table in the package).
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "table-transformer"

    def __init__(
        self,
        hidden_dim=256,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout_rate=0.1,
        num_queries=15,
        num_classes=3,
        image_size=800,
        input_tensor=None,
        name="TableTransformerDetect",
        **kwargs,
    ):
        base = TableTransformerModel(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            num_queries=num_queries,
            image_size=image_size,
            input_tensor=input_tensor,
            name=f"{name}_model",
        )
        last_hidden_state = base.output

        logits = layers.Dense(
            num_classes,
            name="class_labels_classifier",
        )(last_hidden_state)

        bbox = layers.Dense(hidden_dim, activation="relu", name="bbox_predictor_0")(
            last_hidden_state
        )
        bbox = layers.Dense(hidden_dim, activation="relu", name="bbox_predictor_1")(
            bbox
        )
        bbox = layers.Dense(4, name="bbox_predictor_2")(bbox)
        bbox = layers.Activation("sigmoid", name="bbox_sigmoid")(bbox)

        outputs = {"logits": logits, "pred_boxes": bbox}

        super().__init__(inputs=base.input, outputs=outputs, name=name, **kwargs)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.image_size = base.image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "num_encoder_layers": self.num_encoder_layers,
                "num_decoder_layers": self.num_decoder_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout_rate": self.dropout_rate,
                "num_queries": self.num_queries,
                "num_classes": self.num_classes,
                "image_size": self.image_size,
                "input_tensor": self.input_tensor,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def config_from_hf(cls, hf_config):
        return {
            "hidden_dim": hf_config["d_model"],
            "num_heads": hf_config["encoder_attention_heads"],
            "num_encoder_layers": hf_config["encoder_layers"],
            "num_decoder_layers": hf_config["decoder_layers"],
            "dim_feedforward": hf_config["encoder_ffn_dim"],
            "dropout_rate": hf_config["dropout"],
            "num_queries": hf_config["num_queries"],
            "num_classes": hf_num_classes(hf_config) + 1,
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from .convert_table_transformer_hf_to_keras import (
            transfer_table_transformer_weights,
        )

        transfer_table_transformer_weights(keras_model, hf_state_dict)
