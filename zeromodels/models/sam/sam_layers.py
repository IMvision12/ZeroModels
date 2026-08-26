import keras
import numpy as np
from keras import layers, ops


@keras.saving.register_keras_serializable(package="zeromodels")
class SAMAbsolutePositionEmbedding(layers.Layer):
    """Learnable absolute position embeddings for vision encoder patches.

    Adds a learnable 2-D positional embedding grid to the patch embeddings
    produced by the vision encoder. When the target image embedding size
    differs from the pretrained size, the stored embeddings are bilinearly
    resized during weight loading.

    Args:
        hidden_dim (int): Dimensionality of the embedding channels.
        image_embedding_size (int): Spatial size of the square embedding grid.
        data_format (str): One of ``"channels_last"`` or ``"channels_first"``.
            Defaults to ``"channels_last"``.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    References:
        - Segment Anything: https://arxiv.org/abs/2304.02643
    """

    def __init__(
        self, hidden_dim, image_embedding_size, data_format="channels_last", **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.image_embedding_size = image_embedding_size
        self.data_format = data_format

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(
                1,
                self.image_embedding_size,
                self.image_embedding_size,
                self.hidden_dim,
            ),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, hidden_states):
        pos_embed = self.pos_embed
        if self.data_format == "channels_first":
            pos_embed = ops.transpose(pos_embed, (0, 3, 1, 2))
        return hidden_states + pos_embed

    def save_own_variables(self, store):
        super().save_own_variables(store)
        store["image_embedding_size"] = self.image_embedding_size

    def load_own_variables(self, store):
        source_size = int(store["image_embedding_size"][...])

        if source_size == self.image_embedding_size:
            self.pos_embed.assign(store["0"])
            return

        pos_embed = ops.cast(store["0"], dtype="float32")
        pos_embed = ops.image.resize(
            pos_embed,
            size=(self.image_embedding_size, self.image_embedding_size),
            interpolation="bilinear",
            antialias=True,
        )
        self.pos_embed.assign(pos_embed)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "image_embedding_size": self.image_embedding_size,
                "data_format": self.data_format,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAMPromptEncoderLayer(layers.Layer):
    """Prompt encoder for sparse and dense prompts.

    Encodes sparse prompts (points and optionally boxes) into positional
    embeddings and produces a dense embedding for the image. Points are
    always padded with a learned ``not_a_point`` token to match the
    original SAM checkpoint's inference behavior. Box support and mask
    support are opt-in via the ``enable_boxes`` and ``enable_masks``
    construction-time flags: these flags change the expected input
    signature and the shape of ``sparse_embeddings`` so that the
    default configuration matches the original SAM exactly.

    Input signature (all passed as a dict):
        - Always: ``input_points``, ``input_labels``.
        - When ``enable_boxes=True``: ``input_boxes``, ``has_boxes_input``.
        - When ``enable_masks=True``: ``input_mask_embedding``,
          ``has_mask_input``.

    Token counts:
        - ``enable_boxes=False``: ``sparse_embeddings`` has ``N+1``
          tokens (points-only path).
        - ``enable_boxes=True``: ``sparse_embeddings`` has ``N+3``
          tokens (1 pad + N points + 2 box corners or ``not_a_point``
          fallbacks). This is 1 token more than the reference's points+boxes
          path but lets a single model handle both prompt modes.

    Args:
        hidden_dim (int): Dimensionality of the prompt embeddings.
            Defaults to 256.
        image_embedding_size (int): Spatial size of the image embedding grid.
            Defaults to 64.
        image_size (int): Input image resolution used for coordinate
            normalization. Defaults to 1024.
        num_point_embeddings (int): Number of learned point-type embeddings
            (foreground, background, and two box corners). Defaults to 4.
        shared_embedding (Layer): Shared positional embedding layer used to
            encode normalized coordinates. Defaults to None.
        enable_boxes (bool): Whether the layer accepts box prompts.
            Defaults to ``False``.
        enable_masks (bool): Whether the layer accepts mask prompts.
            Defaults to ``False``.
        data_format (str): One of ``"channels_last"`` or ``"channels_first"``.
            Defaults to ``"channels_last"``.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    References:
        - Segment Anything: https://arxiv.org/abs/2304.02643
    """

    def __init__(
        self,
        hidden_dim=256,
        image_embedding_size=64,
        image_size=1024,
        num_point_embeddings=4,
        shared_embedding=None,
        enable_boxes=False,
        enable_masks=False,
        data_format="channels_last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.image_embedding_size = image_embedding_size
        self.image_size = image_size
        self.num_point_embeddings = num_point_embeddings
        self.shared_embedding = shared_embedding
        self.enable_boxes = enable_boxes
        self.enable_masks = enable_masks
        self.data_format = data_format

    def build(self, input_shape):
        self.point_embeddings = []
        for i in range(self.num_point_embeddings):
            w = self.add_weight(
                name=f"point_embed_{i}",
                shape=(1, self.hidden_dim),
                initializer="zeros",
                trainable=True,
            )
            self.point_embeddings.append(w)

        self.not_a_point_embed = self.add_weight(
            name="not_a_point_embed",
            shape=(1, self.hidden_dim),
            initializer="zeros",
            trainable=True,
        )
        self.no_mask_embed = self.add_weight(
            name="no_mask_embed",
            shape=(1, self.hidden_dim),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def _embed_points(self, points, point_labels, pad):
        points = points + 0.5
        if pad:
            batch_size = ops.shape(points)[0]
            point_batch_size = ops.shape(points)[1]
            padding_point = ops.zeros(
                (batch_size, point_batch_size, 1, 2), dtype=points.dtype
            )
            padding_label = -ops.ones((batch_size, point_batch_size, 1), dtype="int32")
            points = ops.concatenate([points, padding_point], axis=2)
            point_labels = ops.concatenate(
                [point_labels, ops.cast(padding_label, point_labels.dtype)], axis=2
            )

        coords_0 = points[..., 0] / float(self.image_size)
        coords_1 = points[..., 1] / float(self.image_size)
        normalized_coords = ops.stack([coords_0, coords_1], axis=-1)
        point_embedding = self.shared_embedding(normalized_coords)

        point_labels_expanded = ops.expand_dims(
            ops.cast(point_labels, "float32"), axis=-1
        )

        not_a_point = ops.broadcast_to(
            self.not_a_point_embed, ops.shape(point_embedding)
        )
        point_embedding = ops.where(
            point_labels_expanded == -1.0, not_a_point, point_embedding
        )

        zeros = ops.zeros_like(point_embedding)
        point_embedding = ops.where(
            point_labels_expanded == -10.0, zeros, point_embedding
        )

        pe_0 = ops.broadcast_to(self.point_embeddings[0], ops.shape(point_embedding))
        point_embedding = ops.where(
            point_labels_expanded == 0.0, point_embedding + pe_0, point_embedding
        )
        pe_1 = ops.broadcast_to(self.point_embeddings[1], ops.shape(point_embedding))
        point_embedding = ops.where(
            point_labels_expanded == 1.0, point_embedding + pe_1, point_embedding
        )
        return point_embedding

    def _embed_boxes(self, boxes):
        boxes = boxes + 0.5
        batch_size = ops.shape(boxes)[0]
        nb_boxes = ops.shape(boxes)[1]
        coords = ops.reshape(boxes, (batch_size, nb_boxes, 2, 2))
        coords_0 = coords[..., 0] / float(self.image_size)
        coords_1 = coords[..., 1] / float(self.image_size)
        normalized_coords = ops.stack([coords_0, coords_1], axis=-1)
        corner_embedding = self.shared_embedding(normalized_coords)
        pe_2 = ops.broadcast_to(
            self.point_embeddings[2],
            ops.shape(corner_embedding[:, :, 0:1, :]),
        )
        pe_3 = ops.broadcast_to(
            self.point_embeddings[3],
            ops.shape(corner_embedding[:, :, 1:2, :]),
        )
        corner_0 = corner_embedding[:, :, 0:1, :] + pe_2
        corner_1 = corner_embedding[:, :, 1:2, :] + pe_3
        return ops.concatenate([corner_0, corner_1], axis=2)

    def call(self, inputs):
        input_points = inputs["input_points"]
        input_labels = inputs["input_labels"]

        batch_size = ops.shape(input_points)[0]

        point_embeddings = self._embed_points(input_points, input_labels, pad=True)

        if self.enable_boxes:
            input_boxes = inputs["input_boxes"]
            has_boxes_input = inputs["has_boxes_input"]

            box_embeddings = self._embed_boxes(input_boxes)
            not_a_point = ops.broadcast_to(
                ops.reshape(self.not_a_point_embed, (1, 1, 1, self.hidden_dim)),
                ops.shape(box_embeddings),
            )
            has_boxes_b = ops.reshape(
                ops.cast(has_boxes_input, box_embeddings.dtype), (-1, 1, 1, 1)
            )
            box_embeddings = (
                has_boxes_b * box_embeddings + (1.0 - has_boxes_b) * not_a_point
            )
            sparse_embeddings = ops.concatenate(
                [point_embeddings, box_embeddings], axis=2
            )
        else:
            sparse_embeddings = point_embeddings

        cf = self.data_format == "channels_first"
        if cf:
            no_mask = ops.reshape(self.no_mask_embed, (1, -1, 1, 1))
            no_mask_dense = ops.broadcast_to(
                no_mask,
                (
                    batch_size,
                    self.hidden_dim,
                    self.image_embedding_size,
                    self.image_embedding_size,
                ),
            )
        else:
            no_mask = ops.reshape(self.no_mask_embed, (1, 1, 1, -1))
            no_mask_dense = ops.broadcast_to(
                no_mask,
                (
                    batch_size,
                    self.image_embedding_size,
                    self.image_embedding_size,
                    self.hidden_dim,
                ),
            )

        if self.enable_masks:
            mask_dense = inputs["input_mask_embedding"]
            has_mask_input = inputs["has_mask_input"]
            has_mask_b = ops.reshape(
                ops.cast(has_mask_input, mask_dense.dtype), (-1, 1, 1, 1)
            )
            dense_embeddings = (
                has_mask_b * mask_dense + (1.0 - has_mask_b) * no_mask_dense
            )
        else:
            dense_embeddings = no_mask_dense

        return {
            "sparse_embeddings": sparse_embeddings,
            "dense_embeddings": dense_embeddings,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "image_embedding_size": self.image_embedding_size,
                "image_size": self.image_size,
                "num_point_embeddings": self.num_point_embeddings,
                "enable_boxes": self.enable_boxes,
                "enable_masks": self.enable_masks,
                "data_format": self.data_format,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAMMaskDecoderLayer(layers.Layer):
    """Mask decoder that predicts segmentation masks and IoU scores.

    Implements a two-way transformer decoder that attends jointly to image
    embeddings and prompt tokens. The decoder upscales the output features
    with transposed convolutions, then generates per-mask predictions via
    lightweight hypernetwork MLPs and estimates mask quality through an
    IoU prediction head.

    Args:
        hidden_dim (int): Dimensionality of the transformer hidden states.
            Defaults to 256.
        num_hidden_layers (int): Number of two-way transformer blocks.
            Defaults to 2.
        num_heads (int): Number of attention heads in each
            transformer block. Defaults to 8.
        mlp_dim (int): Hidden dimensionality of the feed-forward network
            inside each transformer block. Defaults to 2048.
        num_multimask_outputs (int): Number of additional mask predictions
            beyond the single-mask output. Defaults to 3.
        iou_head_depth (int): Number of layers in the IoU prediction MLP.
            Defaults to 3.
        iou_head_hidden_dim (int): Hidden dimensionality of the IoU
            prediction MLP. Defaults to 256.
        attention_downsample_rate (int): Downsampling factor applied to the
            internal dimension of cross-attention layers. Defaults to 2.
        layer_norm_eps (float): Epsilon value for layer normalization.
            Defaults to 1e-6.
        data_format (str): One of ``"channels_last"`` or ``"channels_first"``.
            Defaults to ``"channels_last"``.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    References:
        - Segment Anything: https://arxiv.org/abs/2304.02643
    """

    def __init__(
        self,
        hidden_dim=256,
        num_hidden_layers=2,
        num_heads=8,
        mlp_dim=2048,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        attention_downsample_rate=2,
        layer_norm_eps=1e-6,
        multimask_output=True,
        data_format="channels_last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.attention_downsample_rate = attention_downsample_rate
        self.layer_norm_eps = layer_norm_eps
        self.multimask_output = multimask_output
        self.data_format = data_format

        self.transformer_self_attns = []
        self.transformer_layer_norm1s = []
        self.transformer_cross_attn_token_to_images = []
        self.transformer_layer_norm2s = []
        self.transformer_mlp_lin1s = []
        self.transformer_mlp_lin2s = []
        self.transformer_layer_norm3s = []
        self.transformer_layer_norm4s = []
        self.transformer_cross_attn_image_to_tokens = []

        for i in range(num_hidden_layers):
            prefix = f"transformer_layers_{i}"
            self.transformer_self_attns.append(
                SAMTwoWayAttention(
                    hidden_dim,
                    num_heads,
                    downsample_rate=1,
                    name=f"{prefix}_self_attn",
                )
            )
            self.transformer_layer_norm1s.append(
                layers.LayerNormalization(
                    epsilon=layer_norm_eps, name=f"{prefix}_layer_norm1"
                )
            )
            self.transformer_cross_attn_token_to_images.append(
                SAMTwoWayAttention(
                    hidden_dim,
                    num_heads,
                    downsample_rate=attention_downsample_rate,
                    name=f"{prefix}_cross_attn_token_to_image",
                )
            )
            self.transformer_layer_norm2s.append(
                layers.LayerNormalization(
                    epsilon=layer_norm_eps, name=f"{prefix}_layer_norm2"
                )
            )
            self.transformer_mlp_lin1s.append(
                layers.Dense(mlp_dim, name=f"{prefix}_mlp_lin1")
            )
            self.transformer_mlp_lin2s.append(
                layers.Dense(hidden_dim, name=f"{prefix}_mlp_lin2")
            )
            self.transformer_layer_norm3s.append(
                layers.LayerNormalization(
                    epsilon=layer_norm_eps, name=f"{prefix}_layer_norm3"
                )
            )
            self.transformer_layer_norm4s.append(
                layers.LayerNormalization(
                    epsilon=layer_norm_eps, name=f"{prefix}_layer_norm4"
                )
            )
            self.transformer_cross_attn_image_to_tokens.append(
                SAMTwoWayAttention(
                    hidden_dim,
                    num_heads,
                    downsample_rate=attention_downsample_rate,
                    name=f"{prefix}_cross_attn_image_to_token",
                )
            )

        self.final_attn_token_to_image = SAMTwoWayAttention(
            hidden_dim,
            num_heads,
            downsample_rate=2,
            name="final_attn_token_to_image",
        )
        self.layer_norm_final_attn = layers.LayerNormalization(
            epsilon=1e-6, name="layer_norm_final_attn"
        )

        self.upscale_conv1 = layers.Conv2DTranspose(
            hidden_dim // 4,
            kernel_size=2,
            strides=2,
            data_format=data_format,
            name="upscale_conv1",
        )
        self.upscale_layer_norm = layers.LayerNormalization(
            epsilon=1e-6, name="upscale_layer_norm"
        )
        self.upscale_conv2 = layers.Conv2DTranspose(
            hidden_dim // 8,
            kernel_size=2,
            strides=2,
            data_format=data_format,
            name="upscale_conv2",
        )

        self.output_hypernetworks_mlps_proj_ins = []
        self.output_hypernetworks_mlps_hidden_layers = []
        self.output_hypernetworks_mlps_proj_outs = []
        self._hyper_num_hidden = 3 - 2
        for i in range(self.num_mask_tokens):
            prefix = f"output_hypernetworks_mlps_{i}"
            self.output_hypernetworks_mlps_proj_ins.append(
                layers.Dense(hidden_dim, name=f"{prefix}_proj_in")
            )
            for j in range(self._hyper_num_hidden):
                self.output_hypernetworks_mlps_hidden_layers.append(
                    layers.Dense(hidden_dim, name=f"{prefix}_layers_{j}")
                )
            self.output_hypernetworks_mlps_proj_outs.append(
                layers.Dense(hidden_dim // 8, name=f"{prefix}_proj_out")
            )

        iou_prefix = "iou_prediction_head"
        self.iou_head_proj_in = layers.Dense(
            iou_head_hidden_dim, name=f"{iou_prefix}_proj_in"
        )
        self.iou_head_hidden_layers = []
        for j in range(iou_head_depth - 2):
            self.iou_head_hidden_layers.append(
                layers.Dense(iou_head_hidden_dim, name=f"{iou_prefix}_layers_{j}")
            )
        self.iou_head_proj_out = layers.Dense(
            self.num_mask_tokens, name=f"{iou_prefix}_proj_out"
        )

    def build(self, input_shape):
        self.iou_token = self.add_weight(
            name="iou_token",
            shape=(1, self.hidden_dim),
            initializer="zeros",
            trainable=True,
        )
        self.mask_tokens = self.add_weight(
            name="mask_tokens",
            shape=(self.num_mask_tokens, self.hidden_dim),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        image_embeddings, image_pe, sparse_embeddings, dense_embeddings = inputs

        cf = self.data_format == "channels_first"

        batch_size = ops.shape(image_embeddings)[0]
        point_batch_size = ops.shape(sparse_embeddings)[1]

        output_tokens = ops.concatenate(
            [
                ops.expand_dims(self.iou_token, axis=0),
                ops.expand_dims(self.mask_tokens, axis=0),
            ],
            axis=1,
        )
        output_tokens = ops.broadcast_to(
            ops.expand_dims(output_tokens, axis=0),
            (batch_size, point_batch_size, 1 + self.num_mask_tokens, self.hidden_dim),
        )

        tokens = ops.concatenate([output_tokens, sparse_embeddings], axis=2)

        image_embeddings_with_dense = image_embeddings + dense_embeddings

        if cf:
            num_channels = ops.shape(image_embeddings)[1]
            height = ops.shape(image_embeddings)[2]
            width = ops.shape(image_embeddings)[3]
            image_emb_flat = ops.reshape(
                ops.transpose(image_embeddings_with_dense, (0, 2, 3, 1)),
                (batch_size, height * width, num_channels),
            )
        else:
            height = ops.shape(image_embeddings)[1]
            width = ops.shape(image_embeddings)[2]
            num_channels = ops.shape(image_embeddings)[3]
            image_emb_flat = ops.reshape(
                image_embeddings_with_dense,
                (batch_size, height * width, num_channels),
            )

        image_emb_flat = ops.expand_dims(image_emb_flat, axis=1)
        image_emb_flat = ops.broadcast_to(
            image_emb_flat,
            (batch_size, point_batch_size, height * width, num_channels),
        )

        image_pe_flat = ops.reshape(image_pe, (1, num_channels, height * width))
        image_pe_flat = ops.transpose(image_pe_flat, (0, 2, 1))
        image_pe_flat = ops.expand_dims(image_pe_flat, axis=1)
        image_pe_flat = ops.broadcast_to(
            image_pe_flat,
            (batch_size, point_batch_size, height * width, num_channels),
        )

        queries = tokens
        keys = image_emb_flat

        for i in range(self.num_hidden_layers):
            if i == 0:
                queries = self.transformer_self_attns[i](
                    query=queries, key=queries, value=queries
                )
            else:
                query = queries + tokens
                attn_out = self.transformer_self_attns[i](
                    query=query, key=query, value=queries
                )
                queries = queries + attn_out
            queries = self.transformer_layer_norm1s[i](queries)

            query = queries + tokens
            key = keys + image_pe_flat
            attn_out = self.transformer_cross_attn_token_to_images[i](
                query=query, key=key, value=keys
            )
            queries = queries + attn_out
            queries = self.transformer_layer_norm2s[i](queries)

            mlp_out = self.transformer_mlp_lin1s[i](queries)
            mlp_out = ops.nn.relu(mlp_out)
            mlp_out = self.transformer_mlp_lin2s[i](mlp_out)
            queries = queries + mlp_out
            queries = self.transformer_layer_norm3s[i](queries)

            query = queries + tokens
            key = keys + image_pe_flat
            attn_out = self.transformer_cross_attn_image_to_tokens[i](
                query=key, key=query, value=queries
            )
            keys = keys + attn_out
            keys = self.transformer_layer_norm4s[i](keys)

        query = queries + tokens
        key = keys + image_pe_flat
        attn_out = self.final_attn_token_to_image(query=query, key=key, value=keys)
        queries = queries + attn_out
        queries = self.layer_norm_final_attn(queries)

        iou_token_out = queries[:, :, 0, :]
        mask_tokens_out = queries[:, :, 1 : 1 + self.num_mask_tokens, :]

        if cf:
            keys_spatial = ops.reshape(
                keys,
                (batch_size * point_batch_size, height, width, num_channels),
            )
            keys_spatial = ops.transpose(keys_spatial, (0, 3, 1, 2))
        else:
            keys_spatial = ops.reshape(
                keys,
                (batch_size * point_batch_size, height, width, num_channels),
            )

        upscaled = self.upscale_conv1(keys_spatial)
        if cf:
            upscaled = ops.transpose(upscaled, (0, 2, 3, 1))
        upscaled = self.upscale_layer_norm(upscaled)
        if cf:
            upscaled = ops.transpose(upscaled, (0, 3, 1, 2))
        upscaled = ops.nn.gelu(upscaled, approximate=False)
        upscaled = self.upscale_conv2(upscaled)
        if cf:
            upscaled = ops.transpose(upscaled, (0, 2, 3, 1))
        upscaled = ops.nn.gelu(upscaled, approximate=False)

        up_channels = ops.shape(upscaled)[-1]
        up_height = ops.shape(upscaled)[1]
        up_width = ops.shape(upscaled)[2]

        upscaled_flat = ops.reshape(
            upscaled,
            (batch_size, point_batch_size, up_height * up_width, up_channels),
        )

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            h = self.output_hypernetworks_mlps_proj_ins[i](mask_tokens_out[:, :, i, :])
            h = ops.nn.relu(h)
            for j in range(self._hyper_num_hidden):
                h = ops.nn.relu(
                    self.output_hypernetworks_mlps_hidden_layers[
                        i * self._hyper_num_hidden + j
                    ](h)
                )
            h = self.output_hypernetworks_mlps_proj_outs[i](h)
            hyper_in_list.append(h)
        hyper_in = ops.stack(hyper_in_list, axis=2)

        masks = ops.matmul(hyper_in, ops.transpose(upscaled_flat, (0, 1, 3, 2)))
        masks = ops.reshape(
            masks,
            (batch_size, point_batch_size, self.num_mask_tokens, up_height, up_width),
        )

        iou_pred = self.iou_head_proj_in(iou_token_out)
        iou_pred = ops.nn.relu(iou_pred)
        for hidden_layer in self.iou_head_hidden_layers:
            iou_pred = ops.nn.relu(hidden_layer(iou_pred))
        iou_pred = self.iou_head_proj_out(iou_pred)

        if self.multimask_output:
            masks = masks[:, :, 1:, :, :]
            iou_pred = iou_pred[:, :, 1:]
        else:
            masks = masks[:, :, 0:1, :, :]
            iou_pred = iou_pred[:, :, 0:1]

        return {"pred_masks": masks, "iou_scores": iou_pred}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_hidden_layers": self.num_hidden_layers,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "num_multimask_outputs": self.num_multimask_outputs,
                "iou_head_depth": self.iou_head_depth,
                "iou_head_hidden_dim": self.iou_head_hidden_dim,
                "attention_downsample_rate": self.attention_downsample_rate,
                "layer_norm_eps": self.layer_norm_eps,
                "multimask_output": self.multimask_output,
                "data_format": self.data_format,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAMVisionAttention(layers.Layer):
    """Multi-head attention with decomposed relative position bias.

    Computes standard multi-head self-attention over spatial feature maps
    and optionally adds a decomposed relative positional bias that
    factorizes the 2-D bias into independent height and width components.
    Relative position embeddings are bilinearly resized when the runtime
    resolution differs from the pretrained resolution.

    Args:
        hidden_dim (int): Total dimensionality of the attention output.
        num_heads (int): Number of parallel attention heads.
        qkv_bias (bool): Whether to include bias terms in the QKV
            projection. Defaults to True.
        use_rel_pos (bool): Whether to apply decomposed relative positional
            encoding. Defaults to True.
        input_size (tuple): Spatial ``(height, width)`` of the input feature
            map, required when ``use_rel_pos`` is True. Defaults to None.
        data_format (str): One of ``"channels_last"`` or ``"channels_first"``.
            Defaults to ``"channels_last"``.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    References:
        - Segment Anything: https://arxiv.org/abs/2304.02643
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        qkv_bias=True,
        use_rel_pos=True,
        input_size=None,
        data_format="channels_last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv_bias = qkv_bias
        self.use_rel_pos = use_rel_pos
        self.input_size = input_size
        self.data_format = data_format

        self.qkv = layers.Dense(hidden_dim * 3, use_bias=qkv_bias, name="qkv")
        self.proj = layers.Dense(hidden_dim, name="proj")

    def build(self, input_shape):
        if self.use_rel_pos and self.input_size is not None:
            self.rel_pos_h = self.add_weight(
                name="rel_pos_h",
                shape=(2 * self.input_size[0] - 1, self.head_dim),
                initializer="zeros",
                trainable=True,
            )
            self.rel_pos_w = self.add_weight(
                name="rel_pos_w",
                shape=(2 * self.input_size[1] - 1, self.head_dim),
                initializer="zeros",
                trainable=True,
            )
        super().build(input_shape)

    def _get_rel_pos(self, q_size, k_size, rel_pos):
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        rel_pos_shape = ops.shape(rel_pos)
        if rel_pos_shape[0] != max_rel_dist:
            rel_pos_resized = ops.transpose(
                ops.reshape(rel_pos, (1, rel_pos_shape[0], -1)), (0, 2, 1)
            )
            rel_pos_resized = ops.image.resize(
                ops.expand_dims(rel_pos_resized, axis=-1),
                (ops.shape(rel_pos_resized)[1], max_rel_dist),
                interpolation="bilinear",
            )
            rel_pos_resized = ops.squeeze(rel_pos_resized, axis=-1)
            rel_pos_resized = ops.transpose(
                ops.reshape(rel_pos_resized, (-1, max_rel_dist)), (1, 0)
            )
        else:
            rel_pos_resized = rel_pos

        q_coords = ops.cast(
            ops.expand_dims(ops.arange(q_size), axis=1), dtype="float32"
        ) * max(k_size / q_size, 1.0)
        k_coords = ops.cast(
            ops.expand_dims(ops.arange(k_size), axis=0), dtype="float32"
        ) * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(
            q_size / k_size, 1.0
        )
        relative_coords = ops.cast(relative_coords, dtype="int32")
        return ops.take(rel_pos_resized, relative_coords, axis=0)

    def _get_decomposed_rel_pos(self, query, q_size, k_size):
        query_height, query_width = q_size
        key_height, key_width = k_size
        rel_pos_h = self._get_rel_pos(query_height, key_height, self.rel_pos_h)
        rel_pos_w = self._get_rel_pos(query_width, key_width, self.rel_pos_w)

        batch_size = ops.shape(query)[0]
        dim = ops.shape(query)[2]
        reshaped_query = ops.reshape(
            query, (batch_size, query_height, query_width, dim)
        )
        rel_h = ops.einsum("bhwc,hkc->bhwk", reshaped_query, rel_pos_h)
        rel_w = ops.einsum("bhwc,wkc->bhwk", reshaped_query, rel_pos_w)
        return ops.expand_dims(rel_h, axis=-1) + ops.expand_dims(rel_w, axis=-2)

    def call(self, hidden_states):
        cf = self.data_format == "channels_first"

        batch_size = ops.shape(hidden_states)[0]
        if cf:
            height = ops.shape(hidden_states)[2]
            width = ops.shape(hidden_states)[3]
            hidden_states = ops.transpose(hidden_states, (0, 2, 3, 1))
        else:
            height = ops.shape(hidden_states)[1]
            width = ops.shape(hidden_states)[2]

        qkv = self.qkv(hidden_states)
        qkv = ops.reshape(
            qkv, (batch_size, height * width, 3, self.num_heads, self.head_dim)
        )
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        qkv = ops.reshape(
            qkv, (3, batch_size * self.num_heads, height * width, self.head_dim)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]

        attn_weights = ops.matmul(query * self.scale, ops.transpose(key, (0, 2, 1)))

        if self.use_rel_pos:
            decomposed_rel_pos = self._get_decomposed_rel_pos(
                query, (height, width), (height, width)
            )
            decomposed_rel_pos = ops.reshape(
                decomposed_rel_pos, ops.shape(attn_weights)
            )
            attn_weights = attn_weights + decomposed_rel_pos

        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_output = ops.matmul(attn_weights, value)
        attn_output = ops.reshape(
            attn_output, (batch_size, self.num_heads, height, width, self.head_dim)
        )
        attn_output = ops.transpose(attn_output, (0, 2, 3, 1, 4))
        attn_output = ops.reshape(
            attn_output, (batch_size, height, width, self.hidden_dim)
        )
        attn_output = self.proj(attn_output)

        if cf:
            attn_output = ops.transpose(attn_output, (0, 3, 1, 2))

        return attn_output

    def load_own_variables(self, store):
        if not self.use_rel_pos or self.input_size is None:
            super().load_own_variables(store)
            return

        target_vars = self._trainable_variables + self._non_trainable_variables
        rel_h_idx = None
        rel_w_idx = None
        for i, var in enumerate(target_vars):
            if var is self.rel_pos_h:
                rel_h_idx = i
            elif var is self.rel_pos_w:
                rel_w_idx = i

        source_h = (store[str(rel_h_idx)].shape[0] + 1) // 2
        source_w = (store[str(rel_w_idx)].shape[0] + 1) // 2

        if source_h == self.input_size[0] and source_w == self.input_size[1]:
            super().load_own_variables(store)
            return

        for i, var in enumerate(target_vars):
            if i in (rel_h_idx, rel_w_idx):
                continue
            var.assign(store[str(i)])

        for rel_idx, target_size in [
            (rel_h_idx, 2 * self.input_size[0] - 1),
            (rel_w_idx, 2 * self.input_size[1] - 1),
        ]:
            rel_pos = store[str(rel_idx)]
            rel_pos = ops.cast(rel_pos, dtype="float32")
            source_len = rel_pos.shape[0]
            head_dim = rel_pos.shape[1]
            rel_pos = ops.reshape(rel_pos, (1, source_len, head_dim))
            rel_pos = ops.transpose(rel_pos, (0, 2, 1))
            rel_pos = ops.expand_dims(rel_pos, axis=-1)
            rel_pos = ops.image.resize(
                rel_pos,
                size=(head_dim, target_size),
                interpolation="bilinear",
                antialias=True,
            )
            rel_pos = ops.squeeze(rel_pos, axis=-1)
            rel_pos = ops.transpose(rel_pos, (0, 2, 1))
            rel_pos = ops.reshape(rel_pos, (target_size, head_dim))
            target_var = self.rel_pos_h if rel_idx == rel_h_idx else self.rel_pos_w
            target_var.assign(rel_pos)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "qkv_bias": self.qkv_bias,
                "use_rel_pos": self.use_rel_pos,
                "input_size": self.input_size,
                "data_format": self.data_format,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAMVisionLayer(layers.Layer):
    """Single transformer block in the SAM vision encoder.

    Applies layer-normalized multi-head self-attention followed by a
    two-layer MLP with GELU activation. When ``window_size`` is greater
    than zero, attention is restricted to non-overlapping local windows
    that are padded and unpadded around the attention call.

    Args:
        hidden_dim (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads.
        mlp_dim (int): Hidden dimensionality of the feed-forward MLP.
        qkv_bias (bool): Whether to include bias in the QKV projection.
            Defaults to True.
        use_rel_pos (bool): Whether to use decomposed relative positional
            encoding in the attention layer. Defaults to True.
        window_size (int): Size of the local attention window. Set to 0
            for global attention. Defaults to 0.
        image_size (int): Spatial size of the input feature map used to
            determine the relative position embedding size when
            ``window_size`` is 0. Defaults to 64.
        layer_norm_eps (float): Epsilon value for layer normalization.
            Defaults to 1e-6.
        data_format (str): One of ``"channels_last"`` or ``"channels_first"``.
            Defaults to ``"channels_last"``.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    References:
        - Segment Anything: https://arxiv.org/abs/2304.02643
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        mlp_dim,
        qkv_bias=True,
        use_rel_pos=True,
        window_size=0,
        image_size=64,
        layer_norm_eps=1e-6,
        data_format="channels_last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.qkv_bias = qkv_bias
        self.use_rel_pos = use_rel_pos
        self.window_size = window_size
        self.image_size = image_size
        self.layer_norm_eps = layer_norm_eps
        self.data_format = data_format

        if window_size == 0:
            input_size = (image_size, image_size)
        else:
            input_size = (window_size, window_size)

        self.layer_norm1 = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="layer_norm1"
        )
        self.attn = SAMVisionAttention(
            hidden_dim,
            num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size,
            data_format=data_format,
            name="attn",
        )
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="layer_norm2"
        )
        self.mlp_lin1 = layers.Dense(mlp_dim, name="mlp_lin1")
        self.mlp_lin2 = layers.Dense(hidden_dim, name="mlp_lin2")

    def _window_partition(self, hidden_states, window_size):
        cf = self.data_format == "channels_first"

        batch_size = ops.shape(hidden_states)[0]
        if cf:
            channel = ops.shape(hidden_states)[1]
            height = ops.shape(hidden_states)[2]
            width = ops.shape(hidden_states)[3]
        else:
            height = ops.shape(hidden_states)[1]
            width = ops.shape(hidden_states)[2]
            channel = ops.shape(hidden_states)[3]

        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            if cf:
                hidden_states = ops.pad(
                    hidden_states, [[0, 0], [0, 0], [0, pad_h], [0, pad_w]]
                )
            else:
                hidden_states = ops.pad(
                    hidden_states, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]]
                )
        pad_height = height + pad_h
        pad_width = width + pad_w

        if cf:
            hidden_states = ops.transpose(hidden_states, (0, 2, 3, 1))

        hidden_states = ops.reshape(
            hidden_states,
            (
                batch_size,
                pad_height // window_size,
                window_size,
                pad_width // window_size,
                window_size,
                channel,
            ),
        )
        windows = ops.reshape(
            ops.transpose(hidden_states, (0, 1, 3, 2, 4, 5)),
            (-1, window_size, window_size, channel),
        )

        if cf:
            windows = ops.transpose(windows, (0, 3, 1, 2))

        return windows, (pad_height, pad_width)

    def _window_unpartition(self, windows, window_size, padding_shape, original_shape):
        cf = self.data_format == "channels_first"

        pad_height, pad_width = padding_shape
        height, width = original_shape

        if cf:
            windows = ops.transpose(windows, (0, 2, 3, 1))

        batch_size = ops.shape(windows)[0] // (
            pad_height * pad_width // window_size // window_size
        )
        hidden_states = ops.reshape(
            windows,
            (
                batch_size,
                pad_height // window_size,
                pad_width // window_size,
                window_size,
                window_size,
                -1,
            ),
        )
        hidden_states = ops.reshape(
            ops.transpose(hidden_states, (0, 1, 3, 2, 4, 5)),
            (batch_size, pad_height, pad_width, -1),
        )
        hidden_states = hidden_states[:, :height, :width, :]

        if cf:
            hidden_states = ops.transpose(hidden_states, (0, 3, 1, 2))

        return hidden_states

    def call(self, hidden_states):
        cf = self.data_format == "channels_first"

        residual = hidden_states

        if cf:
            hidden_states = ops.transpose(hidden_states, (0, 2, 3, 1))
        hidden_states = self.layer_norm1(hidden_states)
        if cf:
            hidden_states = ops.transpose(hidden_states, (0, 3, 1, 2))

        if self.window_size > 0:
            if cf:
                height = ops.shape(hidden_states)[2]
                width = ops.shape(hidden_states)[3]
            else:
                height = ops.shape(hidden_states)[1]
                width = ops.shape(hidden_states)[2]
            hidden_states, padding_shape = self._window_partition(
                hidden_states, self.window_size
            )

        hidden_states = self.attn(hidden_states)

        if self.window_size > 0:
            hidden_states = self._window_unpartition(
                hidden_states, self.window_size, padding_shape, (height, width)
            )

        hidden_states = residual + hidden_states

        if cf:
            ln_out = ops.transpose(hidden_states, (0, 2, 3, 1))
        else:
            ln_out = hidden_states
        ln_out = self.layer_norm2(ln_out)
        mlp_out = self.mlp_lin1(ln_out)
        mlp_out = ops.nn.gelu(mlp_out, approximate=False)
        mlp_out = self.mlp_lin2(mlp_out)
        if cf:
            mlp_out = ops.transpose(mlp_out, (0, 3, 1, 2))

        hidden_states = hidden_states + mlp_out
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "qkv_bias": self.qkv_bias,
                "use_rel_pos": self.use_rel_pos,
                "window_size": self.window_size,
                "image_size": self.image_size,
                "layer_norm_eps": self.layer_norm_eps,
                "data_format": self.data_format,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAMImagePositionalEmbeddings(layers.Layer):
    """Grid-based positional embeddings for the image feature map.

    Generates a fixed 2-D coordinate grid normalized to ``[0, 1]`` and
    projects it through a shared positional embedding layer to produce
    dense per-pixel positional encodings for the image feature map.

    Args:
        image_embedding_size (int): Spatial size of the square embedding
            grid.
        shared_embedding (Layer): Shared positional embedding layer used to
            encode the normalized coordinate grid.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    References:
        - Segment Anything: https://arxiv.org/abs/2304.02643
    """

    def __init__(self, image_embedding_size, shared_embedding, **kwargs):
        super().__init__(**kwargs)
        self.image_embedding_size = image_embedding_size
        self.shared_embedding = shared_embedding

    def call(self, inputs):
        size = self.image_embedding_size
        grid = ops.ones((size, size), dtype="float32")
        y_embed = ops.cumsum(grid, axis=0) - 0.5
        x_embed = ops.cumsum(grid, axis=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size
        coords = ops.stack([x_embed, y_embed], axis=-1)
        pe = self.shared_embedding(coords)
        pe = ops.transpose(pe, (2, 0, 1))
        pe = ops.expand_dims(pe, axis=0)
        return pe

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_embedding_size": self.image_embedding_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAMPositionalEmbedding(layers.Layer):
    """Random Fourier feature positional encoding for 2-D coordinates.

    Maps normalized 2-D coordinates to a higher-dimensional space using
    random Fourier features. The input coordinates are scaled, multiplied
    by a fixed random Gaussian projection matrix, and encoded with
    concatenated sine and cosine functions.

    Args:
        num_pos_feats (int): Number of positional features per spatial
            dimension (output dimensionality is ``2 * num_pos_feats``).
            Defaults to 128.
        scale (float): Scaling factor applied to the random projection
            matrix standard deviation. Defaults to 1.0.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    References:
        - Segment Anything: https://arxiv.org/abs/2304.02643
    """

    def __init__(self, num_pos_feats=128, scale=None, **kwargs):
        super().__init__(**kwargs)
        self.num_pos_feats = num_pos_feats
        self.scale = scale if scale is not None else 1.0

    def build(self, input_shape):
        self.positional_embedding = self.add_weight(
            name="positional_embedding",
            shape=(2, self.num_pos_feats),
            initializer=keras.initializers.RandomNormal(stddev=self.scale),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, coordinates):
        coordinates = 2.0 * coordinates - 1.0
        coordinates = ops.cast(coordinates, dtype=self.positional_embedding.dtype)
        coordinates = ops.matmul(coordinates, self.positional_embedding)
        coordinates = 2.0 * np.pi * coordinates
        return ops.concatenate([ops.sin(coordinates), ops.cos(coordinates)], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"num_pos_feats": self.num_pos_feats, "scale": self.scale})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class SAMTwoWayAttention(layers.Layer):
    """Multi-head attention for the mask decoder's two-way transformer.

    Performs standard multi-head attention with separate query, key, and
    value projections followed by an output projection. The internal
    dimensionality can be reduced by a configurable downsample rate to
    lower computation in cross-attention layers of the decoder.

    Args:
        hidden_dim (int): Dimensionality of the input and output features.
        num_heads (int): Number of parallel attention heads.
        downsample_rate (int): Factor by which the internal attention
            dimension is reduced relative to ``hidden_dim``.
            Defaults to 1.
        **kwargs: Additional keyword arguments passed to the `Layer` class.

    References:
        - Segment Anything: https://arxiv.org/abs/2304.02643
    """

    def __init__(self, hidden_dim, num_heads, downsample_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.downsample_rate = downsample_rate
        self.internal_dim = hidden_dim // downsample_rate
        self.head_dim = self.internal_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = layers.Dense(self.internal_dim, name="q_proj")
        self.k_proj = layers.Dense(self.internal_dim, name="k_proj")
        self.v_proj = layers.Dense(self.internal_dim, name="v_proj")
        self.out_proj = layers.Dense(hidden_dim, name="out_proj")

    def call(self, query, key, value, attention_similarity=None):
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        query = self._separate_heads(query)
        key = self._separate_heads(key)
        value = self._separate_heads(value)

        attn_weights = ops.matmul(query, ops.transpose(key, (0, 1, 2, 4, 3)))
        attn_weights = attn_weights * self.scale

        if attention_similarity is not None:
            attn_weights = attn_weights + attention_similarity

        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_output = ops.matmul(attn_weights, value)
        attn_output = self._recombine_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        return attn_output

    def _separate_heads(self, x):
        batch = ops.shape(x)[0]
        point_batch = ops.shape(x)[1]
        n_tokens = ops.shape(x)[2]
        x = ops.reshape(
            x, (batch, point_batch, n_tokens, self.num_heads, self.head_dim)
        )
        return ops.transpose(x, (0, 1, 3, 2, 4))

    def _recombine_heads(self, x):
        batch = ops.shape(x)[0]
        point_batch = ops.shape(x)[1]
        n_tokens = ops.shape(x)[3]
        x = ops.transpose(x, (0, 1, 3, 2, 4))
        return ops.reshape(x, (batch, point_batch, n_tokens, self.internal_dim))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "downsample_rate": self.downsample_rate,
            }
        )
        return config
