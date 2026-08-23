import keras
from keras import layers, utils

from kerasformers.base import BaseModel
from kerasformers.utils import standardize_input_shape

from .sam_config import SamConfig
from .sam_layers import (
    SAMAbsolutePositionEmbedding,
    SAMImagePositionalEmbeddings,
    SAMMaskDecoderLayer,
    SAMPositionalEmbedding,
    SAMPromptEncoderLayer,
    SAMVisionLayer,
)

VISION_OUTPUT_CHANNELS = 256
VISION_PATCH_SIZE = 16
VISION_IMAGE_SIZE = 1024
VISION_WINDOW_SIZE = 14
VISION_LAYER_NORM_EPS = 1e-6
MASK_DECODER_HIDDEN_SIZE = 256
MASK_DECODER_NUM_HIDDEN_LAYERS = 2
MASK_DECODER_NUM_ATTENTION_HEADS = 8
MASK_DECODER_MLP_DIM = 2048
MASK_DECODER_IOU_HEAD_DEPTH = 3
MASK_DECODER_IOU_HEAD_HIDDEN_DIM = 256
PROMPT_ENCODER_HIDDEN_SIZE = 256
PROMPT_ENCODER_MASK_INPUT_CHANNELS = 16
PROMPT_ENCODER_NUM_POINT_EMBEDDINGS = 4


def sam_vision_neck(
    inputs, output_channels, data_format="channels_last", name="vision_encoder_neck"
):
    """Projection neck from vision encoder to mask decoder dimension.

    Reference:
        - `Segment Anything <https://arxiv.org/abs/2304.02643>`_
    """
    cf = data_format == "channels_first"
    x = layers.Conv2D(
        output_channels,
        kernel_size=1,
        use_bias=False,
        data_format=data_format,
        name=f"{name}_conv1",
    )(inputs)
    if cf:
        x = layers.Permute((2, 3, 1), name=f"{name}_ln1_pre_permute")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_layer_norm1")(x)
    if cf:
        x = layers.Permute((3, 1, 2), name=f"{name}_ln1_post_permute")(x)
    x = layers.Conv2D(
        output_channels,
        kernel_size=3,
        padding="same",
        use_bias=False,
        data_format=data_format,
        name=f"{name}_conv2",
    )(x)
    if cf:
        x = layers.Permute((2, 3, 1), name=f"{name}_ln2_pre_permute")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_layer_norm2")(x)
    if cf:
        x = layers.Permute((3, 1, 2), name=f"{name}_ln2_post_permute")(x)
    return x


def sam_vision_encoder(
    pixel_values,
    vision_hidden_size,
    vision_num_hidden_layers,
    vision_num_attention_heads,
    vision_mlp_dim,
    vision_global_attn_indexes,
    image_embedding_size,
    data_format,
):
    """Build the SAM ViT vision encoder + neck.

    Patch embed → absolute positional embedding → ``vision_num_hidden_layers``
    SAM ViT layers (windowed attention except at ``vision_global_attn_indexes``)
    → projection neck. Returns the image embeddings used as the
    decoder's key/value memory.

    Reference:
        - `Segment Anything <https://arxiv.org/abs/2304.02643>`_
    """
    hidden_states = layers.Conv2D(
        vision_hidden_size,
        kernel_size=VISION_PATCH_SIZE,
        strides=VISION_PATCH_SIZE,
        padding="valid",
        use_bias=True,
        data_format=data_format,
        name="vision_encoder_patch_embed_projection",
    )(pixel_values)

    hidden_states = SAMAbsolutePositionEmbedding(
        vision_hidden_size,
        image_embedding_size,
        data_format=data_format,
        name="vision_encoder_pos_embed",
    )(hidden_states)

    for i in range(vision_num_hidden_layers):
        win_size = VISION_WINDOW_SIZE if i not in vision_global_attn_indexes else 0
        hidden_states = SAMVisionLayer(
            vision_hidden_size,
            vision_num_attention_heads,
            vision_mlp_dim,
            qkv_bias=True,
            use_rel_pos=True,
            window_size=win_size,
            image_size=image_embedding_size,
            layer_norm_eps=VISION_LAYER_NORM_EPS,
            data_format=data_format,
            name=f"vision_encoder_layers_{i}",
        )(hidden_states)

    return sam_vision_neck(
        hidden_states,
        VISION_OUTPUT_CHANNELS,
        data_format=data_format,
        name="vision_encoder_neck",
    )


def sam_mask_embedding(
    inputs,
    hidden_dim=256,
    mask_input_channels=16,
    layer_norm_eps=1e-6,
    data_format="channels_last",
    name="mask_embed",
):
    """Embeds dense mask prompts through a small convolutional network.

    Two stride-2 convolutions (each followed by a channel-axis layer
    norm and GELU) downsample the input mask by 4×, then a 1×1
    convolution projects to the prompt encoder hidden size.

    Reference:
        - `Segment Anything <https://arxiv.org/abs/2304.02643>`_
    """
    cf = data_format == "channels_first"
    inner_channels = mask_input_channels // 4

    x = layers.Conv2D(
        inner_channels,
        kernel_size=2,
        strides=2,
        data_format=data_format,
        name=f"{name}_conv1",
    )(inputs)
    if cf:
        x = layers.Permute((2, 3, 1), name=f"{name}_ln1_pre_permute")(x)
    x = layers.LayerNormalization(epsilon=layer_norm_eps, name=f"{name}_layer_norm1")(x)
    if cf:
        x = layers.Permute((3, 1, 2), name=f"{name}_ln1_post_permute")(x)
    x = layers.Activation("gelu", name=f"{name}_gelu_1")(x)

    x = layers.Conv2D(
        mask_input_channels,
        kernel_size=2,
        strides=2,
        data_format=data_format,
        name=f"{name}_conv2",
    )(x)
    if cf:
        x = layers.Permute((2, 3, 1), name=f"{name}_ln2_pre_permute")(x)
    x = layers.LayerNormalization(epsilon=layer_norm_eps, name=f"{name}_layer_norm2")(x)
    if cf:
        x = layers.Permute((3, 1, 2), name=f"{name}_ln2_post_permute")(x)
    x = layers.Activation("gelu", name=f"{name}_gelu_2")(x)

    x = layers.Conv2D(
        hidden_dim,
        kernel_size=1,
        data_format=data_format,
        name=f"{name}_conv3",
    )(x)
    return x


@keras.saving.register_keras_serializable(package="kerasformers")
class SAMModel(BaseModel):
    """SAM vision encoder + neck (no prompt encoder, no mask decoder).

    Wraps the ViT vision encoder used by SAM and exposes the image
    embeddings produced by the projection neck. Use this when you
    want to cache image features across many prompt combinations, or
    to plug a custom decoder on top.

    Reference:
        - `Segment Anything <https://arxiv.org/abs/2304.02643>`_

    Args:
        vision_hidden_size: ViT hidden dimension.
        vision_num_hidden_layers: Number of ViT transformer layers.
        vision_num_attention_heads: Attention heads per layer.
        vision_mlp_dim: MLP hidden dim inside each ViT layer.
        vision_global_attn_indexes: Layer indices that use global
            (non-windowed) attention.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `1024`.
        input_tensor: Optional pre-existing Keras input tensor.
        name: Model name.
    """

    BASE_MODEL_CONFIG = None
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "sam"

    @classmethod
    def config_from_hf(cls, hf_config):
        vc = hf_config["vision_config"]
        image_size = vc.get("image_size", VISION_IMAGE_SIZE)
        return {
            "vision_hidden_size": vc["hidden_size"],
            "vision_num_hidden_layers": vc["num_hidden_layers"],
            "vision_num_attention_heads": vc["num_attention_heads"],
            "vision_mlp_dim": vc["mlp_dim"],
            "vision_global_attn_indexes": list(vc["global_attn_indexes"]),
            "image_size": image_size,
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from kerasformers.models.sam.convert_sam_hf_to_keras import (
            transfer_sam_encoder_weights,
        )

        transfer_sam_encoder_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        vision_hidden_size=768,
        vision_num_hidden_layers=12,
        vision_num_attention_heads=12,
        vision_mlp_dim=3072,
        vision_global_attn_indexes=(2, 5, 8, 11),
        image_size=1024,
        input_tensor=None,
        name="SAMModel",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        image_size = standardize_input_shape(image_size, data_format)

        if input_tensor is not None:
            if not utils.is_keras_tensor(input_tensor):
                pixel_values = layers.Input(
                    tensor=input_tensor, shape=image_size, name="pixel_values"
                )
            else:
                pixel_values = input_tensor
        else:
            pixel_values = layers.Input(shape=image_size, name="pixel_values")

        spatial_size = (
            image_size[1] if data_format == "channels_first" else image_size[0]
        )
        image_embedding_size = spatial_size // VISION_PATCH_SIZE

        image_embeddings = sam_vision_encoder(
            pixel_values,
            vision_hidden_size=vision_hidden_size,
            vision_num_hidden_layers=vision_num_hidden_layers,
            vision_num_attention_heads=vision_num_attention_heads,
            vision_mlp_dim=vision_mlp_dim,
            vision_global_attn_indexes=vision_global_attn_indexes,
            image_embedding_size=image_embedding_size,
            data_format=data_format,
        )

        super().__init__(
            inputs=pixel_values, outputs=image_embeddings, name=name, **kwargs
        )

        self.vision_hidden_size = vision_hidden_size
        self.vision_num_hidden_layers = vision_num_hidden_layers
        self.vision_num_attention_heads = vision_num_attention_heads
        self.vision_mlp_dim = vision_mlp_dim
        self.vision_global_attn_indexes = list(vision_global_attn_indexes)
        self.image_embedding_size = image_embedding_size
        self.image_size = image_size
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_hidden_size": self.vision_hidden_size,
                "vision_num_hidden_layers": self.vision_num_hidden_layers,
                "vision_num_attention_heads": self.vision_num_attention_heads,
                "vision_mlp_dim": self.vision_mlp_dim,
                "vision_global_attn_indexes": self.vision_global_attn_indexes,
                "image_size": self.image_size,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="kerasformers")
class SAMPromptableSegment(BaseModel):
    """SAM full promptable segmentation model.

    Composes :class:`SAMModel` with the prompt encoder and mask
    decoder. Takes an image and a set of prompts (points, optionally
    boxes and dense mask inputs) and returns per-prompt mask logits
    and IoU quality scores.

    Reference:
        - `Segment Anything <https://arxiv.org/abs/2304.02643>`_

    Args:
        vision_hidden_size: ViT hidden dimension.
        vision_num_hidden_layers: Number of ViT transformer layers.
        vision_num_attention_heads: Attention heads per layer.
        vision_mlp_dim: MLP hidden dim inside each ViT layer.
        vision_global_attn_indexes: Layer indices that use global
            attention.
        num_multimask_outputs: Number of mask outputs in the multimask
            head (excluding the single best-mask token).
        multimask_output: If ``True`` returns the three multimask
            tokens, else only the single best-mask token.
        enable_boxes: Whether to expose ``input_boxes`` and
            ``has_boxes_input`` model inputs.
        enable_masks: Whether to expose ``input_masks`` and
            ``has_mask_input`` model inputs.
        image_size: Input image specification. Accepts an integer
            ``N`` (builds an ``N x N x 3`` square input), a 2-tuple
            ``(H, W)`` (assumes 3 channels), or a 3-tuple ordered to
            match the active ``keras.config.image_data_format()``:
            ``(H, W, C)`` for ``channels_last`` or ``(C, H, W)`` for
            ``channels_first``. Defaults to `1024`.
        input_tensor: Optional pre-existing Keras input tensor.
        name: Model name.

    Inputs:
        - ``pixel_values``: ``(batch, H, W, 3)``.
        - ``input_points``: ``(batch, point_batch, num_points, 2)``.
        - ``input_labels``: ``(batch, point_batch, num_points)``
          point labels (``1``: foreground, ``0``: background,
          ``-1``: not-a-point pad, ``-10``: ignore).
        - ``input_boxes`` (when ``enable_boxes=True``): ``(batch,
          point_batch, 4)`` boxes in ``(x1, y1, x2, y2)``.
        - ``has_boxes_input`` (when ``enable_boxes=True``):
          ``(batch, 1)`` flag.
        - ``input_masks`` (when ``enable_masks=True``): ``(batch,
          4*emb, 4*emb, 1)`` dense mask prompt.
        - ``has_mask_input`` (when ``enable_masks=True``):
          ``(batch, 1)`` flag.

    Returns:
        Dict outputs:
        - ``"pred_masks"``: ``(batch, point_batch, 3|1, H', W')``.
        - ``"iou_scores"``: ``(batch, point_batch, 3|1)``.
    """

    BASE_MODEL_CONFIG = None
    config_class = SamConfig
    BASE_WEIGHT_CONFIG = None
    HF_MODEL_TYPE = "sam"

    @classmethod
    def config_from_hf(cls, hf_config):
        vc = hf_config["vision_config"]
        image_size = vc.get("image_size", VISION_IMAGE_SIZE)
        return {
            "vision_hidden_size": vc["hidden_size"],
            "vision_num_hidden_layers": vc["num_hidden_layers"],
            "vision_num_attention_heads": vc["num_attention_heads"],
            "vision_mlp_dim": vc["mlp_dim"],
            "vision_global_attn_indexes": list(vc["global_attn_indexes"]),
            "image_size": image_size,
        }

    @classmethod
    def transfer_from_hf(cls, keras_model, hf_state_dict):
        from kerasformers.models.sam.convert_sam_hf_to_keras import transfer_sam_weights

        transfer_sam_weights(keras_model, hf_state_dict)

    def __init__(
        self,
        vision_hidden_size=768,
        vision_num_hidden_layers=12,
        vision_num_attention_heads=12,
        vision_mlp_dim=3072,
        vision_global_attn_indexes=(2, 5, 8, 11),
        num_multimask_outputs=3,
        multimask_output=True,
        enable_boxes=False,
        enable_masks=False,
        image_size=1024,
        input_tensor=None,
        name="SAMPromptableSegment",
        **kwargs,
    ):
        data_format = keras.config.image_data_format()
        image_size = standardize_input_shape(image_size, data_format)

        spatial_size = (
            image_size[1] if data_format == "channels_first" else image_size[0]
        )
        image_embedding_size = spatial_size // VISION_PATCH_SIZE
        mask_input_size = image_embedding_size * 4

        if data_format == "channels_first":
            image_embed_in_shape = (
                VISION_OUTPUT_CHANNELS,
                image_embedding_size,
                image_embedding_size,
            )
            mask_in_shape = (1, mask_input_size, mask_input_size)
        else:
            image_embed_in_shape = (
                image_embedding_size,
                image_embedding_size,
                VISION_OUTPUT_CHANNELS,
            )
            mask_in_shape = (mask_input_size, mask_input_size, 1)

        if input_tensor is not None:
            if not utils.is_keras_tensor(input_tensor):
                pixel_values = layers.Input(
                    tensor=input_tensor, shape=image_size, name="pixel_values"
                )
            else:
                pixel_values = input_tensor
        else:
            pixel_values = layers.Input(shape=image_size, name="pixel_values")

        input_points = layers.Input(
            shape=(None, None, 2), name="input_points", dtype="float32"
        )
        input_labels = layers.Input(
            shape=(None, None), name="input_labels", dtype="int32"
        )

        input_boxes = None
        has_boxes_input = None
        if enable_boxes:
            input_boxes = layers.Input(
                shape=(None, 4), name="input_boxes", dtype="float32"
            )
            has_boxes_input = layers.Input(
                shape=(1,), name="has_boxes_input", dtype="float32"
            )

        input_masks = None
        has_mask_input = None
        input_mask_embedding = None
        if enable_masks:
            input_masks = layers.Input(
                shape=mask_in_shape, name="input_masks", dtype="float32"
            )
            has_mask_input = layers.Input(
                shape=(1,), name="has_mask_input", dtype="float32"
            )

        image_embeddings = sam_vision_encoder(
            pixel_values,
            vision_hidden_size=vision_hidden_size,
            vision_num_hidden_layers=vision_num_hidden_layers,
            vision_num_attention_heads=vision_num_attention_heads,
            vision_mlp_dim=vision_mlp_dim,
            vision_global_attn_indexes=vision_global_attn_indexes,
            image_embedding_size=image_embedding_size,
            data_format=data_format,
        )

        num_pos_feats = 128
        shared_image_embedding = SAMPositionalEmbedding(
            num_pos_feats=num_pos_feats,
            scale=vision_hidden_size // 2,
            name="shared_image_embedding",
        )

        image_pe_layer = SAMImagePositionalEmbeddings(
            image_embedding_size,
            shared_image_embedding,
            name="image_positional_embeddings",
        )
        image_pe = image_pe_layer(image_embeddings)

        if enable_masks:
            input_mask_embedding = sam_mask_embedding(
                input_masks,
                hidden_dim=PROMPT_ENCODER_HIDDEN_SIZE,
                mask_input_channels=PROMPT_ENCODER_MASK_INPUT_CHANNELS,
                layer_norm_eps=VISION_LAYER_NORM_EPS,
                data_format=data_format,
                name="prompt_encoder_mask_embed",
            )

        prompt_encoder_layer = SAMPromptEncoderLayer(
            hidden_dim=PROMPT_ENCODER_HIDDEN_SIZE,
            image_embedding_size=image_embedding_size,
            image_size=VISION_IMAGE_SIZE,
            num_point_embeddings=PROMPT_ENCODER_NUM_POINT_EMBEDDINGS,
            shared_embedding=shared_image_embedding,
            enable_boxes=enable_boxes,
            enable_masks=enable_masks,
            data_format=data_format,
            name="prompt_encoder",
        )

        def _prompt_inputs_dict():
            d = {"input_points": input_points, "input_labels": input_labels}
            if enable_boxes:
                d["input_boxes"] = input_boxes
                d["has_boxes_input"] = has_boxes_input
            if enable_masks:
                d["input_mask_embedding"] = input_mask_embedding
                d["has_mask_input"] = has_mask_input
            return d

        prompt_results = prompt_encoder_layer(_prompt_inputs_dict())
        sparse_embeddings = prompt_results["sparse_embeddings"]
        dense_embeddings = prompt_results["dense_embeddings"]

        mask_decoder_layer = SAMMaskDecoderLayer(
            hidden_dim=MASK_DECODER_HIDDEN_SIZE,
            num_hidden_layers=MASK_DECODER_NUM_HIDDEN_LAYERS,
            num_heads=MASK_DECODER_NUM_ATTENTION_HEADS,
            mlp_dim=MASK_DECODER_MLP_DIM,
            num_multimask_outputs=num_multimask_outputs,
            iou_head_depth=MASK_DECODER_IOU_HEAD_DEPTH,
            iou_head_hidden_dim=MASK_DECODER_IOU_HEAD_HIDDEN_DIM,
            multimask_output=multimask_output,
            data_format=data_format,
            name="mask_decoder",
        )
        decoder_output = mask_decoder_layer(
            [image_embeddings, image_pe, sparse_embeddings, dense_embeddings]
        )
        pred_masks = decoder_output["pred_masks"]
        iou_scores = decoder_output["iou_scores"]

        main_inputs = {
            "pixel_values": pixel_values,
            "input_points": input_points,
            "input_labels": input_labels,
        }
        if enable_boxes:
            main_inputs["input_boxes"] = input_boxes
            main_inputs["has_boxes_input"] = has_boxes_input
        if enable_masks:
            main_inputs["input_masks"] = input_masks
            main_inputs["has_mask_input"] = has_mask_input

        super().__init__(
            inputs=main_inputs,
            outputs={"pred_masks": pred_masks, "iou_scores": iou_scores},
            name=name,
            **kwargs,
        )

        self.vision_hidden_size = vision_hidden_size
        self.vision_num_hidden_layers = vision_num_hidden_layers
        self.vision_num_attention_heads = vision_num_attention_heads
        self.vision_mlp_dim = vision_mlp_dim
        self.vision_global_attn_indexes = list(vision_global_attn_indexes)
        self.num_multimask_outputs = num_multimask_outputs
        self.multimask_output = multimask_output
        self.enable_boxes = enable_boxes
        self.enable_masks = enable_masks
        self.image_embedding_size = image_embedding_size
        self.mask_input_size = mask_input_size
        self.image_size = image_size
        self.input_tensor = input_tensor

        self._prompt_encoder_layer = prompt_encoder_layer
        self._mask_decoder_layer = mask_decoder_layer
        self._image_pe_layer = image_pe_layer

        image_embeddings_input = layers.Input(
            shape=image_embed_in_shape,
            name="image_embeddings",
            dtype=pixel_values.dtype,
        )
        decoder_side_image_pe = image_pe_layer(image_embeddings_input)
        decoder_prompt_results = prompt_encoder_layer(_prompt_inputs_dict())
        decoder_side_outputs = mask_decoder_layer(
            [
                image_embeddings_input,
                decoder_side_image_pe,
                decoder_prompt_results["sparse_embeddings"],
                decoder_prompt_results["dense_embeddings"],
            ]
        )

        prompt_decoder_inputs = {
            "image_embeddings": image_embeddings_input,
            "input_points": input_points,
            "input_labels": input_labels,
        }
        prompt_encoder_inputs = {
            "input_points": input_points,
            "input_labels": input_labels,
        }
        if enable_boxes:
            prompt_decoder_inputs["input_boxes"] = input_boxes
            prompt_decoder_inputs["has_boxes_input"] = has_boxes_input
            prompt_encoder_inputs["input_boxes"] = input_boxes
            prompt_encoder_inputs["has_boxes_input"] = has_boxes_input
        if enable_masks:
            prompt_decoder_inputs["input_masks"] = input_masks
            prompt_decoder_inputs["has_mask_input"] = has_mask_input
            prompt_encoder_inputs["input_masks"] = input_masks
            prompt_encoder_inputs["has_mask_input"] = has_mask_input

        self.prompt_decoder_model = keras.Model(
            inputs=prompt_decoder_inputs,
            outputs={
                "pred_masks": decoder_side_outputs["pred_masks"],
                "iou_scores": decoder_side_outputs["iou_scores"],
            },
            name=f"{name}_prompt_decoder",
        )

        self.vision_encoder_model = keras.Model(
            inputs=pixel_values,
            outputs=image_embeddings,
            name=f"{name}_vision_encoder",
        )

        self.prompt_encoder_model = keras.Model(
            inputs=prompt_encoder_inputs,
            outputs={
                "sparse_embeddings": prompt_results["sparse_embeddings"],
                "dense_embeddings": prompt_results["dense_embeddings"],
            },
            name=f"{name}_prompt_encoder_model",
        )

    def get_image_embeddings(self, pixel_values):
        """Run only the vision encoder to produce image embeddings.

        Use this to cache image features once and reuse them across
        many prompt combinations via :attr:`prompt_decoder_model`.
        """
        return self.vision_encoder_model(pixel_values)

    def get_prompt_embeddings(
        self,
        input_points,
        input_labels,
        input_boxes=None,
        input_masks=None,
        has_boxes_input=None,
        has_mask_input=None,
    ):
        """Run only the prompt encoder.

        Returns the sparse and dense prompt embeddings without
        invoking the mask decoder.
        """
        inputs = {"input_points": input_points, "input_labels": input_labels}
        if self.enable_boxes:
            inputs["input_boxes"] = input_boxes
            inputs["has_boxes_input"] = has_boxes_input
        if self.enable_masks:
            inputs["input_masks"] = input_masks
            inputs["has_mask_input"] = has_mask_input
        return self.prompt_encoder_model(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_hidden_size": self.vision_hidden_size,
                "vision_num_hidden_layers": self.vision_num_hidden_layers,
                "vision_num_attention_heads": self.vision_num_attention_heads,
                "vision_mlp_dim": self.vision_mlp_dim,
                "vision_global_attn_indexes": self.vision_global_attn_indexes,
                "num_multimask_outputs": self.num_multimask_outputs,
                "multimask_output": self.multimask_output,
                "enable_boxes": self.enable_boxes,
                "enable_masks": self.enable_masks,
                "image_size": self.image_size,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
