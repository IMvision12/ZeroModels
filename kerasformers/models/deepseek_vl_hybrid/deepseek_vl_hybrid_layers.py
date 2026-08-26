import keras
from keras import layers, ops

from kerasformers.models.sam.sam_layers import (
    SAMAbsolutePositionEmbedding,
    SAMVisionLayer,
)


@keras.saving.register_keras_serializable(package="kerasformers")
class DeepseekVLHybridSamEncoder(layers.Layer):
    """SAM / ViTDet-B high-res vision encoder (@1024).

    Reuses the SAM windowed-attention blocks (``SAMVisionLayer``: decomposed 2D
    relative position embeddings, window attention except at the global-attention
    indices) and absolute position embedding. Unlike the functional SAM builder,
    this returns BOTH the final neck output (``last_hidden_state``, 256-d) AND the
    output of the first global-attention block (``global_hidden_state``, hidden-d):
    DeepSeek-VL Hybrid's fusion needs that intermediate state.

    Call args:
        pixel_values: ``(B, 1024, 1024, 3)`` (channels-last).

    Returns:
        ``(last_hidden_state (B, 64, 64, output_channels),
        global_hidden_state (B, 64, 64, hidden_size))``.
    """

    def __init__(
        self,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        image_size=1024,
        patch_size=16,
        output_channels=256,
        window_size=14,
        global_attn_indexes=(2, 5, 8, 11),
        norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.output_channels = output_channels
        self.window_size = window_size
        self.global_attn_indexes = tuple(global_attn_indexes)
        self.norm_eps = norm_eps
        self.grid_size = image_size // patch_size
        self.data_format = keras.config.image_data_format()

        self.patch_embed = layers.Conv2D(
            hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            data_format="channels_last",
            name="patch_embed",
        )
        self.pos_embed = SAMAbsolutePositionEmbedding(
            hidden_size,
            self.grid_size,
            data_format="channels_last",
            name="pos_embed",
        )
        self.blocks = [
            SAMVisionLayer(
                hidden_size,
                num_heads,
                mlp_dim,
                qkv_bias=True,
                use_rel_pos=True,
                window_size=0 if i in self.global_attn_indexes else window_size,
                image_size=self.grid_size,
                layer_norm_eps=norm_eps,
                data_format="channels_last",
                name=f"layers_{i}",
            )
            for i in range(num_layers)
        ]
        # SAM projection neck (768 -> 256), channels-last LayerNorm == HF's
        # channels-first LN over the channel axis.
        self.neck_conv1 = layers.Conv2D(
            output_channels,
            1,
            use_bias=False,
            data_format="channels_last",
            name="neck_conv1",
        )
        self.neck_ln1 = layers.LayerNormalization(epsilon=norm_eps, name="neck_ln1")
        self.neck_conv2 = layers.Conv2D(
            output_channels,
            3,
            padding="same",
            use_bias=False,
            data_format="channels_last",
            name="neck_conv2",
        )
        self.neck_ln2 = layers.LayerNormalization(epsilon=norm_eps, name="neck_ln2")

    def call(self, pixel_values):
        if self.data_format == "channels_first":
            pixel_values = ops.transpose(pixel_values, (0, 2, 3, 1))
        x = self.patch_embed(pixel_values)
        x = self.pos_embed(x)
        global_hidden_state = None
        first_global = self.global_attn_indexes[0]
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == first_global:
                global_hidden_state = x
        neck = self.neck_conv1(x)
        neck = self.neck_ln1(neck)
        neck = self.neck_conv2(neck)
        neck = self.neck_ln2(neck)
        return neck, global_hidden_state

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "output_channels": self.output_channels,
                "window_size": self.window_size,
                "global_attn_indexes": self.global_attn_indexes,
                "norm_eps": self.norm_eps,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class DeepseekVLSamVisionNeck(layers.Layer):
    """High-res neck applied to the intermediate global-attention state.

    Same shape as the SAM encoder's own neck (hidden -> output_channels via a
    1x1 conv + LN + 3x3 conv + LN), but a separate module
    (``high_res_vision_neck``) operating on ``global_hidden_state``.
    """

    def __init__(self, output_channels=256, norm_eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.output_channels = output_channels
        self.norm_eps = norm_eps
        self.conv1 = layers.Conv2D(
            output_channels,
            1,
            use_bias=False,
            data_format="channels_last",
            name="conv1",
        )
        self.layer_norm1 = layers.LayerNormalization(
            epsilon=norm_eps, name="layer_norm1"
        )
        self.conv2 = layers.Conv2D(
            output_channels,
            3,
            padding="same",
            use_bias=False,
            data_format="channels_last",
            name="conv2",
        )
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=norm_eps, name="layer_norm2"
        )

    def call(self, hidden_states):
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {"output_channels": self.output_channels, "norm_eps": self.norm_eps}
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class DeepseekVLSamVisionProj(layers.Layer):
    """Project a high-res feature map to the SigLIP token grid (24x24).

    Bilinearly resizes to ``(4*output_size, 4*output_size)`` then two stride-2
    3x3 convs (output_channels -> *2 -> *4), giving a ``(B, output_size,
    output_size, output_channels*4)`` map. Shared (applied to both the encoder's
    final state and the necked global state).
    """

    def __init__(self, output_channels=256, output_size=24, **kwargs):
        super().__init__(**kwargs)
        self.output_channels = output_channels
        self.output_size = output_size
        self.conv1 = layers.Conv2D(
            output_channels * 2,
            3,
            strides=2,
            padding="same",
            use_bias=False,
            data_format="channels_last",
            name="conv1",
        )
        self.conv2 = layers.Conv2D(
            output_channels * 4,
            3,
            strides=2,
            padding="same",
            use_bias=False,
            data_format="channels_last",
            name="conv2",
        )

    def call(self, features):
        size = 4 * self.output_size
        features = ops.image.resize(
            features,
            (size, size),
            interpolation="bilinear",
            antialias=False,
            data_format="channels_last",
        )
        features = self.conv1(features)
        features = self.conv2(features)
        return features

    def get_config(self):
        config = super().get_config()
        config.update(
            {"output_channels": self.output_channels, "output_size": self.output_size}
        )
        return config


@keras.saving.register_keras_serializable(package="kerasformers")
class DeepseekVLHybridAligner(layers.Layer):
    """Fuse SigLIP (low-res) + SAM (high-res) tokens into the text width.

    ``vision_proj`` and ``high_res_vision_proj`` each map their stream to
    ``out_dim // 2``; the two are concatenated (high-res FIRST, matching HF),
    GELU'd, and mixed by ``proj`` into ``out_dim``.
    """

    def __init__(self, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim
        self.vision_proj = layers.Dense(out_dim // 2, name="vision_proj")
        self.high_res_vision_proj = layers.Dense(
            out_dim // 2, name="high_res_vision_proj"
        )
        self.proj = layers.Dense(out_dim, name="proj")

    def call(self, vision_encodings, high_res_vision_encodings):
        vision_encodings = self.vision_proj(vision_encodings)
        high_res_vision_encodings = self.high_res_vision_proj(high_res_vision_encodings)
        encodings = ops.concatenate(
            [high_res_vision_encodings, vision_encodings], axis=-1
        )
        encodings = ops.gelu(encodings, approximate=False)
        return self.proj(encodings)

    def get_config(self):
        config = super().get_config()
        config.update({"out_dim": self.out_dim})
        return config
