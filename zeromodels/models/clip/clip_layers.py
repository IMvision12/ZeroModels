import keras
from keras import layers, ops

from zeromodels.base.base_attention import fused_attention


@keras.saving.register_keras_serializable(package="zeromodels")
class CLIPAddPositionEmbs(layers.Layer):
    """
    A custom Keras layer that adds learnable position embeddings to input tensors with support for
    flexible grid sizes and optional class/distillation token handling.

    The layer supports three modes of operation:
      1. Standard mode: Creates position embeddings for both patches and a class token.
      2. FlexiViT mode: Creates embeddings only for patches, handling a class token separately.
         In this mode, the layer accepts either a pure patch token input (grid_h * grid_w tokens)
         or an input with a leading class token (grid_h * grid_w + 1 tokens).
      3. DeiT mode: Creates embeddings for patches, a class token, and a distillation token.
         In this case the expected sequence length is grid_h * grid_w + 2 tokens.

    For PiT, the recommended usage is to set `no_embed_class=True` so that the patch tokens
    are first augmented with positional embeddings. Then, the class (or combined class/distillation)
    token is added later in the model pipeline.

    Features:
      - Configurable grid dimensions for position embeddings.
      - Dynamic resizing of position embeddings during loading through bilinear interpolation.
      - Compatible with standard Vision Transformer, FlexiViT, DeiT, and PiT architectures.
      - Handles class and distillation token embeddings appropriately based on mode.
      - When no_embed_class=True, the layer applies positional embeddings only to patch tokens.
        If a class token is present at the beginning, it is preserved and concatenated back after
        the patch tokens are positionally embedded.

    Args:
        grid_h (int): Height of the position embedding grid.
        grid_w (int): Width of the position embedding grid.
        no_embed_class (bool):
            - If False (default), operates in standard mode where position embeddings are learned for
              the entire input (patches plus token(s)).
            - If True, operates in FlexiViT or PiT mode where positional embeddings are applied only to
              the patch tokens. In this mode the input can be either a sequence of patch tokens only
              (grid_h * grid_w tokens) or a sequence with a leading class token (grid_h * grid_w + 1 tokens).
        use_distillation (bool):
            If True, operates in DeiT mode with an additional distillation token.
            When no_embed_class is False, the expected input sequence length is grid_h * grid_w + 2.
            Defaults to False.
        name (str, optional): Name of the layer.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Input Shape:
        3D tensor with shape: `(batch_size, sequence_length, embed_dim)`, where sequence_length should be:
          * Standard mode: grid_h * grid_w + 1 (patch tokens + class token)
          * FlexiViT mode / PiT mode: either grid_h * grid_w (patch tokens only) or grid_h * grid_w + 1
          * DeiT mode: grid_h * grid_w + 2 (patch tokens + class token + distillation token)

    Output Shape:
        Same as the input shape: `(batch_size, sequence_length, embed_dim)`. In FlexiViT or PiT mode,
        if a class token is present at the beginning, it is preserved and positional embeddings are added
        only to the patch tokens.

    The layer supports loading weights from different grid sizes through bilinear interpolation,
    making it flexible for transfer learning and model adaptation scenarios. When loading weights,
    it automatically handles the conversion between different modes (standard / FlexiViT / DeiT),
    adjusting the position embeddings appropriately.
    """

    def __init__(
        self,
        grid_h,
        grid_w,
        no_embed_class=False,
        use_distillation=False,
        resize_mode="bilinear",
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.no_embed_class = no_embed_class
        self.use_distillation = use_distillation
        self.resize_mode = resize_mode

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape, received: {len(input_shape)}D")

        num_patches = self.grid_h * self.grid_w

        if self.no_embed_class:
            if input_shape[1] == num_patches:
                self.skip_cls = False
            elif input_shape[1] == num_patches + 1:
                self.skip_cls = True
            else:
                raise ValueError(
                    f"Input sequence length {input_shape[1]} does not match expected length "
                    f"{num_patches} or {num_patches + 1} (grid: {self.grid_h}x{self.grid_w})"
                )
            self.position_embedding = self.add_weight(
                name="pos_embed",
                shape=(1, num_patches, input_shape[2]),
                initializer="random_normal",
                trainable=True,
            )
        else:
            expected_length = num_patches + (2 if self.use_distillation else 1)
            if input_shape[1] != expected_length:
                raise ValueError(
                    f"Input sequence length {input_shape[1]} does not match expected length "
                    f"{expected_length} (grid: {self.grid_h}x{self.grid_w} + tokens)"
                )
            self.position_embedding = self.add_weight(
                name="pos_embed",
                shape=(1, expected_length, input_shape[2]),
                initializer="random_normal",
                trainable=True,
            )
        super().build(input_shape)

    def call(self, inputs):
        if self.no_embed_class:
            if hasattr(self, "skip_cls") and self.skip_cls:
                cls_token = inputs[:, :1]
                patch_tokens = inputs[:, 1:]
                patch_tokens = patch_tokens + self.position_embedding
                return ops.concatenate([cls_token, patch_tokens], axis=1)
            else:
                return inputs + self.position_embedding
        elif self.use_distillation:
            tokens = inputs[:, :2]
            patch_tokens = inputs[:, 2:]
            token_pos_embed = self.position_embedding[:, :2]
            patch_pos_embed = self.position_embedding[:, 2:]
            tokens = tokens + token_pos_embed
            patch_tokens = patch_tokens + patch_pos_embed
            return ops.concatenate([tokens, patch_tokens], axis=1)
        else:
            return inputs + self.position_embedding

    def compute_output_shape(self, input_shape):
        return input_shape

    def save_own_variables(self, store):
        super().save_own_variables(store)
        store["grid_h"] = self.grid_h
        store["grid_w"] = self.grid_w
        store["no_embed_class"] = self.no_embed_class
        store["use_distillation"] = self.use_distillation

    def load_own_variables(self, store):
        source_h = int(store["grid_h"][...])
        source_w = int(store["grid_w"][...])

        try:
            source_no_embed_class = bool(store["no_embed_class"][...])
        except KeyError:
            source_no_embed_class = False

        try:
            source_use_distillation = bool(store["use_distillation"][...])
        except KeyError:
            source_use_distillation = False

        if source_h == self.grid_h and source_w == self.grid_w:
            self.position_embedding.assign(store["0"])
            return

        pos_embed = store["0"]

        if not source_no_embed_class:
            if source_use_distillation:
                spatial_pos_embed = pos_embed[:, 2:]
                token_pos_embed = pos_embed[:, :2]
            else:
                spatial_pos_embed = pos_embed[:, 1:]
                token_pos_embed = pos_embed[:, :1]
        else:
            spatial_pos_embed = pos_embed

        embed_dim = spatial_pos_embed.shape[-1]

        spatial_pos_embed = ops.cast(spatial_pos_embed, dtype="float32")
        spatial_pos_embed = ops.reshape(
            spatial_pos_embed, [1, source_h, source_w, embed_dim]
        )

        spatial_pos_embed = ops.image.resize(
            spatial_pos_embed,
            size=[self.grid_h, self.grid_w],
            interpolation=self.resize_mode,
            antialias=True,
        )

        spatial_pos_embed = ops.reshape(
            spatial_pos_embed, [1, self.grid_h * self.grid_w, embed_dim]
        )

        if self.no_embed_class:
            pos_embed = spatial_pos_embed
        else:
            if not source_no_embed_class:
                if self.use_distillation:
                    if source_use_distillation:
                        cls_dist_pos_embed = token_pos_embed
                    else:
                        cls_dist_pos_embed = ops.concatenate(
                            [token_pos_embed, token_pos_embed], axis=1
                        )
                else:
                    if source_use_distillation:
                        cls_dist_pos_embed = token_pos_embed[:, :1]
                    else:
                        cls_dist_pos_embed = token_pos_embed
            else:
                if self.use_distillation:
                    cls_dist_pos_embed = ops.zeros((1, 2, embed_dim))
                else:
                    cls_dist_pos_embed = ops.zeros((1, 1, embed_dim))

            pos_embed = ops.concatenate([cls_dist_pos_embed, spatial_pos_embed], axis=1)

        self.position_embedding.assign(pos_embed)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "grid_h": self.grid_h,
                "grid_w": self.grid_w,
                "no_embed_class": self.no_embed_class,
                "use_distillation": self.use_distillation,
                "resize_mode": self.resize_mode,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class CLIPAttention(keras.layers.Layer):
    """Multi-head attention layer for CLIP model implementing scaled dot-product attention.

    This layer implements the multi-head attention mechanism used in the CLIP architecture.
    It projects input tensors into query, key, and value representations, applies
    scaled dot-product attention, and projects the output back to the original dimension.

    Key Features:
        - Independent parallel attention heads for capturing different relationship patterns
        - Scaled dot-product attention with optional attention masking
        - Separate projection matrices for query, key, and value transformations
        - Customizable projection dimensions and number of attention heads
        - Support for sequential inputs with variable sequence lengths

    Args:
        proj_dim (int): Dimension of the projection space. Must be divisible
            by num_heads to ensure even distribution of features across heads
        num_heads (int): Number of parallel attention heads. Each head operates
            on proj_dim/num_heads features
        name_prefix (str, optional): Prefix for naming layer components. Defaults to None
        **kwargs: Additional keyword arguments passed to the parent Layer class

    Input shape:
        - hidden_states: Tensor of shape (batch_size, sequence_length, input_dim)
        - attention_mask: Optional tensor for masking certain positions

    Output shape:
        - Tuple containing tensor of shape (batch_size, sequence_length, proj_dim)

    Notes:
        - The projection dimension (proj_dim) must be divisible by num_heads
        - Each attention head processes proj_dim/num_heads features
        - Implements the standard scaled dot-product attention formula:
          Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
        - Used in CLIP's text and image encoders for contextual feature extraction
    """

    def __init__(self, proj_dim, num_heads, name_prefix=None, **kwargs):
        super().__init__(**kwargs)
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.name_prefix = name_prefix
        self.head_dim = proj_dim // num_heads
        self.scale = self.head_dim**-0.5

        assert proj_dim % num_heads == 0, "proj_dim should be divisible by num_heads"

        q_proj_name = f"{self.name_prefix}_q_proj" if self.name_prefix else "q_proj"
        k_proj_name = f"{self.name_prefix}_k_proj" if self.name_prefix else "k_proj"
        v_proj_name = f"{self.name_prefix}_v_proj" if self.name_prefix else "v_proj"
        out_proj_name = (
            f"{self.name_prefix}_out_proj" if self.name_prefix else "out_proj"
        )

        self.q_proj = keras.layers.Dense(self.proj_dim, use_bias=True, name=q_proj_name)
        self.k_proj = keras.layers.Dense(self.proj_dim, use_bias=True, name=k_proj_name)
        self.v_proj = keras.layers.Dense(self.proj_dim, use_bias=True, name=v_proj_name)
        self.out_proj = keras.layers.Dense(
            self.proj_dim, use_bias=True, name=out_proj_name
        )

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.q_proj.build((None, input_dim))
        self.k_proj.build((None, input_dim))
        self.v_proj.build((None, input_dim))
        self.out_proj.build((None, self.proj_dim))

        self.built = True

    def transpose_for_scores(self, x):
        batch_size = ops.shape(x)[0]
        seq_length = ops.shape(x)[1]
        x = ops.reshape(x, (batch_size, seq_length, self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def call(self, hidden_states, attention_mask=None):
        batch_size = ops.shape(hidden_states)[0]

        x_q = self.q_proj(hidden_states)
        x_k = self.k_proj(hidden_states)
        x_v = self.v_proj(hidden_states)

        x_q = self.transpose_for_scores(x_q)
        x_k = self.transpose_for_scores(x_k)
        x_v = self.transpose_for_scores(x_v)

        x = fused_attention(x_q, x_k, x_v, self.scale, attention_mask)
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (batch_size, -1, self.proj_dim))
        x = self.out_proj(x)

        return (x,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "proj_dim": self.proj_dim,
                "num_heads": self.num_heads,
                "name_prefix": self.name_prefix,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class CLIPVisionModelEmbedding(keras.layers.Layer):
    """Vision Transformer (ViT) embedding layer that processes image patches.

    This layer follows the Vision Transformer architecture by:
    1. Converting an input image into patches
    2. Adding a special class token embedding (similar to BERT's [CLS] token)
    3. Adding learned positional embeddings to provide spatial information

    The input to this layer should be patch embeddings from an image after
    initial projection to the embedding dimension.

    Args:
        width (int): Dimension of the embedding space.
        input_resolution (int): Resolution of the input image (assumes square images).
        patch_size (int): Size of each image patch (assumes square patches).
        data_format: string, either 'channels_last' or 'channels_first',
            specifies the input data format.
        **kwargs: Additional keyword arguments passed to the parent class.

    Inputs:
        A tensor of shape (batch_size, num_patches, width) representing
        the projected patch embeddings from an image.

    Outputs:
        A tensor of shape (batch_size, num_patches + 1, width) containing
        the patch embeddings plus class token, with positional embeddings added.
    """

    def __init__(
        self, width, input_resolution, patch_size, data_format="channels_last", **kwargs
    ):
        super().__init__(**kwargs)
        self.width = width
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.data_format = data_format
        self.num_patches = (input_resolution // patch_size) ** 2
        self.grid_size = input_resolution // patch_size

        self.position_embs = CLIPAddPositionEmbs(
            grid_h=self.grid_size,
            grid_w=self.grid_size,
            no_embed_class=False,
            use_distillation=False,
            name="position_embeddings",
        )

    def build(self, input_shape):
        self.class_embedding = self.add_weight(
            shape=((self.width,)),
            name="class_embedding",
        )

        super().build(input_shape)

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        if self.data_format == "channels_first":
            patch_embeddings = keras.layers.Reshape((self.width, self.num_patches))(
                inputs
            )
            patch_embeddings = keras.layers.Permute((2, 1))(patch_embeddings)
        else:
            patch_embeddings = keras.layers.Reshape((self.num_patches, self.width))(
                inputs
            )
        class_embed = ops.broadcast_to(
            self.class_embedding, (batch_size, 1, self.width)
        )
        embeddings = ops.concatenate([class_embed, patch_embeddings], axis=1)
        embeddings = self.position_embs(embeddings)

        return embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "width": self.width,
                "input_resolution": self.input_resolution,
                "patch_size": self.patch_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class CLIPTextModelEmbedding(keras.layers.Layer):
    """
    A Keras layer that combines token embeddings and positional embeddings for text models.

    This layer is commonly used in transformer-based architectures such as BERT, GPT, and others.
    It performs two key operations:
    1. Converts token IDs to token embeddings using a learned embedding table
    2. Adds positional embeddings to encode position information in the sequence

    The final output is the sum of token embeddings and positional embeddings.

    Args:
        vocab_size (int): Size of the vocabulary, determining the number of unique tokens
        max_seq_len (int): Maximum sequence length to handle
        embed_dim (int): Dimensionality of the embedding vectors
        **kwargs: Additional keyword arguments passed to the parent Layer class

    Input shape:
        Integer tensor of shape (batch_size, sequence_length) with token IDs

    Output shape:
        Float tensor of shape (batch_size, sequence_length, embed_dim)
    """

    def __init__(self, vocab_size, max_seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        self.token_embedding = keras.layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )

        self.position_embedding = keras.layers.Embedding(
            max_seq_len, embed_dim, name="positional_embedding"
        )

    def call(self, inputs):
        token_embeddings = self.token_embedding(inputs)
        batch_size = ops.shape(inputs)[0]
        position_ids = ops.arange(self.max_seq_len, dtype="int32")
        position_ids = ops.expand_dims(position_ids, 0)
        position_embeddings = self.position_embedding(position_ids)
        position_embeddings = ops.tile(position_embeddings, (batch_size, 1, 1))
        return token_embeddings + position_embeddings

    def build(self, input_shape):
        self.token_embedding.build((None, self.max_seq_len))
        self.position_embedding.build((None, self.max_seq_len))
        self.built = True
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.max_seq_len, self.embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "max_seq_len": self.max_seq_len,
                "embed_dim": self.embed_dim,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class CLIPLogitScale(keras.layers.Layer):
    """
    Learnable temperature parameter for scaling logits in CLIP models.

    This layer implements the learnable temperature parameter used in CLIP to scale
    the dot product similarity between image and text embeddings. The temperature
    is initialized with a value that's typically small (default 0.07) and learned
    during training to improve model convergence.

    Args:
        initial_value (float): Initial temperature value. Default is 0.07.
        **kwargs: Additional keyword arguments passed to the parent class.

    Inputs:
        A tuple of `(image_embeddings, text_embeddings)` where:
        - image_embeddings: Tensor of shape `(batch_size, embed_dim)`
        - text_embeddings: Tensor of shape `(batch_size, embed_dim)`

    Outputs:
        A tuple of `(image_logits, text_logits)` where:
        - image_logits: Tensor of shape `(batch_size, batch_size)`
        - text_logits: Tensor of shape `(batch_size, batch_size)`
    """

    def __init__(self, initial_value=0.07, **kwargs):
        super().__init__(**kwargs)
        self.initial_value = initial_value

    def build(self, input_shape):
        if not isinstance(input_shape, list) and len(input_shape) != 2:
            raise ValueError(
                "CLIPLogitScale expects a list of 2 input shapes (image_embeddings, text_embeddings)"
            )

        self.logit_scale = self.add_weight(
            shape=(),
            initializer=keras.initializers.Constant(
                value=ops.log(1 / self.initial_value)
            ),
            trainable=True,
            name="logit_scale",
        )

    def call(self, inputs):
        image_embeddings, text_embeddings = inputs
        logit_scale = ops.exp(self.logit_scale)
        image_logits = (
            ops.matmul(image_embeddings, ops.transpose(text_embeddings)) * logit_scale
        )
        text_logits = ops.transpose(image_logits)
        return image_logits, text_logits
