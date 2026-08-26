import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="zeromodels")
class ConvNeXtGlobalResponseNorm(layers.Layer):
    """
    GRN (Global Response Normalization) layer.
    This layer applies a global normalization to the input tensor, using the square root of the
    sum of squares of the input along the spatial dimensions (height and width) as a global feature.
    The normalized global feature is then used to scale and shift the input tensor.

    The layer automatically determines the number of channels from the input shape.

    Attributes:
        weight (np.ndarray): The weight tensor used for scaling the normalized global feature.
        bias (np.ndarray): The bias tensor used for shifting the input tensor.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=(1, 1, 1, input_shape[-1]),
            initializer=keras.initializers.Zeros(),
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(1, 1, 1, input_shape[-1]),
            initializer=keras.initializers.Zeros(),
        )
        return super().build(input_shape)

    def call(self, hidden_states):
        global_features = ops.sqrt(
            ops.sum(ops.square(hidden_states), axis=(1, 2), keepdims=True)
        )
        norm_features = global_features / (
            ops.mean(global_features, axis=-1, keepdims=True) + 1e-6
        )
        hidden_states = (
            self.weight * (hidden_states * norm_features) + self.bias + hidden_states
        )
        return hidden_states


@keras.saving.register_keras_serializable(package="zeromodels")
class ConvNeXtLayerScale(layers.Layer):
    """Learnable per-channel scale (x * gamma), gamma initialized to layer_scale_init."""

    def __init__(self, layer_scale_init, **kwargs):
        super().__init__(**kwargs)
        self.layer_scale_init = layer_scale_init

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer=keras.initializers.Constant(self.layer_scale_init),
            trainable=True,
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update({"layer_scale_init": self.layer_scale_init})
        return config


@keras.saving.register_keras_serializable(package="zeromodels")
class ConvNeXtStochasticDepth(keras.layers.Layer):
    """Stochastic depth: randomly drops the residual path during training (identity at inference)."""

    def __init__(self, drop_path_rate, seed=None, **kwargs):
        super().__init__(**kwargs)
        if not 0 <= drop_path_rate <= 1:
            raise ValueError(
                f"drop_path_rate should be between 0 and 1, got {drop_path_rate}"
            )
        self.drop_path_rate = drop_path_rate
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            shape = (keras.ops.shape(x)[0],) + (1,) * (len(keras.ops.shape(x)) - 1)
            random_tensor = keep_prob + keras.random.uniform(
                shape, 0, 1, seed=self.seed_generator
            )
            random_tensor = keras.ops.cast(keras.ops.floor(random_tensor), x.dtype)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate, "seed": self.seed})
        return config
