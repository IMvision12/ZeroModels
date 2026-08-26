import keras
from keras import ops


@keras.saving.register_keras_serializable(package="zeromodels")
class ResNetV2StdConv2D(keras.layers.Conv2D):
    """2D Convolution layer with Weight Standardization.

    Weight Standardization standardizes the weights in the convolutional layer
    to have zero mean and unit variance across the spatial and input channel
    dimensions. This can help improve training stability and model performance,
    particularly in conjunction with normalization techniques like Group
    Normalization.

    Arguments:
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the height and width.
            Default is (1, 1).
        padding: One of "valid" or "same" (case-insensitive).
        data_format: A string, one of "channels_last" (default) or "channels_first".
        dilation_rate: An integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
        groups: A positive integer specifying the number of groups in which the
            input is split along the channel axis. Default is 1.
        activation: Activation function to use. Default is None.
        use_bias: Boolean, whether the layer uses a bias vector. Default is True.
        kernel_initializer: Initializer for the kernel weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to kernel weights matrix.
        bias_regularizer: Regularizer function applied to bias vector.
        activity_regularizer: Regularizer function applied to output of the layer.
        kernel_constraint: Constraint function applied to kernel matrix.
        bias_constraint: Constraint function applied to bias vector.
        eps: Small float added to variance to avoid dividing by zero. Default is 1e-8.

    Input shape:
        4D tensor with shape:
        `(batch_size, channels, height, width)` if data_format='channels_first'
        or `(batch_size, height, width, channels)` if data_format='channels_last'.

    Output shape:
        4D tensor with shape:
        `(batch_size, filters, new_height, new_width)` if data_format='channels_first'
        or `(batch_size, new_height, new_width, filters)` if data_format='channels_last'.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        eps=1e-8,
        **kwargs,
    ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        self.eps = eps
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, received: {self.eps}")

    def standardize_kernel(self, kernel):
        kernel_mean = ops.mean(kernel, axis=[0, 1, 2], keepdims=True)
        centered_kernel = kernel - kernel_mean

        kernel_var = ops.mean(
            ops.square(centered_kernel), axis=[0, 1, 2], keepdims=True
        )

        return centered_kernel / ops.sqrt(kernel_var + self.eps)

    def build(self, input_shape):
        super().build(input_shape)
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=keras.initializers.RandomNormal(mean=0.1, stddev=0.1),
                trainable=True,
            )

    def call(self, inputs):
        std_kernel = self.standardize_kernel(self.kernel)
        outputs = ops.conv(
            inputs,
            std_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        if self.use_bias:
            bias_shape = (
                (1, 1, 1, self.filters)
                if self.data_format == "channels_last"
                else (1, self.filters, 1, 1)
            )
            outputs = outputs + ops.reshape(self.bias, bias_shape)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs


@keras.saving.register_keras_serializable(package="zeromodels")
class ResNetV2StochasticDepth(keras.layers.Layer):
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
