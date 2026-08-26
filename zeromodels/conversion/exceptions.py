class WeightTransferError(Exception):
    """Base exception class for weight transfer errors."""

    pass


class WeightMappingError(WeightTransferError):
    """
    Exception raised when a corresponding torch weight cannot be found for a given Keras weight.

    This error typically occurs during weight transfer between Keras and PyTorch models
    when the expected weight mapping is not established or cannot be resolved.
    """

    def __init__(self, keras_weight_name: str, torch_weight_name: str):
        self.keras_weight_name = keras_weight_name
        self.torch_weight_name = torch_weight_name
        self.message = (
            f"Weight mapping error during model conversion:\n"
            f"  - Keras source weight: '{self.keras_weight_name}'\n"
            f"  - Attempted torch mapping: '{self.torch_weight_name}'\n"
            f"No corresponding torch weight could be found for the given Keras weight.\n"
            "Possible causes:\n"
            "  1. Incompatible model architectures\n"
            "  2. Incorrect weight transfer strategy\n"
            "  3. Missing or misnamed layer in the target model"
        )
        super().__init__(self.message)


class WeightShapeMismatchError(WeightTransferError):
    """
    Exception raised when the shapes of corresponding Keras and PyTorch weights do not match.

    This error indicates a fundamental discrepancy in tensor dimensions during
    weight transfer between Keras and PyTorch models.
    """

    def __init__(
        self,
        keras_weight_name: str,
        keras_shape: tuple,
        torch_weight_name: str,
        torch_shape: tuple,
    ):
        self.keras_weight_name = keras_weight_name
        self.keras_shape = keras_shape
        self.torch_weight_name = torch_weight_name
        self.torch_shape = torch_shape

        self.message = (
            f"Weight shape mismatch during model conversion:\n"
            f"  - Keras weight: '{self.keras_weight_name}'\n"
            f"    Shape: {self.keras_shape}\n"
            f"  - Torch weight: '{self.torch_weight_name}'\n"
            f"    Shape: {self.torch_shape}\n"
            "Tensor dimensions are incompatible for direct weight transfer.\n"
            "Possible reasons:\n"
            "  1. Different layer configurations\n"
            "  2. Transposed weight matrices\n"
            "  3. Inconsistent model architecture between frameworks"
        )
        super().__init__(self.message)
