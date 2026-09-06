"""
Model Equivalence Verification Utility for PyTorch, Keras, and Transformers Models

This module provides functionality to verify the equivalence of neural network models
between PyTorch, Keras, and Transformers frameworks. It performs comprehensive testing
of model outputs, optional performance benchmarking, and ImageNet prediction verification.

Key Features:
- Supports comparison between PyTorch and Keras models
- Supports comparison between different Keras models
- Supports comparison between Transformers and Keras models
- Validates model outputs across different batch sizes
- Performs optional performance benchmarking
- Provides detailed test results and diagnostics
- Handles both single and batch inference
- Supports custom tolerance levels for output comparison
- Optional ImageNet prediction testing with predefined test cases
- Detailed error reporting and difference analysis

Dependencies:
- numpy
- torch
- keras (>=3.0)
- transformers (optional, for cross-framework comparisons)
- typing
- time
- os

Example Usage:
    # For PyTorch to Keras comparison with standard tests
    results = verify_cls_model_equivalence(
        model_a=torch_model,           # PyTorch model
        model_b=keras_model,           # Keras model
        input_shape=(224, 224, 3),     # Input shape without batch dimension
        output_specs={"num_classes": 1000},
        comparison_type="torch_to_keras",
        batch_sizes=[1, 4, 8],
        run_performance=True
    )

    # For Transformers to Keras comparison
    results = verify_cls_model_equivalence(
        model_a=hf_model,              # Transformers model
        model_b=keras_model,           # Keras model
        input_shape=(224, 224, 3),     # Input shape without batch dimension
        output_specs={"num_classes": 1000, "hf_output_shape": [1000]},
        comparison_type="hf_to_keras",
        hf_model_config={"input_key": "pixel_values", "use_dict_input": True}
    )

    # For ImageNet prediction testing only
    results = verify_cls_model_equivalence(
        model_a=None,                  # Not required for ImageNet testing
        model_b=keras_model,           # Keras model to test
        input_shape=(224, 224, 3),     # Input shape without batch dimension
        output_specs={"num_classes": 1000},
        test_imagenet_image=True,
        prediction_threshold=0.5
    )

Return Value:
    Returns a dictionary containing test results:
    {
        "standard_input": bool,              # Result of single sample test
        "batch_size_N": bool,                # Results for each batch size
        "standard_input_diff": {             # Optional difference metrics if test fails
            "max_difference": float,
            "mean_difference": float
        },
        "performance": {                     # Optional performance metrics
            "model_a_inference_time": float,
            "model_b_inference_time": float,
            "time_ratio": float
        },
        "imagenet_test": {                   # Optional ImageNet test results
            "image_name": {
                "success": bool,
                "predicted_class": str,
                "expected_class": str,
                "class_matched": bool,
                "confidence": float,
                "threshold": float,
                "threshold_passed": bool
            },
            "all_passed": bool
        }
    }

Notes:
- Input shapes should be specified without the batch dimension
- The function handles necessary tensor transpositions for PyTorch and Transformers inputs
- Performance testing runs multiple inferences to get average timing
- Different random seeds are used for reproducibility
- Custom tolerance levels can be set for numerical comparison
- ImageNet testing includes predefined test cases with expected classes
- For ImageNet testing only, model_a can be None
- Keras model should have preprocessing=True for ImageNet Testing
- For Transformers model comparisons, additional configuration can be provided via hf_model_config
- The transformers library is required for Transformers model comparisons
- Detailed error reporting includes maximum and mean differences when tests fail
"""

from __future__ import annotations

import os
from time import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from keras import ops, utils
from keras.src.applications.imagenet_utils import decode_predictions

from zeromodels.conversion.file_downloader import download_file

if TYPE_CHECKING:
    import torch
    import transformers


def verify_cls_model_equivalence(
    model_a: Union[keras.Model, torch.nn.Module, "transformers.PreTrainedModel", None],
    model_b: keras.Model,
    input_shape: Union[Tuple[int, ...], List[int]],
    output_specs: Dict[str, Any],
    comparison_type: str = "torch_to_keras",
    batch_sizes: tuple = (2,),
    run_performance: bool = True,
    num_performance_runs: int = 5,
    seed: int = 2025,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    test_imagenet_image: bool = False,
    prediction_threshold: float = 0.80,
    hf_model_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Verify equivalence between two models, with optional ImageNet testing.
    For ImageNet testing only, model_a can be None.

    Args:
        model_a: Source model (PyTorch, Keras, or Transformers) or None if only testing ImageNet
        model_b: Target Keras model
        input_shape: Shape of input tensor (excluding batch dimension)
        output_specs: Dictionary containing output specifications
        comparison_type: Type of comparison ('torch_to_keras', 'keras_to_keras', or 'hf_to_keras')
        batch_sizes: List of batch sizes to test
        run_performance: Whether to run performance comparison
        num_performance_runs: Number of runs for performance testing
        seed: Random seed for reproducibility
        atol: Absolute tolerance for numerical comparisons
        rtol: Relative tolerance for numerical comparisons
        test_imagenet_image: Whether to run ImageNet testing only
        prediction_threshold: Confidence threshold for ImageNet predictions
        hf_model_config: Configuration for the Transformers model (vision model params)

    Returns:
        Dictionary containing test results
    """
    import torch

    try:
        import transformers

        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        transformers = None
        TRANSFORMERS_AVAILABLE = False

    results = {}

    if comparison_type == "hf_to_keras" and not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers library not found. Please install with 'pip install transformers'"
        )

    if not test_imagenet_image:
        if comparison_type not in ["torch_to_keras", "keras_to_keras", "hf_to_keras"]:
            raise ValueError(
                "comparison_type must be either 'torch_to_keras', 'keras_to_keras', or 'hf_to_keras'"
            )

        if model_a is None:
            raise ValueError("model_a cannot be None when running comparison tests")

        if comparison_type == "torch_to_keras" and not isinstance(
            model_a, torch.nn.Module
        ):
            raise ValueError(
                "model_a must be a PyTorch model when comparison_type is 'torch_to_keras'"
            )
        elif comparison_type == "keras_to_keras" and not isinstance(
            model_a, keras.Model
        ):
            raise ValueError(
                "model_a must be a Keras model when comparison_type is 'keras_to_keras'"
            )
        elif comparison_type == "hf_to_keras":
            if not TRANSFORMERS_AVAILABLE or not isinstance(
                model_a, transformers.PreTrainedModel
            ):
                raise ValueError(
                    "model_a must be a HuggingFace transformers model when comparison_type is 'hf_to_keras'"
                )

            if hf_model_config is None:
                hf_model_config = {}
                print(
                    "Warning: No HuggingFace config provided. Using default settings."
                )

    if "num_classes" not in output_specs:
        raise ValueError("output_specs must contain 'num_classes' key")

    def test_imagenet_prediction() -> Dict[str, Any]:
        print("\n=== Testing ImageNet Prediction ===")
        imagenet_results = {}

        # Define test cases
        test_cases = [
            {
                "name": "bird",
                "file_name": "indigo_bunting.png",
                "url": "https://raw.githubusercontent.com/IMvision12/ZeroModels/main/assets/bird.png",
                "expected_class": "indigo_bunting",
            },
            {
                "name": "valley",
                "file_name": "valley_image.png",
                "url": "https://raw.githubusercontent.com/IMvision12/ZeroModels/main/assets/valley.png",
                "expected_class": "valley",
            },
            {
                "name": "dam",
                "file_name": "dam_image.png",
                "url": "https://raw.githubusercontent.com/IMvision12/ZeroModels/main/assets/dam.png",
                "expected_class": "dam",
            },
            {
                "name": "space",
                "file_name": "space_image.png",
                "url": "https://raw.githubusercontent.com/IMvision12/ZeroModels/main/assets/space.png",
                "expected_class": "space_shuttle",
            },
            {
                "name": "train",
                "file_name": "train_image.png",
                "url": "https://raw.githubusercontent.com/IMvision12/ZeroModels/main/assets/train.png",
                "expected_class": "bullet_train",
            },
        ]

        for test_case in test_cases:
            try:
                print(f"\nTesting {test_case['name']} image:")

                image_path = download_file(test_case["url"])

                if not os.path.exists(image_path):
                    raise ValueError(
                        f"Failed to download test image: {test_case['name']}"
                    )

                image = utils.load_img(image_path, target_size=input_shape[:2])
                image_array = utils.img_to_array(image)
                x = ops.expand_dims(image_array, axis=0)

                preds = model_b.predict(x)
                decoded_pred = decode_predictions(preds, top=1)[0][0]

                predicted_class = decoded_pred[1]
                class_matched = predicted_class == test_case["expected_class"]

                confidence = decoded_pred[2]
                threshold_passed = (
                    confidence > prediction_threshold if class_matched else False
                )

                test_passed = class_matched and threshold_passed

                imagenet_results[test_case["name"]] = {
                    "success": test_passed,
                    "predicted_class": predicted_class,
                    "expected_class": test_case["expected_class"],
                    "class_matched": class_matched,
                    "confidence": float(confidence),
                    "threshold": prediction_threshold,
                    "threshold_passed": threshold_passed,
                }

                print(f"Expected class: {test_case['expected_class']}")
                print(f"Predicted class: {predicted_class}")
                print(f"Class match: {'PASS' if class_matched else 'FAIL'}")
                print(f"Confidence: {confidence:.4f}")
                print(
                    f"Threshold ({prediction_threshold}) passed: {'PASS' if threshold_passed else 'FAIL'}"
                )
                print(f"Overall test result: {'PASS' if test_passed else 'FAIL'}")

            except Exception as e:
                print(f"FAIL Test failed for {test_case['name']}: {str(e)}")
                imagenet_results[test_case["name"]] = {
                    "success": False,
                    "error": str(e),
                }

        imagenet_results["all_passed"] = all(
            result.get("success", False)
            for name, result in imagenet_results.items()
            if name != "all_passed"
        )

        return imagenet_results

    def get_expected_output_shape(batch_size: int) -> Tuple[int, ...]:
        if comparison_type == "torch_to_keras":
            return (batch_size, output_specs["num_classes"])
        elif comparison_type == "hf_to_keras":
            if "hf_output_shape" in output_specs:
                return (batch_size, *output_specs["hf_output_shape"])
            return (batch_size, output_specs["num_classes"])
        else:  # keras_to_keras
            sample_input = np.zeros([1] + list(input_shape), dtype="float32")
            output_shape = model_a(sample_input).shape[1:]
            return (batch_size, *output_shape)

    def prepare_input(batch_size: int) -> Tuple[Any, Any]:
        if isinstance(input_shape, tuple):
            input_shape_list = list(input_shape)
        else:
            input_shape_list = input_shape

        if comparison_type == "torch_to_keras":
            keras_input = np.random.uniform(
                size=[batch_size, *input_shape_list]
            ).astype("float32")
            torch_input = torch.from_numpy(np.transpose(keras_input, [0, 3, 1, 2]))
            return keras_input, torch_input
        elif comparison_type == "hf_to_keras":
            keras_input = np.random.uniform(
                size=[batch_size, *input_shape_list]
            ).astype("float32")

            if len(input_shape_list) == 3:
                hf_input = torch.from_numpy(np.transpose(keras_input, [0, 3, 1, 2]))
            else:
                hf_input = torch.from_numpy(keras_input)

            input_key = hf_model_config.get("input_key", "pixel_values")
            if hf_model_config.get("use_dict_input", True):
                return keras_input, {input_key: hf_input}
            else:
                return keras_input, hf_input
        else:
            test_input = np.random.uniform(size=[batch_size] + input_shape_list).astype(
                "float32"
            )
            return test_input, test_input

    def get_model_output(
        model: Union[keras.Model, torch.nn.Module, "transformers.PreTrainedModel"],
        input_data: Union[np.ndarray, torch.Tensor, Dict[str, torch.Tensor]],
        expected_shape: Tuple[int, ...],
        is_hf_model: bool = False,
    ) -> np.ndarray:
        if isinstance(model, torch.nn.Module):
            model.eval()
            with torch.no_grad():
                if is_hf_model:
                    if isinstance(input_data, dict):
                        output = model(**input_data)
                    else:
                        output = model(input_data)

                    if hasattr(output, "logits"):
                        output = output.logits
                    elif hasattr(output, "last_hidden_state"):
                        output = output.last_hidden_state
                    else:
                        output = (
                            output[0] if isinstance(output, (tuple, list)) else output
                        )
                else:
                    output = model(input_data)

                output = output.detach().cpu().numpy()
        else:
            output = keras.ops.convert_to_numpy(model(input_data, training=False))

        assert output.shape == expected_shape, (
            f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
        )
        return output

    def test_outputs(output_a: np.ndarray, output_b: np.ndarray) -> bool:
        try:
            np.testing.assert_allclose(output_a, output_b, rtol=rtol, atol=atol)
            return True
        except AssertionError:
            return False

    def test_standard_input() -> bool:
        print("\n=== Testing Standard Input Shape ===")
        np.random.seed(seed)
        torch.manual_seed(seed) if comparison_type in [
            "torch_to_keras",
            "hf_to_keras",
        ] else None
        keras.utils.set_random_seed(seed)

        input_a, input_b = prepare_input(batch_size=1)
        expected_shape = get_expected_output_shape(batch_size=1)

        try:
            is_hf_model = comparison_type == "hf_to_keras"
            output_a = get_model_output(
                model_a,
                input_b
                if comparison_type in ["torch_to_keras", "hf_to_keras"]
                else input_a,
                expected_shape,
                is_hf_model=is_hf_model,
            )
            output_b = get_model_output(model_b, input_a, expected_shape)

            success = test_outputs(output_a, output_b)
            print(
                f"{'PASS' if success else 'FAIL'} Output {'matched' if success else 'mismatched'} "
                f"for input shape {input_shape}"
            )

            if not success:
                max_diff = np.max(np.abs(output_a - output_b))
                mean_diff = np.mean(np.abs(output_a - output_b))
                results["standard_input_diff"] = {
                    "max_difference": float(max_diff),
                    "mean_difference": float(mean_diff),
                }

            return success
        except Exception as e:
            print(f"FAIL Test failed: {str(e)}")
            return False

    def test_batch_processing() -> Dict[str, bool]:
        print("\n=== Testing Different Batch Sizes ===")
        batch_results = {}

        for batch_size in batch_sizes:
            if batch_size == 1:
                continue

            print(f"\nTesting batch size: {batch_size}")
            np.random.seed(seed)
            torch.manual_seed(seed) if comparison_type in [
                "torch_to_keras",
                "hf_to_keras",
            ] else None
            keras.utils.set_random_seed(seed)

            input_a, input_b = prepare_input(batch_size)
            expected_shape = get_expected_output_shape(batch_size)

            try:
                is_hf_model = comparison_type == "hf_to_keras"
                output_a = get_model_output(
                    model_a,
                    input_b
                    if comparison_type in ["torch_to_keras", "hf_to_keras"]
                    else input_a,
                    expected_shape,
                    is_hf_model=is_hf_model,
                )
                output_b = get_model_output(model_b, input_a, expected_shape)

                success = test_outputs(output_a, output_b)
                batch_results[f"batch_size_{batch_size}"] = success
                print(
                    f"{'PASS' if success else 'FAIL'} Output {'matched' if success else 'mismatched'}"
                )

                if not success:
                    max_diff = np.max(np.abs(output_a - output_b))
                    mean_diff = np.mean(np.abs(output_a - output_b))
                    batch_results[f"batch_size_{batch_size}_diff"] = {
                        "max_difference": float(max_diff),
                        "mean_difference": float(mean_diff),
                    }

            except Exception as e:
                batch_results[f"batch_size_{batch_size}"] = False
                print(f"FAIL Test failed for batch size {batch_size}: {str(e)}")

        return batch_results

    def run_performance_test():
        print("\n=== Testing Performance ===")
        input_a, input_b = prepare_input(batch_sizes[0])

        def run_inference(
            model, input_data, is_torch: bool = False, is_hf: bool = False
        ):
            if is_torch:
                model.eval()
                with torch.no_grad():
                    if is_hf and isinstance(input_data, dict):
                        _ = model(**input_data)
                        times = []
                        for _ in range(num_performance_runs):
                            start = time()
                            _ = model(**input_data)
                            times.append(time() - start)
                    else:
                        _ = model(input_data)
                        times = []
                        for _ in range(num_performance_runs):
                            start = time()
                            _ = model(input_data)
                            times.append(time() - start)
            else:
                _ = model(input_data, training=False)
                times = []
                for _ in range(num_performance_runs):
                    start = time()
                    _ = model(input_data, training=False)
                    times.append(time() - start)

            return np.mean(times)

        is_hf = comparison_type == "hf_to_keras"
        time_a = run_inference(
            model_a,
            input_b
            if comparison_type in ["torch_to_keras", "hf_to_keras"]
            else input_a,
            is_torch=comparison_type in ["torch_to_keras", "hf_to_keras"],
            is_hf=is_hf,
        )
        time_b = run_inference(model_b, input_a, is_torch=False)

        return {
            "model_a_inference_time": time_a,
            "model_b_inference_time": time_b,
            "time_ratio": time_b / time_a,
        }

    if test_imagenet_image:
        results["imagenet_test"] = test_imagenet_prediction()

        print("\n=== Test Summary ===")
        print(
            f"ImageNet Tests: {'Passed PASS' if results['imagenet_test']['all_passed'] else 'Failed FAIL'}"
        )
        for name, result in results["imagenet_test"].items():
            if name != "all_passed":
                success = result.get("success", False)
                print(f"  - {name}: {'Passed PASS' if success else 'Failed FAIL'}")
    else:
        results["standard_input"] = test_standard_input()

        if results["standard_input"]:
            results.update(test_batch_processing())

        if run_performance:
            results["performance"] = run_performance_test()
            perf = results["performance"]
            print(
                f"Model A average inference time: {perf['model_a_inference_time']:.4f}s"
            )
            print(
                f"Model B average inference time: {perf['model_b_inference_time']:.4f}s"
            )
            print(f"Time ratio (B/A): {perf['time_ratio']:.2f}x")

        print("\n=== Test Summary ===")
        all_tests = [results["standard_input"]] + [
            v
            for k, v in results.items()
            if k.startswith("batch_size_") and isinstance(v, bool)
        ]
        all_passed = all(all_tests)
        print(
            f"Standard Input Test: {'Passed PASS' if results['standard_input'] else 'Failed FAIL'}"
        )
        print(
            f"Batch Processing Tests: {'Passed PASS' if all_passed else 'Failed FAIL'}"
        )

    return results
