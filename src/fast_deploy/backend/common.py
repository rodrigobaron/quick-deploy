import multiprocessing
from enum import Enum

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from onnxruntime.quantization import QuantType, quantize_dynamic
from typing import List


class WeightType(Enum):
    """Weight conversion type

    Attributes
    ----------
    Int8
    Float16
    """

    Int8 = QuantType.QInt8
    Float16 = "FLOAT16"

    @classmethod
    def from_str(cls, weight_type: str):
        if "int8" == weight_type.lower():
            return WeightType.Int8
        elif "float16" == weight_type.lower():
            return WeightType.Float16
        else:
            raise ValueError


def generic_optimize_onnx(
    onnx_path: str, output_path: str, weight_type: WeightType
) -> None:
    """Apply dynamic quantization to any onnx model.

    This can be used to any model, and also to second optimization step.

    Parameters
    ----------
    onnx_path: str
        path of onnx model to be optimized.
    output_path: str
        path of optimized onnx model.
    weight_type: WeightType
        weight optimization to apply to.
    """
    quantize_dynamic(onnx_path, output_path, weight_type=weight_type.value)


def create_model_for_provider(path: str, provider_to_use: str) -> InferenceSession:
    """Create onnx model for desired provider.

    This is used to run onnx models as standalone.

    Parameters
    ----------
    path: str
        onnx model path.
    provider_to_use: str
        execution provider, eg.: CPUExecutionProvider.

    Returns
    ----------
    InferenceSession
        the onnx model for inference.
    """
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    if provider_to_use == "CPUExecutionProvider":
        options.intra_op_num_threads = multiprocessing.cpu_count()

    providers: List[str] = [provider_to_use]
    return InferenceSession(path, options, providers=providers)
