import multiprocessing
from typing import List

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from onnxruntime.quantization import QuantType, quantize_dynamic


def generic_optimize_onnx(onnx_path: str, output_path: str, quant_type='int8') -> None:
    """Apply dynamic quantization to any onnx model.

    This can be used to any model, and also to second optimization step.

    Parameters
    ----------
    onnx_path: str
        path of onnx model to be optimized.
    output_path: str
        path of optimized onnx model.
    """
    if 'int8' == quant_type:
        qtype = QuantType.QInt8
    elif 'uint8' == quant_type:
        qtype = QuantType.QUInt8
    else:
        raise ValueError(f"unknow QuantType {quant_type}")

    quantize_dynamic(model_input=onnx_path, model_output=output_path, weight_type=qtype)


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
