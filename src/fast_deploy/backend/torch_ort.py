import torch
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.onnx_model_bert import BertOnnxModel
from onnxruntime.quantization import quantize_dynamic, QuantType

import multiprocessing

from pathlib import Path
from typing import OrderedDict


def create_model_for_provider(path: str, provider_to_use: str) -> InferenceSession:
    """"""
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    if provider_to_use == "CPUExecutionProvider":
        options.intra_op_num_threads = multiprocessing.cpu_count()

    provider_to_use = [provider_to_use]
    return InferenceSession(path, options, providers=provider_to_use)


def convert_pytorch(
    model: torch.nn.Module, 
    output_path: str, 
    inputs_pytorch: OrderedDict[str, torch.Tensor], 
    opset_version: int = 12, 
    verbose: bool = False
    ) -> None:
    """"""
    dynamic_axis = {}
    for k in inputs_pytorch.keys():
        dynamic_axis[k] = {0: "batch_size", 1: "sequence"}
    dynamic_axis["output"] = {0: "batch_size"}
    with torch.no_grad():
        torch.onnx.export(
            model,
            args=tuple(inputs_pytorch.values()),
            f=output_path, 
            opset_version=opset_version, 
            do_constant_folding=True,
            input_names=list(inputs_pytorch.keys()),
            output_names=["output"],
            dynamic_axes=dynamic_axis,
            verbose=verbose,
        )


def bert_optimize_onnx(onnx_path: str, output_path: str, use_cuda: bool) -> None:
    optimization_options = FusionOptions("bert")
    optimization_options.enable_gelu_approximation = True  # additional optimization
    optimized_model: BertOnnxModel = optimizer.optimize_model(
        input=onnx_path,
        model_type="bert",
        use_gpu=use_cuda,
        opt_level=1,
        num_heads=0,  # automatic detection
        hidden_size=0,  # automatic detection
        optimization_options=optimization_options,
    )

    # optimized_model.convert_float_to_float16()  # FP32 -> FP16
    # logging.info(f"optimizations applied: {optimized_model.get_fused_operator_statistics()}")
    optimized_model.save_model_to_file(output_path)


def generic_optimize_onnx(onnx_path: str, output_path: str) -> None:
    # quantize_dynamic(onnx_path, output_path, weight_type=QuantType.QInt8)
    quantize_dynamic(onnx_path, output_path, weight_type='FLOAT16')
