import torch
from torch.onnx import export
from pathlib import Path

from onnxruntime_tools import optimizer
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime import (GraphOptimizationLevel, InferenceSession,
                         SessionOptions)

from typing import NamedTuple, Tuple, Optional, List


class ModelSchema(NamedTuple):
    inputs_names: List[str]
    input_shape: Tuple[int..]
    output_names: List[str]
    output_shape: Tuple[int..]
    # transformers optimization
    model_type: Optional[str] = None
    num_heads: Optional[int] = 0
    hidden_size: Optional[int] = 0


def convert_pytorch(model, torch.Model, model_schema: ModelSchema, opset_version: int, output: Path, use_external_format: bool = False):
    """"""    
    with torch.no_grad():
        dummy_input = torch.randn(1, *model_schema.input_shape)

        export(
            model,
            dummy_input,
            f=output.as_posix(),
            input_names=model_schema.input_names,
            output_names=model_schema.output_names,
            do_constant_folding=True,
            use_external_data_format=use_external_format,
            enable_onnx_checker=True,
            opset_version=opset_version,
        )


def create_model_for_provider(model_path: Path, num_threads: int = 1, provider: str = "CPUExecutionProvider"):
    options = SessionOptions()
    options.intra_op_num_threads = num_threads
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    session = InferenceSession(model_path.as_posix(), options, providers=[provider])
    session.disable_fallback()
    return session


def torch_export(model_path: Path, model_output: Path, opset_version=12):
    model_path = Path('./pytorch_model.pt')
    onnx_model_path = Path("onnx/model.onnx")
    model_op_path = Path("onnx/model.opt.onnx")
    model_output = Path("onnx/model.quant.onnx")

    model = torch.load(model_path)
    convert_pytorch(model, model_schema, opset_version=opset_version, onnx_model_path)

    model_type = model_schema.model_type or 'bert'
    num_heads = model_schema.num_heads
    hidden_size = model_schema.hidden_size

    opt_model = optimizer.optimize_model(onnx_model_path.as_posix(), model_type, num_heads=num_heads, hidden_size=hidden_size)
    opt_model.save_model_to_file(model_op_path)

    quantize_dynamic(model_op_path, model_output, weight_type=QuantType.QInt8)
    # onnx_quantized_model = create_model_for_provider(model_output)


if __name__ == '__main__':
    model_schema = ModelSchema(
        input_names: ['input'],
        input_shape: (32, 32, 3),
        output_names: ['output'],
        output_shape: (3,)
    )
    model_path = Path('./pytorch_model.pt')
    model_output = Path("onnx/model.quant.onnx")

    torch_export(model_path, model_output)
