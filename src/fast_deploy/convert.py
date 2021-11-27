import torch
from torch.onnx import export
from pathlib import Path

from onnxruntime_tools import optimizer
from onnxruntime.quantization import quantize_dynamic, QuantType
from fast_deploy.schema import ModelSchema


def convert_pytorch(model: torch.nn.Module, model_schema: ModelSchema, output: Path, opset_version: int, use_external_format: bool = False):
    """"""    
    with torch.no_grad():
        dummy_input = [torch.randint(1, (1, *x)) for x in model_schema.input_shape]
        dynamic_axes = {}
        for input_n in model_schema.input_names:
            dynamic_axes[input_n] = {0: 'batch', 1: 'sequence'}
        
        for output_n in model_schema.output_names:
            dynamic_axes[output_n] = {0: 'batch'}

        export(
            model,
            dummy_input,
            f=output.as_posix(),
            input_names=model_schema.input_names,
            output_names=model_schema.output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            use_external_data_format=use_external_format,
            enable_onnx_checker=True,
            opset_version=opset_version,
        )


def convert_and_optimize(model_path: Path, model_output: Path, model_schema: ModelSchema, opset_version: int = 12):
    """"""
    onnx_model_path = Path("/tmp/model.onnx")
    model = torch.load(model_path)

    #  TODO: transformer optimization 

    convert_pytorch(model, model_schema, onnx_model_path, opset_version=opset_version)
    quantize_dynamic(onnx_model_path, model_output, weight_type=QuantType.QInt8)
