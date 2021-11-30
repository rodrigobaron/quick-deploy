import logging
import torch

from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions

from pathlib import Path
from typing import OrderedDict


def transformers_convert_pytorch(
    model: torch.nn.Module, 
    output_path: str, 
    inputs_pytorch: OrderedDict[str, torch.Tensor], 
    opset_version: int = 12, 
    verbose: bool = False
    ) -> None:
    """Convert an pytorch tansformer model to onnx.
    
    This model conversion is specific for transformers.
    
    Parameters
    ----------
    model: torch.nn.Module
        the pytorch model to convert to.
    output_path: str
        the onnx output filepath
    inputs_pytorch: OrderedDict[str, torch.Tensor]
        the model inputs
    opset_version: int
        the onnx op version to use. Default is 12. 
    verbose: bool
        show detailed logging. Defaul is False.
    """
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


def transformers_optimize_onnx(onnx_path: str, output_path: str, model_family:str, use_cuda: bool) -> None:
    """Transformer model optimization.

    This uses the transformer family (encoder-only, decoder-only and encoder-decoder) pre defined optimizations.

    Parameters
    ----------
    onnx_path: str
        onnx model path.
    output_path: str
        output onnx model path.
    model_family:str
        model family.
    use_cuda: bool
        optimize for cuda.
    """
    optimization_options = FusionOptions(model_family)
    optimization_options.enable_gelu_approximation = True
    optimized_model = optimizer.optimize_model(
        input=onnx_path,
        model_type=model_family,
        use_gpu=use_cuda,
        opt_level=1,
        num_heads=0,
        hidden_size=0,
        optimization_options=optimization_options,
        only_onnxruntime=True
    )

    logging.info(f"optimizations applied: {optimized_model.get_fused_operator_statistics()}")
    optimized_model.save_model_to_file(output_path)
