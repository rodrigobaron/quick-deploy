import logging
from typing import OrderedDict

import tensorflow as tf
import tf2onnx
import torch
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions


def transformers_convert_pytorch(
    model: torch.nn.Module,
    output_path: str,
    inputs_pytorch: OrderedDict[str, torch.Tensor],
    opset_version: int = 12,
    verbose: bool = False,
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


def transformers_convert_tf(
    model: tf.keras.Model,
    output_path: str,
    inputs_tf: OrderedDict[str, tf.Tensor],
    opset_version: int = 12,
    verbose: bool = False,
) -> None:
    """Convert an tensorflow transformers model to onnx.

    This model conversion is specific for transformers.

    Parameters
    ----------
    model: torch.nn.Module
        the tensorflow model to convert to.
    output_path: str
        the onnx output filepath
    inputs_tf: OrderedDict[str, torch.Tensor]
        the model inputs
    opset_version: int
        the onnx op version to use. Default is 12.
    verbose: bool
        show detailed logging. Defaul is False.
    """
    spec = tuple([tf.TensorSpec((None,) + v.shape[1:], v.dtype, name=k) for k, v in inputs_tf.items()])
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=opset_version, output_path=output_path)


def transformers_optimize_onnx(
    onnx_path: str, output_path: str, model_type: str, use_cuda: bool, num_heads: int = 0, hidden_size: int = 0
) -> None:
    """Transformer model optimization.

    This apply custom optimization for transformer model type (encoder-only, decoder-only and encoder-decoder).

    Parameters
    ----------
    onnx_path: str
        onnx model path.
    output_path: str
        output onnx model path.
    model_type:str
        transformer model type. One of [bert, bart, gpt2].
    use_cuda: bool
        optimize for cuda.
    """
    logging.info(f"Optimizing for model type: {model_type}")
    optimization_options = FusionOptions(model_type)
    optimization_options.enable_gelu_approximation = True
    optimized_model = optimizer.optimize_model(
        input=onnx_path,
        model_type=model_type,
        use_gpu=use_cuda,
        opt_level=1,
        num_heads=num_heads,
        hidden_size=hidden_size,
        optimization_options=optimization_options,
        only_onnxruntime=True,
    )

    logging.info(f"optimizations applied: {optimized_model.get_fused_operator_statistics()}")
    optimized_model.save_model_to_file(output_path)
