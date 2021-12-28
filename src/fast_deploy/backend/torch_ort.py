from typing import OrderedDict

import torch


def torch_convert_onnx(
    model: torch.nn.Module,
    output_path: str,
    inputs_pytorch: OrderedDict[str, torch.Tensor],
    opset_version: int = 12,
    verbose: bool = False,
) -> None:
    """Convert an pytorch model to onnx.

    This model conversion is specific for torch.

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
        dynamic_axis[k] = {0: "batch_size"}
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
