from typing import Any, List

from skl2onnx import to_onnx
from skl2onnx.common.data_types import (
    BooleanTensorType,
    DoubleTensorType,
    FloatTensorType,
    Int32TensorType,
    Int64TensorType,
    StringTensorType,
)


def _str_to_type(content: str) -> Any:
    """Get skl2onnx type by string.

    This is used to parse model defintion to skl2onnx I/O

    Parameters
    ----------
    content: str
        The key representation of type.

    Returns
    ----------
    Any:
        The type of the string key.
    """
    content = content.lower().strip()
    if 'float32' == content:
        return FloatTensorType
    if 'float64' == content:
        return DoubleTensorType
    if 'int64' == content:
        return Int64TensorType
    if 'int' == content:
        return Int32TensorType
    if 'bool' == content:
        return BooleanTensorType
    if 'str' == content:
        return StringTensorType

    raise ValueError


def parse_skl_input(shape: List[int], dtype: str) -> Any:
    """Parse input information to desired object.

    This is used to create I/O tensor representation.

    Parameters
    ----------
    shape: tuple
        The desired tensor shape.
    dtype: str
        The desired tensor type.

    Returns
    ----------
    Any:
        The tensor object.
    """
    type_cls = _str_to_type(dtype)
    return type_cls(shape)


def skl_convert_onnx(model, output_path, inputs_type, verbose=False):
    """Convert a SkLearn model to ORT.

    This is used to convert SkLearn model to ORT.

    Parameters
    ----------
    model: SKLearn.Model
        Model to be converted.
    output_path: str
        Path to save the ORT model.
    inputs_type: List[Tuple[str, Any]]
        The model input definition.
    verbose: bool = False
        Show process logs.
    """
    onx = to_onnx(model, initial_types=inputs_type, options={id(model): {'zipmap': False}})
    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())
