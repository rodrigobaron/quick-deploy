from skl2onnx import to_onnx
from skl2onnx.common.data_types import (
    BooleanTensorType,
    DoubleTensorType,
    FloatTensorType,
    Int32TensorType,
    Int64TensorType,
    StringTensorType,
)


def _str_to_type(content):
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


def parse_skl_input(shape, dtype):
    type_cls = _str_to_type(dtype)
    return type_cls(shape)


def skl_convert_onnx(model, output_path, inputs_type, verbose=False):
    onx = to_onnx(model, initial_types=inputs_type, options={id(model): {'zipmap': False}})
    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())
