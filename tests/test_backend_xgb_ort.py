import mock
import pytest
from skl2onnx.common.data_types import (
    BooleanTensorType,
    DoubleTensorType,
    FloatTensorType,
    Int32TensorType,
    Int64TensorType,
    StringTensorType,
)

from quick_deploy.backend.xgb_ort import parse_xgb_input, xgb_convert_onnx


def test_parse_xgb_input():
    bool_tensor = parse_xgb_input([None, 1], 'bool')
    assert isinstance(bool_tensor, BooleanTensorType)
    assert bool_tensor.shape == [None, 1]

    double_tensor = parse_xgb_input([None, 1], 'float64')
    assert isinstance(double_tensor, DoubleTensorType)
    assert double_tensor.shape == [None, 1]

    float_tensor = parse_xgb_input([None, 1], 'float32')
    assert isinstance(float_tensor, FloatTensorType)
    assert float_tensor.shape == [None, 1]

    int32_tensor = parse_xgb_input([None, 1], 'int')
    assert isinstance(int32_tensor, Int32TensorType)
    assert int32_tensor.shape == [None, 1]

    int64_tensor = parse_xgb_input([None, 1], 'int64')
    assert isinstance(int64_tensor, Int64TensorType)
    assert int64_tensor.shape == [None, 1]

    str_tensor = parse_xgb_input([None, 1], 'str')
    assert isinstance(str_tensor, StringTensorType)
    assert str_tensor.shape == [None, 1]

    with pytest.raises(ValueError):
        parse_xgb_input([None, 1], 'xxx')


@mock.patch("quick_deploy.backend.xgb_ort.convert_sklearn")
@mock.patch("quick_deploy.backend.xgb_ort.open")
def test_xgb_convert_onnx(o, m):
    initial_type = [("input", FloatTensorType([None, 1]))]

    xgb_convert_onnx(model='test', output_path='/tmp/xgb', inputs_type=initial_type, verbose=False)

    name, args, kwargs = m.mock_calls[0]

    assert args == ('test',)
    assert kwargs['initial_types'] == initial_type

    options_k = list(kwargs['options'].keys())[0]
    assert kwargs['options'][options_k] == {'zipmap': False}
