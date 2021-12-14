from skl2onnx.common.data_types import (
    FloatTensorType, 
    Int64TensorType,
    Int32TensorType,
    BooleanTensorType,
    StringTensorType,
    DoubleTensorType
)
from skl2onnx import to_onnx

from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes  # noqa
from xgboost import XGBClassifier
# import onnxmltools
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost  # noqa
# import onnxmltools.convert.common.data_types


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
    
    raise ValueError


def parse_xgb_input(shape, dtype):
    type_cls = _str_to_type(dtype)
    return type_cls(shape)


def xgb_convert_onnx(model, output_path, inputs_type, verbose=False):
    # onx = to_onnx(model, initial_types=inputs_type, options={id(model): {'zipmap': False}})

    update_registered_converter(
        XGBClassifier,
        'XGBoostXGBClassifier',
        calculate_linear_classifier_output_shapes, 
        convert_xgboost,
        options={'nocl': [True, False], 'zipmap': [True, False, 'columns']}
    )
    # update_registered_converter(
    # model, 'XGBoostXGBClassifier',
    # calculate_linear_classifier_output_shapes, convert_xgboost,
    # options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})
    onx = convert_sklearn(model, initial_types=inputs_type, options={id(model): {'zipmap': False}})
    # onx = convert_sklearn(
    # model, 'pipeline_xgboost',
    # [('input', FloatTensorType([None, 2]))],
    # target_opset=12)

    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())
