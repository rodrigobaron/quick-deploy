from typing import Any, List, Tuple

from onnxmltools.convert.xgboost.operator_converters.XGBoost import (  # noqa
    convert_xgboost,
)
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import (
    BooleanTensorType,
    DoubleTensorType,
    FloatTensorType,
    Int32TensorType,
    Int64TensorType,
    StringTensorType,
)
from skl2onnx.common.shape_calculator import (  # noqa
    calculate_linear_classifier_output_shapes,
)
from xgboost import XGBClassifier


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


def parse_xgb_input(shape: List[int], dtype: str) -> Any:
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


def xgb_convert_onnx(model: Any, output_path: str, inputs_type: List[Tuple[str, Any]], verbose: bool = False):
    """Convert a XGBoost model to ORT.

    This is used to convert XGBoost model to ORT.

    Parameters
    ----------
    model: Any
        Model to be converted.
    output_path: str
        Path to save the ORT model.
    inputs_type: List[Tuple[str, Any]]
        The model input definition.
    verbose: bool = False
        Show process logs.
    """
    update_registered_converter(
        XGBClassifier,
        'XGBoostXGBClassifier',
        calculate_linear_classifier_output_shapes,
        convert_xgboost,
        options={'nocl': [True, False], 'zipmap': [True, False, 'columns']},
    )
    onx = convert_sklearn(model, initial_types=inputs_type, options={id(model): {'zipmap': False}})

    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())
