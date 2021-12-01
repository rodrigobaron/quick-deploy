import mock
import pytest
from onnxruntime.quantization import QuantType

from fast_deploy.backend.common import (WeightType, create_model_for_provider,
                                        generic_optimize_onnx)


def test_weight_type_from_str():
    assert QuantType.QInt8 == WeightType.from_str("inT8").value
    assert QuantType.QInt8 == WeightType.from_str("Int8").value
    assert QuantType.QInt8 == WeightType.from_str("INT8").value

    assert "FLOAT16" == WeightType.from_str("floAt16").value
    assert "FLOAT16" == WeightType.from_str("Float16").value
    assert "FLOAT16" == WeightType.from_str("FLOAT16").value

    with pytest.raises(ValueError):
        WeightType.from_str("ASd")


@mock.patch("fast_deploy.backend.common.quantize_dynamic")
def test_generic_optimize_onnx(m):
    generic_optimize_onnx(
        onnx_path="tmp/path.onnx",
        output_path="tmp/path.optim.onnx",
        weight_type=WeightType.from_str("float16"),
    )

    generic_optimize_onnx(
        onnx_path="tmp/path2.onnx",
        output_path="tmp/path2.optim.onnx",
        weight_type=WeightType.from_str("int8"),
    )

    name, args, kwargs = m.mock_calls[0]
    assert "tmp/path.onnx" == args[0]
    assert "tmp/path.optim.onnx" == args[1]
    assert "FLOAT16" == kwargs["weight_type"]

    name, args, kwargs = m.mock_calls[1]
    assert "tmp/path2.onnx" == args[0]
    assert "tmp/path2.optim.onnx" == args[1]
    assert QuantType.QInt8 == kwargs["weight_type"]


@mock.patch("fast_deploy.backend.common.InferenceSession")
@mock.patch("fast_deploy.backend.common.multiprocessing")
def test_create_model_for_provider(m, i):

    onnx_model = create_model_for_provider(
        path="tmp/path.optim.onnx", provider_to_use="CPUExecutionProvider"
    )

    m.cpu_count.assert_called()

    onnx_model = create_model_for_provider(
        path="tmp/path2.optim.onnx", provider_to_use="GPUExecutionProvider"
    )

    name, args, kwargs = i.mock_calls[0]
    assert "tmp/path.optim.onnx" == args[0]
    assert ["CPUExecutionProvider"] == kwargs["providers"]

    name, args, kwargs = i.mock_calls[1]
    assert "tmp/path2.optim.onnx" == args[0]
    assert ["GPUExecutionProvider"] == kwargs["providers"]
