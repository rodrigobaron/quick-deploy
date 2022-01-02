import mock
import pytest
from onnxruntime.quantization import QuantType

from quick_deploy.backend.common import create_model_for_provider, generic_optimize_onnx


@mock.patch("quick_deploy.backend.common.quantize_dynamic")
def test_generic_optimize_onnx(m):
    generic_optimize_onnx(onnx_path="tmp/path.onnx", output_path="tmp/path.optim.onnx")

    generic_optimize_onnx(onnx_path="tmp/path2.onnx", output_path="tmp/path2.optim.onnx")

    name, args, kwargs = m.mock_calls[0]
    assert "tmp/path.onnx" == kwargs['model_input']
    assert "tmp/path.optim.onnx" == kwargs['model_output']
    assert QuantType.QInt8 == kwargs["weight_type"]

    name, args, kwargs = m.mock_calls[1]
    assert "tmp/path2.onnx" == kwargs['model_input']
    assert "tmp/path2.optim.onnx" == kwargs['model_output']
    assert QuantType.QInt8 == kwargs["weight_type"]


@mock.patch("quick_deploy.backend.common.InferenceSession")
@mock.patch("quick_deploy.backend.common.multiprocessing")
def test_create_model_for_provider(m, i):

    _ = create_model_for_provider(path="tmp/path.optim.onnx", provider_to_use="CPUExecutionProvider")

    m.cpu_count.assert_called()

    _ = create_model_for_provider(path="tmp/path2.optim.onnx", provider_to_use="CUDAExecutionProvider")

    name, args, kwargs = i.mock_calls[0]
    assert "tmp/path.optim.onnx" == args[0]
    assert ["CPUExecutionProvider"] == kwargs["providers"]

    name, args, kwargs = i.mock_calls[1]
    assert "tmp/path2.optim.onnx" == args[0]
    assert ["CUDAExecutionProvider"] == kwargs["providers"]
