import mock

from fast_deploy.backend.transformers_ort import (
    transformers_convert_pytorch,
    transformers_optimize_onnx,
)
from fast_deploy.utils import parse_transformer_torch_input


@mock.patch("fast_deploy.backend.transformers_ort.torch.onnx.export")
def test_transformers_convert_pytorch(e):
    pipe_model = mock.Mock()

    torch_inputs, onnx_inputs = parse_transformer_torch_input(seq_len=16, batch_size=1, include_token_ids=True)

    transformers_convert_pytorch(model=pipe_model, output_path="tmp/path.onnx", inputs_pytorch=torch_inputs)

    name, args, kwargs = e.mock_calls[0]
    for t in kwargs["args"]:
        assert (1, 16) == t.shape

    assert "tmp/path.onnx" == kwargs["f"]
    assert 12 == kwargs["opset_version"]

    for k in ["input_ids", "token_type_ids", "attention_mask"]:
        assert k in kwargs["input_names"]

    assert ["output"] == kwargs["output_names"]

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence"},
        "token_type_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "output": {0: "batch_size"},
    }

    assert dynamic_axes == kwargs["dynamic_axes"]


@mock.patch("fast_deploy.backend.transformers_ort.optimizer.optimize_model")
def test_transformers_optimize_onnx(o):

    transformers_optimize_onnx(
        onnx_path="tmp/path.onnx",
        output_path="tmp/path.optim.onnx",
        model_type="bert",
        use_cuda=False,
    )
    name, args, kwargs = o.mock_calls[0]

    kwargs_value = {
        "input": "tmp/path.onnx",
        "model_type": "bert",
        "use_gpu": False,
        "opt_level": 1,
        "num_heads": 0,
        "hidden_size": 0,
        "only_onnxruntime": True,
    }

    kwargs.pop("optimization_options")
    assert kwargs == kwargs_value
