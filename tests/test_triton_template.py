import mock
import pytest

from quick_deploy.triton_template import TritonIOConf, TritonIOTypeConf, TritonModelConf


@pytest.fixture
def conf():
    input_ids = TritonIOConf(name='input_ids', data_type=TritonIOTypeConf.INT64, dims=[-1, -1])

    token_type_ids = TritonIOConf(name='token_type_ids', data_type=TritonIOTypeConf.INT64, dims=[-1, -1])

    attention_mask = TritonIOConf(name='attention_mask', data_type=TritonIOTypeConf.INT64, dims=[-1, -1])

    output = TritonIOConf(name='output', data_type=TritonIOTypeConf.FP32, dims=[-1, 10])

    model_input = [input_ids, token_type_ids, attention_mask]
    model_output = [output]

    conf = TritonModelConf(
        workind_directory='tmp/wd',
        model_name="test",
        batch_size=0,
        nb_instance=1,
        use_cuda=False,
        model_inputs=model_input,
        model_outputs=model_output,
    )
    return conf


def test_get_conf(conf):
    inference_conf = """name: "test"
max_batch_size: 0
platform: "onnxruntime_onnx"
default_model_filename: "model.bin"

input [
    {
        name: "input_ids"
        data_type: TYPE_INT64
        dims: [-1, -1]
    },
    {
        name: "token_type_ids"
        data_type: TYPE_INT64
        dims: [-1, -1]
    },
    {
        name: "attention_mask"
        data_type: TYPE_INT64
        dims: [-1, -1]
    }
]

output [
    {
        name: "output"
        data_type: TYPE_FP32
        dims: [-1, 10]
    }
]

instance_group [
    {
      count: 1
      kind: KIND_CPU
    }
]""".strip()
    assert inference_conf == conf.get_conf()


@mock.patch("quick_deploy.triton_template.shutil.copy")
@mock.patch("quick_deploy.triton_template.Path")
def test_write_correct_paths(p, s, conf):
    pipe_tokenizer = mock.Mock()
    conf.write("tmp/path.optim.onnx")

    name, args, kwargs = s.mock_calls[0]

    assert "tmp/path.optim.onnx" == args[0]
    assert "model.bin" in args[1]
