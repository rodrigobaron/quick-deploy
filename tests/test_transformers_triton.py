from fast_deploy.templates.transformer_triton import TransformersConfiguration

import pytest
import mock


@pytest.fixture
def conf():
    conf = TransformersConfiguration(
        model_name='test',
        batch_size=0,
        nb_output=10,
        nb_instance=1,
        include_token_type=True,
        workind_directory='tmp/',
        use_cuda=False
    )
    return conf


def test_model_conf(conf):
    model_conf = """name: "test_onnx_model"
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

output {
    name: "output"
    data_type: TYPE_FP32
    dims: [-1, 10]
}

instance_group [
    {
      count: 1
      kind: KIND_CPU
    }
]""".strip()
    assert model_conf == conf.get_model_conf()


def test_tokenizer_conf(conf):
    tokenizer_conf = """name: "test_onnx_tokenize"
max_batch_size: 0
backend: "python"

input [
    {
        name: "TEXT"
        data_type: TYPE_STRING
        dims: [ -1 ]
    }
]

output [
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

instance_group [
    {
      count: 1
      kind: KIND_CPU
    }
]""".strip()
    assert tokenizer_conf == conf.get_tokenize_conf()


def test_inference_conf(conf):
    inference_conf = """name: "test_onnx_inference"
max_batch_size: 0
platform: "ensemble"

input [
    {
        name: "TEXT"
        data_type: TYPE_STRING
        dims: [ -1 ]
    }
]

output {
    name: "output"
    data_type: TYPE_FP32
    dims: [-1, 10]
}

ensemble_scheduling {
    step [
        {
            model_name: "test_onnx_tokenize"
            model_version: -1
            input_map {
            key: "TEXT"
            value: "TEXT"
        }
        output_map [
            {
                key: "input_ids"
                value: "input_ids"
            },
            {
                key: "token_type_ids"
                value: "token_type_ids"
            },
            {
                key: "attention_mask"
                value: "attention_mask"
            }
        ]
        },
        {
            model_name: "test_onnx_model"
            model_version: -1
            input_map [
                {
                    key: "input_ids"
                    value: "input_ids"
                },
                {
                key: "token_type_ids"
                value: "token_type_ids"
            },
                {
                    key: "attention_mask"
                    value: "attention_mask"
                }
            ]
        output_map {
                key: "output"
                value: "output"
            }
        }
    ]
}""".strip()
    assert inference_conf == conf.get_inference_conf()

@mock.patch('fast_deploy.templates.transformer_triton.shutil.copy')
@mock.patch('fast_deploy.templates.transformer_triton.Path')
def test_create_folders(p, s, conf):
    pipe_tokenizer = mock.Mock()

    conf.create_folders(
        tokenizer=pipe_tokenizer,
        model_path='tmp/path.optim.onnx'
    )

    name, args, kwargs = s.mock_calls[1]

    assert 'tmp/path.optim.onnx' == args[0]
    assert 'model.bin' in args[1]

