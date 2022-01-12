import mock
import numpy as np
import pytest
import torch

from quick_deploy.cli import (
    default_args,
    main,
    main_torch,
    main_tf,
    main_transformers,
    main_skl,
    main_xgb,
    skl_args,
    torch_args,
    tf_args,
    transformers_args,
    xgb_args,
)


class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, obj(b) if isinstance(b, dict) else b)


@pytest.fixture
def transformers_args_fixture():
    return obj(
        {
            'atol': None,
            'cuda': False,
            'framework': 'pt',
            'hidden_size': 0,
            'model': 'test-model',
            'model_type': 'bert',
            'name': 'test',
            'nb_instances': 1,
            'num_heads': 0,
            'output': '/tmp/models',
            'pipeline': 'fill-mask',
            'seq_len': 16,
            'tokenizer': None,
            'verbose': False,
            'workdir': '/tmp/quick_deploy/',
            'quant_type': 'int8',
        }
    )


@pytest.fixture
def torch_args_fixture():
    return obj(
        {
            "atol": None,
            "cuda": False,
            "file": 'test.yaml',
            "model": 'test.pt',
            "name": 'test',
            "nb_instances": 1,
            "no_quant": False,
            "output": '/tmp/tf',
            "verbose": False,
            "workdir": '/tmp/quick_deploy/',
            'quant_type': 'int8',
        }
    )


@pytest.fixture
def tf_args_fixture():
    return obj(
        {
            "atol": None,
            "cuda": False,
            "file": 'test.yaml',
            "model": 'test.pt',
            "name": 'test',
            "nb_instances": 1,
            "no_quant": False,
            "output": '/tmp/tf',
            "verbose": False,
            "workdir": '/tmp/quick_deploy/',
            'quant_type': 'int8',
        }
    )


@pytest.fixture
def skl_args_fixture():
    return obj(
        {
            "atol": None,
            "cuda": False,
            "file": 'test.yaml',
            "model": 'test.bin',
            "name": 'test',
            "nb_instances": 1,
            "output": '/tmp/skl',
            "verbose": False,
            "workdir": '/tmp/quick_deploy/',
        }
    )


@pytest.fixture
def xgb_args_fixture():
    return obj(
        {
            "atol": None,
            "cuda": False,
            "file": 'test.yaml',
            "model": 'test.bin',
            "name": 'test',
            "nb_instances": 1,
            "output": '/tmp/xgb',
            "verbose": False,
            "workdir": '/tmp/quick_deploy/',
        }
    )


def parser_argument_asserts(parser, args_calls, kwargs_calls):
    total_calls = len(parser.add_argument.mock_calls)
    for i in range(total_calls):
        name, args, kwargs = parser.add_argument.mock_calls[i]
        assert args in args_calls
        assert kwargs in kwargs_calls

        k = args_calls.index(args)
        del args_calls[k]
        k = kwargs_calls.index(kwargs)
        del kwargs_calls[k]

    assert 0 == len(args_calls)
    assert 0 == len(kwargs_calls)


transformers_mock = mock.Mock()
transformers_ort_mock = mock.Mock()


@mock.patch("quick_deploy.cli.expanduser")
@mock.patch("quick_deploy.cli.Path")
@mock.patch("quick_deploy.cli.generic_optimize_onnx")
@mock.patch("quick_deploy.cli.TritonModelConf")
@mock.patch.dict(
    "sys.modules", {"transformers": transformers_mock, 'quick_deploy.backend.transformers_ort': transformers_ort_mock}
)
def test_main_transformers(t, g, p, e, transformers_args_fixture):
    # path transformers
    pipeline_mock = mock.Mock(name='pipeline')
    tokenizer_mock = mock.Mock(name='tokenizer')
    model_mock = mock.Mock(name='model')
    model_output_mock = mock.Mock(name='model_output')

    model_mock.return_value = model_output_mock
    model_output_mock.logits = torch.Tensor(np.ones((1, transformers_args_fixture.seq_len)))
    transformers_mock.pipeline.return_value = pipeline_mock
    pipeline_mock.tokenizer = tokenizer_mock
    pipeline_mock.model = model_mock

    tokenizer_mock.model_input_names = ["input_ids", "token_type_ids", "attention_mask"]

    # path backend
    transformers_ort_mock.transformers_convert_pytorch = mock.Mock()

    main_transformers(transformers_args_fixture)

    transformers_ort_mock.transformers_convert_pytorch.assert_called()
    g.assert_called()
    t.assert_called()


torch_mock = mock.Mock()
torch_ort_mock = mock.Mock()


@mock.patch("quick_deploy.cli.expanduser")
@mock.patch("quick_deploy.cli.Path")
@mock.patch("quick_deploy.cli.generic_optimize_onnx")
@mock.patch("quick_deploy.cli.TritonModelConf")
@mock.patch("quick_deploy.cli.open")
@mock.patch("quick_deploy.cli.yaml.safe_load")
@mock.patch.dict("sys.modules", {'torch': torch_mock, 'quick_deploy.backend.torch_ort': torch_ort_mock})
def test_main_torch(y, o, t, g, p, e, torch_args_fixture):
    y.return_value = {
        'kind': 'IOSchema',
        'inputs': [{'name': 'input', 'dtype': 'float32', 'shape': [1, 1, 1]}],
        'outputs': [{'name': 'output', 'dtype': 'float32', 'shape': [1]}],
    }

    torch_mock.inference_mode.return_value = mock.Mock()

    torch_mock.inference_mode.return_value.__enter__ = mock.Mock()
    torch_mock.inference_mode.return_value.__exit__ = mock.Mock()

    # path backend
    torch_ort_mock.torch_convert_onnx = mock.Mock()
    main_torch(torch_args_fixture)

    torch_ort_mock.torch_convert_onnx.assert_called()
    g.assert_called()
    t.assert_called()


tf_ort_mock = mock.Mock()
tf_mock = mock.Mock()


@mock.patch("quick_deploy.cli.expanduser")
@mock.patch("quick_deploy.cli.Path")
@mock.patch("quick_deploy.cli.generic_optimize_onnx")
@mock.patch("quick_deploy.cli.TritonModelConf")
@mock.patch("quick_deploy.cli.open")
@mock.patch("quick_deploy.cli.yaml.safe_load")
@mock.patch.dict("sys.modules", {'tensorflow': tf_mock, 'quick_deploy.backend.tf_ort': tf_ort_mock})
def test_main_tf(y, o, t, g, p, e, tf_args_fixture):
    y.return_value = {
        'kind': 'IOSchema',
        'inputs': [{'name': 'input', 'dtype': 'float32', 'shape': [1, 1, 1]}],
        'outputs': [{'name': 'output', 'dtype': 'float32', 'shape': [1]}],
    }

    # path backend
    tf_ort_mock.tf_convert_onnx = mock.Mock()
    main_tf(tf_args_fixture)

    tf_ort_mock.tf_convert_onnx.assert_called()
    g.assert_called()
    t.assert_called()


skl_ort_mock = mock.Mock()


@mock.patch("quick_deploy.cli.TritonModelConf")
@mock.patch("quick_deploy.cli.open")
@mock.patch("quick_deploy.cli.yaml.safe_load")
@mock.patch("quick_deploy.cli.pickle.load")
@mock.patch.dict("sys.modules", {'quick_deploy.backend.skl_ort': skl_ort_mock})
def test_main_skl(p, y, o, t, skl_args_fixture):
    y.return_value = {
        'kind': 'IOSchema',
        'inputs': [{'name': 'input', 'dtype': 'float32', 'shape': [1, 1, 1]}],
        'outputs': [{'name': 'output', 'dtype': 'float32', 'shape': [1]}],
    }

    o.return_value.__enter__ = mock.Mock()
    o.return_value.__exit__ = mock.Mock()

    # path backend
    skl_ort_mock.skl_convert_onnx = mock.Mock()
    main_skl(skl_args_fixture)

    skl_ort_mock.skl_convert_onnx.assert_called()
    t.assert_called()


xgb_ort_mock = mock.Mock()


@mock.patch("quick_deploy.cli.TritonModelConf")
@mock.patch("quick_deploy.cli.open")
@mock.patch("quick_deploy.cli.yaml.safe_load")
@mock.patch("quick_deploy.cli.pickle.load")
@mock.patch.dict("sys.modules", {'quick_deploy.backend.xgb_ort': xgb_ort_mock})
def test_main_xgb(p, y, o, t, xgb_args_fixture):
    y.return_value = {
        'kind': 'IOSchema',
        'inputs': [{'name': 'input', 'dtype': 'float32', 'shape': [1, 1, 1]}],
        'outputs': [{'name': 'output', 'dtype': 'float32', 'shape': [1]}],
    }

    o.return_value.__enter__ = mock.Mock()
    o.return_value.__exit__ = mock.Mock()

    # path backend
    xgb_ort_mock.xgb_convert_onnx = mock.Mock()
    main_xgb(xgb_args_fixture)

    xgb_ort_mock.xgb_convert_onnx.assert_called()
    t.assert_called()


def test_default_args():
    parser = mock.Mock()
    default_args(parser)

    args_calls = [
        ('-n', '--name'),
        ('-m', '--model'),
        ('-o', '--output'),
        ('-w', '--workdir'),
        ('--nb-instances',),
        ('--cuda',),
        ('-v', '--verbose'),
        ('--atol',),
        ('--custom-module',),
        ('--quant-type',),
    ]
    kwargs_calls = [
        {'required': True, 'help': 'model name'},
        {'required': True, 'help': 'model path'},
        {'required': True, 'help': 'path used to export models'},
        {'default': '~/.quick_deploy/', 'help': 'model path'},
        {'default': 1, 'help': '# of model instances', 'type': int},
        {'action': 'store_true', 'help': 'use cuda optimization'},
        {'action': 'store_true', 'help': 'display detailed information'},
        {'default': None, 'help': 'test outputs when convert', 'type': float},
        {'default': None, 'help': 'use custom module path'},
        {'choices': ['int8', 'uint8'], 'default': 'int8', 'help': 'set quantization weights type'},
    ]

    parser_argument_asserts(parser, args_calls, kwargs_calls)


def test_transformers_args():
    parser = mock.Mock()
    transformers_args(parser)

    args_calls = [
        ('-t', '--tokenizer'),
        ('--model-type',),
        ('--framework',),
        ('--num-heads',),
        ('--hidden-size',),
        ('-p', '--pipeline'),
        ('--seq-len',),
    ]
    kwargs_calls = [
        {'help': 'tokenizer path'},
        {
            'help': 'custom optimization for transformer model type. One of [bert, bart, gpt2]',
            'choices': ['bert', 'bart', 'gpt2'],
        },
        {'type': str, 'choices': ['pt', 'tf'], 'default': 'pt', 'help': 'Framework for loading the model'},
        {'default': 0, 'help': 'number of heads (not needed for bert)', 'type': int},
        {'default': 0, 'help': 'the weights size (not needed for bert)', 'type': int},
        {'required': True, 'help': "pipeline task, eg: 'text-classification'"},
        {'default': 16, 'help': 'sequence length to optimize', 'type': int},
    ]

    parser_argument_asserts(parser, args_calls, kwargs_calls)


def test_torch_args():
    parser = mock.Mock()
    torch_args(parser)

    args_calls = [('-f', '--file'), ('--no-quant',)]
    kwargs_calls = [
        {'required': True, 'help': 'model IO configuration.'},
        {'action': 'store_true', 'help': 'avoid quant optimization'},
    ]

    parser_argument_asserts(parser, args_calls, kwargs_calls)


def test_tf_args():
    parser = mock.Mock()
    tf_args(parser)

    args_calls = [('-f', '--file'), ('--no-quant',)]
    kwargs_calls = [
        {'required': True, 'help': 'model IO configuration.'},
        {'action': 'store_true', 'help': 'avoid quant optimization'},
    ]

    parser_argument_asserts(parser, args_calls, kwargs_calls)


def test_skl_args():
    parser = mock.Mock()
    skl_args(parser)

    args_calls = [
        ('-f', '--file'),
    ]
    kwargs_calls = [
        {'required': True, 'help': 'model IO configuration.'},
    ]

    parser_argument_asserts(parser, args_calls, kwargs_calls)


def test_xgb_args():
    parser = mock.Mock()
    xgb_args(parser)

    args_calls = [
        ('-f', '--file'),
    ]
    kwargs_calls = [
        {'required': True, 'help': 'model IO configuration.'},
    ]

    parser_argument_asserts(parser, args_calls, kwargs_calls)


@mock.patch("quick_deploy.cli.argparse")
@mock.patch("quick_deploy.cli.Path")
@mock.patch("quick_deploy.cli.expanduser")
def test_main(e, p, m):
    def get_subparsers(m):
        for i in range(len(m.mock_calls)):
            name, args, kwargs = m.mock_calls[i]
            if 'ArgumentParser().add_subparsers().add_parser' == name:
                yield args[0]

    main()
    subparsers = list(get_subparsers(m))

    assert "transformers" in subparsers
    assert "torch" in subparsers
    assert "sklearn" in subparsers
    assert "xgboost" in subparsers
