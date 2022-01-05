import logging

import mock
import pytest

import numpy as np
from quick_deploy.utils import (
    get_provider,
    parse_transformer_torch_input,
    setup_logging,
    slugify,
    parse_torch_input,
    parse_transformer_tf_input,
    parse_tf_input
)


def test_slugify():
    assert "my_model" == slugify("My Model")


def test_setup_logging():
    vals = {
        "format": "%(asctime)s %(levelname)-8s %(message)s",
        "datefmt": "%m/%d/%Y %H:%M:%S",
        "level": 20,
    }
    m = mock.Mock()
    logging.basicConfig = m

    setup_logging(logging.INFO)
    name, args, kwargs = m.mock_calls[0]
    assert kwargs == vals


def test_get_provider():
    import torch

    m = mock.Mock()
    m.cuda = False

    provider_to_use = get_provider(m)
    assert "CPUExecutionProvider" == provider_to_use

    c = mock.Mock()
    c.return_value = True
    torch.cuda.is_available = c

    m.cuda = True

    provider_to_use = get_provider(m)
    assert "CUDAExecutionProvider" == provider_to_use

    with pytest.raises(AssertionError) as e:
        c = mock.Mock()
        c.return_value = False
        torch.cuda.is_available = c

        m.cuda = True

        provider_to_use = get_provider(m)
        e_error = "CUDA is not available. Please check your CUDA installation"
        assert e_error == str(e.value)


def test_parse_transformer_torch_input():
    torch_inputs, onnx_inputs = parse_transformer_torch_input(seq_len=16, batch_size=1, include_token_ids=True)
    input_keys = list(torch_inputs.keys())
    for k in ["input_ids", "token_type_ids", "attention_mask"]:
        assert k in input_keys

    for k in input_keys:
        assert torch_inputs[k].shape == onnx_inputs[k].shape
        assert torch_inputs[k].shape == (1, 16)

    torch_inputs, onnx_inputs = parse_transformer_torch_input(seq_len=256, batch_size=32, include_token_ids=False)
    input_keys = list(torch_inputs.keys())
    for k in ["input_ids", "attention_mask"]:
        assert k in input_keys

    for k in input_keys:
        assert torch_inputs[k].shape == onnx_inputs[k].shape
        assert torch_inputs[k].shape == (32, 256)


def test_parse_torch_input():
    inputs_pytorch, inputs_onnx = parse_torch_input(shape=(1,), batch_size=1, use_cuda=False)

    assert (1, 1) == tuple(inputs_pytorch['input'].shape)
    assert (1, 1) == inputs_onnx['input'].shape
    assert np.float32 == inputs_onnx['input'].dtype


def test_parse_transformer_tf_input():

    inputs_tf, onnx_inputs = parse_transformer_tf_input(
            batch_size=1, seq_len=16, include_token_ids=True
    )

    input_keys = list(inputs_tf.keys())
    for k in ["input_ids", "token_type_ids", "attention_mask"]:
        assert k in input_keys

    for k in input_keys:
        assert inputs_tf[k].shape == onnx_inputs[k].shape
        assert inputs_tf[k].shape == (1, 16)

    inputs_tf, onnx_inputs = parse_transformer_torch_input(seq_len=256, batch_size=32, include_token_ids=False)
    input_keys = list(inputs_tf.keys())
    for k in ["input_ids", "attention_mask"]:
        assert k in input_keys

    for k in input_keys:
        assert inputs_tf[k].shape == onnx_inputs[k].shape
        assert inputs_tf[k].shape == (32, 256)


def test_parse_tf_input():
    inputs_tf, inputs_onnx = parse_tf_input(shape=(1,), batch_size=1)

    assert (1, 1) == tuple(inputs_tf['input'].shape)
    assert (1, 1) == inputs_onnx['input'].shape
    assert np.float32 == inputs_onnx['input'].dtype