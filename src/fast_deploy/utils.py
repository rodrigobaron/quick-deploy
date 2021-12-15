import logging
from argparse import Namespace
from collections import OrderedDict
from typing import OrderedDict as OrderedDictType
from typing import Tuple

import numpy as np
import torch
import tensorflow as tf

import re
import unidecode


def slugify(text):
    text = unidecode.unidecode(text).lower()
    return re.sub(r'[\W_]+', '_', text)


def setup_logging(level: int = logging.INFO) -> None:
    """Setup the logging to desired level and format the message.

    This filter the logging by the level and format the message.

    Parameters
    ----------
    level: int
        The logging level.
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )


def get_provider(args: Namespace) -> str:
    """Get provider by common argparse parameters.

    This return the execution ort provide: 'CUDAExecutionProvider' or  'CPUExecutionProvider'.

    Parameters
    ----------
    args: Namespace
        The argparse namespace.

    Returns
    ----------
    str
        'CUDAExecutionProvider' or  'CPUExecutionProvider'.
    """
    if args.cuda:
        assert torch.cuda.is_available(), "CUDA is not available. Please check your CUDA installation"
        return "CUDAExecutionProvider"

    return "CPUExecutionProvider"


def parse_transformer_torch_input(
    seq_len: int, batch_size: int, include_token_ids: bool
) -> Tuple[OrderedDictType[str, torch.Tensor], OrderedDictType[str, np.ndarray]]:
    """Get transformers model input as torch and onnx.

    This is used to create torch and onnx model input configuration.

    Parameters
    ----------
    seq_len: int
        Transformer model sequence length.
    include_token_ids: bool
        Check if need return the token ids also.

    Returns
    ----------
    Tuple[Tuple[Dict[str, Tensor]], Tuple[Dict[str, Tensor]]]
        A tuple of same shape for torch and onnx.
    """
    shape = (batch_size, seq_len)
    inputs_pytorch: OrderedDict[str, torch.Tensor] = OrderedDict()
    inputs_pytorch["input_ids"] = torch.randint(high=100, size=shape, dtype=torch.long)
    if include_token_ids:
        inputs_pytorch["token_type_ids"] = torch.ones(size=shape, dtype=torch.long)
    inputs_pytorch["attention_mask"] = torch.ones(size=shape, dtype=torch.long)
    inputs_onnx: OrderedDict[str, np.ndarray] = OrderedDict(
        {k: np.ascontiguousarray(v.detach().cpu().numpy()) for k, v in inputs_pytorch.items()}
    )
    return inputs_pytorch, inputs_onnx


def parse_torch_input(
    shape: Tuple[int], batch_size: int
) -> Tuple[OrderedDictType[str, torch.Tensor], OrderedDictType[str, np.ndarray]]:
    """Get model input as torch and onnx.

    This is used to create torch and onnx model input configuration.

    Parameters
    ----------
    seq_len: int
        Transformer model sequence length.
    include_token_ids: bool
        Check if need return the token ids also.

    Returns
    ----------
    Tuple[Tuple[Dict[str, Tensor]], Tuple[Dict[str, Tensor]]]
        A tuple of same shape for torch and onnx.
    """
    shape = (batch_size,) + shape
    inputs_pytorch: OrderedDict[str, torch.Tensor] = OrderedDict()
    inputs_pytorch["input"] = torch.randint(high=100, size=shape, dtype=torch.long).type(torch.float)
    inputs_onnx: OrderedDict[str, np.ndarray] = OrderedDict(
        {k: np.ascontiguousarray(v.detach().cpu().numpy()) for k, v in inputs_pytorch.items()}
    )
    return inputs_pytorch, inputs_onnx


def parse_transformer_tf_input(
    seq_len: int, batch_size: int, include_token_ids: bool
) -> Tuple[OrderedDictType[str, tf.Tensor], OrderedDictType[str, np.ndarray]]:
    """Get transformers model input as tf and onnx.

    This is used to create tf and onnx model input configuration.

    Parameters
    ----------
    seq_len: int
        Transformer model sequence length.
    include_token_ids: bool
        Check if need return the token ids also.

    Returns
    ----------
    Tuple[Tuple[Dict[str, Tensor]], Tuple[Dict[str, Tensor]]]
        A tuple of same shape for tf and onnx.
    """
    shape = (batch_size, seq_len)
    inputs_tf: OrderedDict[str, tf.Tensor] = OrderedDict()
    inputs_tf["input_ids"] = tf.random.uniform(shape=shape, maxval=100, dtype=tf.int64)
    if include_token_ids:
        inputs_tf["token_type_ids"] = tf.ones(shape=shape, dtype=tf.int64)
    inputs_tf["attention_mask"] = tf.ones(shape=shape, dtype=tf.int64)
    inputs_onnx: OrderedDict[str, np.ndarray] = OrderedDict(
        {k: np.ascontiguousarray(v.numpy()) for k, v in inputs_tf.items()}
    )
    return inputs_tf, inputs_onnx


def parse_tf_input(
    shape: Tuple[int], batch_size: int
) -> Tuple[OrderedDictType[str, tf.Tensor], OrderedDictType[str, np.ndarray]]:
    """Get model input as tensorflow and onnx.

    This is used to create tensorflow and onnx model input configuration.

    Parameters
    ----------
    seq_len: int
        Transformer model sequence length.
    include_token_ids: bool
        Check if need return the token ids also.

    Returns
    ----------
    Tuple[Tuple[Dict[str, Tensor]], Tuple[Dict[str, Tensor]]]
        A tuple of same shape for tensorflow and onnx.
    """
    shape = (batch_size,) + shape
    inputs_tf: OrderedDict[str, tf.Tensor] = OrderedDict()
    inputs_tf["input"] = tf.random.uniform(shape=shape, maxval=100, dtype=tf.float32)
    inputs_onnx: OrderedDict[str, np.ndarray] = OrderedDict(
        {k: np.ascontiguousarray(v.detach().cpu().numpy()) for k, v in inputs_tf.items()}
    )
    return inputs_tf, inputs_onnx
