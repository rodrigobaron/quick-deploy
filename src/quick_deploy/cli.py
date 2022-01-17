import argparse
import logging
import pickle
from os.path import expanduser
from pathlib import Path

import numpy as np
import yaml

from quick_deploy.backend.common import create_model_for_provider, generic_optimize_onnx
from quick_deploy.triton_template import TritonIOConf, TritonIOTypeConf, TritonModelConf
from quick_deploy.utils import (
    get_provider,
    parse_torch_input,
    parse_tf_input,
    parse_transformer_tf_input,
    parse_transformer_torch_input,
    setup_logging,
)


def main_transformers(args):
    """Command to parse transformers models"""
    from transformers import pipeline

    from quick_deploy.backend.transformers_ort import (
        transformers_convert_pytorch,
        transformers_convert_tf,
        transformers_optimize_onnx,
    )

    onnx_model_path = Path(f"{expanduser(args.workdir)}/transformer_{args.name}.onnx").as_posix()
    onnx_optim_model_path = Path(f"{expanduser(args.workdir)}/transformer_{args.name}.optim.onnx").as_posix()

    provider_to_use = get_provider(args)

    pipe = pipeline(args.pipeline, model=args.model, tokenizer=args.tokenizer, framework=args.framework)
    pipe_tokenizer = pipe.tokenizer
    pipe_model = pipe.model

    if args.cuda:
        pipe_model.cuda()

    input_names = pipe_tokenizer.model_input_names
    include_token_ids = "token_type_ids" in input_names

    output_np: np.ndarray = None
    if "pt" == args.framework:
        import torch

        pipe_model.eval()
        inputs_pytorch, inputs_onnx = parse_transformer_torch_input(
            batch_size=1, seq_len=args.seq_len, include_token_ids=include_token_ids, use_cuda=args.cuda
        )

        with torch.inference_mode():
            output = pipe_model(**inputs_pytorch)
            output = output.logits
            output_np = output.detach().cpu().numpy()

        transformers_convert_pytorch(
            model=pipe_model, output_path=onnx_model_path, inputs_pytorch=inputs_pytorch, verbose=args.verbose
        )
    elif "tf" == args.framework:
        inputs_tf, inputs_onnx = parse_transformer_tf_input(
            batch_size=1, seq_len=args.seq_len, include_token_ids=include_token_ids
        )

        output = pipe_model(inputs_tf)
        output = output.logits
        output_np = output.numpy()

        transformers_convert_tf(
            model=pipe_model, output_path=onnx_model_path, inputs_tf=inputs_tf, verbose=args.verbose
        )
    else:
        raise ValueError

    del pipe_model

    if args.atol is not None:
        onnx_model = create_model_for_provider(path=onnx_model_path, provider_to_use=provider_to_use)
        output_onnx = onnx_model.run(None, inputs_onnx)
        assert np.allclose(a=output_onnx, b=output_np, atol=args.atol)

    to_optimize: str = onnx_model_path
    if args.model_type is not None:
        transformers_optimize_onnx(
            onnx_path=onnx_model_path,
            output_path=onnx_optim_model_path,
            model_type=args.model_type,
            use_cuda=args.cuda,
            num_heads=args.num_heads,
            hidden_size=args.hidden_size,
        )
        to_optimize = onnx_optim_model_path

    generic_optimize_onnx(onnx_path=to_optimize, output_path=onnx_optim_model_path, quant_type=args.quant_type)

    if args.atol is not None:
        onnx_model = create_model_for_provider(path=onnx_optim_model_path, provider_to_use=provider_to_use)
        output_onnx_optimised = onnx_model.run(None, inputs_onnx)
        assert np.allclose(a=output_onnx_optimised, b=output_np, atol=args.atol)

    input_ids = TritonIOConf(name='input_ids', data_type=TritonIOTypeConf.INT64, dims=[-1, -1])

    token_type_ids = TritonIOConf(name='token_type_ids', data_type=TritonIOTypeConf.INT64, dims=[-1, -1])

    attention_mask = TritonIOConf(name='attention_mask', data_type=TritonIOTypeConf.INT64, dims=[-1, -1])

    output_shape = [-1 for _ in output_np.shape[:-1]]
    output_shape = output_shape + [output_np.shape[-1]]
    output = TritonIOConf(name='output', data_type=TritonIOTypeConf.FP32, dims=output_shape)

    model_input = [input_ids, attention_mask]
    model_output = [output]

    if include_token_ids:
        model_input.insert(1, token_type_ids)

    model_conf = TritonModelConf(
        workind_directory=args.output,
        model_name=args.name,
        batch_size=0,
        nb_instance=1,
        use_cuda=args.cuda,
        model_inputs=model_input,
        model_outputs=model_output,
    )
    model_conf.write(onnx_optim_model_path)


def main_torch(args):
    """Command to parse torch models"""
    import torch

    from quick_deploy.backend.torch_ort import torch_convert_onnx

    torch_model = torch.load(args.model)
    if args.cuda:
        torch_model.cuda()

    torch_model.eval()
    provider_to_use = get_provider(args)

    with open(args.file, "r") as stream:
        io_conf = yaml.safe_load(stream)

    assert "IOSchema" == io_conf['kind']

    model_input = []
    for m_input in io_conf['inputs']:
        t_conf = TritonIOConf(
            name=m_input['name'], data_type=TritonIOTypeConf.from_str(m_input['dtype']), dims=[-1] + m_input['shape']
        )
        model_input.append(t_conf)

    model_output = []
    for m_output in io_conf['outputs']:
        t_conf = TritonIOConf(
            name=m_output['name'],
            data_type=TritonIOTypeConf.from_str(m_output['dtype']),
            dims=[-1] + m_output['shape'],
        )
        model_output.append(t_conf)

    input_shape = tuple(model_input[-1].dims[1:])
    inputs_pytorch, inputs_onnx = parse_torch_input(shape=input_shape, batch_size=1, use_cuda=args.cuda)

    with torch.inference_mode():
        output = torch_model(inputs_pytorch['input'])
        output_np: np.ndarray = output.detach().cpu().numpy() if torch.is_tensor(output) else output

    onnx_model_path = Path(f"{expanduser(args.workdir)}/torch_{args.name}.onnx").as_posix()
    onnx_optim_model_path = Path(f"{expanduser(args.workdir)}/torch_{args.name}.optim.onnx").as_posix()

    torch_convert_onnx(
        model=torch_model, output_path=onnx_model_path, inputs_pytorch=inputs_pytorch, verbose=args.verbose
    )
    del torch_model

    if args.atol is not None:
        onnx_model = create_model_for_provider(path=onnx_model_path, provider_to_use=provider_to_use)
        output_onnx = onnx_model.run(None, inputs_onnx)
        assert np.allclose(a=output_onnx, b=output_np, atol=args.atol)

    if args.no_quant:
        onnx_optim_model_path = onnx_model_path
    else:
        generic_optimize_onnx(onnx_path=onnx_model_path, output_path=onnx_optim_model_path, quant_type=args.quant_type)

    model_conf = TritonModelConf(
        workind_directory=args.output,
        model_name=args.name,
        batch_size=0,
        nb_instance=1,
        use_cuda=args.cuda,
        model_inputs=model_input,
        model_outputs=model_output,
    )
    model_conf.write(onnx_optim_model_path)


def main_tf(args):
    """Command to parse torch models"""
    import tensorflow as tf
    from quick_deploy.backend.tf_ort import tf_convert_onnx

    tf_model = tf.keras.models.load_model(args.model)
    provider_to_use = get_provider(args)

    with open(args.file, "r") as stream:
        io_conf = yaml.safe_load(stream)

    assert "IOSchema" == io_conf['kind']

    model_input = []
    for m_input in io_conf['inputs']:
        t_conf = TritonIOConf(
            name=m_input['name'], data_type=TritonIOTypeConf.from_str(m_input['dtype']), dims=[-1] + m_input['shape']
        )
        model_input.append(t_conf)

    model_output = []
    for m_output in io_conf['outputs']:
        t_conf = TritonIOConf(
            name=m_output['name'],
            data_type=TritonIOTypeConf.from_str(m_output['dtype']),
            dims=[-1] + m_output['shape'],
        )
        model_output.append(t_conf)

    input_shape = tuple(model_input[-1].dims[1:])
    inputs_tf, inputs_onnx = parse_tf_input(shape=input_shape, batch_size=1)

    output = tf_model.predict(inputs_tf['input'])
    output_np: np.ndarray = None
    if args.cuda:
        output_np = output.cpu().numpy()
    else:
        output_np = output

    onnx_model_path = Path(f"{expanduser(args.workdir)}/tf_{args.name}.onnx").as_posix()
    onnx_optim_model_path = Path(f"{expanduser(args.workdir)}/tf_{args.name}.optim.onnx").as_posix()

    tf_convert_onnx(model=tf_model, output_path=onnx_model_path, verbose=args.verbose)
    del tf_model

    if args.atol is not None:
        onnx_model = create_model_for_provider(path=onnx_model_path, provider_to_use=provider_to_use)
        output_onnx = onnx_model.run(None, inputs_onnx)
        assert np.allclose(a=output_onnx, b=output_np, atol=args.atol)

    if args.no_quant:
        onnx_optim_model_path = onnx_model_path
    else:
        generic_optimize_onnx(onnx_path=onnx_model_path, output_path=onnx_optim_model_path, quant_type=args.quant_type)

    model_conf = TritonModelConf(
        workind_directory=args.output,
        model_name=args.name,
        batch_size=0,
        nb_instance=1,
        use_cuda=args.cuda,
        model_inputs=model_input,
        model_outputs=model_output,
    )
    model_conf.write(onnx_optim_model_path)


def main_skl(args):
    """Command to parse sklearn models."""
    from quick_deploy.backend.skl_ort import parse_skl_input, skl_convert_onnx

    with open(args.model, "rb") as p_file:
        model = pickle.load(p_file)

    with open(args.file, "r") as stream:
        io_conf = yaml.safe_load(stream)

    assert "IOSchema" == io_conf['kind']

    model_input = []
    initial_type = []
    for m_input in io_conf['inputs']:
        t_conf = TritonIOConf(
            name=m_input['name'], data_type=TritonIOTypeConf.from_str(m_input['dtype']), dims=[-1] + m_input['shape']
        )
        model_input.append(t_conf)
        initial_type.append((m_input['name'], parse_skl_input([None] + m_input['shape'], m_input['dtype'])))

    model_output = []
    for m_output in io_conf['outputs']:
        t_conf = TritonIOConf(
            name=m_output['name'], data_type=TritonIOTypeConf.from_str(m_output['dtype']), dims=m_output['shape']
        )
        model_output.append(t_conf)

    onnx_model_path = Path(f"{expanduser(args.workdir)}/skl_{args.name}.onnx").as_posix()

    skl_convert_onnx(model=model, output_path=onnx_model_path, inputs_type=initial_type, verbose=args.verbose)

    model_conf = TritonModelConf(
        workind_directory=args.output,
        model_name=args.name,
        batch_size=0,
        nb_instance=1,
        use_cuda=False,
        model_inputs=model_input,
        model_outputs=model_output,
    )
    model_conf.write(onnx_model_path)


def main_xgb(args):
    """Command to parse xgboost models."""
    from quick_deploy.backend.xgb_ort import parse_xgb_input, xgb_convert_onnx

    with open(args.model, "rb") as p_file:
        model = pickle.load(p_file)

    with open(args.file, "r") as stream:
        io_conf = yaml.safe_load(stream)

    assert "IOSchema" == io_conf['kind']
    onnx_model_path = Path(f"{expanduser(args.workdir)}/xgb_{args.name}.onnx").as_posix()

    model_input = []
    initial_type = []
    for m_input in io_conf['inputs']:
        t_conf = TritonIOConf(
            name=m_input['name'], data_type=TritonIOTypeConf.from_str(m_input['dtype']), dims=[-1] + m_input['shape']
        )
        model_input.append(t_conf)
        initial_type.append((m_input['name'], parse_xgb_input([None] + m_input['shape'], m_input['dtype'])))

    model_output = []
    for m_output in io_conf['outputs']:
        t_conf = TritonIOConf(
            name=m_output['name'], data_type=TritonIOTypeConf.from_str(m_output['dtype']), dims=m_output['shape']
        )
        model_output.append(t_conf)

    xgb_convert_onnx(model=model, output_path=onnx_model_path, inputs_type=initial_type, verbose=args.verbose)

    model_conf = TritonModelConf(
        workind_directory=args.output,
        model_name=args.name,
        batch_size=0,
        nb_instance=1,
        use_cuda=False,
        model_inputs=model_input,
        model_outputs=model_output,
    )
    model_conf.write(onnx_model_path)


def default_args(parser):
    """Parse common arguments between commands."""
    parser.add_argument("-n", "--name", required=True, help="model name")
    parser.add_argument("-m", "--model", required=True, help="model path")
    parser.add_argument("-o", "--output", required=True, help="path used to export models")
    parser.add_argument("-w", "--workdir", default="~/.quick_deploy/", help="model path")
    parser.add_argument("--nb-instances", default=1, help="# of model instances", type=int)
    parser.add_argument("--cuda", action="store_true", help="use cuda optimization")
    parser.add_argument("-v", "--verbose", action="store_true", help="display detailed information")
    parser.add_argument("--atol", default=None, help="test outputs when convert", type=float)
    parser.add_argument("--custom-module", default=None, help="use custom module path")
    parser.add_argument(
        "--quant-type",
        default="int8",
        help="set quantization weights type",
        choices=["int8", "uint8"],
    )


def transformers_args(parser_tra):
    """Parse transformers arguments."""
    parser_tra.add_argument("-t", "--tokenizer", help="tokenizer path")
    parser_tra.add_argument(
        "--model-type",
        help="custom optimization for transformer model type. One of [bert, bart, gpt2]",
        choices=["bert", "bart", "gpt2"],
    )
    parser_tra.add_argument(
        "--framework",
        type=str,
        choices=["pt", "tf"],
        default="pt",
        help="Framework for loading the model",
    )
    parser_tra.add_argument("--num-heads", default=0, help="number of heads (not needed for bert)", type=int)
    parser_tra.add_argument("--hidden-size", default=0, help="the weights size (not needed for bert)", type=int)
    parser_tra.add_argument(
        "-p",
        "--pipeline",
        required=True,
        help="pipeline task, eg: 'text-classification'",
    )
    parser_tra.add_argument("--seq-len", default=16, help="sequence length to optimize", type=int)
    parser_tra.set_defaults(func=main_transformers)


def torch_args(parser_torch):
    """Parse torch arguments."""
    parser_torch.add_argument("-f", "--file", required=True, help="model IO configuration.")
    parser_torch.add_argument("--no-quant", action="store_true", help="avoid quant optimization")
    parser_torch.set_defaults(func=main_torch)


def tf_args(parser_tf):
    """Parse tensorflow arguments."""
    parser_tf.add_argument("-f", "--file", required=True, help="model IO configuration.")
    parser_tf.add_argument("--no-quant", action="store_true", help="avoid quant optimization")
    parser_tf.set_defaults(func=main_tf)


def skl_args(parser_skl):
    parser_skl.add_argument("-f", "--file", required=True, help="model IO configuration.")
    parser_skl.set_defaults(func=main_skl)


def xgb_args(parser_xgb):
    parser_xgb.add_argument("-f", "--file", required=True, help="model IO configuration.")
    parser_xgb.set_defaults(func=main_xgb)


def main():
    """Entry-point function."""

    parser = argparse.ArgumentParser(
        description="Optimize and deploy machine learning models fast as possible!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="transformers, torch, tf, sklearn, xgboost")
    subparsers.required = True

    # transformers arguments and func binding
    parser_tra = subparsers.add_parser("transformers")
    default_args(parser_tra)
    transformers_args(parser_tra)

    # torch arguments and func binding
    parser_torch = subparsers.add_parser("torch")
    default_args(parser_torch)
    torch_args(parser_torch)

    # tf arguments and func binding
    parser_tf = subparsers.add_parser("tf")
    default_args(parser_tf)
    tf_args(parser_tf)

    # sklearn arguments and func binding
    parser_skl = subparsers.add_parser("sklearn")
    default_args(parser_skl)
    skl_args(parser_skl)

    # xgboost arguments and func binding
    parser_xgb = subparsers.add_parser("xgboost")
    default_args(parser_xgb)
    xgb_args(parser_xgb)

    args = parser.parse_args()
    setup_logging(level=logging.INFO if args.verbose else logging.WARNING)

    if args.custom_module is not None:
        import sys

        sys.path.insert(0, args.custom_module)

    Path(expanduser(args.workdir)).mkdir(parents=True, exist_ok=True)
    Path(expanduser(args.output)).mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
