import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
import pickle

from fast_deploy.backend.common import create_model_for_provider, generic_optimize_onnx
from fast_deploy.backend.transformers_ort import (
    transformers_convert_pytorch,
    transformers_optimize_onnx,
)
from fast_deploy.backend.torch_ort import torch_convert_pytorch
from fast_deploy.backend.skl_ort import parse_skl_input, skl_convert_onnx

from fast_deploy.triton_template import TritonIOTypeConf, TritonIOConf, TritonModelConf
from fast_deploy.utils import get_provider, parse_transformer_torch_input, parse_torch_input, setup_logging


def main_transformers(args):
    from transformers import pipeline

    provider_to_use = get_provider(args)

    pipe = pipeline(args.pipeline, model=args.model, tokenizer=args.tokenizer)
    pipe_tokenizer = pipe.tokenizer
    pipe_model = pipe.model

    if args.cuda:
        pipe_model.cuda()

    pipe_model.eval()
    input_names = pipe_tokenizer.model_input_names
    include_token_ids = "token_type_ids" in input_names

    inputs_pytorch, inputs_onnx = parse_transformer_torch_input(
        batch_size=1, seq_len=args.seq_len, include_token_ids=include_token_ids
    )

    with torch.inference_mode():
        output = pipe_model(**inputs_pytorch)
        output = output.logits
        output_pytorch: np.ndarray = output.detach().cpu().numpy()

    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    onnx_model_path = Path(f"{args.workdir}/transformer_{args.name}.onnx").as_posix()
    onnx_optim_model_path = Path(f"{args.workdir}/transformer_{args.name}.optim.onnx").as_posix()

    transformers_convert_pytorch(
        model=pipe_model, output_path=onnx_model_path, inputs_pytorch=inputs_pytorch, verbose=args.verbose
    )
    del pipe_model

    if args.atol is not None:
        onnx_model = create_model_for_provider(path=onnx_model_path, provider_to_use=provider_to_use)
        output_onnx = onnx_model.run(None, inputs_onnx)
        assert np.allclose(a=output_onnx, b=output_pytorch, atol=args.atol)

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

    generic_optimize_onnx(onnx_path=to_optimize, output_path=onnx_optim_model_path)

    if args.atol is not None:
        onnx_model = create_model_for_provider(path=onnx_optim_model_path, provider_to_use=provider_to_use)
        output_onnx_optimised = onnx_model.run(None, inputs_onnx)
        assert np.allclose(a=output_onnx_optimised, b=output_pytorch, atol=args.atol)
        
    input_ids = TritonIOConf(
        name='input_ids',
        data_type=TritonIOTypeConf.INT64,
        dims=[-1, -1]
    )

    token_type_ids= TritonIOConf(
        name='token_type_ids',
        data_type=TritonIOTypeConf.INT64,
        dims=[-1, -1]
    )

    attention_mask= TritonIOConf(
        name='attention_mask',
        data_type=TritonIOTypeConf.INT64,
        dims=[-1, -1]
    )

    output_shape = [-1 for _ in output_pytorch.shape[:-1]]
    output_shape = output_shape + [output_pytorch.shape[-1]]
    output= TritonIOConf(
        name='output',
        data_type=TritonIOTypeConf.FP32,
        dims=output_shape
    )

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
        model_outputs=model_output
    )
    model_conf.write(onnx_optim_model_path)


def main_torch(args):
    torch_model = torch.load(args.model)
    if args.cuda:
        torch_model.cuda()

    torch_model.eval()

    with open(args.file, "r") as stream:
        io_conf = yaml.safe_load(stream)

    assert "IOSchema" == io_conf['kind']

    model_input = []
    for m_input in io_conf['inputs']:
        t_conf = TritonIOConf(
            name=m_input['name'],
            data_type=TritonIOTypeConf.from_str(m_input['dtype']),
            dims=[-1] + m_input['shape']
        )
        model_input.append(t_conf)
    
    model_output = []
    for m_output in io_conf['outputs']:
        t_conf = TritonIOConf(
            name=m_output['name'],
            data_type=TritonIOTypeConf.from_str(m_output['dtype']),
            dims=[-1] + m_output['shape']
        )
        model_output.append(t_conf)

    input_shape = tuple(model_input[-1].dims[1:])
    inputs_pytorch, inputs_onnx = parse_torch_input(shape=input_shape, batch_size=1)

    with torch.inference_mode():
        output = torch_model(inputs_pytorch['input'])
        output_pytorch: np.ndarray = output.detach().cpu().numpy()

    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    onnx_model_path = Path(f"{args.workdir}/torch_{args.name}.onnx").as_posix()
    onnx_optim_model_path = Path(f"{args.workdir}/torch_{args.name}.optim.onnx").as_posix()

    torch_convert_pytorch(
        model=torch_model, output_path=onnx_model_path, inputs_pytorch=inputs_pytorch, verbose=args.verbose
    )
    del torch_model

    if args.atol is not None:
        onnx_model = create_model_for_provider(path=onnx_model_path, provider_to_use=provider_to_use)
        output_onnx = onnx_model.run(None, inputs_onnx)
        assert np.allclose(a=output_onnx, b=output_pytorch, atol=args.atol)

    if args.no_quant:
        onnx_optim_model_path = onnx_model_path
    else:
        generic_optimize_onnx(onnx_path=onnx_model_path, output_path=onnx_optim_model_path)
    
    model_conf = TritonModelConf(
        workind_directory=args.output,
        model_name=args.name,
        batch_size=0,
        nb_instance=1,
        use_cuda=args.cuda,
        model_inputs=model_input,
        model_outputs=model_output
    )
    model_conf.write(onnx_optim_model_path)


def main_skl(args):
    with open(args.model, "rb") as p_file:
        model = pickle.load(p_file)

    with open(args.file, "r") as stream:
        io_conf = yaml.safe_load(stream)

    assert "IOSchema" == io_conf['kind']

    model_input = []
    initial_type = []
    for m_input in io_conf['inputs']:
        t_conf = TritonIOConf(
            name=m_input['name'],
            data_type=TritonIOTypeConf.from_str(m_input['dtype']),
            dims=[-1] + m_input['shape']
        )
        model_input.append(t_conf)
        initial_type.append(
            (
                m_input['name'], 
                parse_skl_input([None] + m_input['shape'], m_input['dtype'])
            )
        )
    
    model_output = []
    for m_output in io_conf['outputs']:
        t_conf = TritonIOConf(
            name=m_output['name'],
            data_type=TritonIOTypeConf.from_str(m_output['dtype']),
            dims=m_output['shape']
        )
        model_output.append(t_conf)

    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    onnx_model_path = Path(f"{args.workdir}/skl_{args.name}.onnx").as_posix()

    skl_convert_onnx(
        model=model, output_path=onnx_model_path, inputs_type=initial_type, verbose=args.verbose
    )

    model_conf = TritonModelConf(
        workind_directory=args.output,
        model_name=args.name,
        batch_size=0,
        nb_instance=1,
        use_cuda=False,
        model_inputs=model_input,
        model_outputs=model_output
    )
    model_conf.write(onnx_model_path)


def default_args(parser):
    parser.add_argument("-n", "--name", required=True, help="model name")
    parser.add_argument("-m", "--model", required=True, help="model path")
    parser.add_argument("-o", "--output", required=True, help="path used to export models")
    parser.add_argument("-w", "--workdir", default="env/", help="model path")
    parser.add_argument("--nb-instances", default=1, help="# of model instances", type=int)
    parser.add_argument("--cuda", action="store_true", help="use cuda optimization")
    parser.add_argument("-v", "--verbose", action="store_true", help="display detailed information")
    parser.add_argument("--atol", default=None, help="test outputs when convert", type=float)


def transformers_args(parser_tra):
    parser_tra.add_argument("-t", "--tokenizer", help="tokenizer path")
    parser_tra.add_argument(
        "--model-type",
        help="custom optimization for transformer model type. One of [bert, bart, gpt2]",
        choices=["bert", "bart", "gpt2"],
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
    parser_torch.add_argument("-f", "--file", required=True, help="model IO configuration.")
    parser_torch.add_argument("--no-quant", action="store_true", help="avoid quant optimization")
    parser_torch.set_defaults(func=main_torch)


def main():
    parser = argparse.ArgumentParser(
        description="Optimize and deploy machine learning models fast as possible!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(help="transformers help", dest="transformers, torch")
    subparsers.required = True

    # transformers arguments and func binding
    parser_tra = subparsers.add_parser("transformers")
    default_args(parser_tra)
    transformers_args(parser_tra)

    # torch arguments and func binding
    parser_torch = subparsers.add_parser("torch")
    default_args(parser_torch)
    torch_args(parser_torch)

    # sklearn arguments and func binding
    parser_skl = subparsers.add_parser("sklearn")
    default_args(parser_skl)
    parser_skl.add_argument("-f", "--file", required=True, help="model IO configuration.")
    parser_skl.set_defaults(func=main_skl)
    

    args = parser.parse_args()
    setup_logging(level=logging.INFO if args.verbose else logging.WARNING)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
