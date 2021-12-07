import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from fast_deploy.backend.common import create_model_for_provider, generic_optimize_onnx
from fast_deploy.backend.transformers_ort import (
    transformers_convert_pytorch,
    transformers_optimize_onnx,
)
from fast_deploy.templates.transformer_triton import TransformersConfiguration
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

    conf = TransformersConfiguration(
        model_name=args.name,
        batch_size=0,
        nb_output=output_pytorch.shape[1],
        nb_instance=1,
        include_token_type=include_token_ids,
        workind_directory=args.output,
        use_cuda=args.cuda,
    )
    conf.create_folders(tokenizer=pipe_tokenizer, model_path=onnx_optim_model_path)


def main_torch(args):
    torch_model = torch.load(args.model)
    if args.cuda:
        torch_model.cuda()

    torch_model.eval()

    input_shape = tuple([int(k.strip()) for k in args.input_shape.split(",")])
    inputs_pytorch, inputs_onnx = parse_torch_input(shape=input_shape, batch_size=1)

    with torch.inference_mode():
        output = torch_model(inputs_pytorch['input'])
        output_pytorch: np.ndarray = output.detach().cpu().numpy()

    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    onnx_model_path = Path(f"{args.workdir}/torch_{args.name}.onnx").as_posix()
    onnx_optim_model_path = Path(f"{args.workdir}/torch_{args.name}.optim.onnx").as_posix()

    from fast_deploy.backend.torch_ort import torch_convert_pytorch

    torch_convert_pytorch(
        model=torch_model, output_path=onnx_model_path, inputs_pytorch=inputs_pytorch, verbose=args.verbose
    )
    del torch_model

    if args.atol is not None:
        onnx_model = create_model_for_provider(path=onnx_model_path, provider_to_use=provider_to_use)
        output_onnx = onnx_model.run(None, inputs_onnx)
        assert np.allclose(a=output_onnx, b=output_pytorch, atol=args.atol)

    generic_optimize_onnx(onnx_path=onnx_model_path, output_path=onnx_optim_model_path)

    conf = GenericConfiguration(
        model_name=args.name,
        batch_size=0,
        nb_output=output_pytorch.shape[1],
        nb_instance=1,
        include_token_type=include_token_ids,
        workind_directory=args.output,
        use_cuda=args.cuda,
    )
    conf.create_folders(model_path=onnx_optim_model_path)


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
    parser_torch.add_argument("--input-shape", required=True, help="model shape. eg: 3, 256, 256")
    parser_torch.set_defaults(func=main_torch)

    args = parser.parse_args()
    # import pdb; pdb.set_trace()
    setup_logging(level=logging.INFO if args.verbose else logging.WARNING)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
