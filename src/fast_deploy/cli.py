import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from fast_deploy.backend.common import (
    WeightType,
    create_model_for_provider,
    generic_optimize_onnx,
)
from fast_deploy.backend.transformers_ort import (
    transformers_convert_pytorch,
    transformers_optimize_onnx,
)
from fast_deploy.templates.transformer_triton import TransformersConfiguration
from fast_deploy.utils import get_provider, parse_transformer_torch_input, setup_logging


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

    transformers_convert_pytorch(model=pipe_model, output_path=onnx_model_path, inputs_pytorch=inputs_pytorch)

    onnx_model = create_model_for_provider(path=onnx_model_path, provider_to_use=provider_to_use)
    output_onnx = onnx_model.run(None, inputs_onnx)
    assert np.allclose(a=output_onnx, b=output_pytorch, atol=1e-1)

    transformers_optimize_onnx(
        onnx_path=onnx_model_path,
        output_path=onnx_optim_model_path,
        model_family=args.tran_family,
        use_cuda=False,
    )
    generic_optimize_onnx(
        onnx_path=onnx_optim_model_path,
        output_path=onnx_optim_model_path,
        weight_type=WeightType.from_str(args.weight_type),
    )
    onnx_model = create_model_for_provider(path=onnx_optim_model_path, provider_to_use=provider_to_use)

    # output_onnx_optimised = onnx_model.run(None, inputs_onnx)
    # assert np.allclose(a=output_onnx_optimised, b=output_pytorch, atol=7e-1)

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


def default_args(parser):
    parser.add_argument("-n", "--name", required=True, help="model name")
    parser.add_argument("-m", "--model", required=True, help="model path")
    parser.add_argument("-o", "--output", required=True, help="path used to export models")
    parser.add_argument("-w", "--workdir", default="env/", help="model path")
    parser.add_argument("--nb-instances", default=1, help="# of model instances", type=int)
    parser.add_argument("--cuda", action="store_true", help="use cuda optimization")
    parser.add_argument("-v", "--verbose", action="store_true", help="display detailed information")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize and deploy machine learning models fast as possible!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(help="transformers help", dest="transformes")
    subparsers.required = True

    parser_tra = subparsers.add_parser("transformers")
    default_args(parser_tra)
    parser_tra.add_argument("-t", "--tokenizer", help="tokenizer path")
    parser_tra.add_argument(
        "-f",
        "--tran-family",
        default="bert",
        help="transformer family. One of [bert, gpt2, t5]",
        nargs="*",
        choices=["bert", "gpt2", "t5"],
    )
    parser_tra.add_argument(
        "-p",
        "--pipeline",
        required=True,
        help="pipeline task, eg: 'text-classification'",
    )
    parser_tra.add_argument("-s", "--seq-len", default=16, help="sequence length to optimize")
    parser_tra.set_defaults(func=main_transformers)
    parser_tra.add_argument(
        "-c",
        "--weight-type",
        default="float16",
        help="Weight type. One of [int8, float16]",
        nargs="*",
        choices=["int8", "float16"],
    )

    args = parser.parse_args()
    setup_logging(level=logging.INFO if args.verbose else logging.WARNING)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
