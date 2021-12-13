from fast_deploy.triton_template import transformers_configuration

from transformers import pipeline

from fast_deploy.utils import parse_transformer_torch_input
from fast_deploy.backend.transformers_ort import (
    transformers_convert_pytorch
)
from pathlib import Path
import torch
import numpy as np


pipe = pipeline("fill-mask", "bert-base-uncased")
pipe_tokenizer = pipe.tokenizer
pipe_model = pipe.model
dataset = [{"text": "The goal of life is [MASK]."}]

onnx_model_path = Path("env/transformer_my-bert-base.onnx").as_posix()
tokenizer_path = Path("env/tokenizer").as_posix()

inputs_pytorch, inputs_onnx = parse_transformer_torch_input(
    batch_size=1, seq_len=16, include_token_ids=True
)

with torch.inference_mode():
    output = pipe_model(**inputs_pytorch)
    output = output.logits
    output_pytorch: np.ndarray = output.detach().cpu().numpy()
    print(output_pytorch)

transformers_convert_pytorch(
    model=pipe_model, output_path=onnx_model_path, inputs_pytorch=inputs_pytorch, verbose=False
)

pipe_tokenizer.save_pretrained(str(tokenizer_path))

transformers_configuration(
    model_name="test_model",
    batch_size=0,
    nb_output_shape=[-1, output_pytorch.shape[-1]],
    nb_instance=1,
    include_token_type=True,
    workind_directory="/home/rodrigo/triton_test2",
    use_cuda=False,
    model_path=onnx_model_path,
    tokenize_path=tokenizer_path
)
