from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
from datasets import load_dataset, load_metric
from scipy.special import softmax
from transformers import pipeline
from utils import (
    LMPerformanceBenchmark,
    OnnxPipeline,
    OnnxPerformanceBenchmark
)

pipe = pipeline("text-classification", "bert-base-uncased")
dataset = [{"text": "The goal of life is [MASK]."}]

lmpb = LMPerformanceBenchmark(pipe, dataset)
lmpb.run_benchmark()

onnx_model_path = Path("env/transformer_my-bert-base.onnx").as_posix()
onnx_quantized_model_path = Path("env/transformer_my-bert-base.optim.onnx").as_posix()

onnx_model = create_model_for_provider(onnx_model_path, provider_to_use="CPUExecutionProvider")
pipe_onnx = OnnxPipeline(onnx_model, pipe.tokenizer)
pb_onnx = OnnxPerformanceBenchmark(pipe_onnx, dataset, model_path=onnx_model_path, name="Onnx")
pb_onnx.run_benchmark()

onnx_quantized_model = create_model_for_provider(onnx_quantized_model_path, provider_to_use="CPUExecutionProvider")
pipe_quant_onnx = OnnxPipeline(onnx_quantized_model, pipe.tokenizer)
pb_quant_onnx = OnnxPerformanceBenchmark(
    pipe_quant_onnx, dataset, model_path=onnx_quantized_model_path, name="Optimized Onnx"
)
pb_quant_onnx.run_benchmark()
