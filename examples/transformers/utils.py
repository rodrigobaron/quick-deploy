from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
from datasets import load_dataset, load_metric
from scipy.special import softmax
from transformers import pipeline

from quick_deploy.backend.common import create_model_for_provider
from quick_deploy.benchmark import PerformanceBenchmark


class LMPerformanceBenchmark(PerformanceBenchmark):
    def compute_accuracy(self):
        return {}

    def time_pipeline(self):
        model_inputs = next(iter(self.dataset))["text"]
        latencies = []
        _ = self.pipeline(model_inputs)
        for _ in range(100):
            start_time = perf_counter()
            _ = self.pipeline(model_inputs)
            latency = perf_counter() - start_time
            latencies.append(latency)
        time_avg_ms = 1000 * np.mean(latencies)
        time_std_ms = 1000 * np.std(latencies)
        print(f"Average latency (ms) - {time_avg_ms:.2f} +\\- {time_std_ms:.2f}")
        return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms}


class OnnxPipeline:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, query):
        model_inputs = self.tokenizer(query, return_tensors="pt")
        inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
        logits = self.model.run(None, inputs_onnx)[0][0, :]
        probs = softmax(logits)
        pred_idx = np.argmax(probs).item()
        return [{"label": f"{pred_idx}", "score": probs[pred_idx]}]


class OnnxPerformanceBenchmark(LMPerformanceBenchmark):
    def __init__(
        self,
        pipeline: Callable,
        dataset: Callable,
        model_path: Path,
        name: str = "baseline",
    ):
        super().__init__(pipeline, dataset, name)
        self.model_path = model_path

    def compute_size(self):
        size_mb = Path(self.model_path).stat().st_size / (1024 * 1024)
        print(f"Model size (MB) - {size_mb:.2f}")
        return {"size_mb": size_mb}
