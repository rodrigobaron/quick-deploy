from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
from datasets import Dataset, load_metric


class PerformanceBenchmark:
    def __init__(self, pipeline: Callable, dataset: Dataset, name: str = "baseline"):
        self.pipeline = pipeline
        self.dataset = dataset
        self.name = name

    def compute_accuracy(self):
        pass

    def compute_size(self):
        pass

    def time_pipeline(self):
        pass

    def run_benchmark(self):
        metrics = {"name": self.name}
        metrics["size"] = self.compute_size()
        metrics["latency"] = self.time_pipeline()
        metrics["accuracy"] = self.compute_accuracy()
        return metrics


class LMPerformanceBenchmark(PerformanceBenchmark):
    def compute_accuracy(self):
        accuracy_score = load_metric("accuracy")
        intents = self.dataset.features["intent"]

        preds, labels = [], []
        for example in self.dataset:
            pred = self.pipeline(example["text"])[0]["label"]
            label = example["intent"]
            preds.append(intents.str2int(pred))
            labels.append(label)
        accuracy = accuracy_score.compute(predictions=preds, references=labels)
        print(f"Accuracy on test set - {accuracy['accuracy']:.3f}")
        return accuracy

    def time_pipeline(self):
        model_inputs = next(iter(self.dataset))["text"]
        latencies = []
        # Warmup
        for _ in range(10):
            _ = self.pipeline(model_inputs)
        # Timed run
        for _ in range(100):
            start_time = perf_counter()
            _ = self.pipeline(model_inputs)
            latency = perf_counter() - start_time
            latencies.append(latency)
        # Compute run statistics
        time_avg_ms = 1000 * np.mean(latencies)
        time_std_ms = 1000 * np.std(latencies)
        print(f"Average latency (ms) - {time_avg_ms:.2f} +\\- {time_std_ms:.2f}")
        return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms}


class OnnxPerformanceBenchmark(LMPerformanceBenchmark):
    def __init__(
        self,
        pipeline: Callable,
        dataset: Dataset,
        model_path: Path,
        name: str = "baseline",
    ):
        super().__init__(pipeline, dataset, name)
        self.model_path = model_path

    def compute_size(self):
        size_mb = Path(self.model_path).stat().st_size / (1024 * 1024)
        print(f"Model size (MB) - {size_mb:.2f}")
        return {"size_mb": size_mb}
