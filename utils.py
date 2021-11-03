from typing import Callable, Iterable

class PerformanceBenchmark:
    def __init__(self, pipeline: Callable, dataset: Iterable, name: str = "baseline"):
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
        metrics = {'name' self.name}
        metrics['size'] = self.compute_size()
        metrics['latency'].update(self.time_pipeline())
        metrics['accuracy'].update(self.compute_accuracy())
        return metrics