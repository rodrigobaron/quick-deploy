from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
from datasets import load_dataset, load_metric
from scipy.special import softmax
from transformers import pipeline

from fast_deploy.backend.common import create_model_for_provider
from fast_deploy.benchmark import PerformanceBenchmark


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
        return {}
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


ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""


pipe = pipeline("summarization", "facebook/bart-base")
dataset = [{"text": ARTICLE}]

# lmpb = LMPerformanceBenchmark(pipe, dataset)
# lmpb.run_benchmark()

onnx_model_path = Path("env/transformer_my-bart-base.onnx").as_posix()
onnx_quantized_model_path = Path("env/transformer_my-bart-base.optim.onnx").as_posix()

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
