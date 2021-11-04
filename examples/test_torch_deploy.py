import sys
sys.path.insert(0, '../')

from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline
)
from datasets import load_dataset

from app.session import create_model_for_provider
from fast_deploy.utils import (
    LMPerformanceBenchmark,
    OnnxPerformanceBenchmark,
)
from fast_deploy.pipeline import OnnxPipeline
from fast_deploy.schema import ModelSchema
from fast_deploy.convert import  convert_and_optimize


if __name__ == '__main__':
    bert_ckpt = "transformersbook/bert-base-uncased-finetuned-clinc"
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_ckpt)
    bert_model = (AutoModelForSequenceClassification
                .from_pretrained(bert_ckpt).to("cpu"))
    pipe = TextClassificationPipeline(model=bert_model, tokenizer=bert_tokenizer)
    clinc = load_dataset("clinc_oos", "plus")

    # pb = LMPerformanceBenchmark(pipe, clinc["test"])
    # perf_metrics = pb.run_benchmark()
    # print(perf_metrics)
    
    model_schema = ModelSchema(
        input_names = ['input_ids', 'attention_mask', 'token_type_ids'],
        input_shape = [(7,), (7,), (7,)],
        output_names = ['output_0'],
        output_shape = (1,),
    )

    model_path = Path('../env/bert.pt')
    model_output = Path("../env/model.quant.onnx")

    convert_and_optimize(model_path, model_output, model_schema)
    onnx_quantized_model = create_model_for_provider(model_output)

    intents = clinc["test"].features["intent"]
    pipe = OnnxPipeline(onnx_quantized_model, bert_tokenizer, intents=intents)
    
    pb = OnnxPerformanceBenchmark(pipe, clinc["test"], model_path = model_output, name = "Optimized")
    perf_metrics = pb.run_benchmark()
