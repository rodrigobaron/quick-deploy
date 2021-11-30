import sys
sys.path.insert(0, '../')
from collections import OrderedDict

from pathlib import Path
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline
)
from datasets import load_dataset

# from app.session import create_model_for_provider
from src.fast_deploy.utils import (
    LMPerformanceBenchmark,
    OnnxPerformanceBenchmark,
)

from typing import Dict, Tuple, List
from src.fast_deploy.pipeline import OnnxPipeline
# from fast_deploy.schema import ModelSchema
# from fast_deploy.convert import  convert_and_optimize
from src.fast_deploy.backend.torch_ort import convert_pytorch, create_model_for_provider, transformers_optimize_onnx, generic_optimize_onnx
from src.fast_deploy.templates.transformer_triton import Configuration


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
    
    # model_schema = ModelSchema(
    #     input_names = ['input_ids', 'attention_mask', 'token_type_ids'],
    #     input_shape = [(7,), (7,), (7,)],
    #     output_names = ['output_0'],
    #     output_shape = (1,),
    # )

    # model_path = Path('../env/bert.pt')
    # model_output = Path("../env/model.quant.onnx")

    # convert_and_optimize(model_path, model_output, model_schema)
    # onnx_quantized_model = create_model_for_provider(model_output)

    # intents = clinc["test"].features["intent"]
    # pipe = OnnxPipeline(onnx_quantized_model, bert_tokenizer, intents=intents)
    
    # pb = OnnxPerformanceBenchmark(pipe, clinc["test"], model_path = model_output, name = "Optimized")
    # perf_metrics = pb.run_benchmark()
    
    bert_model.eval()

    def prepare_input(
    seq_len: int, batch_size: int, include_token_ids: bool
) -> Tuple[Dict[str, torch.Tensor], Dict[str, np.ndarray]]:
        shape = (batch_size, seq_len)
        inputs_pytorch: OrderedDict[str, torch.Tensor] = OrderedDict()
        inputs_pytorch["input_ids"] = torch.randint(high=100, size=shape, dtype=torch.long)
        if include_token_ids:
            inputs_pytorch["token_type_ids"] = torch.ones(size=shape, dtype=torch.long)
        inputs_pytorch["attention_mask"] = torch.ones(size=shape, dtype=torch.long)
        inputs_onnx: Dict[str, np.ndarray] = {
            k: np.ascontiguousarray(v.detach().cpu().numpy()) for k, v in inputs_pytorch.items()
        }
        return inputs_pytorch, inputs_onnx
    
    input_names: List[str] = bert_tokenizer.model_input_names
    # logging.info(f"axis: {input_names}")
    include_token_ids = "token_type_ids" in input_names
    # tensor_shapes = list(zip(args.batch_size, args.seq_len))
    tensor_shapes = list(zip((1,1,1), (16,16,16)))
    inputs_pytorch, inputs_onnx = prepare_input(
        batch_size=tensor_shapes[-1][0], seq_len=tensor_shapes[-1][1], include_token_ids=include_token_ids
    )

    with torch.inference_mode():
        output = bert_model(**inputs_pytorch)
        output = output.logits  # extract the value of interest
        output_pytorch: np.ndarray = output.detach().cpu().numpy()

    # logging.info(f"[Pytorch] input shape {inputs_pytorch['input_ids'].shape}")
    # logging.info(f"[Pytorch] output shape: {output_pytorch.shape}")
    # create onnx model and compare results

    # model_path = Path('../env/bert.pt')
    onnx_model_path = Path("../env/model.onnx").as_posix()
    onnx_optim_fp16_path = Path("../env/model.quant.onnx").as_posix()

    convert_pytorch(model=bert_model, output_path=onnx_model_path, inputs_pytorch=inputs_pytorch)
    onnx_model = create_model_for_provider(path=onnx_model_path, provider_to_use='CPUExecutionProvider')
    output_onnx = onnx_model.run(None, inputs_onnx)
    assert np.allclose(a=output_onnx, b=output_pytorch, atol=1e-1)


    intents = clinc["test"].features["intent"]
    pipe = OnnxPipeline(onnx_model, bert_tokenizer, intents=intents)
    
    # pb = OnnxPerformanceBenchmark(pipe, clinc["test"], model_path = onnx_model_path, name = "Non Optimized")
    # perf_metrics = pb.run_benchmark()
    # print(perf_metrics)




    bert_optimize_onnx(
        onnx_path=onnx_model_path,
        output_path=onnx_optim_fp16_path,
        use_cuda=False,
    )
    generic_optimize_onnx(
        onnx_path=onnx_optim_fp16_path,
        output_path=onnx_optim_fp16_path
    )
    onnx_model = create_model_for_provider(path=onnx_optim_fp16_path, provider_to_use="CPUExecutionProvider")
    # run the model (None = get all the outputs)
    output_onnx_optimised = onnx_model.run(None, inputs_onnx)
    try:
        assert np.allclose(a=output_onnx_optimised, b=output_pytorch, atol=5e-1)
    except:
        import pdb; pdb.set_trace()
    # pipe = OnnxPipeline(onnx_model, bert_tokenizer, intents=intents)
    
    # pb = OnnxPerformanceBenchmark(pipe, clinc["test"], model_path = onnx_optim_fp16_path, name = "Optimized")
    # perf_metrics = pb.run_benchmark()
    # print(perf_metrics)


    conf = Configuration(
        model_name='transformer',
        batch_size=0,
        nb_output=output_pytorch.shape[1],
        nb_instance=1,
        include_token_type=include_token_ids,
        workind_directory='triton_models',
    )
    conf.create_folders(tokenizer=bert_tokenizer, model_path=onnx_optim_fp16_path)



    # docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models1
    # nvcr.io/nvidia/tritonserver:21.11-py3-min

    # docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /home/rodrigo/projects/fastdeploy/env:/models nvcr.io/nvidia/tritonserver:21.11-py3-min tritonserver --model-repository=/models