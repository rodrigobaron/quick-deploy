# Quick-Deploy

<p align="center">
    <a href="https://github.com/rodrigobaron/quick-deploy/actions/workflows/build.yaml">
        <img alt="Build" src="https://github.com/rodrigobaron/quick-deploy/actions/workflows/build.yaml/badge.svg">
    </a>
    <a href="https://github.com/rodrigobaron/quick-deploy/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/rodrigobaron/quick-deploy.svg?color=blue">
    </a>
    <a href="https://github.com/rodrigobaron/quick-deploy/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/rodrigobaron/quick-deploy.svg">
    </a>
</p>

<h3 align="center">
    Optimize and deploy machine learning models fast and easy as possible.
</h3>

quick-deploy provide tools to optimize, convert and deploy machine learning models as fast inference API (low latency and high throughput) by [Triton Inference Server](https://github.com/triton-inference-server/server) using [Onnx Runtime](https://github.com/microsoft/onnxruntime) backend. It support 🤗 transformers, PyToch, Tensorflow, SKLearn and XGBoost models.


## Get Started

Let's see an quick example by deploying bert transformers for GPU inference. quick-deploy already have support 🤗 transformers so we can specify the path of pretrained model or just the name from the Hub:

```bash
$ quick-deploy transformers \
    -n my-bert-base \
    -p text-classification \
    -m bert-base-uncased \
    -o ./models \
    --model-type bert \
    --seq-len 128 \
    --cuda
```

The command above created the deployment artifacts by optimizing and converting the model to Onxx. Next just run the inference server:
```bash
$ docker run -it --rm \
    --gpus all \
    --shm-size 256m \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -v $(pwd)/models:/models nvcr.io/nvidia/tritonserver:21.11-py3 \
    tritonserver --model-repository=/models

```

Now we can use tritonclient which uses gRPC calls to consume our model:
```python
import numpy as np
import tritonclient.http
from scipy.special import softmax
from transformers import BertTokenizer, TensorType


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model_name = "my_bert_base"
url = "127.0.0.1:8000"
model_version = "1"
batch_size = 1

text = "The goal of life is [MASK]."
tokens = tokenizer(text=text, return_tensors=TensorType.NUMPY)

triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)
assert triton_client.is_model_ready(
    model_name=model_name, model_version=model_version
), f"model {model_name} not yet ready"

input_ids = tritonclient.http.InferInput(name="input_ids", shape=(batch_size, 9), datatype="INT64")
token_type_ids = tritonclient.http.InferInput(name="token_type_ids", shape=(batch_size, 9), datatype="INT64")
attention = tritonclient.http.InferInput(name="attention_mask", shape=(batch_size, 9), datatype="INT64")
model_output = tritonclient.http.InferRequestedOutput(name="output", binary_data=False)

input_ids.set_data_from_numpy(tokens['input_ids'] * batch_size)
token_type_ids.set_data_from_numpy(tokens['token_type_ids'] * batch_size)
attention.set_data_from_numpy(tokens['attention_mask'] * batch_size)

response = triton_client.infer(
    model_name=model_name,
    model_version=model_version,
    inputs=[input_ids, token_type_ids, attention],
    outputs=[model_output],
)

token_logits = response.as_numpy("output")
print(token_logits)
```

**Note:** This does only model deployment the tokenizer and post-processing should be done in the client side. The full tansformers deployment is comming soon.

For more use cases please check the [examples](examples) page.

## Install

Before install make sure to install just the target model eg.: "torch", "sklearn" or "all". There two options to use quick-deploy, by docker container:
```bash
$ docker run --rm -it rodrigobaron/quick-deploy:0.1.1-all --help
```

or install the python library `quick-deploy`:

```bash
$ pip install quick-deploy[all]
```

**Note:** This will install the full vesion `all`.

## Contributing

Please follow the [Contributing](CONTRIBUTING.md) guide.

## License

[Apache License 2.0](LICENSE)
