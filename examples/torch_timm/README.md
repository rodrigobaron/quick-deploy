# Torch ImageNet Classifier

In this example we'll cover the deployment of an `Resnet18` for the classical `ImagetNet` dataset to demonstrate how easy is move machine learning model to a production ready soluction using **Quick-Deploy**.

## Install

Instal the dependencies:
```bash
$ pip install quick-deploy[torch] timm torchvision Pillow tritonclient geventhttpclient 
```

## Model Training

First of all we need download a pretrained model ([timm_resnet18_download.py](timm_resnet18_download.py)):

```python
import timm;
import torch;

# load pretrained model and save
model = timm.create_model('resnet18', pretrained=True)
torch.save(model, 'resnet18.pt')

```

## Model Definition

Before move to production we need supply the IO definition ([resnet18.yaml](resnet18.yaml)), we know the model expect a 224x224 channel first image input and 1000 classes:

```yaml
kind: IOSchema
inputs:
  - name: input
    dtype: float32
    shape: [3, 224, 224]
outputs:
  - name: output
    shape: [1000]
    dtype: float32
```

## Deploy

Now with the model trained `resnet18.pt` and with the IO definition `resnet18.yaml` we can use **Quick-Deploy**:

```bash
$ quick-deploy torch \
    -n resnet18 \
    -m resnet18.pt \
    -o ./models \
    -f resnet18.yaml \
    --no-quant
```

The arguments is prety straightforward. Now we run the `triton inference sever` ([run_inference_server.sh](run_inference_server.sh)), in this example using docker:

## Spin Up the Server 

```bash
$ docker run -it --rm \
    --shm-size 256m \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -v $(pwd)/models:/models nvcr.io/nvidia/tritonserver:21.11-py3 \
    tritonserver --model-repository=/models
```

With the server running we can consume our model using `tritonclient` ([timm_resnet18_triton.py](timm_resnet18_triton.py)):

## Consume the Model

```python
import numpy as np
import json
from PIL import Image
import torchvision.transforms as transforms

import tritonclient.http
from scipy.special import softmax

# setup the server endpoint and models
model_name = "resnet18"
url = "127.0.0.1:8000"
model_version = "1"
batch_size = 1

# load image and apply the pre-processing transformations
image = Image.open("image.png")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

transformations = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
]

for t in transformations:
    image = t(image)
image = image.numpy().reshape((1,) + image.shape)

# verify if the model is ready to consume
triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)
assert triton_client.is_model_ready(
    model_name=model_name, model_version=model_version
), f"model {model_name} not yet ready"

# Define the input/output
model_input = tritonclient.http.InferInput(name="input", shape=(batch_size, 3, 224, 224), datatype="FP32")
model_output = tritonclient.http.InferRequestedOutput(name="output", binary_data=False)

# set the data and call the model
model_input.set_data_from_numpy(image)
response = triton_client.infer(
    model_name=model_name, model_version=model_version, inputs=[model_input], outputs=[model_output]
)

output = response.as_numpy("output")
cls_pred = np.argmax(softmax(output))

# check the class list
with open('imagenet1000_clsidx_to_labels.txt', 'r') as f:
    cls_list = json.load(f)

print(f"Image prediction: {cls_list[str(cls_pred)]}!")

```

Example display:  

![Image](image.png)

```
Image prediction: golden retriever!
```
