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
