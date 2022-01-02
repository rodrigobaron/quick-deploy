import numpy as np
import tritonclient.http
from scipy.special import softmax

model_name = "resnet18"
url = "127.0.0.1:8000"
model_version = "1"
batch_size = 1

x = np.random.randn(1, 3, 224, 224)

triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)
assert triton_client.is_model_ready(
    model_name=model_name, model_version=model_version
), f"model {model_name} not yet ready"

model_input = tritonclient.http.InferInput(name="input", shape=(batch_size, 3, 224, 224), datatype="FP32")
model_output = tritonclient.http.InferRequestedOutput(name="output", binary_data=False)

model_input.set_data_from_numpy(x)
response = triton_client.infer(
    model_name=model_name, model_version=model_version, inputs=[model_input], outputs=[model_output]
)

output = response.as_numpy("output")
