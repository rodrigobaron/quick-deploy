import numpy as np
import tritonclient.http


model_name = f"test_model_inferece"
url = "127.0.0.1:8000"
model_version = "1"
batch_size = 1

text = "The goal of life is [MASK]."

triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)
assert triton_client.is_model_ready(
    model_name=model_name, model_version=model_version
), f"model {model_name} not yet ready"

query = tritonclient.http.InferInput(name="TEXT", shape=(batch_size,), datatype="BYTES")
mask_token_index = tritonclient.http.InferInput(name="mask_token_index", shape=(batch_size,), datatype="BYTES")
model_score = tritonclient.http.InferRequestedOutput(name="mask_token", binary_data=False)

query.set_data_from_numpy(np.asarray([text] * batch_size, dtype=object))
mask_token_index.set_data_from_numpy(np.asarray([6] * batch_size, dtype=object))
response = triton_client.infer(
    model_name=model_name, model_version=model_version, inputs=[query], outputs=[model_score]
)

print(response.as_numpy("mask_token"))