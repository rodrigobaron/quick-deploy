import numpy as np
import tritonclient.http

from sklearn.datasets import load_iris
import random


# load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# pick a random example
example_idx = random.randint(0, len(X))
example_X = X[example_idx]
example_y = y[example_idx]

# setup the server endpoint and models
model_name = "iris_cls"
url = "127.0.0.1:8000"
model_version = "1"
batch_size = 1

# verify if the model is ready to consume
triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)

assert triton_client.is_model_ready(
    model_name=model_name, model_version=model_version
), f"model {model_name} not yet ready"

# Define the input/output
model_input = tritonclient.http.InferInput(name="input", shape=(batch_size, 4), datatype="FP32")
model_label = tritonclient.http.InferRequestedOutput(name="label", binary_data=False)
model_proba = tritonclient.http.InferRequestedOutput(name="probabilities", binary_data=False)

# set the data and call the model
model_input.set_data_from_numpy(np.array([example_X]).astype(np.float32))
response = triton_client.infer(
    model_name=model_name, model_version=model_version, inputs=[model_input], outputs=[model_label, model_proba]
)

# get model results
label_pred = response.as_numpy("label")[0]
probs = response.as_numpy("probabilities")[0]
label_prob = probs[label_pred]

print(f"Model predicted {label_pred}: with score of {label_prob}, the truth is {example_y}!")
