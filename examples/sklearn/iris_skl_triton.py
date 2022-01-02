import numpy as np
import tritonclient.http
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

model_name = "iris_cls"
url = "127.0.0.1:8000"
model_version = "1"
batch_size = 1

triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)
assert triton_client.is_model_ready(
    model_name=model_name, model_version=model_version
), f"model {model_name} not yet ready"

model_input = tritonclient.http.InferInput(name="input", shape=(batch_size, 4), datatype="FP32")
model_label = tritonclient.http.InferRequestedOutput(name="label", binary_data=False)
model_proba = tritonclient.http.InferRequestedOutput(name="probabilities", binary_data=False)

model_input.set_data_from_numpy(np.array([X_test[0]]).astype(np.float32))
response = triton_client.infer(
    model_name=model_name, model_version=model_version, inputs=[model_input], outputs=[model_label, model_proba]
)

output1 = response.as_numpy("label")
output2 = response.as_numpy("probabilities")
print(output1)
print(output2)
