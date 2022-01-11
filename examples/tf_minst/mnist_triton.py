import numpy as np
import tensorflow as tf
import tritonclient.http
from scipy.special import softmax


# setup the server endpoint and models
model_name = "mnist"
url = "127.0.0.1:8000"
model_version = "1"
batch_size = 1

# load dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# take a example
label_example = test_labels[1]
image_example = test_images[1].reshape(-1, 28 * 28) / 255.0

# verify if the model is ready to consume
triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)
assert triton_client.is_model_ready(
    model_name=model_name, model_version=model_version
), f"model {model_name} not yet ready"

# Define the input/output
model_input = tritonclient.http.InferInput(name="dense_input", shape=(batch_size, 784), datatype="FP32")
model_output = tritonclient.http.InferRequestedOutput(name="dense_1", binary_data=False)

# set the data and call the model
model_input.set_data_from_numpy(image_example.astype(np.float32))
response = triton_client.infer(
    model_name=model_name, model_version=model_version, inputs=[model_input], outputs=[model_output]
)

output = response.as_numpy("dense_1")
class_pred = np.argmax(softmax(output))

print(f"Model predicted {class_pred}, truth is {label_example}!")
